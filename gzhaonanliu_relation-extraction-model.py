import pickle

import tensorflow as tf

from tensorflow import data as tfdata

from tensorflow.keras import Model,layers



label_train = pickle.load(open('../input/relation-extraction-data-preprocessing/label_train.pkl','rb'))

sen_train = pickle.load(open('../input/relation-extraction-data-preprocessing/sen_train.pkl','rb'))

ent_train = pickle.load(open('../input/relation-extraction-data-preprocessing/ent_train.pkl','rb'))

mask_train = pickle.load(open('../input/relation-extraction-data-preprocessing/mask_train.pkl','rb'))

label_test = pickle.load(open('../input/relation-extraction-data-preprocessing/label_test.pkl','rb'))

sen_test = pickle.load(open('../input/relation-extraction-data-preprocessing/sen_test.pkl','rb'))

ent_test = pickle.load(open('../input/relation-extraction-data-preprocessing/ent_test.pkl','rb'))

mask_test = pickle.load(open('../input/relation-extraction-data-preprocessing/mask_test.pkl','rb'))
batch_size = 128

drop_rate = 0.5

learning_rate = 0.008

display_step = 100

num_classes = 19

filter_nums = 128

filter_size = 2

pool_size = 9

hidden_size = 300

training_steps = 5000
# Create TF Model.

class AttenConv(Model):

    # Set layers.

    def __init__(self):

        super(AttenConv, self).__init__()

        self.conv1 = layers.Conv1D(filter_nums, filter_size,2, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l1(0.01))

        self.maxpool1 = layers.MaxPool1D(pool_size)

        self.flatten = layers.Flatten()

        # Fully connected layer.

        self.fc1 = layers.Dense(hidden_size,kernel_regularizer=tf.keras.regularizers.l1(0.01))

        # Apply Dropout (if is_training is False, dropout is not applied).

        self.dropout = layers.Dropout(rate=drop_rate)

        # Output layer, class prediction.

        self.out = layers.Dense(num_classes)



    # Set forward pass.

    def call(self,sen,mask,w1,w2,is_training=False):

        atten_w1 = layers.Attention()(inputs=[w1,sen])

        atten_w2 = layers.Attention()(inputs=[w2,sen])

#         atten_w1 = layers.Attention()(inputs=[w1,sen],mask=[tf.fill([sen.shape[0],1],True),mask])

#         atten_w2 = layers.Attention()(inputs=[w2,sen],mask=[tf.fill([sen.shape[0],1],True),mask])

        concat = tf.concat([atten_w1,atten_w2,sen],1) # (20, 92, 768)

#         concat = tf.concat([atten_w1,atten_w2,tf.einsum('ij,ijk->ijk',tf.cast(mask,tf.float32),sen)],1) # (20, 92, 768)

        x = self.conv1(concat) # (20, (92-filter_size+1)/2, filter_nums)

        x = self.maxpool1(x) # (20, (92-filter_size+1)/2/pool_size, filter_nums)

        x = self.flatten(x) # (20, filter_nums*(92-filter_size+1)/2/pool_size)

        x = self.fc1(x) # (20, hidden_size)

        x = self.dropout(x, training=is_training) # training进行drop，否则不drop

        x = self.out(x) # (20, num_classes)

        x = tf.nn.softmax(x) # (20, num_classes)

        return x

    

    
# Build neural network model.

net = AttenConv()

# Cross-Entropy Loss.

cce = tf.keras.losses.CategoricalCrossentropy()

# Stochastic gradient descent optimizer.

optimizer = tf.optimizers.Adagrad(learning_rate)

# optimizer = tf.optimizers.SGD(learning_rate)



# Optimization process. 

def run_optimization(sen,mask,w1,w2,label):

    # Wrap computation inside a GradientTape for automatic differentiation.

    with tf.GradientTape() as g:

        # Forward pass.

        pred = net(sen,mask,w1,w2,is_training=True)

        # Compute loss.

        loss = cce(label,pred)

        

    gradients = g.gradient(loss, net.trainable_variables)

    optimizer.apply_gradients(zip(gradients, net.trainable_variables))

    return loss



# Accuracy metric.

def metrics(y_pred, y_true):

    from sklearn.metrics import f1_score,recall_score,accuracy_score

    # Predicted class is the index of highest score in prediction vector (i.e. argmax).

    y_pred,y_true = tf.argmax(y_pred, 1), tf.argmax(y_true, 1)

    return accuracy_score(y_true,y_pred),recall_score(y_true,y_pred,average='macro'),f1_score(y_true,y_pred,average='macro')
train_data = tfdata.Dataset.from_tensor_slices((sen_train,mask_train,ent_train,label_train))

train_data = train_data.shuffle(1000).repeat().batch(batch_size).prefetch(1)
for step,(sen,mask,entity,label) in enumerate(train_data.take(training_steps), 1):

    w1,w2 =tf.split(entity,2,1)

    loss = run_optimization(sen,mask,w1,w2,label)

    if step % display_step == 0:

        w1_t,w2_t =tf.split(ent_test,2,1)

        pred = net(sen_test,mask_test,w1_t,w2_t)

        acc,rec,f1 = metrics(pred, label_test)

        print("step: %i, training_loss: %f, accuracy: %f, recall: %f, f1: %f" % (step, loss, acc,rec,f1))