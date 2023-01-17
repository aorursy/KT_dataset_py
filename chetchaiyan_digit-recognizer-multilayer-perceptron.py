import pandas as pd
import tensorflow as tf

train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')
print("Train Dataset")
print(train_dataset[:5])
print("Test Dataset")
print(test_dataset[:5])
print("Train Size : {}".format(train_dataset.shape[0]))
print("Test Size : {}".format(test_dataset.shape[0]))
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer

validation_ratio = 0.1

pixel_scaler = MinMaxScaler()
one_hot = LabelBinarizer()
one_hot.fit(range(10))
train_y = one_hot.transform(train_dataset['label'])
train_x = train_dataset.drop('label', axis=1)
train_x = pixel_scaler.fit_transform(train_x)
test_x = pixel_scaler.transform(test_dataset)

split = int(train_dataset.shape[0] * (1 - validation_ratio))

valid_x = train_x[split:train_x.shape[0]]
valid_y = train_y[split:train_x.shape[0]]
train_x = train_x[0:split]
train_y = train_y[0:split]
print("Train Size : {}".format(train_x.shape[0]))
print("Validation Size : {}".format(valid_x.shape[0]))
class DigitRecognizerMLP(object):
    def __init__(self, feature_size=784, label_size=10):
        self.feature_size = feature_size
        self.label_size = label_size
    
    def build_input(self):
        model_x = tf.placeholder(tf.float32, [None, self.feature_size])
        model_y = tf.placeholder(tf.int32, [None, self.label_size])
        model_lr = tf.placeholder(tf.float32)
        return model_x, model_y, model_lr
    
    def build_mlp(self, model_x, model_y):
        hidden_1 = tf.layers.dense(model_x, 1024, activation=tf.nn.relu)
        hidden_2 = tf.layers.dense(hidden_1, 512, activation=tf.nn.relu)
        hidden_3 = tf.layers.dense(hidden_2, 256, activation=tf.nn.relu)
        hidden_4 = tf.layers.dense(hidden_2, 128, activation=tf.nn.relu)
        logits = tf.layers.dense(hidden_3, self.label_size, activation=None)
        return logits
    
    def build_output(self, model_y, logits):
        output = tf.argmax(logits, axis=1)
        loss = tf.losses.softmax_cross_entropy(model_y, logits)
        accuracy = tf.reduce_mean( tf.cast( tf.equal( tf.argmax(model_y, axis=1), output ), tf.float32) )
        return output, loss, accuracy
    
    def model_opt(self, model_lr, loss):
        opt = tf.train.AdamOptimizer(model_lr).minimize(loss)
        return opt
    
    def make_nn(self):
        tf.reset_default_graph()
        self.model_x, self.model_y, self.model_lr = self.build_input()
        self.logits = self.build_mlp(self.model_x, self.model_y)
        self.output, self.loss, self.accuracy = self.build_output(self.model_y, self.logits)
        self.opt = self.model_opt(self.model_lr, self.loss)
        
    def train(self, epoch, learning_rate, train_x, train_y, batch_size, valid_x, valid_y, get_batch_func, print_every):
        t_loss_list = []
        v_loss_list = []
        accuracy_list = []
        saver = tf.train.Saver()
        with tf.Session() as sess :
            sess.run(tf.global_variables_initializer())
            counter = 0
            for e in range(epoch) :
                for x,y in get_batch_func(train_x, train_y, batch_size):
                    feed_dict = {self.model_x:x, self.model_y:y, self.model_lr:learning_rate}
                    loss, _ = sess.run([self.loss, self.opt], feed_dict=feed_dict)
                    if counter % print_every == 0 :
                        feed_dict = {self.model_x:valid_x, self.model_y:valid_y, self.model_lr:learning_rate}
                        v_loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                        print("Epoch: {}/{}, Step: {}, T Loss: {:.4f}, V Loss: {:.4f}, Accuracy: {:.4f}".format(e+1, 
                                    epoch, counter, loss, v_loss, accuracy))
                        t_loss_list.append( loss )
                        v_loss_list.append( v_loss )
                        accuracy_list.append( accuracy )
                        save_path = saver.save(sess, "tensor/model.ckpt")
                    counter += 1
        return t_loss_list, v_loss_list, accuracy_list
    
    def predict(self, predict_x):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "tensor/model.ckpt")
            feed_dict = {self.model_x:predict_x}
            output = sess.run(self.output, feed_dict=feed_dict)
            return output
# batch create function
def get_batch(feature, label, batch_size):
    num_batch = feature.shape[0] // batch_size
    for i in range(num_batch):
        start = i * batch_size
        end = (i + 1) * batch_size
        x = feature[start:end]
        y = label[start:end]
        yield x, y
learning_rate = 0.0001
batch_size = 64
epoch = 50
print_every = 1000

mlp = DigitRecognizerMLP()
mlp.make_nn()
t_loss, v_loss, acc = mlp.train(epoch, learning_rate, train_x, train_y, batch_size, valid_x, valid_y, get_batch, print_every)
mlp = DigitRecognizerMLP()
mlp.make_nn()
predict = mlp.predict(test_x)
import csv

with open('predict.csv', 'w') as csv_file :
    writer = csv.writer(csv_file)
    writer.writerow(['ImageId', 'Label'])
    for i in range(len(predict)) :
        writer.writerow([str(i+1), str(predict[i])])
    csv_file.close()
