import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
df.describe()
f,ax = plt.subplots(1,3,figsize=(15,6))

ax[0].imshow(test.iloc[0].reshape(28,28),cmap='binary')

ax[1].imshow(test.iloc[1].reshape(28,28),cmap='binary')

ax[2].imshow(test.iloc[2].reshape(28,28),cmap='binary')
#one-hot formula

np.array([np.array([int(i == label) for i in range(10)]) for label in [5,2,3,9]])

#basically generate a numpy array for all elements (5,2,and 3), then doing a loop of 10 loops

#for 10 loops, do a check if current i == label, return 0 if false, 1 if true



#this will create a one-hot encoded number
labels_encoded = np.array([np.array([int(i == label) for i in range(10)]) for label in df.iloc[:,0].values])
dataset = df.drop('label',axis=1)

# convert from [0:255] => [0.0:1.0]

dataset = np.multiply(dataset.values.astype(np.float32), 1.0 / 255.0)

test = np.multiply(test.values.astype(np.float32), 1.0 / 255.0)



dataset.shape,labels_encoded.shape
train_size = 40000

validation_size = 2000
train = dataset[:train_size]

train_targets = labels_encoded[:train_size]

validation = dataset[train_size:]

validation_targets = labels_encoded[train_size:]



train.shape, train_targets.shape, validation.shape, validation_targets.shape, test.shape 
input_size = 784

output_size = 10

hidden_layer_size = 2000



tf.reset_default_graph()
#set the input and targets placeholders

inputs = tf.placeholder(tf.float32,[None,input_size])

targets = tf.placeholder(tf.float32,[None,output_size])
#setting the 1st layer

w_1 = tf.get_variable('w_1',[input_size,hidden_layer_size])

b_1 = tf.get_variable('b_1',[hidden_layer_size])



#to deal with non-linearity, activation function must be applied on every outputs

#I use relu on this model, you can choose another function like sigmoid,tanH,or softmax

o_1 = tf.nn.relu(tf.matmul(inputs,w_1) + b_1)
#setting the 2nd layer

w_2 = tf.get_variable('w_2',[hidden_layer_size,hidden_layer_size])

b_2 = tf.get_variable('b_2',[hidden_layer_size])



#still using relu

o_2 = tf.nn.relu(tf.matmul(o_1,w_2) + b_2)
#setting the 3rd layer

w_3 = tf.get_variable('w_3',[hidden_layer_size,hidden_layer_size])

b_3 = tf.get_variable('b_3',[hidden_layer_size])



#still using relu

o_3 = tf.nn.relu(tf.matmul(o_2,w_3) + b_3)
#setting the 4th layer

w_4 = tf.get_variable('w_4',[hidden_layer_size,hidden_layer_size])

b_4 = tf.get_variable('b_4',[hidden_layer_size])



#still using relu

o_4 = tf.nn.relu(tf.matmul(o_3,w_4) + b_4)
#setting the 5th layer

w_5 = tf.get_variable('w_5',[hidden_layer_size,hidden_layer_size])

b_5 = tf.get_variable('b_5',[hidden_layer_size])



#still using relu

o_5 = tf.nn.relu(tf.matmul(o_4,w_5) + b_5)
#setting the outputs layer

w_6 = tf.get_variable('w_6',[hidden_layer_size,output_size])

b_6 = tf.get_variable('b_6',[output_size])



#for the last part, I don't use any activation function, later I'll use softmax on it

outputs = tf.matmul(o_5,w_6)+ b_6
loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=targets)

mean_loss = tf.reduce_mean(loss)
optimize = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(mean_loss)



#argmax simply return the index of the outputs to be compare with the targets

#example(output = [0,0,1] will return 2, targets = [0,0,1] return 2, so the result = True or 1)

#for all samples, let's say 60 out of 100 is true, so the accuracy = 60/100 = 60%

out_equal_target = tf.equal(tf.argmax(outputs,1),tf.argmax(targets,1))

accuracy = tf.reduce_mean(tf.cast(out_equal_target,tf.float32))
sess = tf.InteractiveSession()

initializer = tf.global_variables_initializer()

sess.run(initializer)
#to make the learning process running in timely manner, batching is a good practice.

batch_size = 150

batch_number = train.shape[0]//batch_size

max_epoch = 15

prev_validation_loss = 9999999.
for epoch_counter in range(max_epoch):

    curr_epoch_loss = 0

    start = 0

    end = start+batch_size

    

    #batch training

    for batch_counter in range(batch_number):

        #set the input and target batch equals to defined size

        input_batch = train[start:end]

        target_batch = train_targets[start:end]

        start = end

        end = start+batch_size

        

        #running optimizer, feeding the model with current batch dataset

        #the model will continously set the weight and biases with forward and back propagation

        _, batch_loss = sess.run([optimize,mean_loss],

                                 feed_dict={inputs:input_batch, targets:target_batch}                                )

        curr_epoch_loss += batch_loss

    

    curr_epoch_loss /= batch_number

    

    #validation, forward propagate only, to see the accuracy on the model using new dataset     

    val_loss, val_accuracy = sess.run([mean_loss,accuracy],

                                 feed_dict={inputs:validation, targets:validation_targets})

    

    print('Epoch '+str(epoch_counter+1)+

         '. Training loss: '+'{0:.3f}'.format(curr_epoch_loss)+

         '. Validation loss: '+'{0:.3f}'.format(val_loss)+

         '. Validation accuracy: '+'{0:.2f}'.format(val_accuracy*100.)+'%')

    

    #the rule to prevent overfitting

    #1st, we already set max epoch to prevent the model continously iterating causing overfit

    #2nd, if validation loss starts increasing, we need to stop learning

    if val_loss > prev_validation_loss:

        break

    

    prev_validation_loss = val_loss

print('End of Training')
predict = tf.argmax(tf.nn.softmax(outputs),1)
#forward propagate using trained model to get prediction results

predictions = predict.eval(feed_dict={inputs: test})
submission = pd.DataFrame({

    'ImageId': range(1,len(predictions)+1),

    'Label': predictions

})
submission.head(3)
submission.to_csv('submission.csv',index=False)