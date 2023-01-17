import pandas as pd

import numpy as np

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import os 
column_names = ['buying', 'maint', 'doors','persons','lug_boot','safety','class']

data = pd.read_csv("/kaggle/input/car-classification/Car Classification.txt",names=column_names)

data.head()
data_vars = data.columns.values.tolist()

Y = ['class']

X = [v for v in data_vars if v not in Y]



x_predictors= data[X]

y_target = data[Y]



from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 

x_buying = le.fit_transform(x_predictors['buying'])

x_maint = le.fit_transform(x_predictors['maint'])

x_lugboot = le.fit_transform(x_predictors['lug_boot'])

x_safety = le.fit_transform(x_predictors['safety'])

y_target = le.fit_transform(y_target.values.ravel()) 
data['buying'] = x_buying

data['maint'] = x_maint

data['lug_boot'] = x_lugboot

data['safety'] = x_safety

data['class'] = y_target

data["doors"] = [ 5 if x=="5more" else x for x in data["doors"]]

data["persons"] = [ 5 if x=="more" else x for x in data["persons"]]
data
def get_target_vector(car_class):

    if(car_class==0):

        return [1.0,0.0,0.0,0.0]

    

    if(car_class==1):

        return [0.0,1.0,0.0,0.0]

    

    if(car_class==2):

        return [0.0,0.0,1.0,0.0]

    

    if(car_class==3):

        return [0.0,0.0,0.0,1.0]

    

    return [0.0,0.0,0.0,0.0]



def get_vectors_list(targets):

    targets_vector_list = [] 

    for target in targets:

        target_in_vector = get_target_vector(target)

        targets_vector_list.append(target_in_vector)

        

    return targets_vector_list
from sklearn.model_selection import train_test_split
data_vars = data.columns.values.tolist()

Y = ['class']

X = [v for v in data_vars if v not in Y]

X_train, X_test, Y_train, Y_test = train_test_split(data[X],data[Y],test_size = 0.3, random_state=0)
#Init variables

n_input = 6 # input layer (6 inputs )

n_hidden1 = 60  # 1st hidden layer

n_hidden2 = 30  # 2nd hidden layer

n_hidden3 = 15  # 3rd hidden layer

n_output = 4  # output layer (4 cars classes)



# Hyper parameters

learning_rate = 1e-4

n_iterations = 15000

batch_size = 128

dropout = 0.5



# NN Input and Target Values

X = tf.placeholder("float", [None, n_input])

Y = tf.placeholder("float", [None, n_output])

keep_prob = tf.placeholder(tf.float32)



weights = {

    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),

    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),

    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),

    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),

}



biases = {

    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),

    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),

    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),

    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))

}



# Define the NN layers 

layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])

layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])

layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])

layer_drop = tf.nn.dropout(layer_3, keep_prob)

output_layer = tf.matmul(layer_3, weights['out']) + biases['out']



cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
for i in range(n_iterations):

    

    rand_idx = np.random.choice(len(X_train), size=batch_size)

    X_train_array = X_train.to_numpy()

    Y_train_array = Y_train.to_numpy()

    rand_x = X_train_array[rand_idx]

    rand_y = Y_train_array[rand_idx]

    rand_y_in_vectors = get_vectors_list(rand_y)

    

    sess.run(train_step, feed_dict={X: rand_x, Y: rand_y_in_vectors, keep_prob: dropout})



    # print loss and accuracy

    if i % 100 == 0:

        minibatch_loss, minibatch_accuracy = sess.run(

            [cross_entropy, accuracy],

            feed_dict={X: rand_x, Y: rand_y_in_vectors, keep_prob: 1.0}

            )

        print(

            "Iteration",

            str(i),

            "\t| Loss =",

            str(minibatch_loss),

            "\t| Accuracy =",

            str(minibatch_accuracy)

            )

        
# Check accuracy of the NN with the test data

rand_idx = np.random.choice(len(X_train), size=batch_size)

X_test_array = X_test.to_numpy()

Y_test_array = Y_test.to_numpy()

rand_y_in_vectors = get_vectors_list(Y_test_array)

test_accuracy = sess.run(accuracy, feed_dict={X: X_test_array, Y: rand_y_in_vectors, keep_prob: 1.0})

print("\nAccuracy on test set:", test_accuracy)