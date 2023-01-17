# Imports

import tensorflow as tf

import numpy as np

from numpy import genfromtxt



# Taking iris data from sklearn

from sklearn import datasets

# 

#

from sklearn.cross_validation import train_test_split

import sklearn
# Writing out data to the following files:

#    cs-training.csv

#    cs-testing.csv

def buildDataFromIris():

    iris = datasets.load_iris()

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

    f=open('cs-training.csv','w')

    for i,j in enumerate(X_train):

        k=np.append(np.array(y_train[i]),j   )

        f.write(",".join([str(s) for s in k]) + '\n')

    f.close()

    f=open('cs-testing.csv','w')

    for i,j in enumerate(X_test):

        k=np.append(np.array(y_test[i]),j   )

        f.write(",".join([str(s) for s in k]) + '\n')

    f.close()

    

    

# Convert to one hot

def convertOneHot(data):

    y=np.array([int(i[0]) for i in data])

    y_onehot=[0]*len(y)

    for i,j in enumerate(y):

        y_onehot[i]=[0]*(y.max() + 1)

        y_onehot[i][j]=1

    return (y,y_onehot)





# Let's trying building

buildDataFromIris()

# Reading in the data

data = genfromtxt('cs-training.csv',delimiter=',')  # Training data

test_data = genfromtxt('cs-testing.csv',delimiter=',')  # Test data





x_train=np.array([ i[1::] for i in data])

y_train,y_train_onehot = convertOneHot(data)



# Okay, what did we do?

#

#  The following shows 4 rows of orig dataset:

#

#   SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species

#   5.7,             2.9,           4.2,          1.3,         Iris-versicolor

#   7.6,             3.0,           6.6,          2.1,         Iris-virginica

#   5.6,             3.0,           4.5,          1.5,         Iris-versicolor

#   5.1,             3.5,           1.4,          0.2,         Iris-setosa

#   

#

#  What we've done is placed the values 5.7, 2.9, ... up to but not 

#  including the Species into x_train

#  

#     x_train[0:4]  ... 

#

#         array([[ 5.7,  2.9,  4.2,  1.3], 

#                [ 7.6,  3. ,  6.6,  2.1],

#                [ 5.6,  3. ,  4.5,  1.5],

#                [ 5.1,  3.5,  1.4,  0.2]])

#

#

#     y_train[0:4]  ...

#   

#         array([1, 2, 1, 0]) # 0 is Iris-setosa, 1 is Iris-versicolor, 2 is Iris-virginica

#

#

#     y_train_onehot[0:4]

#

#        [[0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]

#

#     So now you have the following:

#         [0, 1, 0]  represents 1, which is Iris-versicolor

#         [0, 0, 1]  represents 2, which is Iris-virginica

#         [0, 1, 0]  represents 1, which is Iris-versicolor (again)

#         [1, 0, 0]  represents 0, which is Iris-setosa

#

#      ...note our one hot encoding is actually flipped. [1,0,0] == 0

#         [0,1,0] == 1

#         [0,0,1] == 2

#         That's okay, as long as it's consistent 









# Doing a similiar conversion for the test data.

x_test=np.array([ i[1::] for i in test_data])

y_test,y_test_onehot = convertOneHot(test_data)





#  A number of features, 4 in this example

#  B = 3 species of Iris (setosa, virginica and versicolor)

A=data.shape[1]-1 # Number of features, Note first is y

B=len(y_train_onehot[0])

tf_in = tf.placeholder("float", [None, A]) # Features

tf_weight = tf.Variable(tf.zeros([A,B]))

tf_bias = tf.Variable(tf.zeros([B]))

tf_softmax = tf.nn.softmax(tf.matmul(tf_in,tf_weight) + tf_bias)





# Training via backpropagation

tf_softmax_correct = tf.placeholder("float", [None,B])

tf_cross_entropy = -tf.reduce_sum(tf_softmax_correct*tf.log(tf_softmax))



# Train using tf.train.GradientDescentOptimizer

tf_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(tf_cross_entropy)



# Add accuracy checking nodes

tf_correct_prediction = tf.equal(tf.argmax(tf_softmax,1), tf.argmax(tf_softmax_correct,1))

tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))





# Recreate logging dir after each run. 

# Note, when using TensorBoard, and saving our model

# we will recreate the results...removing the old

import shutil, os, sys

TMPDir='./tenIrisSave'

try:

 shutil.rmtree(TMPDir)

except:

 print("Tmp Dir did not exist")

 os.mkdir(TMPDir, 755 )





# Fix for error messages

NUM_CORES = 4  # Choose how many cores to use.

sess = tf.Session(

    config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,

                   intra_op_parallelism_threads=NUM_CORES))





# Initialize and run

# You probably want tf.InteractiveSession() for iPython/jupyter

#sess = tf.Session()

sess = tf.InteractiveSession()

init = tf.initialize_all_variables()

sess.run(init)

# Build the summary operation based on the TF collection of Summaries.

tf.train.write_graph(sess.graph_def, TMPDir + '/logsd','graph.pbtxt')



#acc = tf.scalar_summary("Accuracy:", tf_accuracy)

tf.scalar_summary("Accuracy:", tf_accuracy)

tf.histogram_summary('weights', tf_weight)

tf.histogram_summary('bias', tf_bias)

tf.histogram_summary('softmax', tf_softmax)

tf.histogram_summary('accuracy', tf_accuracy)



# 

summary_op = tf.merge_all_summaries()

summary_writer = tf.train.SummaryWriter(TMPDir + '/logs',sess.graph)

# You must reset, or on 2nd run of iPython/jupyter you'll get errors

# ref: http://stackoverflow.com/a/35424017/904032

tf.reset_default_graph()





# This is for saving all our work

saver = tf.train.Saver([tf_weight,tf_bias])



# Now we're ready to train and test I model

# 

# We'll go through 100 iterations. We'll test the trained

# data against the test data, and the first time 100% accuracy

# and the test data, we'll save our model to the following:

#

#          ./tenIrisSave/saveOne        

#

# After run 26, we reach 1.0 or 100% accuracy

#

#       Tmp Dir did not exist  <-- Recreating, will always get this msg.

#       Run 0,0.3199999928474426

#       Run 1,0.30000001192092896

#       Run 2,0.3799999952316284

#       ...

#       Run 25,0.699999988079071

#       Run 26,1.0   <--------------- Save this run

#       saving

#

k=[]

saved=0

for i in range(100):

    sess.run(tf_train_step, 

              feed_dict={tf_in: x_train, tf_softmax_correct: y_train_onehot})

# Print accuracy

    result = sess.run(tf_accuracy, 

              feed_dict={tf_in: x_test, tf_softmax_correct: y_test_onehot})

    print("Run {},{}".format(i,result))

    k.append(result)

    

# Create Graphs

    summary_str = sess.run(summary_op,

              feed_dict={tf_in: x_test, tf_softmax_correct: y_test_onehot})

    summary_writer.add_summary(summary_str, i)

    if result == 1 and saved == 0:

        saved=1

        print("saving")

        saver.save(sess,"./tenIrisSave/saveOne")



k=np.array(k)

print(np.where(k==k.max()))

print("Max: {}".format(k.max()))



# Below is an example of restoring this

# session and running it to get specific results.

#

# When restoring, make sure you've defined the same variables

# created, when you built the model.

# ...Reference: http://stackoverflow.com/a/33763208/904032

#

saver = tf.train.Saver([tf_weight,tf_bias])

saver.restore(sess, "./tenIrisSave/saveOne")

result = sess.run(tf_accuracy, feed_dict={tf_in: x_test, 

                                          tf_softmax_correct: y_test_onehot})

print("Result: {}".format(result))





ans = sess.run(tf_softmax, feed_dict={tf_in: x_test})

print(x_test[0:3])

print("Correct prediction\n",ans[0:3])



# If we look at the Iris.csv file we see the following:

#

#   6.1,2.8,4.7,1.2,Iris-versicolor

#   5.7,3.8,1.7,0.3,Iris-setosa

#   7.7,2.6,6.9,2.3,Iris-virginica

# 

#   Correct prediction

#    [[  6.18339442e-02   8.63519728e-01   7.46462867e-02]  == [0,1,0]

#    [  9.98805881e-01   1.19409396e-03   3.25464861e-13]   == [1,0,0] 

#    [  1.52262430e-07   4.49669518e-04   9.99550164e-01]]  == [0,0,1]

#





# So, let's translate the encodings above, and notice we have confired

# a correct prediction.

#

#     Here's the translation:

#         [0, 1, 0]  represents 1, which is Iris-versicolor

#         [1, 0, 0]  represents 0, which is Iris-setosa

#         [0, 0, 1]  represents 2, which is Iris-virginica

# %%bash

# tensorboard --logdir=$(pwd)/tenIrisSave