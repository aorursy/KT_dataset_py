# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Hello TensorFlow

#Basic operation addition of variables and constants demostrated.

import tensorflow as tf



# first, create a TensorFlow constant

const = tf.constant(2.0, name="const")

    

# create TensorFlow variables

b = tf.Variable(2.0, name='b')

c = tf.Variable(1.0, name='c')



# now create some operations

e = tf.add(c, const, name='e')



# setup the variable initialisation

init_op = tf.global_variables_initializer()



# start the session

with tf.Session() as sess:

    # initialise the variables

    sess.run(init_op)

    # compute the output of the graph

    a_out = sess.run(e)

    #print variables

    print("Variable e is {}".format(a_out))

    print("Variable b is {}".format(b.eval()))
# Hello World for SVM



from sklearn import svm



#Added dataset of first-bit-hot (o/p is 1 if first bit is 1)

# x for input set

x=[[0,0,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0],[1,1,1]]

# y for result set

y=[0,0,0,1,1,1]



# Test set

x_test = [[0,1,0],[1,1,0]]



# Declare model

model = svm.SVC(kernel='linear',C=1,gamma=1)

# Train Model

model.fit(x,y)

model.score(x,y)



# Make predication

predicted = model.predict(x_test)

print(predicted)

#Hello Matlab Plot



import matplotlib.pyplot as plt



# default linear plot no.s 1,2,3,4

# Add x lablel and y label

plt.plot([1,2,3,4])

plt.ylabel('some numbers')

plt.show()



# Dot plot no.s with custom axis representation.

plt.plot([1,2,3,4], [1,4,9,16], 'ro')

plt.axis([0, 6, 0, 20])

plt.show()



# plot 1,2,3,4 no.s with Dot representation.

# show axis from 0 -> 5 both x and y

plt.plot([1,2,3,4],[1,2,3,4],'o')

plt.axis([0,5,0,5])

plt.xlabel("X Axis")

plt.ylabel("Y Axis")

plt.show()