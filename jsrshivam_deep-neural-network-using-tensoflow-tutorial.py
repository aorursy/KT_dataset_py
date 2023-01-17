import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors

import os
os.listdir('../input/')


data = pd.read_csv('../input/blobs.csv')



print(data.groupby('class').head(2))

X = data[['feature1', 'feature2']].values

y = data['class'].values
colors=['green','blue']

cmap = matplotlib.colors.ListedColormap(colors)

#Plot the figure

plt.figure()

plt.title('Non-linearly separable classes')

plt.scatter(X[:,0], X[:,1], c=y,

           marker= 'o', s=50,cmap=cmap,alpha = 0.5 )

plt.show()
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

_ = lr_model.fit(X, y)
def plot_decision_boundary( X, y, model):

    plt.clf()

    # Set min and max values and give it some padding

    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1

    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1   

    colors=['blue','green']

    cmap = matplotlib.colors.ListedColormap(colors)   

    h = 0.01

    # Generate a grid of points with distance h between them

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid

    A = model(np.c_[xx.ravel(), yy.ravel()])

    A = A.reshape(xx.shape)

    # Plot the contour and training examples

    plt.contourf(xx, yy, A, cmap="spring")

    plt.ylabel('x2')

    plt.xlabel('x1')

    plt.scatter(X[0, :], X[1, :], c=y, s=8,cmap=cmap)

    plt.title("Decision Boundary for learning rate:")

    plt.show()
plot_decision_boundary(X.T,y,lambda x: lr_model.predict(x))
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25,random_state = 25)



print("shape of X_train:{} shape 0f Y_train:{}".format(X_train.shape, Y_train.shape))  

print("shape of X_test:{} shape 0f Y_test:{}".format(X_test.shape, Y_test.shape))  
X_train = X_train.T



Y_train = Y_train.reshape(1, len(Y_train))



X_test = X_test.T



Y_test = Y_test.reshape(1, len(Y_test))







print("shape of X_train:{} shape 0f Y_train:{} after transformation".format(X_train.shape, Y_train.shape))
layer_dims = [2,9,9,1]
import tensorflow as tf
def placeholders(num_features):

  A_0 = tf.placeholder(dtype = tf.float64, shape = ([num_features,None]))

  Y = tf.placeholder(dtype = tf.float64, shape = ([1,None]))

  return A_0,Y
def initialize_parameters_deep(layer_dims):

  L = len(layer_dims)

  parameters = {}

  print("Initializing parameters \n L : {}".format(L))

  for l in range(1,L):

    print("l : {}".format(l))

    parameters['W' + str(l)] = tf.Variable(initial_value=tf.random_normal([layer_dims[l], layer_dims[l-1]], dtype=tf.float64)* 0.01)

    parameters['b' + str(l)]=tf.Variable(initial_value=tf.zeros([layer_dims[l],1],dtype=tf.float64) * 0.01)

  print(parameters)

  return parameters 
def linear_forward_prop(A_prev,W,b, activation):

  Z = tf.add(tf.matmul(W, A_prev), b)

  if activation == "sigmoid":

    A = Z

  elif activation == "relu":

    A = tf.nn.relu(Z)                                           #apply relu activation on Z using tf.nn.relu() function

  return A
def l_layer_forwardProp(A_0, parameters):

  A = A_0

  L = len(parameters)//2

  for l in range(1,L):

    A_prev = A

    A = linear_forward_prop(A_prev,parameters['W' + str(l)],parameters['b' + str(l)], "relu")

  A_final = linear_forward_prop(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid" )

  return A_final
def deep_layer_network( X_data,y_data,  layer_dims, learning_rate, num_iter, X_test=np.nan,Y_test=np.nan):

  num_features = layer_dims[0]

  A_0, Y =  placeholders(num_features)                                    #call placeholder function to initialize placeholders A_0 and Y

  parameters = initialize_parameters_deep(layer_dims)                                    #Initialse Weights and bias using initialize_parameters_deep() with layer_dims as parameters  

  Z_final =   l_layer_forwardProp(A_0, parameters)                                     #call the function l_layer_forwardProp() to define the final output

  cost =   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z_final,labels=Y))

  #define the mean cost using sigmoid cross entropy function

  train_net = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  #call tensorflow's gradient descent optimizer function with minimize cost  

  init = tf.global_variables_initializer()    

  

  with tf.Session() as sess:

    sess.run(init)

    for i in range(num_iter):

      _,c = sess.run([train_net, cost], feed_dict={A_0: X_data, Y: y_data})

      if i % 1000 == 0:

        print(c)

    if X_test.any() and Y_test.any():

        correct_prediction = tf.equal(tf.round(tf.sigmoid(Z_final)), Y)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    

        print("Accuracy on test set:", accuracy.eval({A_0: X_test, Y: Y_test}))

    with open("Output.txt", "w") as text_file:

      text_file.write("c= %f\n" % c)

    params = sess.run(parameters)

  return params
parameters = deep_layer_network(X_train,Y_train, layer_dims, learning_rate=0.3, num_iter=10000, X_test=X_test, Y_test= Y_test)    #fill the parameters 
def predict(A_0, parameters):

  with tf.Session() as sess:

    Z = l_layer_forwardProp(A_0, parameters)

    A = sess.run(tf.round(tf.sigmoid(Z)))

  return A
plot_decision_boundary(X.T,y,lambda x: predict(x.T,parameters))
print(X.T.shape, y.shape, X_train.shape, Y_train.reshape(-1).shape)
plot_decision_boundary(X_train,Y_train.reshape(-1),lambda x: predict(x.T,parameters))