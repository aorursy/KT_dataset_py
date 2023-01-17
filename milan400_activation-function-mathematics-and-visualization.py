import numpy as np

import matplotlib.pyplot as plt



from sklearn.datasets import make_classification, make_circles

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Dense
def create_data():

  x_1 = np.random.randn(100,2) * 0.1 + 0.5

  x_2 = np.random.randn(100,2) * 0.1 + np.array([0.5,-0.5])

  x_4 = np.random.randn(100,2) * 0.1 + np.array([-0.5,0.5])

  x_3 = np.random.randn(100,2) * 0.1 - 0.5

  x = np.concatenate((x_1, x_2, x_3, x_4))

  

  y_1 = np.ones(100)

  y_2 = np.ones(100) * 0

  y_4 = np.ones(100) * 0

  y_3 = np.ones(100)

  y = np.concatenate((y_1,y_2,y_3, y_4))

  y = y.reshape((-1,1))

  

  return x, y



def plot_data(x, y):



  plt.plot(x[y[:,0] == 1,0], x[y[:,0]==1,1], 'bx')

  plt.plot(x[y[:,0]==0,0], x[y[:,0]==0,1], 'ro')



def plot_line(c_1, c_2, c):

  # c_1*x + c_2*y + c = 0 



  plt.ylim(-1.2,1.2)

  plt.xlim(-1.2,1.2)

  x = np.linspace(-2,2,100)

  y = (-c_1*x-c)/c_2

  

  plt.plot(x, y)



def plot_lines(W, b):

  for col, c in zip(W.T, b):

    plot_line(col[0], col[1], c)

    

x_train, y_train = create_data()

plot_data(x_train,y_train)
import tensorflow as tf

#print(tf.__version__)



tf.keras.backend.clear_session()



tf.random.set_seed(10)

inputs = tf.keras.Input(shape=(2,))

x = tf.keras.layers.Dense(2, activation=tf.nn.tanh)(inputs)

outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



plot_data(x_train ,y_train)

b = model.weights[1].numpy()

w = model.weights[0].numpy()

plot_lines(w,b)
plot_data(x_train ,y_train)

b = model.weights[1].numpy().reshape((1,-1))

w = model.weights[0].numpy()

plot_line(w[0,0], w[1,0], b[0,0])    

plot_line(w[0,1], w[1,1], b[0,1]) 
model.fit(x_train, y_train, batch_size = 16, epochs=100)
plot_data(x_train ,y_train)

b = model.weights[1].numpy()

w = model.weights[0].numpy()

plot_lines(w,b)
activations = model.get_layer(name='dense')(x_train)

plot_data(activations.numpy(), y_train)

b = model.weights[3].numpy().reshape((1,-1))

w = model.weights[2].numpy()

plot_line(w[0,0], w[1,0], b[0,0])  

#plot_line(w[0,1], w[1,1], b[0,1])  
X, y = make_circles(n_samples=400, factor=0.3, noise=.05)



reds = y == 0

blues = y == 1



plt.scatter(X[reds, 0], X[reds, 1], c='red', s=20, edgecolor='k')

plt.scatter(X[blues, 0], X[blues, 1], c='blue', s=20, edgecolor='k')
model2 = Sequential()

model2.add(Dense(3,input_dim=2, activation='sigmoid'))

model2.add(Dense(1, activation='sigmoid'))



model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model2.fit(X, y, epochs=1000,)
w1 = []

for i in model2.layers:

    w1.append(i.get_weights())
#first hidden layer

wih11 = w1[0][0][0][0]

wih12 = w1[0][0][0][1]

wih13 = w1[0][0][0][2]



wih21 = w1[0][0][1][0]

wih22 = w1[0][0][1][1]

wih23 = w1[0][0][1][2]



bh1 = w1[0][1][0]

bh2 = w1[0][1][1]

bh3 = w1[0][1][2]



#output layer

who1 = w1[1][0][0][0]

who2 = w1[1][0][1][0]

who3 = w1[1][0][2][0]



bho0 = w1[1][1][0]
x = np.arange(-1,1.5)



plt.plot(x, ((-x*wih11)-bh1)/wih21, c='y')

plt.plot(x, ((-x*wih12)-bh2)/wih22, c='g')

plt.plot(x, ((-x*wih13)-bh3)/wih23, c='k')



plt.scatter(X[reds, 0], X[reds, 1], c='red', s=20, edgecolor='k')

plt.scatter(X[blues, 0], X[blues, 1], c='blue', s=20, edgecolor='k')
