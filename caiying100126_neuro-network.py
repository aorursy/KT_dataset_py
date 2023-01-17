import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.datasets import make_moons, make_circles

from sklearn.preprocessing import scale, PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score as r2

import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

%matplotlib inline
# set the number of samples to take for each toy dataset

n_samples = 1000

# set the proportion of toy data to hold out for testing

test_size = 0.3

# set the random seed to make the experiment reproducible 

random_seed = 5153

np.random.seed(random_seed)
# define a function

f = lambda x: np.cos(np.sqrt(x))**2

# choose some points from the function - this is our toy dataset 

X = np.random.permutation(np.linspace(0, 50, n_samples))

Y = f(X)

# create training and testing data from this set of points

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
# plot the toy data

fig, ax = plt.subplots()

ax.scatter(X_train, Y_train, color='green')

ax.legend()

ax.set(xlabel='X', ylabel='Y', title='Toy data set for regression')

plt.show()
%%time

# degree of the polynomial model - play around with this!

degree = 4

# add polynomial features to the data and fit a ridge regression model

simple_model = make_pipeline(PolynomialFeatures(degree), Ridge())

simple_model.fit(X_train.reshape((X_train.shape[0], 1)), Y_train)

# use our model to predict in the range we want

X_range = np.linspace(0, 50, 500).reshape((500, 1))

y_pred = simple_model.predict(X_range)



# we plot the model (degree 4 polynomial) against the data

fig, ax = plt.subplots()

ax.scatter(X_train, Y_train, label='Training data')

ax.plot(X_range, y_pred, color='r', label='Degree ' + str(degree) + ' polynomial model')

ax.legend(loc='best')

ax.set(xlabel='X', ylabel='Y', title='Toy regression data set')

plt.show()
# evaluate the model

print('Train R2:', simple_model.score(X_test.reshape((X_test.shape[0], 1)), Y_test))

print('Test R2:', r2(Y_test, simple_model.predict(X_test.reshape((X_test.shape[0], 1)))))
# number of hidden nodes

H = 500

# input dimension

input_dim = 1



# create sequential multi-layer perceptron

model = Sequential()

# layer 0

model.add(keras.layers.Dense(500,input_dim = 1,activation='tanh'))



# layer 1

model.add(keras.layers.Dense(1,activation='tanh'))



# configure the model

model.compile(optimizer='adam',

              loss='mean_squared_error',

             metrics=['acc'])
model.summary()
%%time

# fit the model

# model.fit(...)

model.fit(X_train,Y_train,epochs=20)
# use our model to predict in the range we want

X_range = np.linspace(0, 50, 500)

y_pred = model.predict(X_range)



# we plot the model (degree 4 polynomial) against the data

fig, ax = plt.subplots()

ax.scatter(X_train, Y_train, label='Training data')

ax.plot(X_range, y_pred, color='r', label='MLP with one hidden layer')

ax.legend(loc='best')

ax.set(xlabel='X', ylabel='Y', title='Toy regression data set')

plt.show()
# evaluate the training and testing performance of your model 

# note: you should extract check both the loss function and your evaluation metric

score = model.evaluate(X_train, Y_train, verbose=0)

print('Train loss:', score)

print('Train R2:', r2(Y_train, model.predict(X_train)))
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score)

print('Test R2:', r2(Y_test, model.predict(X_test)))
H2 = [10, 50, 100, 150, 300, 500]

Train_loss=[]

Train_R2=[]

Test_loss=[]

Test_R2=[]



for i in H2:



    model = Sequential()

    model.add(keras.layers.Dense(i,input_dim = 1,activation='tanh'))

    model.add(keras.layers.Dense(1,activation='tanh'))

    model.compile(optimizer='adam',

                  loss='mean_squared_error',

                 metrics=['acc'])

    model.fit(X_train,Y_train,epochs=20)

    X_range = np.linspace(0, 50, 500)

    y_pred = model.predict(X_range)

   

    score_train = model.evaluate(X_train, Y_train, verbose=0)

    Train_loss.append(score_train)

    Train_R2.append(r2(Y_train, model.predict(X_train)))

    print('Train loss:', score_train)

    print('Train R2:', r2(Y_train, model.predict(X_train)))

    

    score_test = model.evaluate(X_test, Y_test, verbose=0)

    Test_loss.append(score_test)

    Test_R2.append(r2(Y_test, model.predict(X_test)))

    print('Test loss:', score_test)

    print('Test R2:', r2(Y_test, model.predict(X_test)))



print("=====Train loss list is====",Train_loss)

print("=====Train R2 list is====",Train_R2)

print("=====Test loss list is====",Test_loss)

print("=====Test R2 list is====",Test_R2)
Train_loss=[[0.1115021870817457, 0.0], [0.0782939025759697, 0.0], [0.059992337013993945, 0.0],

 [0.05777185938188008, 0.0], [0.056267819106578826, 0.0], [0.05865261882543564, 0.0]]

Test_loss=[[0.09530642668406168, 0.0], [0.07291402826706568, 0.0], [0.0532765114804109, 0.0], 

           [0.058304697622855506, 0.0], [0.05366655692458153, 0.0], [0.06053324937820435, 0.0]]



for x in Train_loss:

    x.remove(0.0)

    

for x in Test_loss:

    x.remove(0.0)
print(Train_loss)

print(Test_loss)
# convert R2 list into a list of lists 

def extractDigits(list): 

    return [[el] for el in list] 



Train_R2=[0.09139561109933192, 0.3620018969780652, 0.5111369332448601,

          0.5292310788457846, 0.5414871294283854, 0.5220539836204734]

Test_R2=[0.13092891581135735, 0.314798070616674, 0.5159118872428279, 

         0.5483444443265537, 0.5397474522970052, 0.5486338542197504]



Train_R2= extractDigits(Train_R2)

Test_R2=extractDigits(Test_R2)

print(Train_R2)

print(Test_R2)
# Plot the train/test performace against the number of hidden nodes, H

H2 = [10, 50, 100, 150, 300, 500]

plt.figure(figsize=(12, 6), dpi=100)

plt.subplot(2,1,1)

plt.plot(H2, Train_R2, 'r', label='train R2')

plt.plot(H2, Test_R2, 'b', label='test R2')

plt.title('Train and Test R2')

plt.legend()



plt.figure(figsize=(12, 6), dpi=100)

plt.subplot(2,1,2)

plt.plot(H2, Train_loss, 'r', label='train loss')

plt.plot(H2, Test_loss, 'b', label='test loss')

plt.title('Train and test Loss')

plt.legend()

plt.show()
af = ['linear', 'sigmoid', 'tanh']

Training_loss=[]

Training_R2=[]

Testing_loss=[]

Testing_R2=[]



for i in af:



    model = Sequential()

    model.add(keras.layers.Dense(500,input_dim = 1,activation=i))

    model.add(keras.layers.Dense(1,activation=i))

    model.compile(optimizer='adam',

                  loss='mean_squared_error',

                 metrics=['acc'])

    model.fit(X_train,Y_train,epochs=20)

    X_range = np.linspace(0, 50, 500)

    y_pred = model.predict(X_range)

   

    score_train = model.evaluate(X_train, Y_train, verbose=0)

    Training_loss.append(score_train)

    Training_R2.append(r2(Y_train, model.predict(X_train)))

    print('Training loss:', score_train)

    print('Training R2:', r2(Y_train, model.predict(X_train)))

    

    score_test = model.evaluate(X_test, Y_test, verbose=0)

    Testing_loss.append(score_test)

    Testing_R2.append(r2(Y_test, model.predict(X_test)))

    print('Testing loss:', score_test)

    print('Testing R2:', r2(Y_test, model.predict(X_test)))



print("=====Training loss list is====",Training_loss)

print("=====Training R2 list is====",Training_R2)

print("=====Testing loss list is====",Testing_loss)

print("=====Testing R2 list is====",Testing_R2)
Training_loss=[[0.13970782433237347, 0.0], [0.09516122468880245, 0.0], [0.050909366948263986, 0.0]]

Testing_loss= [[0.11489185671011606, 0.0], [0.09387640635172526, 0.0], [0.04909474655985832, 0.0]]



for x in Training_loss:

    x.remove(0.0)

    

for x in Testing_loss:

    x.remove(0.0)



print(Training_loss)

print(Testing_loss)
# convert R2 list into a list of lists 



Training_R2=[-0.13844530324353643, 0.22455415770822051, 0.5851518614284135]



Testing_R2=[-0.016575348907371668, 0.16937155615724664, 0.565604464921087]



Training_R2= extractDigits(Training_R2)

Testing_R2=extractDigits(Testing_R2)

print(Training_R2)

print(Testing_R2)
# Plot the train/test performace against these three activation functions



af = ['linear', 'sigmoid', 'tanh']

plt.figure(figsize=(12, 6), dpi=100)

plt.subplot(2,1,1)

plt.plot(af, Training_R2, 'r', label='train R2')

plt.plot(af, Testing_R2, 'b', label='test R2')

plt.title('Train and Test R2')

plt.legend()



plt.figure(figsize=(12, 6), dpi=100)

plt.subplot(2,1,2)

plt.plot(af, Training_loss, 'r', label='train loss')

plt.plot(af, Testing_loss, 'b', label='test loss')

plt.title('Train and test Loss')

plt.legend()

plt.show()
# number of hidden nodes

H = 50

# input dimension

input_dim = 1



# create sequential multi-layer perceptron

model2 = Sequential()

# layer 0

model2.add(keras.layers.Dense(50,input_dim = 1,activation='tanh'))



# layer 1

model2.add(keras.layers.Dense(50,activation='tanh'))



# layer 2

model2.add(keras.layers.Dense(50,activation='tanh'))



# layer 3

model2.add(keras.layers.Dense(50,activation='tanh'))



# layer 4

model2.add(keras.layers.Dense(50,activation='tanh'))



# layer 5

model2.add(keras.layers.Dense(50,activation='tanh'))



# layer 6

model2.add(keras.layers.Dense(1,activation='tanh'))



# configure the model

model2.compile(optimizer='adam',

              loss='mean_squared_error',

             metrics=['acc'])
%%time

# fit the model

# fit the model

# model.fit(...)

model2.fit(X_train,Y_train,epochs=20)
# use our model to predict in the range we want

X_range = np.linspace(0, 50, 500)

y_pred = model2.predict(X_range)



# we plot the model (degree 4 polynomial) against the data

fig, ax = plt.subplots()

ax.scatter(X_train, Y_train, label='Training data')

ax.plot(X_range, y_pred, color='r', label='MLP with one hidden layer')

ax.legend(loc='best')

ax.set(xlabel='X', ylabel='Y', title='Toy regression data set')

plt.show()
# evaluate the training and testing performance of your model 

# note: you should extract check both the loss function and your evaluation metric

score = model2.evaluate(X_train, Y_train, verbose=0)

print('Train loss:', score)

print('Train R2:', r2(Y_train, model2.predict(X_train)))
score = model2.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score)

print('TEST R2:', r2(Y_test, model2.predict(X_test)))
# Besides tanh activation function, I also tried Sigmoid activation function

# First let`s look at the wide but shallow network with Sigmoid function:

# set H=500, Layer= 2

# number of hidden nodes

H3 = 500

# input dimension

input_dim = 1



# create sequential multi-layer perceptron

model3 = Sequential()

# layer 0

model3.add(keras.layers.Dense(500,input_dim = 1,activation='sigmoid'))



# layer 1

model3.add(keras.layers.Dense(500,activation='sigmoid'))



# layer 2

model3.add(keras.layers.Dense(1,activation='sigmoid'))





# configure the model

model3.compile(optimizer='adam',

              loss='mean_squared_error',

             metrics=['acc'])
%%time

# fit the model

# fit the model

# model.fit(...)

model3.fit(X_train,Y_train,epochs=20)


# use our model to predict in the range we want

X_range = np.linspace(0, 50, 500)

y_pred = model3.predict(X_range)



# we plot the model (degree 4 polynomial) against the data

fig, ax = plt.subplots()

ax.scatter(X_train, Y_train, label='Training data(using Sigmoid, wide but shallow network)')

ax.plot(X_range, y_pred, color='r', label='MLP with one hidden layer')

ax.legend(loc='best')

ax.set(xlabel='X', ylabel='Y', title='Toy regression data set')

plt.show()
# evaluate the training and testing performance of model3 

# note: you should extract check both the loss function and your evaluation metric

score = model3.evaluate(X_train, Y_train, verbose=0)

print('Train loss of Sigmoid AF,wide but shallow network:', score)

print('Train R2 of Sigmoid AF,wide but shallow network::', r2(Y_train, model3.predict(X_train)))



score = model3.evaluate(X_test, Y_test, verbose=0)

print('Test loss of Sigmoid AF,wide but shallow network::', score)

print('Test R2 of Sigmoid AF,wide but shallow network::', r2(Y_test, model3.predict(X_test)))
# Now let`s try Sigmoid with Narrow but Deep network

# set H=10, layer=15

# number of hidden nodes

H4 = 10

# input dimension

input_dim = 1



# create sequential multi-layer perceptron

model4 = Sequential()

# layer 0

model4.add(keras.layers.Dense(10,input_dim = 1,activation='sigmoid'))



# layer 1

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 2

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 3

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 4

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 5

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 6

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 7

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 8

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 9

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 10

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 11

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 12

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 13

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 14

model4.add(keras.layers.Dense(10,activation='sigmoid'))



# layer 15

model4.add(keras.layers.Dense(1,activation='sigmoid'))



# configure the model

model4.compile(optimizer='adam',

              loss='mean_squared_error',

             metrics=['acc'])

%%time

# fit the model

# fit the model

# model.fit(...)

model4.fit(X_train,Y_train,epochs=20)
# use our model to predict in the range we want

X_range = np.linspace(0, 50, 500)

y_pred = model4.predict(X_range)



# we plot the model (degree 4 polynomial) against the data

fig, ax = plt.subplots()

ax.scatter(X_train, Y_train, label='Training data using sigmoid AF (narrow but deep network)')

ax.plot(X_range, y_pred, color='r', label='MLP with one hidden layer')

ax.legend(loc='best')

ax.set(xlabel='X', ylabel='Y', title='Toy regression data set')

plt.show()


# evaluate the training and testing performance of your model 

# note: you should extract check both the loss function and your evaluation metric

score = model4.evaluate(X_train, Y_train, verbose=0)

print('Train loss of Sigmoid AF,narrow but deep network:', score)

print('Train R2 of Sigmoid AF,narrow but deep network:', r2(Y_train, model4.predict(X_train)))



score = model4.evaluate(X_test, Y_test, verbose=0)

print('Test loss of Sigmoid AF,narrow but deep network:', score)

print('Test R2 of Sigmoid AF,narrow but deep network:', r2(Y_test, model4.predict(X_test)))

# now let`s try another non-linear activation function-->Relu

# first we try wide but shallow network

# set the H=500, Layer=2

# number of hidden nodes

H5 = 500

# input dimension

input_dim = 1



# create sequential multi-layer perceptron

model5 = Sequential()

# layer 0

model5.add(keras.layers.Dense(500,input_dim = 1,activation='relu'))



# layer 1

model5.add(keras.layers.Dense(500,activation='relu'))



# layer 2

model5.add(keras.layers.Dense(1,activation='relu'))





# configure the model

model5.compile(optimizer='adam',

              loss='mean_squared_error',

             metrics=['acc'])
%%time

# fit the model

# fit the model

# model.fit(...)

model5.fit(X_train,Y_train,epochs=20)
# use our model to predict in the range we want

X_range = np.linspace(0, 50, 500)

y_pred = model5.predict(X_range)



# we plot the model (degree 4 polynomial) against the data

fig, ax = plt.subplots()

ax.scatter(X_train, Y_train, label='Training data of relu AF with wide and shallow network')

ax.plot(X_range, y_pred, color='r', label='MLP with one hidden layer')

ax.legend(loc='best')

ax.set(xlabel='X', ylabel='Y', title='Toy regression data set using relu AF(wide and shallow network)')

plt.show()



# evaluate the training and testing performance of your model 

# note: you should extract check both the loss function and your evaluation metric

score = model5.evaluate(X_train, Y_train, verbose=0)

print('Train loss of Relu AF(wide and shallow network):', score)

print('Train R2 of Relu AF(wide and shallow network):', r2(Y_train, model5.predict(X_train)))



score = model5.evaluate(X_test, Y_test, verbose=0)

print('Test loss of Relu AF(wide and shallow network):', score)

print('Test R2 of Relu AF(wide and shallow network):', r2(Y_test, model5.predict(X_test)))
# now I try narrow but deep network using relu activation function

# set H=10, Layer=15



H6 = 10

# input dimension

input_dim = 1



# create sequential multi-layer perceptron

model6 = Sequential()

# layer 0

model6.add(keras.layers.Dense(10,input_dim = 1,activation='relu'))



# layer 1

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 2

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 3

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 4

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 5

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 6

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 7

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 8

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 9

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 10

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 11

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 12

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 13

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 14

model6.add(keras.layers.Dense(10,activation='relu'))



# layer 15

model6.add(keras.layers.Dense(1,activation='relu'))



# configure the model

model6.compile(optimizer='adam',

              loss='mean_squared_error',

             metrics=['acc'])

%%time

# fit the model

# fit the model

# model.fit(...)

model6.fit(X_train,Y_train,epochs=20)
# use our model to predict in the range we want

X_range = np.linspace(0, 50, 500)

y_pred = model6.predict(X_range)



# we plot the model (degree 4 polynomial) against the data

fig, ax = plt.subplots()

ax.scatter(X_train, Y_train, label='Training data of relu AF(narrow but deeep network)')

ax.plot(X_range, y_pred, color='r', label='MLP with one hidden layer')

ax.legend(loc='best')

ax.set(xlabel='X', ylabel='Y', title='Toy regression data set of relu AF(narrow but deeep network)')

plt.show()
# evaluate the training and testing performance of your model 

# note: you should extract check both the loss function and your evaluation metric

score = model6.evaluate(X_train, Y_train, verbose=0)

print('Train loss of relu AF(narrow but deeep network):', score)

print('Train R2 of relu AF(narrow but deeep network):', r2(Y_train, model6.predict(X_train)))



score = model6.evaluate(X_test, Y_test, verbose=0)

print('Test loss of relu AF(narrow but deeep network):', score)

print('Test R2 of relu AF(narrow but deeep network):', r2(Y_test, model6.predict(X_test)))