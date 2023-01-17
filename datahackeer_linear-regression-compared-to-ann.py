import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import linear_model

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train = train.dropna() 

test = test.dropna() 
x_train = train.as_matrix(['x'])

y_train = train.as_matrix(['y'])

x_test = test.as_matrix(['x'])

y_test = test.as_matrix(['y'])
# Create the regression model

lm = linear_model.LinearRegression()

# and train it

lm.fit(x_train,y_train)



#safe our Values for the regressionline

coef = lm.coef_

intercept = lm.intercept_
plt.scatter(x_train,y_train)

line=coef*x_train+intercept

fig=plt.plot(x_train,line,c="red",linewidth=4)

plt.show()
y_pred =lm.predict(x_test)

MSE1=0



for i in range(len(y_pred)):

    MSE1+=((y_pred[i]-y_test[i])**2)



MSE1/300
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train_sc = sc.fit_transform(x_train)

x_test_sc = sc.transform(x_test)

y_train_sc = sc.transform(y_train)



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

#classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'linear', input_dim = 1))

#classifier.add(Dropout(p = 0.5))

# Adding the output layer



classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))



# Compiling the ANN

classifier.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(x_train_sc, y_train_sc, batch_size = 10, epochs =17, verbose=0)

print("finished")
#MSE on the test data

y_pred2 = classifier.predict(x_test)

MSE2=0



for i in range(len(y_pred2)):

    MSE2+=((y_pred2[i]-y_test[i])**2)



print(MSE2/300)



#MSE on the training data

y_train_pred = classifier.predict(x_train)

MSE=0





for i in (300,600):

    MSE+=((y_train_pred[i]-y_train[i-300])**2)

MSE/700
classifier.get_weights()
coef
plt.scatter(x_test,y_test)

plt.scatter(x_test,y_pred2,c="red")

plt.title("ANN x_test")



plt.show()



plt.scatter(x_train,y_train)

fig2=plt.scatter(x_train,y_train_pred,c="red")

plt.title("ANN x_train")

plt.show()



#l. regression model on train data again

plt.scatter(x_train,y_train)

line=coef*x_train+intercept

fig=plt.plot(x_train,line,c="red",linewidth=4)

plt.title("l. regression model on traindata")

plt.show()
#weight of the Neuron

print(classifier.get_weights()[0][0])

#steepnes of l. regression model

print(coef[0])