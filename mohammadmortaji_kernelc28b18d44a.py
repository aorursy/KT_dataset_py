# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#check the data

data = pd.read_csv('/kaggle/input/titanic/train.csv')

x_train = data.drop('Survived',axis = 1)

y_train = data['Survived']

x_train.head()
y_train.head()
#what is cabin?

x_train['Cabin'].describe()
#preprocessing



# no need for name,ticket id, passengerid

x_train = x_train.drop(['Name','PassengerId','Ticket'],axis = 1)



#it should be importnat if they have/not have cabin

x_train['Cabin'] = pd.isna(x_train['Cabin'])



#sex and embarked should be categorical

sex = pd.get_dummies(x_train['Sex'])

embark = pd.get_dummies(x_train['Embarked'],prefix = 'embark')

x_train = x_train.drop(['Sex','Embarked'],axis = 1)

x_train = pd.concat([x_train,sex,embark],axis = 1)



#now let's take a look

x_train.head()
#converting data to np array

x_train = x_train.to_numpy(dtype = float)

x_train.shape
y_train = y_train.to_numpy(dtype = float)

y_train.shape
model = keras.models.Sequential()

model.add (keras.layers.Dense(2,input_shape = (11,)))

model.add (keras.layers.Dense(1,activation = 'softmax'))

model.compile(optimizer = 'adam',

             loss = 'mse')

model.summary()
model.fit(x_train,y_train,verbose = 0)
#There are nan values in input. Why?

data = pd.read_csv('/kaggle/input/titanic/train.csv')

x_train = data.drop('Survived',axis = 1)

# we have nan values in origial data; on what columns?

pd.isna(x_train).any()
#let's see how much nan does we have on each column:

x_train.isnull().sum()
#We need to remove "age" as well, then remove the nan rows of Embarked.

#preprocessing



# no need for name,ticket id, passengerid

x_train = x_train.drop(['Name','PassengerId','Ticket','Age'],axis = 1)

#it should be importnat if they have/not have cabin

x_train['Cabin'] = pd.isna(x_train['Cabin'])



#sex and embarked should be categorical

sex = pd.get_dummies(x_train['Sex'])

embark = pd.get_dummies(x_train['Embarked'],prefix = 'embark')

x_train = x_train.drop(['Sex','Embarked'],axis = 1)

x_train = pd.concat([x_train,sex,embark],axis = 1)

x_train = x_train.dropna(axis = 0)

#converting data to np array

x_train = x_train.to_numpy(dtype = float)

x_train.shape, y_train.shape
model = keras.models.Sequential()

model.add (keras.layers.Dense(2,input_shape = (10,)))

model.add (keras.layers.Dense(1,activation = 'softmax'))

model.compile(optimizer = 'adam',

             loss = 'mse')

model.summary()

model.fit(x_train,y_train,verbose = 0)
model.fit(x_train,y_train,

         epochs = 10,verbose = 0)
model_v2 = keras.models.Sequential()

model_v2.add (keras.layers.Dense(2,input_shape = (10,)))

model_v2.add (keras.layers.Dense(4))

model_v2.add (keras.layers.Dense(1,activation = 'softmax'))

model_v2.compile(optimizer = 'adam',

             loss = 'mse')

model_v2.summary()

model_v2.fit(x_train,y_train,verbose = 0)
model_v2.fit(x_train,y_train,

         epochs = 10,verbose = 0)
#the model should be able to overfit on 1 piece of data

x_train_ov = x_train[0:1]

y_train_ov = y_train[0:1]

model.fit(x_train_ov,y_train_ov,verbose = 0)
model.fit(x_train_ov,y_train_ov,

         epochs = 10,verbose = 0)
model_v2.fit(x_train_ov,y_train_ov,verbose = 0)
model_v2.fit(x_train_ov,y_train_ov,

            epochs = 10,verbose = 0)
#trying linear activation

model_v3 = keras.models.Sequential()

model_v3.add (keras.layers.Dense(2,input_shape = (10,)))

model_v3.add (keras.layers.Dense(4))

model_v3.add (keras.layers.Dense(1,activation = 'linear'))

model_v3.compile(optimizer = 'adam',

             loss = 'mse')

model_v3.summary()

model_v3.fit(x_train,y_train,verbose = 0)
model_v3.fit(x_train_ov,y_train_ov,

            epochs = 10,verbose = 0)
model_v3.fit(x_train_ov,y_train_ov,

            epochs = 10000,verbose = 0)
model_v3.fit(x_train,y_train,

             validation_split=0.2,

             epochs = 1000,verbose = 0)
#let's evaluate our model



#build data for training/evaluating

X,Y = x_train,y_train

x_val = X[:len(X)//7]

x_train = X[len(X)//7:]

y_val = Y[:len(Y)//7]

y_train = Y[len(Y)//7:]



x_val.shape, x_train.shape, y_val.shape ,y_train.shape

#making an early stop callback



e_s = keras.callbacks.EarlyStopping()
model_v3.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,

             callbacks=[e_s],verbose = 0)
#evaluating on validation data



prediction = model_v3.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#relu activation



model_v4 = keras.models.Sequential()

model_v4.add (keras.layers.Dense(2,input_shape = (10,)))

model_v4.add (keras.layers.Dense(4))

model_v4.add (keras.layers.Dense(1,activation = 'relu'))

model_v4.compile(optimizer = 'adam',

             loss = 'mse')

model_v4.summary()

model_v4.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)



prediction = model_v4.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#softmax with binary cross enthropy as loss



model_v5 = keras.models.Sequential()

model_v5.add (keras.layers.Dense(2,input_shape = (10,)))

model_v5.add (keras.layers.Dense(4))

model_v5.add (keras.layers.Dense(1,activation = 'softmax'))

model_v5.compile(optimizer = 'adam',

             loss = 'binary_crossentropy')

model_v5.summary()

model_v5.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)



prediction = model_v5.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#relu with binary cross enthropy as loss



model_v6 = keras.models.Sequential()

model_v6.add (keras.layers.Dense(2,input_shape = (10,)))

model_v6.add (keras.layers.Dense(4))

model_v6.add (keras.layers.Dense(1,activation = 'relu'))

model_v6.compile(optimizer = 'adam',

             loss = 'binary_crossentropy')

model_v6.summary()

model_v6.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)



prediction = model_v6.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#no, should change the hyperparameters again





#relu with mean absolute erorr as loss



model_v8 = keras.models.Sequential()

model_v8.add (keras.layers.Dense(2,input_shape = (10,)))

model_v8.add (keras.layers.Dense(4))

model_v8.add (keras.layers.Dense(1,activation = 'relu'))

model_v8.compile(optimizer = 'adam',

             loss = 'mean_absolute_error')

model_v8.summary()

model_v8.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)



prediction = model_v8.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#relu for all layers with mean absolute erorr as loss



model_v9 = keras.models.Sequential()

model_v9.add (keras.layers.Dense(2,input_shape = (10,),activation = 'relu'))

model_v9.add (keras.layers.Dense(4,activation = 'relu'))

model_v9.add (keras.layers.Dense(1,activation = 'relu'))

model_v9.compile(optimizer = 'adam',

             loss = 'mean_absolute_error')

model_v9.summary()

model_v9.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)



prediction = model_v9.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#relu for all layers, sigmoid for last layer with mean absolute erorr as loss



model_v10 = keras.models.Sequential()

model_v10.add (keras.layers.Dense(2,input_shape = (10,),activation = 'relu'))

model_v10.add (keras.layers.Dense(4,activation = 'relu'))

model_v10.add (keras.layers.Dense(1,activation = 'sigmoid'))

model_v10.compile(optimizer = 'adam',

             loss = 'mean_absolute_error')

model_v10.summary()

model_v10.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)



prediction = model_v10.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#relu for all layers, sigmoid for last layer with binary_crossentropy as loss



model_v10 = keras.models.Sequential()

model_v10.add (keras.layers.Dense(2,input_shape = (10,),activation = 'relu'))

model_v10.add (keras.layers.Dense(4,activation = 'relu'))

model_v10.add (keras.layers.Dense(1,activation = 'sigmoid'))

model_v10.compile(optimizer = 'adam',

             loss = 'binary_crossentropy')

model_v10.summary()

model_v10.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)



prediction = model_v10.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#relu for all layers, sigmoid for last layer with binary_crossentropy as loss (overfit)





model_v10.fit(x_train_ov,y_train_ov,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)



prediction = model_v10.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#bigger v_10



model_v11 = keras.models.Sequential()

model_v11.add (keras.layers.Dense(10,input_shape = (10,),activation = 'relu'))

model_v11.add (keras.layers.Dense(10,activation = 'relu'))

model_v11.add (keras.layers.Dense(1,activation = 'sigmoid'))

model_v11.compile(optimizer = 'adam',

             loss = 'binary_crossentropy')

model_v11.summary()

model_v11.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)



prediction = model_v11.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#bigger v_10, none linear



model_v12 = keras.models.Sequential()

model_v12.add (keras.layers.Dense(100,input_shape = (10,),activation = 'relu'))

model_v12.add (keras.layers.Dense(1,activation = 'sigmoid'))

model_v12.compile(optimizer = 'adam',

             loss = 'binary_crossentropy')

model_v12.summary()

model_v12.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)



prediction = model_v12.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#smaller v_12



model_v13 = keras.models.Sequential()

model_v13.add (keras.layers.Dense(30,input_shape = (10,),activation = 'relu'))

model_v13.add (keras.layers.Dense(1,activation = 'sigmoid'))

model_v13.compile(optimizer = 'adam',

             loss = 'binary_crossentropy')

model_v13.summary()

model_v13.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)



prediction = model_v13.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
# one layer network?



model_v14 = keras.models.Sequential()

model_v14.add (keras.layers.Dense(1,input_shape = (10,),activation = 'sigmoid'))

model_v14.compile(optimizer = 'adam',

             loss = 'binary_crossentropy')

model_v14.summary()

model_v14.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,

             verbose = 0)



prediction = model_v14.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#smaller v_13



model_v14 = keras.models.Sequential()

model_v14.add (keras.layers.Dense(20,input_shape = (10,),activation = 'relu'))

model_v14.add (keras.layers.Dense(1,activation = 'sigmoid'))

model_v14.compile(optimizer = 'adam',

             loss = 'binary_crossentropy')

model_v14.summary()

model_v14.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)



prediction = model_v14.predict(x_val)

true = y_val.reshape(prediction.shape)



plt.scatter(prediction,true)
#plot hist



model_v14 = keras.models.Sequential()

model_v14.add (keras.layers.Dense(20,input_shape = (10,),activation = 'relu'))

model_v14.add (keras.layers.Dense(1,activation = 'sigmoid'))

model_v14.compile(optimizer = 'adam',

             loss = 'binary_crossentropy')

model_v14.summary()

h = model_v14.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 1000,verbose = 0)

pd.DataFrame(h.history).plot()
model_v14 = keras.models.Sequential()

model_v14.add (keras.layers.Dense(1,input_shape = (10,),activation = 'relu'))

model_v14.add (keras.layers.Dense(1,activation = 'sigmoid'))

model_v14.compile(optimizer = 'adam',

             loss = 'binary_crossentropy')

model_v14.summary()

h = model_v14.fit(x_train,y_train,

             validation_data=(x_val,y_val),

             epochs = 100000,verbose = 2)



pd.DataFrame(h.history).plot()
#using logistic regression model



from sklearn.linear_model import LogisticRegressionCV

model_v15 = LogisticRegressionCV()

model_v15.fit(x_train,y_train)

model_v15.score(x_val,y_val)
#reading test data, preprocessing and finding the prediction

x_test = pd.read_csv('/kaggle/input/titanic/test.csv')

x_test = x_test.drop(['Name','PassengerId','Ticket','Age'],axis = 1)

#it should be importnat if they have/not have cabin

x_test['Cabin'] = pd.isna(x_test['Cabin'])



#sex and embarked should be categorical

sex = pd.get_dummies(x_test['Sex'])

embark = pd.get_dummies(x_test['Embarked'],prefix = 'embark')

x_test = x_test.drop(['Sex','Embarked'],axis = 1)

x_test = pd.concat([x_test,sex,embark],axis = 1)

x_test = x_test.dropna(axis = 0)

output = pd.DataFrame(model_v15.predict(x_test))

output.to_csv('./result.csv')