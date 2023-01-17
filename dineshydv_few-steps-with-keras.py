# Basic libraries

import numpy as np

import pandas as pd



# model libraries

from sklearn.model_selection import train_test_split

import tensorflow

import keras

#from sklearn.ensemble import RandomForestClassifier

#from sklearn.model_selection import KFold

#from sklearn.model_selection import GridSearchCV



# model accuracy check

from sklearn import metrics



# plotting libraries

import matplotlib.pyplot as plt

import seaborn as sns

# read the data

train_file = '../input/digit-recognizer/train.csv'

test_file = '../input/digit-recognizer/test.csv'



df_train = pd.read_csv(train_file)

df_test = pd.read_csv(test_file)

df_train.head()
df_train.shape
# columns pixel0....pixel785 are independent variable of a digit

# column label contains the digit (dependent variable)



df_train.columns
df_test.shape
df_test.columns
# no null values in train dataset



df_train.isnull().values.any()
df_train.info()
# print the frequency of each label



print(df_train['label'].value_counts())

sns.countplot(df_train['label'])
plt.figure(figsize=(12,4))

for i in range(30):  

    plt.subplot(3, 10, i+1)

    plt.imshow(df_train.drop(['label'],axis=1).values[i].reshape(28,28) )

    plt.axis('off')

plt.show()



# print corresponding labels:

print(list(df_train['label'].loc[0:9]))

print(list(df_train['label'].loc[10:19]))

print(list(df_train['label'].loc[20:29]))
plt.figure(figsize=(12,4))

for i in range(30):  

    plt.subplot(3, 10, i+1)

    plt.imshow(df_test.values[i].reshape(28,28) )

    plt.axis('off')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['label'],axis=1),

                                                   df_train['label'],

                                                   test_size = 0.2,

                                                   random_state=13)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
X_train.head()
y_train.head()
# y_train = keras.utils.to_categorical(y_train)

# y_test = keras.utils.to_categorical(y_test)



n_cols = X_train.shape[1]

print("Number of input columns: {0}".format(n_cols))



n_features = len(y_train.unique())

print("Number of output features: {0}".format(n_features))

# let's convert labels to categorical variable

y_train = keras.utils.to_categorical(y_train, n_features)

y_test = keras.utils.to_categorical(y_test, n_features)
y_train, y_test
# Let's do the same with X variables. 

# data has pixels with max values of 255. So will divide values with 255 to scale the data

X_train = X_train / 255

X_test = X_test / 255
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop

from keras.optimizers import SGD



m = Sequential()

m.add(Dense(512,activation='relu',input_shape=(n_cols,)))

m.add(Dropout(0.5))

m.add(Dense(256,activation='relu'))

m.add(Dropout(0.5))

m.add(Dense(128,activation='relu'))

m.add(Dropout(0.5))

m.add(Dense(64,activation='relu'))

m.add(Dropout(0.5))

m.add(Dense(n_features,activation='softmax'))



m.summary()
m.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#m.compile(optimizer=SGD(),loss='categorical_crossentropy',metrics=['accuracy'])

n_batch_size = 256

n_epochs = 200

history = m.fit(X_train, y_train, batch_size=n_batch_size, epochs=n_epochs, validation_data=(X_test,y_test))

plt.plot(history.history['val_acc'],'b')

plt.plot(history.history['acc'],'r')
plt.plot(history.history['val_loss'],'b')

plt.plot(history.history['loss'],'r')
y_pred = np.round(m.predict(X_test)).astype('int64')

y_pred
# remove the categories from y_pred

# select the indix with the maximum probability

y_pred1 = np.argmax(y_pred,axis = 1)

y_pred1
# do the same for y_test

# select the index with the maximum probability

y_test1 = np.argmax(y_test,axis = 1)

y_test1
# print the frequency of each label



y = pd.DataFrame(y_pred1)

y.columns=['label']

print(y['label'].value_counts())



y[y['label'] < 0] = 0

y[y['label'] > 9] = 9



sns.countplot(y['label'])
print('Accuracy score for y_test: ', metrics.accuracy_score(y_test1,y_pred1))
pd.DataFrame(metrics.confusion_matrix(y_test1,y_pred1))
# combine actual and predicted in a single df

X_test['actual'] = y_test1

X_test['pred'] = y_pred1
X_test_err = X_test[X_test['actual'] != X_test['pred']]

print(X_test_err.shape[0],"predictions are wrong")
for i in range (10):

    act=X_test_err['actual'].values[i]

    prd=X_test_err['pred'].values[i]

    print("actual {0} ; predicted {1}".format(act,prd))

    plt.figure(figsize=(1,1))

    plt.imshow(X_test_err.drop(['actual','pred'], axis=1).values[i].reshape(28,28))

    plt.axis("off")

    plt.show()
# # normalize the input data

df_test = df_test / 255

pred = m.predict(df_test)

pred
# select the index with the maximum probability

pred = np.argmax(pred,axis = 1)

pred


for i in range(11):  

    print('Prediction {0}'.format(pred[i]))

    plt.figure(figsize=(1,1))

    plt.imshow(df_test.values[i].reshape(28,28) )

    plt.axis('off')

    plt.show()

pred = pd.Series(pred,name="Label")
submit = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pred],axis = 1)



submit.to_csv("cnn_mnist_fewsteps_keras.csv",index=False)