import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.style.use('fivethirtyeight')
df = pd.read_csv('../input/drug-classification/drug200.csv')

df.head()
print('No. of missing values in Dataframe is: ' + str(df.isnull().sum().sum()))
df.columns
plt.figure(1,figsize=(20,20))

n = 0

for col in ['Sex', 'BP', 'Cholesterol','Drug']:

  n+=1

  plt.subplot(2,2,n)

  plt.subplots_adjust(hspace=0.3,wspace = 0.3)

  sns.scatterplot(x = 'Na_to_K',y = 'Age',data = df,hue = col,s = 100)

  plt.title(col)

  plt.legend(loc = 0)
df['Drug'] = pd.Categorical(df['Drug'])

df['Code'] = df['Drug'].cat.codes

df.head()
temp = pd.get_dummies(df[['Sex', 'BP', 'Cholesterol']])

temp.head()
df = pd.concat([df,temp],axis = 1)

df.head()
df = df.drop(['Sex', 'BP', 'Cholesterol'],axis = 1)

df.head()
df.columns
X = df[['Age', 'Na_to_K', 'Sex_F', 'Sex_M', 'BP_HIGH', 'BP_LOW','BP_NORMAL', 'Cholesterol_HIGH', 'Cholesterol_NORMAL']]

y = df['Code']

from sklearn.model_selection import train_test_split as tts

X_train,X_test,y_train,y_test = tts(X, y, test_size = 0.33, random_state = 7)

for setd,name in zip([X_train,X_test,y_train,y_test],['X_train','X_test','y_train','y_test']):

  print(name + ' shape : ' + str(setd.shape))
import tensorflow as tf

from tensorflow import keras
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 128,input_shape = (9,),activation = 'relu'))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units = 64,activation = 'relu'))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units = 32,activation = 'relu'))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units = 5,activation= 'softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics= ['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train, epochs = 1500,batch_size = 128)
model.summary()
# list all data in history

print(history.history.keys())
# summarize history for accuracy

plt.plot(history.history['sparse_categorical_accuracy'])

plt.rcParams["figure.figsize"] = (16,6)

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

plt.clf()
# summarize history for loss

plt.plot(history.history['loss'])

plt.rcParams["figure.figsize"] = (16,6)

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

plt.clf()
test_loss,test_accuracy = model.evaluate(X_test,y_test)

print('Test Loss ' + str(test_loss))

print('Test Accuracy ' + str(test_accuracy))