import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

train_set = pd.read_csv('../input/digit-recognizer/train.csv')

sample_set = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

test_set = pd.read_csv('../input/digit-recognizer/test.csv')

test = test_set.values
test.shape
train_set.head()

train_set.shape

X = train_set.drop(['label'], axis=1)

y = train_set.iloc[:,0]

train_set.dtypes

plt.bar(y.value_counts().index,y.value_counts())

print(X.describe())



#Train Test Split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=42)
#check for the null values

print(X_train.isnull().any().describe())

print(X_test.isnull().any().describe())

#Normalization

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)



#One Hot Encoding

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

#Model

model = tf.keras.Sequential()

model.add(Dense(units = 784/2, activation = 'relu', input_dim=784))

model.add(Dense(units = 784/4, activation = 'relu'))

model.add(Dense(units = 784/8, activation = 'relu'))

model.add(Dense(units = 784/16, activation = 'relu'))

model.add(Dense(units = 784/32, activation = 'relu'))

model.add(Dense(units = 10, activation = 'softmax'))
#Compiling the model and fitting

model.compile(loss="categorical_crossentropy",

              optimizer="sgd",

              metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size=100, epochs=100)
#prediction

predicted = model.predict(X_test)

y_head = predicted.argmax(axis=1).reshape(-1,1)

#Evaluation

print(accuracy_score(y_test.argmax(axis=1), y_head))

cm = confusion_matrix(y_test.argmax(axis=1),y_head)

sns.heatmap(cm, annot=True)
# Making Predictions on Test Data

y_head_test = model.predict(test)

result_test = y_head_test.argmax(axis=1)

#Visualising predictions

for i in range(1,5):

    index = np.random.randint(1,28001)

    plt.subplot(3,3,i)

    plt.imshow(test[index].reshape(28,28))

    plt.title("Predicted Label : {} ".format(result_test[index]))

plt.subplots_adjust(hspace = 1.2, wspace = 1.2)

plt.show()
result_test = pd.Series(result_test,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result_test],axis = 1)

submission.to_csv("submission.csv",index=False)