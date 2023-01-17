# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_df.sample(11)
train_df.iloc[1,1:].values.reshape(28,28)
x = np.random.randint(1000)

plt.imshow(train_df.iloc[x,1:].values.reshape(28,28),cmap = 'gray')

plt.title('Label :'.format({str(train_df.iloc[1,0])}))

plt.title(train_df.iloc[x,0])

plt.colorbar()

plt.show()
fig = plt.figure(figsize=(10,10))

for i in range(1,10):

    x = np.random.randint(1000)

    fig.add_subplot(3,3,i)

    plt.imshow(train_df.iloc[x,1:].values.reshape(28,28),cmap = 'gray')

    plt.title('Label :'.format({str(train_df.iloc[x,0])}))

    plt.title(train_df.iloc[x,0])

    plt.axis('off')

    plt.show()
from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid = train_test_split(train_df.iloc[:,1:].values,train_df.loc[:,'label'].values,test_size = 0.20,random_state=11)
X_train = X_train/255.0

X_valid = X_valid/255.0
X_train

X_valid
np.unique(train_df.loc[:,'label'].values)
from keras.utils import to_categorical
y_train,y_valid=to_categorical(y_train, num_classes=10),to_categorical(y_valid, num_classes=10)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

model = Sequential()

model.add(Dense(100, input_shape= (784, )))

model.add(Dense(200, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(200, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(200, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(200, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(200, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))
model.summary()
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
history = model.fit(X_train,y_train,epochs=10,validation_data=(X_train,y_train))
plt.figure(figsize= (12,6))

hist = history.history

plt.plot(range(1,11),hist['accuracy'],label = 'Accuracy')

plt.plot(range(1,11),hist['val_accuracy'],label = 'Val_Acc')

plt.legend()

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.show()
loss,acc = model.evaluate(X_valid,y_valid)

print('Loss:',loss,'\nAccuracy:',acc)
model.predict(X_valid[10].reshape(-1,784)),np.argmax(model.predict(X_valid[10].reshape(-1,784)))
x = np.random.randint(1000)

plt.imshow(X_valid[x].reshape(28, 28), cmap='gray')

plt.title("True Label: {} Predict: {}".format(np.argmax(y_valid[x]), np.argmax(model.predict(X_valid[x].reshape(-1, 784)))))

plt.colorbar()

plt.show()
X_test = test_df.values.reshape(-1, 784)

X_test.shape
sub_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

sub_df.head()
pred = np.argmax(model.predict(X_test), axis = 1)
pred.shape
sub_df['Label'] = pred
sub_df.head()
sub_df.to_csv('mnist_su.csv', index=False)