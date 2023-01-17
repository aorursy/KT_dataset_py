import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from keras import layers, models

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

import os

print(os.listdir("../input"))
#Getting training and testing data in a pandas dataframe

train = pd.read_csv("../input/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashion-mnist_test.csv")
#Training Set consists of 60000 samples with one label field and 28*28 image pixels

train.shape
#Test Set consists of 10000 samples with one label field and 28*28 image pixels

test.shape
#No Null Values in training data set

train.isnull().values.any()
#No Null Values in test data set

test.isnull().values.any()
#Quick look into training data

train.head()
#Quick look into testing data

test.head()
#Maximium fileds has 255 as Max and 0 as Min as these are pixel values

#Labels are from 0 to 9, i.e. 10 values 

train.describe()
#Labels from 0 to 9 are following Items 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
X = train.drop('label',axis=1)

y = train['label']



X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=101)
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(np.array(X.iloc[i]).reshape(28,28), cmap=plt.cm.binary)

    plt.xlabel(class_names[y[i]])

plt.show()
# Rescaling training and test data to range 0 to 1 by dividing them by 255

X_train = X_train / 255

X_val = X_val / 255
model = models.Sequential()

model.add(layers.Dense(128,activation='relu',input_shape=(784,)))

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
model.fit(X_train,y_train,epochs=15,batch_size=128,validation_data=(X_val,y_val))
history_dict = model.history.history
history_dict.keys()
loss_value = history_dict['acc']

val_loss_value = history_dict['val_acc']
epochs = range(1,16)

plt.plot(epochs, loss_value, 'bo', label='Training loss')

plt.plot(epochs, val_loss_value, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
X_test = test.drop('label',axis=1) / 255

y_test = test['label']

predictions = model.predict(X_test)
predict_labels = []

for pred in predictions:

    predict_labels.append(np.argmax(pred))
mat = confusion_matrix(y_test, predict_labels)

plt.figure(figsize=(10, 10))

sns.set()

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,

            xticklabels=np.unique(y_test),

            yticklabels=np.unique(y_test))

plt.xlabel('true label')

plt.ylabel('predicted label')

plt.show()
print(classification_report(predict_labels,y_test))
test_loss, test_acc = model.evaluate(X_test,y_test)

print(test_loss, test_acc)