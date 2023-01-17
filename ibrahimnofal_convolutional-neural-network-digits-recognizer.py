from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix


import seaborn as sns
import matplotlib.pyplot as plt

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
submission=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train.head()
print("Data Describs :", train.describe())
print("Data Info :" ,train.info())
print("Data Columns :" ,train.columns)
print("Data Describs :", test.describe())
print("Data Info :" ,test.info())
print("Data Columns :" ,test.columns)
test.shape,train.shape
plt.figure(figsize=(10,5))
ax=sns.countplot(train.label,palette='icefire')
plt.title('Number of Digits Class');
labels=np.unique(train.label)
labels
img_size = 28

train_piksel = np.array(train.drop("label",axis=1))
test_piksel = np.array(test)
test_piksel = test_piksel.reshape(test.shape[0],img_size,img_size,1)

plt.subplot(1,2,1)
plt.imshow(train_piksel[1].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(train_piksel[15].reshape(img_size, img_size))
plt.axis('off')
plt.show()
import keras
from sklearn.model_selection import train_test_split
num_classes = 10
img_size=28
X_train, X_test, Y_train, Y_test = train_test_split(train_piksel,train.loc[:,"label"],test_size=0.3)

X_train = X_train.reshape(X_train.shape[0],img_size,img_size,1)
X_test = X_test.reshape(X_test.shape[0],img_size,img_size,1)

Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

print("X_train.shape :",X_train.shape)
print("X_test.shape :",X_test.shape)
print("Y_train.shape :",Y_train.shape)
print("Y_test.shape :",Y_test.shape)
# Some examples
# plot some samples
plt.figure(figsize=(10,10))
for i in range(10):
    img=X_train[i][:,:,0]
    img=img.reshape(28,28)
    ax = plt.subplot(5, 5, i+1)
    plt.imshow(img)
    plt.axis('off');
 
# Model Libraries
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ReduceLROnPlateau
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

from keras.layers import MaxPooling2D,Input
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import Model

def get_model():
    In = Input(shape=(28,28,1))
    x = Conv2D(32, (5,5), padding="same", activation="relu")(In)
    x = Conv2D(32, (5,5), activation="relu")(x)
    x = Conv2D(32, (5,5), activation="relu")(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (5,5), padding="same", activation="relu")(x)
    x = Conv2D(64, (5,5), activation="relu")(x)
    x = Conv2D(64, (5,5), activation="relu")(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    
    Out = Dense(10, activation="softmax")(x)
    
    model = Model(In, Out)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model



model=get_model()
model.summary()
epochs = 20
batch_size = 32

# Compile the model
model.compile(optimizer = optimizer ,
              loss = "categorical_crossentropy",
              metrics=["accuracy"])
history=model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size)
# evaluate the model
_, train_acc = model.evaluate(X_train, Y_train, verbose=0)
_, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.legend()
# plot accuracy during training
print('\n')
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.legend()
plt.show()
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(Y_test,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
predictions=model.predict(X_test)
# First prediction
index = 42
print(predictions[index])
print(f"Max value (probability of prediction): {np.max(predictions[index])}")
print(f"Sum: {np.sum(predictions[index])}")
print(f"Max index: {np.argmax(predictions[index])}")
print(f"Predicted label: {labels[np.argmax(predictions[index])]}")
# Some examples
# plot some samples
plt.figure(figsize=(10,10))
for i in range(10):
    img=X_test[i][:,:,0]
    img=img.reshape(28,28)
    ax = plt.subplot(5, 5, i+1)
    plt.axis('off');
    pred_prob, true_label = predictions[i], labels[i]

  # Get the pred label
    pred_label = labels[np.argmax(predictions[i])]

  # Plot image & remove ticks
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

  # Change the colour of the title depending on if the prediction is right or wrong
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"
  
  # Change plot title to be predicted, probability of prediction and truth label
    plt.title("{} {:2.0f}% {}".format(pred_label,
                                    np.max(pred_prob)*100,
                                    ''),
                                    color=color)
 
#set ids as ImageId and predict label 
ids= submission.drop("Label",axis=1)
predict = model.predict(test_piksel)
predict = np.argmax(predict,axis = 1) 

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'ImageId' : ids.ImageId, 'Label': predict})
output.to_csv('submission.csv', index=False)
output.head()
