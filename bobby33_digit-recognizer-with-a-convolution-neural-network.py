
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,Dropout
from keras.layers.core import Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping ,ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import SGD, RMSprop
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/train.csv")
df_test=pd.read_csv("../input/test.csv")
df_train.info()
df_train.head()

predictors=np.array(df_train.drop(columns=['label']))
target=np.array(df_train["label"])
test=np.array(df_test)
images=predictors.reshape(-1,28,28)
images[0].shape
f, axs = plt.subplots(1,10,figsize=(15,15))  
axs.ravel()
for k in range(0,10):
    axs[k].imshow(images[k],cmap="gray_r")
image_three=df_train[df_train["label"]==3]
image_three=np.array(image_three.drop(columns=['label']))
image_three=image_three.reshape(-1,28,28)
f, axs = plt.subplots(5,5,figsize=(15,15))  
axs.ravel()
i=0
for k in range(0,5):
    for j in range(0,5):
        axs[j,k].imshow(image_three[i],cmap="gray_r")
        i+=1
model_test = Sequential()
model_test.add(Conv2D(50, kernel_size=5, padding="same",input_shape=(28, 28, 1),activation='relu'))
image=image_three[0]
plt.imshow(image,cmap="gray_r")
image=image.reshape(-1,28,28,1)
image.shape
conv_image=model_test.predict(image)
conv_image=np.squeeze(conv_image,axis=0)
conv_image.shape
f, axs = plt.subplots(5,10,figsize=(15,15))  
axs.ravel()
i=0
for k in range(0,10):
    for j in range(0,5):
        axs[j,k].imshow(conv_image[:,:,i],cmap="gray_r")
        i+=1
model_test.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
conv_image=model_test.predict(image)
conv_image=np.squeeze(conv_image,axis=0)
conv_image.shape
f, axs = plt.subplots(5,10,figsize=(15,15))  
axs.ravel()
i=0
for k in range(0,10):
    for j in range(0,5):
        axs[j,k].imshow(conv_image[:,:,i],cmap="gray_r")
        i+=1
predictors=predictors/255
test=test/255
target=to_categorical(target)
print(target)
predictors=predictors.reshape(-1,28,28,1)
test=test.reshape(-1,28,28,1)
X_train, X_test, y_train, y_test = train_test_split(predictors,target,test_size=0.2, random_state=42)
model = Sequential()
model.add(Conv2D(50, kernel_size=5, padding="same",input_shape=(28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(80, kernel_size=5, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs= 15 , batch_size=200, validation_split = 0.2)
result_test=model.predict(predictors)
# select the index with the maximum probability
result_test = np.argmax(result_test,axis=1)
result_test[456]
plt.imshow(predictors[456].reshape(28,28),cmap="gray_r")
results=model.predict(test)
results=np.argmax(results,axis=1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("result.csv",index=False)