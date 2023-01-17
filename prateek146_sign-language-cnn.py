import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dropout,Dense,Conv2D,MaxPooling2D,Flatten,LeakyReLU
train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')
test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')
train.head()
train.shape
labels=train['label'].values
train.drop('label',axis=1,inplace=True)
plt.figure(figsize=(18,8))
sns.countplot(x=labels)

images=train.values
images=np.array([np.reshape(i,(28,28)) for i in images])
images=np.array([i.flatten() for i in images])
images.shape
from IPython.display import Image
Image('../input/sign-language-mnist/amer_sign2.png')
from sklearn.preprocessing import LabelBinarizer
labelbinarizer=LabelBinarizer()
labels=labelbinarizer.fit_transform(labels)
plt.imshow(images[0].reshape(28,28))
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(images,labels,test_size=0.3,random_state=101)
train_X=train_X/255
test_X=test_X/255
train_X=train_X.reshape(train_X.shape[0],28,28,1)
test_X=test_X.reshape(test_X.shape[0],28,28,1)
train_X.shape
plt.imshow(train_X[0].reshape(28,28))
epochs=50
num_classes=24
batch_size=128
model=Sequential()
# Block 1
model.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))
model.add(LeakyReLU())
model.add(Conv2D(32,3, padding  ="same"))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64,3, padding  ="same"))
model.add(LeakyReLU())
model.add(Conv2D(64,3, padding  ="same"))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(num_classes,activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

history=model.fit(train_X,train_y,validation_data=(test_X,test_y),epochs=epochs,batch_size=batch_size)

labels_test=test['label'].values
test.drop('label',axis=1,inplace=True)

images_test=test.values
images_test=np.array([np.reshape(i,(28,28)) for i in images_test])
images_test=np.array([i.flatten() for i in images_test])
images_test.shape

labels_test=labelbinarizer.fit_transform(labels_test)

images_test=images_test.reshape(images_test.shape[0],28,28,1)

y_pred=model.predict(images_test)

from sklearn.metrics import accuracy_score
accuracy_score(labels_test, y_pred.round())
