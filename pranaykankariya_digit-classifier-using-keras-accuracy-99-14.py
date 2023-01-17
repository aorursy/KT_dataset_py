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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score,precision_score,recall_score,plot_precision_recall_curve
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential,Model
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,GlobalMaxPooling2D,BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.optimizers import RMSprop,Adam
from keras.utils import to_categorical

#Load the Data
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
x_train = train.drop(labels=['label'],axis=1)
y_train = train['label']
#Count 
count = y_train.value_counts()
plt.figure(figsize=(8,6))
sns.countplot(y_train)
print(count)
#Normalize
x_train = x_train/255.0
test = test/255.0

#Reshape
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
#Images
index = np.random.randint(0,42000)
img = x_train[index][:,:,0]
plt.imshow(img)
plt.title(y_train[index])
#Split Data
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=42)
model = Sequential()

model.add(Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(10,activation='softmax'))

model.compile(RMSprop(lr=0.001,rho=0.9),loss='sparse_categorical_crossentropy',metrics=['acc'])
model.summary()

train_datagen = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=False,
                                   fill_mode='nearest')
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train,y_train,batch_size=128)

#Callbacks
earlystop = EarlyStopping(monitor='val_loss',patience=2,verbose=1)
learning_reduce = ReduceLROnPlateau(patience=2,monitor="val_acc",verbose=1,min_lr=0.00001,factor=0.5)
callbacks = [earlystop,learning_reduce]
history = model.fit_generator(train_generator,epochs=30,verbose=1,validation_data=(x_val,y_val),callbacks=callbacks)
#Visualize Training
def plot_graphs(history, string):
    plt.figure(figsize=(8,8))
    plt.plot(history.history[string])
    plt.plot(history.history["val_"+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string,"val_"+string])
    plt.show()
plot_graphs(history,'acc')
plot_graphs(history,'loss')
loss,accuracy = model.evaluate(x_val,y_val,verbose=1)
print("Validation Loss: ",loss)
print("Validation Accuracy: ",accuracy)
y_pred = model.predict(x_val)
y_pred = np.argmax(y_pred,axis=1)

confusion = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10,10))
sns.heatmap(confusion, annot=True,cmap="Blues",fmt='.1f')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
recall = recall_score(y_val,y_pred,average='micro')
precision = precision_score(y_val,y_pred,average='micro')
print("Precision: ",precision)
print("Recall: ",recall)
predict = model.predict(test)
predict = np.argmax(predict,axis=1)
result = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
result['Label'] = predict
result.to_csv("submission.csv",index=False)