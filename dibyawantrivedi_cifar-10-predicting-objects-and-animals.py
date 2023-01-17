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
from tensorflow.keras.datasets import cifar10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train.shape
#scaling the data

x_test=x_test/255

x_train=x_train/255
from tensorflow.keras.utils import to_categorical

y_c_train=to_categorical(y_train)

y_c_test=to_categorical(y_test)
y_c_train.shape
#building the model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout



model=Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (32,32,3)))



model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (32,32,3)))



model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(256,activation='relu'))



model.add(Dropout(0.5))

#output layer

model.add(Dense(10,activation='softmax'))
#compiling the model

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping

early_stop=EarlyStopping(monitor='val_accuracy',patience=2)



model.fit(x_train,y_c_train,epochs=25,validation_data=(x_test,y_c_test),callbacks=[early_stop])
model.history.history
df=pd.DataFrame(model.history.history)
df[['accuracy','val_accuracy']].plot()

df[['loss','val_loss']].plot()
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix

preds=model.predict_classes(x_test)

print(classification_report(y_test,preds))

plt.figure(figsize=(12,6))

sns.heatmap(confusion_matrix(y_test,preds),annot=True)
my_img=x_test[15]

plt.imshow(my_img)

preds=model.predict_classes(my_img.reshape(1,32,32,3))
print(preds)
y_test[15]