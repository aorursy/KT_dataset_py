# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data_train=pd.read_csv('../input/train.csv')
data_train.head(5)
digit=data_train['label']
data_train=data_train.drop('label',1)
from keras.utils import to_categorical
y=to_categorical(digit,num_classes=10)
data_train = data_train.values.reshape(-1,28,28,1)
plt.imshow(data_train[100][:,:,0])
data_train.shape
data_train=data_train.astype('float32')/255
from keras.layers import Conv2D, MaxPooling2D,Dropout,Dense, Flatten
from keras.models import Sequential
model=Sequential()
model.add(Conv2D(224,3,activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(224,3,activation='relu'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Conv2D(160,3,activation='relu'))
model.add(Conv2D(160,3,activation='relu'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Conv2D(160,2,activation='relu'))
model.add(Conv2D(160,2,activation='relu'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(len(set(digit)),activation='softmax'))
model.compile(optimizer = 'Adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(data_train,y,validation_split=0.1,epochs=5,batch_size=32, verbose=1)
tst=pd.read_csv('../input/test.csv')
tst = tst.values.reshape(-1,28,28,1)
tst=tst.astype('float32')/255
pred=model.predict(tst)
acc=[]
for i in pred:
    acc.append(np.argmax(i))
id=list(range(1,len(tst)+1))
predictions_csv={'ImageId':id,'Label':acc}
predictions_csv=pd.DataFrame(predictions_csv,
                      columns=['ImageId','Label'])
predictions_csv.to_csv('submission_3.csv',header=True,mode='a', index=False)
