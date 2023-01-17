import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

os.listdir("../input")
train = pd.read_csv('../input/train.csv')

test =  pd.read_csv('../input/test.csv')
train.head(3)
test.head(3)
x_train = train.drop('label',axis=1)

y_train = train.drop(x_train.columns,axis=1)

x_train = np.array(x_train)

y_train = np.array(y_train)
test = np.array(test)
x_train.shape
x_train = x_train.reshape(42000,28,28,1)
x_train.shape,y_train.shape,test.shape
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization

import tensorflow as tf
tf.reset_default_graph()
model = Sequential()



model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

model.add(BatchNormalization())



model.add(Conv2D(32,kernel_size=3,activation='relu'))

model.add(BatchNormalization())



model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

    

model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

    

model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

    

model.add(Dense(10, activation='softmax'))
model.summary()
model.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,

              y_train,

              batch_size=256,

              epochs=15,

              callbacks=None,

              validation_split=.1)
pred = model.predict(test.reshape(28000,28,28,1))
from sklearn.metrics import accuracy_score
pred_ix = [list(row).index(np.max(row))for row in pred]
sub = pd.read_csv('../input/sample_submission.csv')
sub.head()
label = np.array(sub['Label'])
label = pred_ix
label = pd.Series(label,name ="Label" )
sample_submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), label], axis=1)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "mnist_results1.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(sample_submission)