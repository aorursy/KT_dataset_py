import numpy as np 

import pandas as pd 

import os

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv',error_bad_lines=False)

train
str_ = train['Size']

enc_str_ = [0 for i in range(len(str_))]

for i in range(len(str_)):

    if(str_[i]=='Medium'):

        enc_str_[i]=1

    elif(str_[i]=='Small'):

        enc_str_[i]=2

    elif(str_[i]=='Big'):

        enc_str_[i]=3

    else:

        str_[i]=-1

train['Size'] = enc_str_

train.head()
dec_ = train['Score']

int_ = [0 for i in range(len(dec_))]

for i in range(len(int_)):

    int_[i] = int(round(dec_[i]))

train['Score'] = int_

train.head()
X = train['Difficulty']

y = 0.0

z = 0

for i in range(len(X)):

    if(X[i]!='?'):

        y = y + float(X[i])

        z = z+1

y = y/z

for i in range(len(X)):

    if(X[i]=='?'):

        X[i] = float(y)

    else:

        X[i] = float(X[i])

Y = [0 for x in range(len(X))]

for i in range(len(Y)):

    Y[i] = int(100*round(X[i],2))

train['Difficulty']=Y

train.head()
X = train['Number of Quantities']

Y = [0 for x in range(len(X))]

for x in range(len(X)):

    if(X[x]=='?'):

        Y[x]=2

    else:

        Y[x] = int(X[x])

train['Number of Quantities']=Y



X = train['Number of Insignificant Quantities']

Y = [0 for x in range(len(X))]

print(X.value_counts())

for x in range(len(X)):

    if(X[x]=='?'):

        Y[x]=0

    else:

        Y[x]=int(X[x])

train['Number of Insignificant Quantities']=Y



X = train['Total Number of Words']

Y = [0 for x in range(len(X))]

for x in range(len(Y)):

    if(X[x]=='?'):

        Y[x]=20

    else:

        Y[x]=int(X[x])

train['Total Number of Words']=Y



X = train['Number of Special Characters']

Y = [0 for x in range(len(X))]

print(X.value_counts())

for x in range(len(Y)):

    if(X[x]=='?'):

        Y[x]=3

    else:

        Y[x]=int(X[x])

train['Number of Special Characters']=Y

train.head()
train.info()
Y = train['Class']

train_ = train.drop(['ID', 'Class'], axis=1)

train_
train_=train_.apply(pd.to_numeric, errors='coerce').dropna()

train_
from keras.layers import Conv2D , MaxPooling2D , Dense, Flatten , Input , Dropout

from keras.models import Sequential , Model

import keras

import tensorflow as tf

from PIL import Image

from keras.models import model_from_json

import os

from keras.preprocessing.image import ImageDataGenerator

from keras import utils as np_utils  

from matplotlib import image
Y = np.asarray(Y)

Y = np_utils.to_categorical(Y)

X = np.asarray(train_)

X=np.reshape(X, (371,11,1))
NN_model = Sequential()

NN_model.add(Dense(63, input_shape=(11,1), activation='relu'))

NN_model.add(Dense(63, activation='relu'))

NN_model.add(Dense(63, activation='relu'))

NN_model.add(Flatten())

NN_model.add(Dense(6, activation='softmax'))

NN_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

NN_model.fit(X , Y , batch_size = 20 , epochs = 300 , validation_split = 0.1)
test = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", sep=',')

#test=test.drop(columns=['Number of Quantities','Total Number of Words','Number of Sentences'])

test
str_ = test['Size']

enc_str_ = [0 for i in range(len(str_))]

for i in range(len(str_)):

    if(str_[i]=='Medium'):

        enc_str_[i]=1

    elif(str_[i]=='Small'):

        enc_str_[i]=2

    elif(str_[i]=='Big'):

        enc_str_[i]=3

    else:

        str_[i]=-1

test['Size'] = enc_str_



dec_ = test['Score']

int_ = [0 for i in range(len(dec_))]

for i in range(len(int_)):

    int_[i] = int(round(dec_[i]))

test['Score'] = int_



X = test['Difficulty']

y = 0.0

z = 0

for i in range(len(X)):

    if(X[i]!='?'):

        y = y + float(X[i])

        z = z+1

y = y/z

for i in range(len(X)):

    if(X[i]=='?'):

        X[i] = float(y)

    else:

        X[i] = float(X[i])

Y = [0 for x in range(len(X))]

for i in range(len(Y)):

    Y[i] = int(100*round(X[i],2))

test['Difficulty']=Y



X = test['Number of Quantities']

Y = [0 for x in range(len(X))]

for x in range(len(X)):

    if(X[x]=='?'):

        Y[x]=2

    else:

        Y[x] = int(X[x])

test['Number of Quantities']=Y



X = test['Number of Insignificant Quantities']

Y = [0 for x in range(len(X))]

print(X.value_counts())

for x in range(len(X)):

    if(X[x]=='?'):

        Y[x]=0

    else:

        Y[x]=int(X[x])

test['Number of Insignificant Quantities']=Y



X = test['Total Number of Words']

Y = [0 for x in range(len(X))]

for x in range(len(Y)):

    if(X[x]=='?'):

        Y[x]=20

    else:

        Y[x]=int(X[x])

test['Total Number of Words']=Y



X = test['Number of Special Characters']

Y = [0 for x in range(len(X))]

print(X.value_counts())

for x in range(len(Y)):

    if(X[x]=='?'):

        Y[x]=3

    else:

        Y[x]=int(X[x])

test['Number of Special Characters']=Y



test
Q = test['ID']

test=test.drop({'ID'},axis=1)

X = np.asarray(test)

X = np.reshape(X,(159,11,1))

Yt = NN_model.predict(X)

d = {'ID':Q, 'Class':np.argmax(Yt,axis=1)}

submission = pd.DataFrame(data=d)

submission
submission.to_csv('output.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(submission, title = "Download CSV file", filename = "data.csv"):

    csv = submission.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(submission)