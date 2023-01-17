# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/names"))
df_male = pd.read_csv('../input/dinosaur-island/dinos.txt',header=None)

df_male.head()
examples = []

for name in df_male[0]:

    examples.append(list(name.lower()))
def one_hot_tr(name):

    name_oh = []

    for ch in name:

        temp = np.zeros([27,])

        temp[ord(ch)-ord('a')] = 1

        name_oh.append(temp)

    return name_oh
def one_hot_ts(name):

    name_oh = []

    for ch in name[1:]:

        temp = np.zeros([27,])

        temp[ord(ch)-ord('a')] = 1

        name_oh.append(temp)

    return name_oh
def sequence_maker(examples):

    X = np.zeros([len(examples),40,27])

    Y = np.zeros([len(examples),40,27])

    for i,name in enumerate(examples):

        print(name)

        try :

            temp1 = np.array(one_hot_tr(name))

        except :

            continue

        X[i,0:temp1.shape[0],:] = temp1

        X[i,temp1.shape[0]:,26] = 1

        temp2 = np.array(one_hot_ts(name))

        Y[i,0:temp2.shape[0],:] = temp2

        Y[i,temp2.shape[0]:,26] = 1

    return X,Y
X,Y = sequence_maker(examples)
X.shape
def oh_decode(oh):

    name = []

    for i in range(oh.shape[0]):

        if np.argmax(oh[i])<26 :

            ch = char(ord('a')+np.argmax(oh[i]))

        else :

            ch = ' '

    return name
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM,SimpleRNN
model = Sequential()

model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))

model.add(LSTM(50, return_sequences=True))

model.add(Dense(27, activation='softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=500)
from keras.models import load_model

model = load_model('../input/model1h5/my_model(1).h5')
def forward_prop(model,in_seed):

    test_x = np.zeros([1,40,27])

    for i in range(len(in_seed)):

        test_x[0][i][ord(in_seed[i])-ord('a')] = 1

    c = len(in_seed)-1

    name = in_seed

    while True :

        if c == 39 :

            print("yes")

            break

        temp_y = model.predict(test_x)[0][c]

        test_y = chr(ord('a') + np.random.choice(list(range(27)),p=temp_y.ravel()))

        temp_y = np.zeros([27,])

        temp_y[ord(test_y)-ord('a')] = 1

        test_x[0][c+1] = temp_y  

        c = c + 1

        if ord(test_y)-ord('a') < 26 :

            name = name + test_y

        else :

            break

    return name 
for ch in range(26):

    print(forward_prop(model,chr(ord('a')+ch)))
for _ in range(5):

    print((forward_prop(model,'ke')))
model.save('my_model.h5')
from IPython.display import FileLink, FileLinks

FileLinks('.') 