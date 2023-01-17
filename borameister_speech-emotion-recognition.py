# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
path = '../input/ravdessfractals'

# d = {'W':[], 'N':[], 'F':[], 'D':[], 'S':[], 'A':[], 'J':[]}
d = {'Angry':[], 'Calm':[], 'Disgust':[], 'Fear':[], 'Happy':[], 'Neutral':[], 'Sad':[], 'Suprise':[]}
for e in os.listdir(path):
	for file in os.listdir(os.path.join(path, e)):
		img = cv2.imread(os.path.join(path,e,file),cv2.IMREAD_GRAYSCALE)
		a = np.reshape(img, (480, -1))[58:427, 80:576] # (369,496)
		a = cv2.resize(a, (int(a.shape[1]/5), int(a.shape[0]/5)) )
		b = a.ravel()
        
		d[e].append(b)
for i,key in zip(range(len(d)),d):
    globals()[key] = pd.DataFrame(np.array(d[key]))

    globals()[key]['label'] = i
    
    col_list = list(globals()[key].columns[-1:])+list(globals()[key].columns[:-1])
    globals()[key] = globals()[key][col_list]
# Concatenate
df = pd.concat([globals()[df] for df in d], ignore_index=True)
df = df.reset_index(drop=True)

# TTS
x = df.drop(['label'], axis=1).values/255
x = x.reshape(-1, a.shape[0], a.shape[1], 1)

y = to_categorical(df.label, num_classes=df.label.nunique())
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#check shapes

tr = np.bincount(np.argmax(y_train, axis=1))
ts = np.bincount(np.argmax(y_test, axis=1))

print('Number of Samples for Each Label\n')

splits = [', '.join(['train='+ str(i), 'test='+ str(j)]) for i,j in zip(tr,ts)]
for i,j in zip(range(len(splits)), splits):
    print('label= {} : {}'.format(i,j))
# FW 1

def model1():
    model = Sequential()
    model.add(Conv2D(32, 5, activation='relu', input_shape=x.shape[1:]))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))


    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(df.label.nunique(), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
# FW 2
def model2():

    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', input_shape=x.shape[1:]))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(df.label.nunique(), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
# FW 3
def model3():

    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', input_shape=x.shape[1:]))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(df.label.nunique(), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
# FW 4
def model4():

    model = Sequential()
    model.add(Conv2D(32, 5, activation='relu', input_shape=x.shape[1:]))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(df.label.nunique(), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
# FW 5
def model5():

    model = Sequential()
    model.add(Conv2D(60, 8, activation='relu', input_shape=x.shape[1:]))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))
    model.add(Conv2D(120, 5, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))
    model.add(Conv2D(240, 5, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(df.label.nunique(), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
# model1()
model2()
# model3()
# model4()
# model5()


model.fit(x_train, y_train, batch_size=32, epochs=25, verbose=1, validation_split=0.25)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
y_pred = model.predict(x_test)

y_predicted = np.argmax(y_pred, axis = 1) 
y_true = np.argmax(y_test, axis = 1) 
# print(y_predicted.shape)
# print(y_true.shape)

from sklearn.metrics import accuracy_score, f1_score
print('test accuracy: ',accuracy_score(y_true, y_predicted))
print('f1 score: ',f1_score(y_true, y_predicted, average='macro'))
cm = confusion_matrix(y_true, y_predicted)
plt.figure(figsize=(12,8))
plt.yticks(rotation=60)

g=sns.heatmap(cm, annot=True, linewidths=0.2, cmap="Reds",linecolor="black",  fmt= '.1f',
            xticklabels=d.keys())
g.set_yticklabels(d.keys(), rotation=0)
plt.title('  -  '.join([':'.join([j,str(i)]) for i,j in zip(np.bincount(y_true), d.keys())]))
plt.show()

