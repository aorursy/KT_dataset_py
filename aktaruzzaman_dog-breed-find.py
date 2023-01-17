import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
import cv2
import glob


%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
def train_impath():
    labels = pd.read_csv('../input/labels.csv')
    lab = labels.iloc[:,0].tolist()

    data_dir = []
    for l in lab:
        t_dir = "../input/train/"+l+".jpg"
        data_dir.append(t_dir)
    return data_dir
def test_impath():
    labels = pd.read_csv('../input/sample_submission.csv')
    lab = labels.iloc[:,0].tolist()

    data_dir = []
    for l in lab:
        t_dir = "../input/test/"+l+".jpg"
        data_dir.append(t_dir)
    return data_dir
train_dir = train_impath()
test_dir = test_impath()
# data: str-> name of the dataSet
def im_read(data_dir):
    df = []
    for path in data_dir[0:100]:
        img = cv2.imread(path,0)
        img = cv2.resize(img,(80,80)).reshape(80,80,1)
        df.append(img.tolist())
    return np.array(df, dtype=float).reshape(-1,80,80,1)/255
t = im_read(train_dir) #test dataSet
ts = im_read(test_dir) #train dataSet
lebels = pd.read_csv('../input/labels.csv')
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()
enc = OneHotEncoder()
#Labels
leb = lebels.iloc[:,1].value_counts().index.sort_values()
leb = [[l] for l in leb]
#Lebel Encoding
le = le.fit(leb)
lebels.breed = le.transform(lebels.breed)
#OneHotEncoding
leb_enc = le.transform(leb)
leb_enc = [[l] for l in leb_enc]
breed = lebels.breed.tolist()
breed = [[b] for b in breed]
enc = enc.fit(leb_enc)
breed_enc = enc.transform(breed).toarray()
X = t.copy()
del t
Y = breed_enc[0:100]
X_ts = ts.copy()
del ts
#import packages
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D,Flatten, LeakyReLU
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='linear', input_shape = (80,80,1)))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
#                  activation ='linear', input_shape = (200,200,1)))
# model.add(LeakyReLU(alpha=.001))
# model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = "linear"))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.5))
model.add(Dense(256, activation = "linear"))
model.add(LeakyReLU(alpha=.001))
#model.add(Dropout(0.5))
model.add(Dense(120, activation = "relu"))
from keras import optimizers
opt = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer = opt , loss = 'mse', metrics=['mae','accuracy'])
model.fit(X, Y, epochs=1, batch_size=100,validation_split = 0.2)
Y_ts = model.predict(X_ts)
labels = pd.read_csv('../input/labels.csv')
columns = labels.iloc[:,1].value_counts().index.sort_values()
ans = pd.DataFrame(Y_ts, columns = columns)
sub = pd.read_csv('../input/sample_submission.csv')
ids = pd.DataFrame(sub.id)
result = pd.concat([ids, ans], axis=1, sort=False)
result.to_csv('sub1.csv',index=False)