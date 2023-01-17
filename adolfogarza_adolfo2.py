# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD, Adam, RMSprop, Adadelta
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

import numpy as np

from sklearn.model_selection import train_test_split
#Read data for training

trainpath = '../input/learn-together/train.csv'

raw_dataset = pd.read_csv(trainpath)



# encode class values as integers

encoder = LabelEncoder()

encoder.fit(raw_dataset['Cover_Type'])

encoded_Y = encoder.transform(raw_dataset['Cover_Type'])



dummy_y = np_utils.to_categorical(encoded_Y)

dy = pd.DataFrame(dummy_y, columns=['CT1', 'CT2','CT3', 'CT4', 'CT5', 'CT6', 'CT7'])



raw_dataset = pd.concat([raw_dataset, dy], axis = 1)
#raw_dataset.describe().T
# Generate dummy data



X_train = ['Elevation', 'Aspect', 'Slope',

            'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

            'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

            'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',

             ]



y_train = ['CT1', 'CT2','CT3', 'CT4', 'CT5', 'CT6', 'CT7']

xdata = raw_dataset[X_train]

ydata = raw_dataset[y_train]



print(xdata.shape, ydata.shape)
raw_dataset.shape
X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.2, shuffle=True, random_state=1)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
X_test.head()
train_stats = X_train.describe()

#train_stats.pop('Cover_Type')

train_stats = train_stats.transpose()

train_stats
#Normailiza los datos para centrarlos alrededor del 0

def norm(x):

    return (x - train_stats['min']) / (train_stats['max']-train_stats['min'])



#Convierte los datos a normalizados.

normed_train_data = norm(X_train)

normed_test_data = norm(X_test)
normed_train_data.head()
model = Sequential()

model.add(Dense(64, activation='relu', input_dim=10))

model.add(Dense(128, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='softmax'))

model.add(Dense(7, activation='softmax'))
#optim = SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=False)

optim = Adam(lr=0.0016, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#optim = Adadelta(lr=0.001, rho=0.95, epsilon=None, decay=0.0)

#optim = RMSprop(lr=0.000001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',

              optimizer=optim,

              metrics=['accuracy'])
model.fit(normed_train_data, y_train,

          epochs=256,

          batch_size=32, verbose=1)
model.summary()
score = model.evaluate(normed_test_data, y_test, batch_size=32)
print(score)
print('Loss: ', score[0], 'acc: ', 100*score[1],'%')
#Read data for test

trainpath2 = '../input/learn-together/test.csv'

eval_X = pd.read_csv(trainpath2)
feat2 = ['Elevation', 'Aspect', 'Slope',

            'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

            'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

            'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',

             ]

eval_X[feat2].head(10)
normed_eval_X = norm(eval_X)
eval_y = model.predict_classes(normed_eval_X[feat2], verbose=1,batch_size=32)+1
eval_y
eval_X.head()
eval_X.set_index('Id', inplace= True)
pd.DataFrame({'Id': eval_X.index,

                       'Cover_Type': eval_y})
output = pd.DataFrame({'Id': eval_X.index,

                       'Cover_Type': eval_y})

output.to_csv('../input/submission.csv', index=False)
outpath = '../input/submission.csv'

outsubm = pd.read_csv(outpath)

outsubm.head(20)
outsubm['Cover_Type'].value_counts() 
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))