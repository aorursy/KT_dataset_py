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
train_df = pd.read_csv('../input/train_V2.csv')
train_df.head()
cols_to_keep = [

#     'Id', # Unique identifier of person

    'killPlace', # May be required

    'kills', # more kills more chances to survive

    'matchDuration', # Longer to live

    'matchType', # In which match longest lived

    'maxPlace', # Continuous Roaming in map

#     'rankPoints', # Needs to be combined with matchType for more effect

    'rideDistance', # More vehicle use, less chances to get killed

    'swimDistance', # More travel, less chances to get killed

    'walkDistance', # More travel, less chances to get killed

    'winPlacePerc', # Label

]
for col in train_df.columns:

    if col not in cols_to_keep:

        train_df = train_df.drop(col, axis=1)
train_df.head()
train_df['matchType'].unique()
dict_map = {

    'squad-fpp': 0,

    'duo': 1,

    'solo-fpp': 2,

    'squad': 3,

    'duo-fpp': 4,

    'solo': 5,

    'normal-squad-fpp': 6,

    'crashfpp': 7,

    'flaretpp': 8,

    'normal-solo-fpp': 9,

    'flarefpp': 10,

    'normal-duo-fpp': 11,

    'normal-duo': 12,

    'normal-squad': 13,

    'crashtpp': 14,

    'normal-solo': 15

}
train_df['matchType'] = train_df['matchType'].map(dict_map)
train_df.head()
from sklearn import preprocessing

x = train_df.drop('winPlacePerc', axis=1).values

y = train_df['winPlacePerc'].values

min_max_scaler = preprocessing.MinMaxScaler()

x = min_max_scaler.fit_transform(x)

df = pd.DataFrame(x)
df.head()
print(x.shape)

print(y.shape)



from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.optimizers import SGD



dropout = False

dropoutVal = 0.2

epochs = 2

batch_size = 512
def model():

    model = Sequential()

    model.add(Dense(500, input_shape=(8, ), activation='relu'))

    if dropout:

        model.add(Dropout(dropoutVal))

    model.add(Dense(300, activation='relu'))

    if dropout:

        model.add(Dropout(dropoutVal))

    model.add(Dense(50, activation='relu'))

    if dropout:

        model.add(Dropout(dropoutVal))

    model.add(Dense(1, activation='relu'))

    

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model
model =  model()
history = model.fit(xTrain, yTrain, epochs=epochs, validation_data=(xTest, yTest), batch_size=batch_size)
test_df = pd.read_csv('../input/test_V2.csv')
test_df.columns
index_df = test_df['Id']

for col in test_df.columns:

    if col not in cols_to_keep:

        test_df = test_df.drop(col, axis=1)
test_df['matchType'] = test_df['matchType'].map(dict_map)
test_df.head()
t = test_df.values

min_max_scaler = preprocessing.MinMaxScaler()

t = min_max_scaler.fit_transform(t)
pred = model.predict(t)
final_sub = []

for p in pred:

    final_sub.append(p[0])



submission = pd.DataFrame({

    'Id': index_df,

    'winPlacePerc': final_sub

})
submission.to_csv('submission.csv', index=False)