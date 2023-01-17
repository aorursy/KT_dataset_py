import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from tqdm import tqdm
warnings.filterwarnings('ignore')
train_data = pd.read_csv('../input/train.csv')
print (train_data.shape)
train_data.head()
test_data = pd.read_csv('../input/test.csv')
print (test_data.shape)
test_data.head()
test_data['winPlacePerc'] = 0
train_data.describe()
test_data_id = test_data['Id']
train_data = train_data.drop(['Id', 'groupId', 'matchId'], 1)
test_data = test_data.drop(['Id', 'groupId', 'matchId'], 1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
test_data = sc.fit_transform(test_data)
test_data = sc.transform(test_data)
y_train = train_data[['winPlacePerc']]
X_train = train_data.drop(['winPlacePerc'], 1)
X_train.shape
X_test = test_data
from keras import Sequential
from keras.layers import Dense, Dropout, Input
model = Sequential()
model.add(Dense(80,input_dim=X_train.shape[1],activation='selu'))
model.add(Dense(160,activation='selu'))
model.add(Dense(320,activation='selu'))
model.add(Dropout(0.1))
model.add(Dense(160,activation='selu'))
model.add(Dense(80,activation='selu'))
model.add(Dense(40,activation='selu'))
model.add(Dense(20,activation='selu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=70,batch_size=100000)
prediction = model.predict(X_test)
prediction = prediction.ravel()
prediction = pd.Series(prediction, name='winPlacePerc')
prediction
test_data['winPlacePerc'] = prediction
submission = test_data[['Id', 'winPlacePerc']]
submission.to_csv('pubg_submission.csv', index = False)
