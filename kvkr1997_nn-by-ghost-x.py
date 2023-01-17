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
data=pd.read_csv('../input/train_V2.csv')
#adding team_size and match_size as features
data=data.merge(data['groupId'].value_counts().to_frame(),left_on='groupId',right_index=True)
data=data.merge(data['matchId'].value_counts().to_frame(),left_on='matchId',right_index=True)
data['team_size']=data['groupId_y']
data['match_size']=data['matchId_y']
data = data.drop("groupId_y", axis=1)
data = data.drop("matchId_y", axis=1)
data = data.drop("matchId_x", axis=1)
data = data.drop("groupId_x", axis=1)

data['duo']=(data['matchType']=='duo') | (data['matchType']=='duo-fpp') | (data['matchType']=='normal-duo') | (data['matchType']=='normal-duo-fpp')
data['squad']=(data['matchType']=='squad') | (data['matchType']=='squad-fpp') | (data['matchType']=='normal-squad-fpp') | (data['matchType']=='normal-squad')
data['solo']=(data['matchType']=='solo') | (data['matchType']=='solo-fpp') | (data['matchType']=='normal-solo') | (data['matchType']=='normal-solo-fpp')
data['others']=(data['matchType']=='crashfpp') | (data['matchType']=='flaretpp') | (data['matchType']=='flarefpp') | (data['matchType']=='crashtpp')

data['duo']=data['duo'].astype(int)
data['squad']=data['squad'].astype(int)
data['solo']=data['solo'].astype(int)
data['others']=data['others'].astype(int)
from sklearn.model_selection import train_test_split
data=data.dropna()
X=data.drop(['Id','groupId','matchId','winPlacePerc','killPoints','matchDuration','rankPoints','winPoints','matchType'],axis=1)
y=data['winPlacePerc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
from keras.models import Sequential
from keras.layers import Dense,Dropout
model=Sequential()

#model.add(Dropout(0.3, input_shape=(X_train.shape[1],)))
model.add(Dense(units=64,activation='relu',input_dim=X_train.shape[1]))
#model.add(Dropout(0.5))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=4,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['accuracy'],)

model.fit(X_train,y_train,epochs=5,batch_size=128)
predictions=model.predict(X_test)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,predictions)
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
out1=pd.read_csv('../input/test_V2.csv')
out=out1.copy()
#adding team_size and match_size as features
out=out.merge(out['groupId'].value_counts().to_frame(),left_on='groupId',right_index=True)
out=out.merge(out['matchId'].value_counts().to_frame(),left_on='matchId',right_index=True)
out['team_size']=out['groupId_y']
out['match_size']=out['matchId_y']
out = out.drop("groupId_y", axis=1)
out = out.drop("matchId_y", axis=1)
out = out.drop("matchId_x", axis=1)
out = out.drop("groupId_x", axis=1)
out['duo']=(out['matchType']=='duo') | (out['matchType']=='duo-fpp') | (out['matchType']=='normal-duo') | (out['matchType']=='normal-duo-fpp')
out['squad']=(out['matchType']=='squad') | (out['matchType']=='squad-fpp') | (out['matchType']=='normal-squad-fpp') | (out['matchType']=='normal-squad')
out['solo']=(out['matchType']=='solo') | (out['matchType']=='solo-fpp') | (out['matchType']=='normal-solo') | (out['matchType']=='normal-solo-fpp')
out['others']=(out['matchType']=='crashfpp') | (out['matchType']=='flaretpp') | (out['matchType']=='flarefpp') | (out['matchType']=='crashtpp')
out['duo']=out['duo'].astype(int)
out['squad']=out['squad'].astype(int)
out['solo']=out['solo'].astype(int)
out['others']=out['others'].astype(int)

X_out=out.drop(['Id','groupId','matchId','killPoints','matchDuration','rankPoints','winPoints','matchType'],axis=1)

predictions_out=model.predict(X_out)
X_out['winPlacePerc']=predictions_out
final_out=out.merge(X_out['winPlacePerc'].to_frame(),left_index=True,right_index=True)

final_out
final_out[['Id','winPlacePerc']].to_csv('submission.csv',index=False)

