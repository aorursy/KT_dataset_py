# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
match = pd.read_csv('/kaggle/input/ipl/matches.csv')
delivery = pd.read_csv('/kaggle/input/ipl/deliveries.csv')
innings = pd.read_csv('/kaggle/input/random-data/innings-dataset.csv')
innings.sample(5)
# 1. Venue avg score
# 2. Batting team avg score 
# 3. Bowling team avg run conceeded
delivery = delivery.merge(match,left_on='match_id',right_on='id')
delivery = delivery[delivery['inning']==1]
runs_per_stadium = delivery.groupby('venue').sum()['total_runs']
match_per_stadium = match['venue'].value_counts()
avg_score_stadium = runs_per_stadium/match_per_stadium
avg_score_stadium = avg_score_stadium.reset_index()
avg_score_stadium.rename(columns={'index':'venue',0:'avg_score_stadium'},inplace=True)
avg_score_stadium.head()
innings = innings.merge(avg_score_stadium,on='venue')
runs_avg_batting_team = innings.drop_duplicates(subset=['mid']).groupby('bat_team').mean()['total']
runs_avg_bowling_team = innings.drop_duplicates(subset=['mid']).groupby('bowl_team').mean()['total']
runs_avg_batting_team = runs_avg_batting_team.reset_index()
runs_avg_bowling_team = runs_avg_bowling_team.reset_index()
runs_avg_batting_team.rename(columns={'index':'bat_team',0:'avg_runs_batting_team'},inplace=True)
runs_avg_bowling_team.rename(columns={'index':'bowl_team',0:'avg_runs_bowling_team'},inplace=True)
innings = innings.merge(runs_avg_batting_team,on='bat_team')
innings = innings.merge(runs_avg_bowling_team,on='bowl_team')
innings.rename(columns={'total_y':'avg_batting_team_score','total':'avg_bowling_team_score'},inplace=True)
innings.rename(columns={'total_x':'total'},inplace=True)
innings.corr()['total']
innings.drop(columns=['mid','date','batsman','bowler','striker','non-striker'],axis=1,inplace=True)
innings.head()
innings.shape
innings['bat_team'].unique()
obsolete_teams = ['Kochi Tuskers Kerala','Pune Warriors','Deccan Chargers','Rising Pune Supergiants','Gujarat Lions','Rising Pune Supergiant']
innings = innings[~(innings['bat_team'].isin(obsolete_teams)) & ~(innings['bowl_team'].isin(obsolete_teams))]
innings.head()
innings.shape

from sklearn.preprocessing import LabelEncoder
venue_encoder = LabelEncoder()
team_encoder = LabelEncoder()
innings['venue'] = venue_encoder.fit_transform(innings['venue'])
innings['bat_team'] = team_encoder.fit_transform(innings['bat_team'])
innings['bowl_team'] = team_encoder.transform(innings['bowl_team'])
innings.sample(5)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first')
X = innings.drop(['total','venue','bat_team','bowl_team'],axis=1)
X_trans = ohe.fit_transform(innings[['venue','bat_team','bowl_team']]).toarray()
X_trans.shape
X = np.hstack((X,X_trans))
X.shape
y = innings['total'].values
new.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[0]
X = scaler.fit_transform(X)
X[0]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
X_train.shape
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Dropout
import kerastuner as kt
from tensorflow.keras import regularizers
def build_model(hp):
    
    model = Sequential()
    counter = 0
    
    for i in range(hp.Int('num_layers',min_value=1,max_value=10)):
        if counter == 0 :
            model.add(
                Dense(
                    units=hp.Int('unit' + str(i),min_value=8,max_value=256,step=8),
                    activation=hp.Choice('activation',values=['relu','tanh','sigmoid','selu','elu']),
                    input_dim=X_train.shape[1],
                    kernel_regularizer = regularizers.l2(hp.Choice('reg' + str(i),values=[1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3]))
                )
            )
            model.add(Dropout(hp.Choice('dropout'+str(i),values=[0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9])))
        else:
            model.add(
                Dense(
                    units=hp.Int('unit' + str(i),min_value=8,max_value=256,step=8),
                    activation=hp.Choice('activation',values=['relu','tanh','sigmoid','selu','elu']),
                    kernel_regularizer = regularizers.l2(hp.Choice('reg' + str(i),values=[1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3]))
                )
            )
            model.add(Dropout(hp.Choice('dropout'+str(i),values=[0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9])))
        counter = counter + 1
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer=hp.Choice('opti',['rmsprop','adam','sgd','adagrad','adadelta','nadam','adamax']),
                 loss = 'mse',
                 metrics=['mse'])
    return model  
tuner = kt.RandomSearch(build_model,objective='mse',max_trials=3,directory='somedir',project_name='ipl')
tuner.search(X_train,y_train,epochs=5,validation_data=(X_test,y_test))
tuner.get_best_hyperparameters()[0].values
model = tuner.get_best_models(num_models=1)[0]
model.fit(X_train,y_train,epochs=500,initial_epoch=6,validation_data=(X_test,y_test))
history = model.history.history
import matplotlib.pyplot as plt
plt.plot(history['mse'])
plt.plot(history['val_mse'])
model.evaluate(X_test,y_test)
import pickle
pickle.dump()