import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
%matplotlib inline
data = pd.read_csv('../input/data.csv')
sample = pd.read_csv('../input/sample_submission.csv')
data['season_year'] = data.season.apply(lambda e: e.split('-')[0])
data['season_sec'] = data.season.apply(lambda e: e.split('-')[1])

data['home_away'] = data.matchup.apply(lambda e: e.split(' ')[1])
data[['season_year', 'season_sec', 'home_away']].head()
data = data.drop(['season','game_date', 'matchup', 
                            'team_id', 'team_name', 'action_type', 'game_event_id',
                 'game_id'], axis =1 )
data.head()
def OneHot(series):
    label_encoder = LabelEncoder()
    values = array(series)
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return(pd.DataFrame(onehot_encoded))
combined_shot_type = OneHot(data['combined_shot_type'])
period = OneHot(data['period'])
playoffs = OneHot(data['playoffs'])
shot_type = OneHot(data['shot_type'])
shot_zone_area = OneHot(data['shot_zone_area'])
shot_zone_basic = OneHot(data['shot_zone_basic'])
shot_zone_range = OneHot(data['shot_zone_range'])
opponent = OneHot(data['opponent'])
season_year = OneHot(data['season_year'])
season_sec = OneHot(data['season_sec'])
home_away = OneHot(data['home_away'])
data = data.drop(['combined_shot_type','period', 'playoffs', 
                            'shot_type', 'shot_zone_area', 'shot_zone_basic', 
                  'shot_zone_range', 'opponent', 'season_year', 'season_sec',
                 'home_away'], axis =1 )
data.head()
data_1 = pd.concat([combined_shot_type, shot_type,
                 shot_zone_area, shot_zone_basic, shot_zone_range,
                 opponent, season_year, season_sec, home_away],axis= 1)
data_1.head()
data_2 = pd.concat([data, data_1], axis=1)
data_2.head()
train = data_2[data.shot_made_flag.notnull()]
test=data_2[data['shot_made_flag'].isnull()]
test[["shot_made_flag","shot_id"]].head()
X_train =  train.drop(['shot_made_flag'], axis = 1)
Y_train = train["shot_made_flag"]
X_test = test.drop(['shot_made_flag'], axis = 1)
sub_shot_id = test["shot_id"]
X_train.shape
model = Sequential()
model.add(Dense(95, input_dim=109, activation='relu', init ="he_normal"))
model.add(Dense(95, input_dim=95, activation='relu', init ="he_normal"))
model.add(Dense(1, input_dim=10, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, epochs=1000, batch_size=1000)
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 1.0])
acc_ax.set_ylim([0.0, 1.0])

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['acc'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
predicts=model.predict(X_test)
predicts
submission = pd.DataFrame({'shot_id' : np.reshape(np.array(sub_shot_id),(5000)), 
                          'shot_made_flag' : np.reshape(predicts,(5000))})
submission["shot_made_flag"] = submission.shot_made_flag.round(5)
submission.head()
submission.to_csv("sub.csv",index=False)