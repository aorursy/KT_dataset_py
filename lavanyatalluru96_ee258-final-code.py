import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os
#import download
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
%matplotlib inline
from keras.models import Sequential
from keras.layers import Dense
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import time
from keras.models import load_model
from keras import optimizers
import random
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import array
from sklearn.metrics import mean_absolute_error
data = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")
data.info() 
plt.figure(figsize = (20,15))
sn.heatmap(data.corr(), annot = True ,fmt='.3f',cmap = 'Blues')
plt.show()
copy_data=data.copy()
train = pd.DataFrame()
train['heals'] =  copy_data['heals']/copy_data.groupby('matchId')['heals'].transform('sum')
train['walkDistance'] =  copy_data['walkDistance']/copy_data.groupby('matchId')['walkDistance'].transform('sum')
train['boosts'] =  copy_data['boosts']/copy_data.groupby('matchId')['boosts'].transform('sum')
train['weaponsAcquired'] =  copy_data['weaponsAcquired']/copy_data.groupby('matchId')['weaponsAcquired'].transform('sum')
train['kills'] =  copy_data['kills']/copy_data.groupby('matchId')['kills'].transform('sum')
train['DBNOs'] =  copy_data['DBNOs']/copy_data.groupby('matchId')['DBNOs'].transform('sum')
train['killPlace'] =  copy_data['killPlace']/copy_data.groupby('matchId')['killPlace'].transform('max')
train['headshotKills'] =  copy_data['headshotKills']/copy_data.groupby('matchId')['headshotKills'].transform('sum')
train['damageDealt'] =  copy_data['damageDealt']/copy_data.groupby('matchId')['damageDealt'].transform('sum')
train['killStreaks'] =  copy_data['killStreaks']/copy_data.groupby('matchId')['killStreaks'].transform('max')
train['winPlacePerc'] = copy_data['winPlacePerc']
train=train.fillna(0)
final_test = pd.DataFrame()
final_test['heals'] =  test['heals']/test.groupby('matchId')['heals'].transform('sum')
final_test['walkDistance'] =  test['walkDistance']/test.groupby('matchId')['walkDistance'].transform('sum')
final_test['boosts'] =  test['boosts']/test.groupby('matchId')['boosts'].transform('sum')
final_test['weaponsAcquired'] =  test['weaponsAcquired']/test.groupby('matchId')['weaponsAcquired'].transform('sum')
final_test['kills'] =  test['kills']/test.groupby('matchId')['kills'].transform('sum')
final_test['DBNOs'] =  test['DBNOs']/test.groupby('matchId')['DBNOs'].transform('sum')
final_test['killPlace'] =  test['killPlace']/test.groupby('matchId')['killPlace'].transform('max')
final_test['headshotKills'] =  test['headshotKills']/test.groupby('matchId')['headshotKills'].transform('sum')
final_test['damageDealt'] =  test['damageDealt']/test.groupby('matchId')['damageDealt'].transform('sum')
final_test['killStreaks'] =  test['killStreaks']/test.groupby('matchId')['killStreaks'].transform('max')
final_test=final_test.fillna(0)
train.head()
plt.figure(figsize = (20,15))
sn.heatmap(train.corr(), annot = True ,fmt='.3f',cmap = 'Blues')
plt.show()
plt.plot(train["heals"],train["winPlacePerc"],'.')
plt.xlabel("Percentage of heals used by the player")
plt.ylabel("winPlacePerc")
plt.title("% of Heals used Vs winPlacePerc")
plt.show()
healsVswin = train[["heals","winPlacePerc"]]

healsVswin['% of Heals used'] = pd.cut(healsVswin['heals'], [0, 0.2, 0.4,0.5,0.6,0.7,0.8,0.9,1], labels=['0-20%','20-40%', '40-50%', '50-60%','60-70%','70-80%','80-90%','90-100%'])

plt.figure(figsize=(15,8))
sn.boxplot(x="% of Heals used", y="winPlacePerc", data=healsVswin)
plt.show()
plt.plot(train["walkDistance"],train["winPlacePerc"],'.')
plt.xlabel("Percentage of distance walked by the player")
plt.ylabel("winPlacePerc")
plt.title("% of distance walked Vs winPlacePerc")
plt.show()
plt.figure(figsize=(7,5))
plt.title("walk distance",fontsize=15)
sn.distplot(train['walkDistance'])
plt.xlabel("% of Distance walked")
plt.title("Density function of distance walked")
plt.show()
plt.plot(train["boosts"],train["winPlacePerc"],'.')
plt.xlabel("Percentage of boosts used by the player")
plt.ylabel("winPlacePerc")
plt.title("% of boosts used Vs winPlacePerc")
plt.show()
plt.figure(figsize=(7,5))
plt.title("Boosts",fontsize=15)
sn.distplot(train['boosts'])
plt.xlabel("Percentage of boosts used by the players")
plt.title("Density function of the boosts used")
plt.show()
plt.plot(train["weaponsAcquired"],train["winPlacePerc"],'.')
plt.xlabel("Percentage of weaponsAcquired by the player")
plt.ylabel("winPlacePerc")
plt.title("% of weaponsAcquired Vs winPlacePerc")
plt.show()
weaponsAcquiredVswin = train[["weaponsAcquired","winPlacePerc"]]

weaponsAcquiredVswin['% of weaponsAcquired'] = pd.cut(weaponsAcquiredVswin['weaponsAcquired'], [0, 0.1,0.15,0.21,0.22,0.23,0.25,0.3,0.35,0.4,0.45,0.5], labels=['0-10%','10-15%','15-21%','21-22%','22-23%','23-25%','25-30%','30-35%','35-40%','40-45%','45-50%'])

plt.figure(figsize=(15,8))
sn.boxplot(x="% of weaponsAcquired", y="winPlacePerc", data=weaponsAcquiredVswin)
plt.show()
plt.plot(train["kills"],train["winPlacePerc"],'.')
plt.xlabel("Percentage of enemies killed by the player")
plt.ylabel("winPlacePerc")
plt.title("% of enemies killed Vs winPlacePerc")
plt.show()
plt.figure(figsize=(7,5))
plt.title("Kills",fontsize=15)
sn.distplot(train['kills'])
plt.xlabel("Percentage of enemies killed by the player")
plt.title("Density function of players killed")
plt.show()
plt.plot(train["damageDealt"],train["winPlacePerc"],'.')
plt.xlabel("Percentage of damage dealt by the player")
plt.ylabel("winPlacePerc")
plt.title("% of damage dealt Vs winPlacePerc")
plt.show()
plt.figure(figsize=(7,5))
plt.title("Damage Dealt",fontsize=15)
sn.distplot(train['damageDealt'])
plt.xlabel("Percentage of damage dealt by the player")
plt.title("Density function of Damage dealt")
plt.show()
plt.plot(train["killStreaks"],train["winPlacePerc"],'.')
plt.xlabel("Percentage of enemies killed in short time by the player")
plt.ylabel("winPlacePerc")
plt.title("% Percentage of enemies killed in short time Vs winPlacePerc")
plt.show()
killStreaksVswin = train[["killStreaks","winPlacePerc"]]
killStreaksVswin['% of killStreaks'] = pd.cut(killStreaksVswin['killStreaks'], [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], labels=['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'])
plt.figure(figsize=(15,8))
sn.boxplot(x="% of killStreaks", y="winPlacePerc", data=killStreaksVswin)
plt.show()
plt.plot(train["DBNOs"],train["winPlacePerc"],'.')
plt.xlabel("Percentage of enemy players knocked by the player")
plt.ylabel("winPlacePerc")
plt.title("% Percentage of enemy players knocked Vs winPlacePerc")
plt.show()
DBNOsVswin = train[["DBNOs","winPlacePerc"]]
DBNOsVswin['% of DBNOs'] = pd.cut(DBNOsVswin['DBNOs'], [0, 0.1,0.2,0.3,0.4,1], labels=['0-10%','10-20%','20-30%','30-40%','40-100%'])
plt.figure(figsize=(15,8))
sn.boxplot(x="% of DBNOs", y="winPlacePerc", data=DBNOsVswin)
plt.show()
plt.plot(train["killPlace"],train["winPlacePerc"],'.')
plt.xlabel("Ranking of the player in the match")
plt.ylabel("winPlacePerc")
plt.title("Ranking of the player in the match Vs winPlacePerc")
plt.show()
killPlaceVswin = train[["killPlace","winPlacePerc"]]
killPlaceVswin['% of killPlace'] = pd.cut(killPlaceVswin['killPlace'], [0, 0.1,0.2,0.3,0.4,0.6,0.8,1], labels=['0-10%','10-20%','20-30%','30-40%','40-60%','60-80%','80-100%'])
plt.figure(figsize=(15,8))
sn.boxplot(x="% of killPlace", y="winPlacePerc", data=killPlaceVswin)
plt.show()
plt.plot(train["headshotKills"],train["winPlacePerc"],'.')
plt.xlabel("% of enemies killed by headshots by the player")
plt.ylabel("winPlacePerc")
plt.title("% of enemies killed by headshots by the player Vs winPlacePerc")
plt.show()
headshotKillsVswin = train[["headshotKills","winPlacePerc"]]
headshotKillsVswin['% of headshotKills'] = pd.cut(headshotKillsVswin['headshotKills'], [0, 0.1,0.2,0.3,0.4,0.6,0.8,1], labels=['0-10%','10-20%','20-30%','30-40%','40-60%','60-80%','80-100%'])
plt.figure(figsize=(15,8))
sn.boxplot(x="% of headshotKills", y="winPlacePerc", data=headshotKillsVswin)
plt.show()
import numpy as np
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,1,figsize=(15,5))
    # summarize history for MAE
    axs.plot(range(1,len(model_history.history['mean_absolute_error'])+1),model_history.history['mean_absolute_error'])
    axs.plot(range(1,len(model_history.history['val_mean_absolute_error'])+1),model_history.history['val_mean_absolute_error'])
    axs.set_title('Model MAE')
    axs.set_ylabel('mean_absolute_error')
    axs.set_xlabel('Epoch')
    axs.set_xticks(np.arange(1,len(model_history.history['mean_absolute_error'])+1),len(model_history.history['mean_absolute_error'])/10)
    axs.legend(['train', 'val'], loc='best')
    plt.show()
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(train, test_size=1/12, random_state=42)
Y_val1 = X_val[['winPlacePerc']]
Y_val = Y_val1.values
X_val1 = X_val.drop(columns=["winPlacePerc"])
X_val = X_val1.values
final_test = final_test.values

Y_train1 = X_train[['winPlacePerc']]
Y_train = Y_train1.values
X_train1 = X_train.drop(columns=["winPlacePerc"])
X_train = X_train1.values
[m,n] = X_train.shape
X_train_set1 = [];Y_train_set1=[];X_train_set2 = [];Y_train_set2=[];X_train_set3 = [];Y_train_set3=[];
X_train_set4 = [];Y_train_set4=[];X_train_set5 = [];Y_train_set5=[];

for i in range(m):
    rn1 = random.randint(0,m-1)
    X_train_set1.append(X_train[rn1])
    Y_train_set1.append(Y_train[rn1])
    
    rn1 = random.randint(0,m-1)
    X_train_set2.append(X_train[rn1])
    Y_train_set2.append(Y_train[rn1])
    
    rn1 = random.randint(0,m-1)
    X_train_set3.append(X_train[rn1])
    Y_train_set3.append(Y_train[rn1])
    
    rn1 = random.randint(0,m-1)
    X_train_set4.append(X_train[rn1])
    Y_train_set4.append(Y_train[rn1])
    
    rn1 = random.randint(0,m-1)
    X_train_set5.append(X_train[rn1])
    Y_train_set5.append(Y_train[rn1])
    
    
X_train_set1 = array(X_train_set1); Y_train_set1 = array(Y_train_set1)
X_train_set2 = array(X_train_set2); Y_train_set2 = array(Y_train_set2)
X_train_set3 = array(X_train_set3); Y_train_set3 = array(Y_train_set3)
X_train_set4 = array(X_train_set4); Y_train_set4 = array(Y_train_set4)
X_train_set5 = array(X_train_set5); Y_train_set5 = array(Y_train_set5)
hidden_neurons = 10

model = Sequential()
model.add(Dense(10, input_dim=10, activation='relu'))
for i in range(3):
    model.add(Dense(10, activation='relu'))
model.add(Dense(1,activation = 'relu'))
adam = optimizers.Adam(lr=0.0001)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
model_info=model.fit(X_train, Y_train, batch_size=100,nb_epoch=30,validation_data=(X_val,Y_val))
plot_model_history(model_info)
model1 = Sequential()
model1.add(Dense(10, input_dim=10, activation='relu'))
for i in range(3):
    model1.add(Dense(10, activation='relu'))
model1.add(Dense(1,activation = 'relu'))
adam = optimizers.Adam(lr=0.0001)
model1.compile(loss='mean_absolute_error', optimizer=adam, metrics=['mae'])
callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=5),
             ModelCheckpoint(filepath='best_model_set1', monitor='val_mean_absolute_error', save_best_only=True)]
model1_info = model1.fit(X_train_set1, Y_train_set1, batch_size=100,callbacks=callbacks,nb_epoch=30,validation_data=(X_val,Y_val))
plot_model_history(model1_info)
model2 = Sequential()
model2.add(Dense(10, input_dim=10, activation='relu'))
for i in range(3):
    model2.add(Dense(10, activation='relu'))
model2.add(Dense(1,activation = 'relu'))
adam = optimizers.Adam(lr=0.0001)
model2.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=5),
             ModelCheckpoint(filepath='best_model_set2', monitor='val_mean_absolute_error',save_best_only=True)]
model2_info = model2.fit(X_train_set2, Y_train_set2, batch_size=100,nb_epoch=30,callbacks=callbacks,validation_data=(X_val,Y_val))
plot_model_history(model2_info)
model3 = Sequential()
model3.add(Dense(10, input_dim=10, activation='relu'))
for i in range(3):
    model3.add(Dense(10, activation='relu'))
model3.add(Dense(1,activation = 'relu'))
adam = optimizers.Adam(lr=0.0001)
model3.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=5),
             ModelCheckpoint(filepath='best_model_set3', monitor='val_mean_absolute_error', save_best_only=True)]
model3_info = model3.fit(X_train_set3, Y_train_set3, batch_size=100,nb_epoch=30,callbacks=callbacks,validation_data=(X_val,Y_val))
plot_model_history(model3_info)
model4 = Sequential()
model4.add(Dense(10, input_dim=10, activation='relu'))
for i in range(3):
    model4.add(Dense(10, activation='relu'))
model4.add(Dense(1,activation = 'relu'))
adam = optimizers.Adam(lr=0.0001)
model4.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=5),
             ModelCheckpoint(filepath='best_model_set4', monitor='val_mean_absolute_error', save_best_only=True)]
model4_info = model4.fit(X_train_set4, Y_train_set4, batch_size=100,nb_epoch=30,callbacks=callbacks,validation_data=(X_val,Y_val))
plot_model_history(model4_info)
model5 = Sequential()
model5.add(Dense(10, input_dim=10, activation='relu'))
for i in range(3):
    model5.add(Dense(10, activation='relu'))
model5.add(Dense(1,activation = 'relu'))
adam = optimizers.Adam(lr=0.0001)
model5.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=5),
             ModelCheckpoint(filepath='best_model_set5', monitor='val_mean_absolute_error', save_best_only=True)]
model5_info = model5.fit(X_train_set5, Y_train_set5, batch_size=100,nb_epoch=30,callbacks=callbacks,validation_data=(X_val,Y_val))
plot_model_history(model5_info)
from keras.models import load_model
Optimal_Model1 = load_model('best_model_set1')
Optimal_Model2 = load_model('best_model_set2')
Optimal_Model3 = load_model('best_model_set3')
Optimal_Model4 = load_model('best_model_set4')
Optimal_Model5 = load_model('best_model_set5')

val_score1 = Optimal_Model1.evaluate(X_val,Y_val);
print("\nValdidation loss and Mean Absolute error of model1 is ",val_score1)
val_score2 = Optimal_Model2.evaluate(X_val,Y_val);
print("\nValdidation loss and Mean Absolute error of model2 is ",val_score2)
val_score3 = Optimal_Model3.evaluate(X_val,Y_val);
print("\nValdidation loss and Mean Absolute error of model3 is ",val_score3)
val_score4 = Optimal_Model4.evaluate(X_val,Y_val);
print("\nValdidation loss and Mean Absolute error of model4 is ",val_score4)
val_score5 = Optimal_Model5.evaluate(X_val,Y_val);
print("\nValdidation loss and Mean Absolute error of model5 is ",val_score5)


y_val_predict1 = Optimal_Model1.predict(X_val)
y_val_predict2 = Optimal_Model2.predict(X_val)
y_val_predict3 = Optimal_Model3.predict(X_val)
y_val_predict4 = Optimal_Model4.predict(X_val)
y_val_predict5 = Optimal_Model5.predict(X_val)


y_val_predict = (y_val_predict1 + y_val_predict2 + y_val_predict3 + y_val_predict4 + y_val_predict5)/5
predicted_Bagging = mean_absolute_error(Y_val, y_val_predict)

y_baseline_predict = model.predict(X_val)
predicted_no_Bagging = mean_absolute_error(Y_val, y_baseline_predict)

print("Mean Absolute error (NO Bagging) ", predicted_no_Bagging)
print("Mean Absolute error (With Bagging) ", predicted_Bagging)
Y_test_predict = model.predict(final_test)
Y_test_predict1 = Optimal_Model1.predict(final_test)
Y_test_predict2 = Optimal_Model2.predict(final_test)
Y_test_predict3 = Optimal_Model3.predict(final_test)
Y_test_predict4 = Optimal_Model4.predict(final_test)
Y_test_predict5 = Optimal_Model5.predict(final_test)

Y_test_Bagging = (Y_test_predict1 + Y_test_predict2 + Y_test_predict3 + Y_test_predict4 + Y_test_predict5)/5 
Y_test_Bagging_dataframe = pd.DataFrame(Y_test_Bagging)

Y_test_Bagging_dataframe.columns = ['winPlacePerc']

subm_df = pd.read_csv('../input/sample_submission_V2.csv')
subm_df['winPlacePerc'] = Y_test_Bagging_dataframe
subm_df.to_csv('sample_submission_new.csv', index=False)
