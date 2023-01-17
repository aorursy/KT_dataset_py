import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
#from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import load_model
import gc, sys
gc.enable()
def feature_engineering(is_train=True):
    # When this function is used for the training data, load train_V2.csv :
    if is_train: 
        print("processing train_V2.csv")
        df = pd.read_csv('../input/train_V2.csv')
        
        # Only take the samples with matches that have more than 1 player 
        # there are matches with no players or just one player ( those samples could affect our model badly) 
        df = df[df['maxPlace'] > 1]
    
    # When this function is used for the test data, load test_V2.csv :
    else:
        print("processing test_V2.csv")
        df = pd.read_csv('../input/test_V2.csv')
        
    # Make a new feature indecating the total distance a player cut :
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
          

    # Process the 'rankPoints' feature by replacing any value of (-1) to be (0) :
    df['rankPoints'] = np.where(df['rankPoints'] <= 0 ,0 , df['rankPoints'])                                
    

    target = 'winPlacePerc'
    # Get a list of the features to be used
    features = list(df.columns)
    
    # Remove some features from the features list :
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchDuration")
    features.remove("matchType")
    
    y = None
    
    # If we are processing the training data, process the target
    # (group the data by the match and the group then take the mean of the target) 
    if is_train: 
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        # Remove the target from the features list :
        features.remove(target)
    
    # Make new features indicating the mean of the features ( grouped by match and group ) :
    print("get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    
    # If we are processing the training data let df_out = the grouped  'matchId' and 'groupId'
    if is_train: df_out = agg.reset_index()[['matchId','groupId']]
    # If we are processing the test data let df_out = 'matchId' and 'groupId' without grouping 
    else: df_out = df[['matchId','groupId']]
    
    # Merge agg and agg_rank (that we got before) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the max value of the features for each group ( grouped by match )
    print("get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    # Merge the new (agg and agg_rank) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the minimum value of the features for each group ( grouped by match )
    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    # Merge the new (agg and agg_rank) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the number of players in each group ( grouped by match )
    print("get group size feature")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
     
    # Merge the group_size feature with df_out :
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the mean value of each features for each match :
    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    
    # Merge the new agg with df_out :
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    # Make new features indicating the number of groups in each match :
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    
    # Merge the match_size feature with df_out :
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    # Drop matchId and groupId
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
    
    # X is the output dataset (without the target) and y is the target :
    X = np.array(df_out, dtype=np.float64)
    
    
    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y
%%time
# Process the training data :
X, y = feature_engineering(True)
# Scale the data to be in the range (-1 , 1)
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit(X)
print("X", X.shape, X.max(), X.min())
scaler.transform(X)
print("X", X.shape, X.max(), X.min())
y = y * 2 - 1
print("y", y.shape, y.max(), y.min())
#Let's build a model
model = Sequential()
model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='relu'))
#model.add(LeakyReLU(0.1))
#model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
#model.add(LeakyReLU(0.1))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
#model.add(LeakyReLU(0.1))
#model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
#model.add(LeakyReLU(0.1))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))

model.summary()
optimizer = optimizers.Adam(lr=0.01, epsilon=1e-8, decay=1e-4, amsgrad=False)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
history = model.fit(x=X, y=y, batch_size=1000,
             epochs=20, verbose=1, callbacks=callbacks_list,
             validation_split=0.2, validation_data=None, shuffle=True,
             class_weight=None, sample_weight=None, initial_epoch=0,
             steps_per_epoch=None, validation_steps=None)

del X, y
#gc.collect()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation mae values
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Mean Absolute Error')
plt.ylabel('Mean absolute error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#Processing test data
X, _ = feature_engineering(False)
scaler.transform(X)
print("x_test", X.shape, X.max(), X.min())
np.clip(X, out=X, a_min=-1, a_max=1)
print("x_test", X.shape, X.max(), X.min())
%%time
pred = model.predict(X)
del X
pred = pred.reshape(-1)
pred = (pred + 1) / 2

# pred = (pred + 1) / 2
df_test = pd.read_csv('../input/test_V2.csv')


print("Correcting winPlacePerc")
for i in range(len(df_test)):
    winPlacePerc = pred[i]
    maxPlace = int(df_test.iloc[i]['maxPlace'])
    if maxPlace == 0:
        winPlacePerc = 0.0
    elif maxPlace == 1:
        winPlacePerc = 1.0
    else:
        gap = 1.0 / (maxPlace - 1)
        winPlacePerc = round(winPlacePerc / gap) * gap
    
    if winPlacePerc < 0: winPlacePerc = 0.0
    if winPlacePerc > 1: winPlacePerc = 1.0    
    pred[i] = winPlacePerc
df_test['winPlacePerc'] = pred
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)