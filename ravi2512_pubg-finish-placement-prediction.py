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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import BatchNormalization,Dropout
import catboost
import lightgbm as lgb
Train_org = pd.read_csv('../input/train.csv')
Test_org = pd.read_csv('../input/test.csv')
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
Train = reduce_mem_usage(Train_org)
Test = reduce_mem_usage(Test_org)
del Train_org
del Test_org
Train.columns
Train.shape
Train.corr()['winPlacePerc'].sort_values()
Train=Train.drop(['maxPlace','numGroups'],axis=1)
Test=Test.drop(['maxPlace','numGroups'],axis=1)
sns.set(font_scale=1)
f, ax = plt.subplots(figsize=(15, 15))
hm = sns.heatmap(Train.corr(), cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8})
plt.show()
# Feature Engineering
Train["distance"] = Train["rideDistance"]+Train["walkDistance"]+Train["swimDistance"]
Train["skill"] = Train["headshotKills"]+Train["roadKills"]
Test["distance"] = Test["rideDistance"]+Test["walkDistance"]+Test["swimDistance"]
Test["skill"] = Test["headshotKills"]+Test["roadKills"]
Train_size = Train.groupby(['matchId','groupId']).size().reset_index(name='group_size')
Test_size = Test.groupby(['matchId','groupId']).size().reset_index(name='group_size')

Train_mean = Train.groupby(['matchId','groupId']).mean().reset_index()
Test_mean = Test.groupby(['matchId','groupId']).mean().reset_index()

Train_max = Train.groupby(['matchId','groupId']).max().reset_index()
Test_max = Test.groupby(['matchId','groupId']).max().reset_index()

Train_min = Train.groupby(['matchId','groupId']).min().reset_index()
Test_min = Test.groupby(['matchId','groupId']).min().reset_index()

#Train_median = Train.groupby(['matchId','groupId']).median().reset_index()
#Test_median = Test.groupby(['matchId','groupId']).median().reset_index()
Train_match_mean = Train.groupby(['matchId']).mean().reset_index()
Test_match_mean = Test.groupby(['matchId']).mean().reset_index()

Train = pd.merge(Train, Train_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
Test = pd.merge(Test, Test_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
del Train_mean
del Test_mean

Train = pd.merge(Train, Train_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
Test = pd.merge(Test, Test_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
del Train_max
del Test_max

Train = pd.merge(Train, Train_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
Test = pd.merge(Test, Test_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
del Train_min
del Test_min

#Train = pd.merge(Train, Train_median, suffixes=["", "_median"], how='left', on=['matchId', 'groupId'])
#Test = pd.merge(Test, Test_median, suffixes=["", "_median"], how='left', on=['matchId', 'groupId'])
#del Train_median
#del Test_median

Train = pd.merge(Train, Train_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
Test = pd.merge(Test, Test_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
del Train_match_mean
del Test_match_mean

Train = pd.merge(Train, Train_size, how='left', on=['matchId', 'groupId'])
Test = pd.merge(Test, Test_size, how='left', on=['matchId', 'groupId'])
del Train_size
del Test_size

target = 'winPlacePerc'
train_columns = list(Test.columns)
train_columns.remove("Id")
train_columns.remove("matchId")
train_columns.remove("groupId")
train_columns.remove("Id_mean")
train_columns.remove("Id_max")
train_columns.remove("Id_min")
#train_columns.remove("Id_medain")
train_columns.remove("Id_match_mean")
len(train_columns)
X = Train[train_columns]
Y = Test[train_columns]
T = Train[target]
#from sklearn import preprocessing
#scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X)

#X = scaler.transform(X)
#Y = scaler.transform(Y)
del Train
del Test
x_train, x_test, t_train, t_test = train_test_split(X, T, test_size = 0.15, random_state = 1)
model = Sequential()
model.add(Dense(256, kernel_initializer='he_normal', input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(64, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, kernel_initializer='normal', activation='linear'))
from keras import optimizers
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
optimizer = optimizers.Adam(lr=0.05, epsilon=1e-8, decay=1e-4, amsgrad=False)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10, verbose=0):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule, verbose)

lr_sched = step_decay_schedule(initial_lr=0.1, decay_factor=0.9, step_size=1, verbose=1)
early_stopping = EarlyStopping(monitor='val_mean_absolute_error', mode = 'min', patience=10, verbose=1)
saved_model = model.fit(x_train, t_train, 
                 validation_data=(x_test, t_test),
                 epochs=80,
                 batch_size=65536,
                 callbacks=[lr_sched,early_stopping], 
                 verbose=1)
plt.plot(saved_model.history['loss'])
plt.plot(saved_model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation mae values
plt.plot(saved_model.history['mean_absolute_error'])
plt.plot(saved_model.history['val_mean_absolute_error'])
plt.title('Mean Abosulte Error')
plt.ylabel('Mean absolute error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

y_pred=model.predict(Y)
submission=pd.DataFrame()
submission['Id']=Test_org['Id']
submission['winPlacePerc']=y_pred
submission.to_csv('submission.csv',index=False)
