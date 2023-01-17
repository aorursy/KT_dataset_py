import os

import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn import preprocessing



from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras import optimizers

from keras.models import load_model
df_train = pd.read_csv('../input/train.csv').astype('float32')

df_test = pd.read_csv('../input/test.csv')
# save memory

def reduce_mem_usage(df):

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

    return df
df_train = reduce_mem_usage(df_train)

df_test = reduce_mem_usage(df_test)
df_train["distance"] = df_train["rideDistance"]+df_train["walkDistance"]+df_train["swimDistance"]

df_train["skill"] = df_train["headshotKills"]+df_train["roadKills"]

df_test["distance"] = df_test["rideDistance"]+df_test["walkDistance"]+df_test["swimDistance"]

df_test["skill"] = df_test["headshotKills"]+df_test["roadKills"]
# feature of each group

df_train_size = df_train.groupby(['matchId','groupId']).size().reset_index(name='group_size')

df_test_size = df_test.groupby(['matchId','groupId']).size().reset_index(name='group_size')



df_train_mean = df_train.groupby(['matchId','groupId']).mean().reset_index()

df_test_mean = df_test.groupby(['matchId','groupId']).mean().reset_index()



df_train_max = df_train.groupby(['matchId','groupId']).max().reset_index()

df_test_max = df_test.groupby(['matchId','groupId']).max().reset_index()



df_train_min = df_train.groupby(['matchId','groupId']).min().reset_index()

df_test_min = df_test.groupby(['matchId','groupId']).min().reset_index()
# feature of each match

df_train_match_mean = df_train.groupby(['matchId']).mean().reset_index()

df_test_match_mean = df_test.groupby(['matchId']).mean().reset_index()



df_train = pd.merge(df_train, df_train_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])

df_test = pd.merge(df_test, df_test_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])

del df_train_mean

del df_test_mean



df_train = pd.merge(df_train, df_train_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])

df_test = pd.merge(df_test, df_test_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])

del df_train_max

del df_test_max



df_train = pd.merge(df_train, df_train_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])

df_test = pd.merge(df_test, df_test_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])

del df_train_min

del df_test_min



df_train = pd.merge(df_train, df_train_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])

df_test = pd.merge(df_test, df_test_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])

del df_train_match_mean

del df_test_match_mean



df_train = pd.merge(df_train, df_train_size, how='left', on=['matchId', 'groupId'])

df_test = pd.merge(df_test, df_test_size, how='left', on=['matchId', 'groupId'])

del df_train_size

del df_test_size



target = 'winPlacePerc'

train_columns = list(df_test.columns)
# remove some columns 

train_columns.remove("Id")

train_columns.remove("matchId")

train_columns.remove("groupId")

train_columns.remove("Id_mean")

train_columns.remove("Id_max")

train_columns.remove("Id_min")

train_columns.remove("Id_match_mean")
# team skill level is more important than personal skill level 

# remove the features of each player, just select the features of group and match

train_columns_new = []

for name in train_columns:

    if '_' in name:

        train_columns_new.append(name)

train_columns = train_columns_new    

print(train_columns)
X = df_train[train_columns]

Y = df_test[train_columns]

T = df_train[target]



del df_train
x_train, x_test, t_train, t_test = train_test_split(X, T, test_size = 0.2, random_state = 1234)



scaler = preprocessing.QuantileTransformer().fit(x_train)



x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)

Y = scaler.transform(Y)



print("x_train", x_train.shape, x_train.min(), x_train.max())

print("x_test", x_test.shape, x_test.min(), x_test.max())

print("Y", Y.shape, Y.min(), Y.max())
model = Sequential()

model.add(Dense(512, kernel_initializer='he_normal', input_dim=x_train.shape[1], activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(Dense(256, kernel_initializer='he_normal', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
optimizer = optimizers.Adam(lr=0.01, epsilon=1e-8, decay=1e-4, amsgrad=False)



model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10, verbose=0):

    # create a LearningRateScheduler with step decay schedule

    def schedule(epoch):

        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    

    return LearningRateScheduler(schedule, verbose)



lr_sched = step_decay_schedule(initial_lr=0.1, decay_factor=0.9, step_size=1, verbose=1)

early_stopping = EarlyStopping(monitor='val_mean_absolute_error', mode = 'min', patience=4, verbose=1)
history = model.fit(x_train, t_train, 

                 validation_data=(x_test, t_test),

                 epochs=30,

                 batch_size=32768,

                 callbacks=[lr_sched,early_stopping], 

                 verbose=1)
# plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# plot training & validation mae values

plt.plot(history.history['mean_absolute_error'])

plt.plot(history.history['val_mean_absolute_error'])

plt.title('Mean Abosulte Error')

plt.ylabel('Mean absolute error')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
pred = model.predict(Y)

pred = pred.ravel()



df_test['winPlacePercPred'] = np.clip(pred, a_min=0, a_max=1)





aux = df_test.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()

aux.columns = ['matchId','groupId','winPlacePerc']

df_test = df_test.merge(aux, how='left', on=['matchId','groupId'])

    

submission = df_test[['Id', 'winPlacePerc']]



submission.to_csv('submission.csv', index=False)