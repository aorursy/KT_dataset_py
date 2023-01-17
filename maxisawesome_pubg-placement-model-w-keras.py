import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os, gc, sys
gc.enable()


print(os.listdir("../input"))
train_data = pd.read_csv("../input/train_V2.csv")
test_data = pd.read_csv("../input/test_V2.csv")

# there's like one nan value so get rid of it
train_data = train_data.dropna()
test_data = test_data.dropna()
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

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

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

train_data = reduce_mem_usage(train_data)
test_data = reduce_mem_usage(test_data)
def feature_engineering(test, train, is_train=True):
    if is_train: 
        print("processing train.csv")
        df = train
        df = df[df['maxPlace'] > 1]
    else:
        print("processing test.csv")
        df = test
    
    # df = reduce_mem_usage(df)
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    
    # df = df[:100]
    
    print("remove some columns")
    target = 'winPlacePerc'
    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    
    features.remove("matchType")
    
    # matchType = pd.get_dummies(df['matchType'])
    # df = df.join(matchType)    
    
    y = None
    
    print("get target")
    if is_train: 
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print("get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    if is_train: df_out = agg.reset_index()[['matchId','groupId']]
    else: df_out = df[['matchId','groupId']]

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    # print("get group sum feature")
    # agg = df.groupby(['matchId','groupId'])[features].agg('sum')
    # agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    # df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    # df_out = df_out.merge(agg_rank, suffixes=["_sum", "_sum_rank"], how='left', on=['matchId', 'groupId'])
    
    # print("get group sum feature")
    # agg = df.groupby(['matchId','groupId'])[features].agg('sum')
    # agg_rank = agg.groupby('matchId')[features].agg('sum')
    # df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    # df_out = df_out.merge(agg_rank.reset_index(), suffixes=["_sum", "_sum_pct"], how='left', on=['matchId', 'groupId'])
    
    print("get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group size feature")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    # print("get match type feature")
    # agg = df.groupby(['matchId'])[matchType.columns].agg('mean').reset_index()
    # df_out = df_out.merge(agg, suffixes=["", "_match_type"], how='left', on=['matchId'])
    
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    X = np.array(df_out, dtype=np.float64)
    
    feature_names = list(df_out.columns)

    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y, feature_names 
INPUT_DIR = "../input/"
x_train, y_train, names = feature_engineering(test_data, train_data, True)
x_test, _, _ = feature_engineering(test_data, train_data, False)
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit(x_train)
#scaler = preprocessing.QuantileTransformer().fit(x)
#scaler.transform(x)
print("x_train before scaling", x_train.shape, x_train.max(), x_train.min())
scaler.transform(x_train)
print("x_train after scaling", x_train.shape, x_train.max(), x_train.min())

# x_train_split, y_train split is our test set
# x_test_split, y_test_split is our val set
# This is because we've split the training set
x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1234)
split = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1234)

#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)
#Y = scaler.transform(Y)

print("x_train_split after split", x_train_split.shape, x_train_split.min(), x_train_split.max())
print("x_test_split after split", x_test_split.shape, x_test_split.min(), x_test_split.max())
#print("Y", Y.shape, Y.min(), Y.max())
from keras import regularizers
start_vec_size = x_train_split.shape[1]
model = Sequential()
model.add(Dense(start_vec_size*2, input_shape=(start_vec_size,)))
model.add(LeakyReLU(.1))
model.add(BatchNormalization())
model.add(Dropout(.2))

model.add(Dense(256))
model.add(LeakyReLU(.1))
model.add(BatchNormalization())
model.add(Dropout(.2))

model.add(Dense(256))
model.add(LeakyReLU(.1))
model.add(BatchNormalization())
model.add(Dropout(.2))

model.add(Dense(128))
model.add(LeakyReLU(.1))
model.add(BatchNormalization())
model.add(Dropout(.2))

model.add(Dense(64))
model.add(LeakyReLU(.1))
model.add(BatchNormalization())

model.add(Dense(1, activation="sigmoid"))
model.summary()
optimizer = optimizers.Adam(lr=0.01, epsilon=1e-8, decay=1e-4, amsgrad=False, clipnorm=1.)
early_stopping = EarlyStopping(monitor='val_mean_absolute_error', mode = 'min', patience=4, verbose=1, restore_best_weights=True)
model.compile(optimizer=optimizer,
              loss="mse",
              metrics=["mae"])
model.fit(x_train_split, y_train_split,
          validation_data=(x_test_split, y_test_split),
          epochs=8, batch_size=1024,
          callbacks=[early_stopping])
# .0486 --> dropout at .3, no first star_vec_size*2 layer
# .0319 w/ 256 layers
# uhhh less with all 64 layers
# .033 with x2 256 256 128 64 1

