from time import time



import numpy as np

import pandas as pd

import tensorflow.keras as keras

import tensorflow.keras.layers as L

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.callbacks import EarlyStopping



pd.set_option('display.max_columns', None)
test_size = 0.05

batch_size = 20000

env = 'kaggle'



if env == 'kaggle':

    test_size = 0.0

    batch_size = 100000
def get_path(env='local'):

    """Env is either 'colab', 'local' or 'kaggle'"""

    if env == 'local':

        return 'pubg-finish-placement-prediction'

    

    if env == 'colab':

        from google.colab import drive

        drive.mount('/content/gdrive')

        return '/content/gdrive/My Drive/ML/datasets/pubg-placement-competition'

    

    if env == 'kaggle':

        return '../input'

    

    raise ValueError('Wrong argument `env`')

    

path = get_path(env)
def reduce_mem_usage(df):

    # iterate through all the columns of a dataframe and modify the data type

    #   to reduce memory usage.        

    

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
class timeit:

    def __enter__(self):

        self.start = time()



    def __exit__(self, type, value, traceback):

        print(f'Done in {time() - self.start:.2f} sec')
def load_data(file):

    """Load data from csv file and reduce its size."""

    print('Loading data...')

    with timeit():

        df = pd.read_csv(f'{path}/{file}')

    print(f'Rows loaded: {len(df.index)}')

    

    print('\nReducing mem usage...')

    with timeit():

        df = reduce_mem_usage(df)

        

    return df
def ds_solo_split(df):

    """Split to Duo-Squad and Solo data."""

    print('Extracting Duo-Squad data...')

    with timeit():

        df_ds = df[df['matchType'].str.contains('squad') | df['matchType'].str.contains('duo')]

    print(f'Duo-Squad: {len(df_ds.index)} rows\n')



    print('Extracting Solo data...')

    with timeit():

        df_solo = df[~df.Id.isin(set(df_ds.Id.values.flatten()))]

    print(f'Solo: {len(df_solo.index)} rows')

    

    return df_ds, df_solo
def train_test_split_by_match(df, test_size):

    """Smart data split, to avoid breaking one match into two datasets."""

    print('Splitting dataframe to train and test...')

    with timeit():

        train_matches, test_matches = train_test_split(df.matchId.unique(), test_size=test_size)

        df_train = df[df.matchId.isin(train_matches)]

        df_test = df[df.matchId.isin(test_matches)]

    print(f'Splitted: train={len(df_train.index)}, test={len(df_test.index)}')

    return df_train, df_test
def process_df(df):

    """Feature engineering for all data (Solo-Duo-Squad)."""

    groupby_matchId = df.groupby('matchId')



    df = df.assign(

        walkDistanceLog=np.log(df.walkDistance+1),

        walkDistanceSqrt=np.sqrt(df.walkDistance),

        walkDistanceSq=df.walkDistance**2,

        walkDistanceCube=df.walkDistance**3,

        walkDistancePerDuration= df.walkDistance/df.matchDuration,

        walkDistancePerc=groupby_matchId.walkDistance.rank(pct=True).values,

        walkDistanceRate=df.kills/groupby_matchId.walkDistance.transform(np.sum),

        walkDistanceHealsRatio=df.walkDistance / df.heals,

        walkDistanceKillsRatio=df.walkDistance / df.kills,

        totalDistance=0.25*df.rideDistance + df.walkDistance + df.swimDistance,

        maxPlaceByNumGroups=df.maxPlace / df.numGroups,

        maxPlaceMinusNumGroups=df.maxPlace - df.numGroups,

        headshotKillsRatio=df.headshotKills/df.kills,

        killHeadshotRatio=df.kills/df.headshotKills,

        killStreakKillRatio=df.killStreaks/df.kills,

        killPlaceMaxPlaceRatio=df.killPlace / df.maxPlace,

        killsWalkDistanceRatio=df.kills / df.walkDistance,

        killPlaceKillsRatio=df.killPlace/df.kills,

        killPerc=groupby_matchId.kills.rank(pct=True).values,

        killPlacePerc=groupby_matchId.killPlace.rank(pct=True).values,

        killsRate=df.kills/groupby_matchId.kills.transform(np.sum),

        weaponsAcquiredRank=groupby_matchId.weaponsAcquired.rank(pct=True).values,

    )

    

    df = df.assign(

        killKillsRatio2=df.killPerc / df.walkDistancePerc,

        walkDistanceKillsRatio2=df.walkDistancePerc / df.killPerc,

        totalDistanceWeaponsAcqRatio=df.totalDistance / df.weaponsAcquired,

        totalDistancePerDuration=df.totalDistance / df.matchDuration,

        totalDistanceRank=df.groupby('matchId').totalDistance.rank(pct=True).values,

        killPlaceWalkDistanceRatio2=df.walkDistancePerc / df.killPlacePerc,

        killPlaceKillsRatio2=df.killPlacePerc / df.killPerc,

        walkDistanceKillsRatio3=df.walkDistancePerc / df.kills,

        walkDistanceKillsRatio4=df.kills / df.walkDistancePerc,

        walkDistanceKillsRatio5=df.killPerc / df.walkDistance,

        walkDistanceKillsRatio6=df.walkDistance / df.killPerc

    )

    

    return df





def process_solo(df, has_y=True):

    """Feature engineering for Solo data."""

    df = process_df(df)

    no_solo_cols = ['DBNOs','revives','assists','teamKills',]

    df = df.drop(no_solo_cols, axis=1)

    return finalize(df, has_y)

  



def process_ds(df, has_y=True):

    """Feature engineering for Duo-Squad data."""

    df = process_df(df)

    groupby_groupId = df.groupby('groupId')

    df = df.assign(

        groupWalkDistanceMean=groupby_groupId.walkDistance.transform(np.mean),

        groupWalkDistanceSum=groupby_groupId.walkDistance.transform(np.sum),

        groupWalkDistanceMax=groupby_groupId.walkDistance.transform(np.max),

        groupKillsMean=groupby_groupId.kills.transform(np.mean),

        groupKillsSum=groupby_groupId.kills.transform(np.sum),

        groupKillsMax=groupby_groupId.kills.transform(np.max),

        groupHealsMean=groupby_groupId.heals.transform(np.mean),

        groupHealsSum=groupby_groupId.heals.transform(np.sum),

        groupHealsMax=groupby_groupId.heals.transform(np.max),

        groupDamageMean=groupby_groupId.damageDealt.transform(np.mean),

        groupDamageSum=groupby_groupId.damageDealt.transform(np.sum),

        groupDamageMax=groupby_groupId.damageDealt.transform(np.max),

        groupTotalDistanceMean=groupby_groupId.totalDistance.transform(np.mean),

        groupTotalDistanceSum=groupby_groupId.totalDistance.transform(np.sum),

        groupTotalDistanceMax=groupby_groupId.totalDistance.transform(np.max),

        groupWeaponsAcquiredMean=groupby_groupId.weaponsAcquired.transform(np.mean),

        groupWeaponsAcquiredSum=groupby_groupId.weaponsAcquired.transform(np.sum),

        groupWeaponsAcquiredMax=groupby_groupId.weaponsAcquired.transform(np.max),

        groupRevives=groupby_groupId.revives.transform(np.sum),

        groupKnocks=groupby_groupId.DBNOs.transform(np.sum),

    )

    return finalize(df, has_y)
def finalize(df, has_y=True):

    """Last preparations before passing to NN."""

    df = df.drop([

        'Id',

        'matchId',

        'groupId',

        'matchType',

    ], axis=1)

    

    df[df == np.Inf] = np.NaN

    df[df == np.NINF] = np.NaN

    df = df.fillna(0)

    

    if not has_y:

        return df

    

    X = df.drop('winPlacePerc', axis=1)

    y = df[['winPlacePerc']].values.ravel()

    return X, y
def fit(model, *args, **kwargs):

    if env == 'kaggle':

        kwargs.pop('validation_data', None)

    model.fit(*args, **kwargs)
def postprocess(df, y_pred):

    """Get more realistic predictions."""

    df = df.copy()

    df['winPlacePercPred'] = y_pred

    df['meanWinPlacePred'] = df.groupby('groupId').winPlacePercPred.transform(np.mean)

    df['placement'] = df.groupby('matchId').meanWinPlacePred.rank(method='dense', ascending=False)

    y_final = (df.numGroups - df.placement) / (df.numGroups - 1)

    y_final[y_final.isna()] = 0.0

    return y_final.values
df = load_data('train_V2.csv')
df_ds, df_solo = ds_solo_split(df)
env = 'kaggle'
df_ds_train, df_ds_test = train_test_split_by_match(df_ds, test_size=test_size)
print('Processing train Duo-Squad dataset...')

with timeit():

    X_ds_train, y_ds_train = process_ds(df_ds_train)



print('\nProcessing test Duo-Squad dataset...')

with timeit():

    X_ds_test, y_ds_test = process_ds(df_ds_test)
scaler_ds = StandardScaler()

print('Fitting Duo-Squad standard scaling...')

with timeit():

    scaler_ds.fit(X_ds_train)
def get_model(input_shape):

    model = keras.models.Sequential()

    model.add(L.InputLayer(input_shape=input_shape))

    model.add(L.Dense(64, activation='relu'))

    model.add(L.BatchNormalization())

    model.add(L.Dense(32, activation='relu'))

    model.add(L.BatchNormalization())

    model.add(L.Dense(16, activation='relu'))

    model.add(L.Dense(1, activation='sigmoid', kernel_initializer='normal'))

    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model
model_ds = get_model([len(X_ds_train.columns)])
validation_data = []

if env != 'kaggle':

    validation_data = [

        scaler_ds.transform(X_ds_test), 

        y_ds_test

    ]



fit(

    model_ds,

    scaler_ds.transform(X_ds_train), 

    y_ds_train, 

    epochs=100, 

    validation_data=validation_data, 

    batch_size=batch_size, 

    callbacks=[

        EarlyStopping('loss', min_delta=0.0001, patience=20)

    ]

)
if env != 'kaggle':

    y_ds_pred = model_ds.predict(scaler_ds.transform(X_ds_test))

    y_ds_pred_post = postprocess(df_ds_test, y_ds_pred)

    print('MAE DS:', mean_absolute_error(y_ds_test, y_ds_pred_post))
df_solo_train, df_solo_test = train_test_split_by_match(df_solo, test_size=test_size)
print('Processing train Solo dataset...')

with timeit():

    X_solo_train, y_solo_train = process_solo(df_solo_train)



print('\nProcessing train Solo dataset...')

with timeit():

    X_solo_test, y_solo_test = process_solo(df_solo_test)
scaler_solo = StandardScaler()

scaler_solo.fit(X_solo_train)
model_solo = get_model([len(X_solo_train.columns)])
validation_data = []

if env != 'kaggle':

    validation_data = [

        scaler_solo.transform(X_solo_test), 

        y_solo_test

    ]



fit(

    model_solo,

    scaler_solo.transform(X_solo_train), 

    y_solo_train, 

    epochs=200, 

    validation_data=validation_data, 

    batch_size=batch_size, 

    callbacks=[

        EarlyStopping('loss', min_delta=0.0001, patience=20)

    ]

)
if env != 'kaggle':

    y_solo_pred = model_solo.predict(scaler_solo.transform(X_solo_test))

    y_solo_pred_post = postprocess(df_solo_test, y_solo_pred)

    print('MAE SOLO:', mean_absolute_error(y_solo_test, y_solo_pred_post))
if env != 'kaggle':

    y_all_pred = np.hstack([y_ds_pred_post.flatten(), y_solo_pred_post.flatten()])

    y_all_test = np.hstack([y_ds_test, y_solo_test])

    print('MAE ALL:', mean_absolute_error(y_all_test, y_all_pred))
df_final = load_data('test_V2.csv')

print()

df_final_ds, df_final_solo = ds_solo_split(df_final)
X_final_ds = process_ds(df_final_ds, False)

y_final_ds_pred = model_ds.predict(scaler_ds.transform(X_final_ds))

df_final_ds_pred = df_final_ds.assign(winPlacePerc=postprocess(df_final_ds, y_final_ds_pred))
X_final_solo = process_solo(df_final_solo, False)

y_final_solo_pred = model_solo.predict(scaler_solo.transform(X_final_solo))

df_final_solo_pred = df_final_solo.assign(winPlacePerc=postprocess(df_final_solo, y_final_solo_pred))
df_final_all = df_final_ds_pred[['Id', 'winPlacePerc']].append(df_final_solo_pred[['Id', 'winPlacePerc']])

df_result = (df_final[['Id']].set_index('Id').join(df_final_all.set_index('Id'), on='Id'))
df_result.to_csv('submission.csv')
# model_ds.save('model_ds.h5')

# model_solo.save('model_solo.h5')