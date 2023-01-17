import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import gc

import os

import sys



sns.set_style('darkgrid')

sns.set_palette('bone')

pd.options.display.float_format = '{:,.3f}'.format



print(os.listdir("../input"))
def toTapleList(list1,list2):

    return list(itertools.product(list1,list2))
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

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

                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                #    df[col] = df[col].astype(np.float16)

                #el

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
%%time

train = pd.read_csv('../input/train_V2.csv')

train = reduce_mem_usage(train)

test = pd.read_csv('../input/test_V2.csv')

test = reduce_mem_usage(test)

print(train.shape, test.shape)
train.info()
null_cnt = train.isnull().sum().sort_values()

print('null count:', null_cnt[null_cnt > 0])

# dropna

train.dropna(inplace=True)
train.describe(include=np.number).drop('count').T
all_data = train.append(test, sort=False).reset_index(drop=True)

del train, test

gc.collect()
def fillInf(df, val):

    numcols = df.select_dtypes(include='number').columns

    cols = numcols[numcols != 'winPlacePerc']

    df[df == np.Inf] = np.NaN

    df[df == np.NINF] = np.NaN

    for c in cols: df[c].fillna(val, inplace=True)
all_data['_totalDistance'] = all_data['rideDistance'] + all_data['walkDistance'] + all_data['swimDistance']

all_data['_healthItems'] = all_data['heals'] + all_data['boosts']

all_data['_headshotKillRate'] = all_data['headshotKills'] / all_data['kills']

all_data['_killPlaceOverMaxPlace'] = all_data['killPlace'] / all_data['maxPlace']

all_data['_killsOverWalkDistance'] = all_data['kills'] / all_data['walkDistance']

#all_data['_killsOverDistance'] = all_data['kills'] / all_data['_totalDistance']

#all_data['_walkDistancePerSec'] = all_data['walkDistance'] / all_data['matchDuration']



fillInf(all_data, 0)
match = all_data.groupby('matchId')

all_data['killsPerc'] = match['kills'].rank(pct=True).values

all_data['killPlacePerc'] = match['killPlace'].rank(pct=True).values

all_data['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values

#all_data['damageDealtPerc'] = match['damageDealt'].rank(pct=True).values



all_data['walkPerc_killsPerc'] = all_data['walkDistancePerc'] / all_data['killsPerc']

#all_data['walkPerc_kills'] = all_data['walkDistancePerc'] / all_data['kills']

#all_data['kills_walkPerc'] = all_data['kills'] / all_data['walkDistancePerc']
#all_data.drop(['killStreaks','DBNOs'], axis=1, inplace=True)

all_data.drop(['boosts','heals','revives','assists'], axis=1, inplace=True)

all_data.drop(['headshotKills','roadKills','vehicleDestroys','teamKills'], axis=1, inplace=True)

all_data.drop(['rideDistance','swimDistance','matchDuration'], axis=1, inplace=True)

all_data.drop(['rankPoints','killPoints','winPoints'], axis=1, inplace=True)
match = all_data.groupby(['matchId'])

group = all_data.groupby(['matchId','groupId','matchType'])



# target feature (max, min)

agg_col = list(all_data.columns)

exclude_agg_col = ['Id','matchId','groupId','matchType','maxPlace','numGroups','winPlacePerc']

for c in exclude_agg_col:

    agg_col.remove(c)

print(agg_col)



# target feature (sum)

sum_col = ['kills','killPlace','damageDealt','walkDistance','_healthItems']
''' match sum, match max, match mean, group sum

'''

match_data = pd.concat([

    match.size().to_frame('m.players'), 

    match[sum_col].sum().rename(columns=lambda s: 'm.sum.' + s), 

    match[sum_col].max().rename(columns=lambda s: 'm.max.' + s),

    match[sum_col].mean().rename(columns=lambda s: 'm.mean.' + s)

    ], axis=1).reset_index()

match_data = pd.merge(match_data, 

    group[sum_col].sum().rename(columns=lambda s: 'sum.' + s).reset_index())

match_data = reduce_mem_usage(match_data)



print(match_data.shape)
''' ranking of kills and killPlace in each match

'''

minKills = all_data.sort_values(['matchId','groupId','kills','killPlace']).groupby(

    ['matchId','groupId','kills']).first().reset_index().copy()

for n in np.arange(5):

    c = 'kills_' + str(n) + '_Place'

    nKills = (minKills['kills'] == n)

    minKills.loc[nKills, c] = minKills[nKills].groupby(['matchId'])['killPlace'].rank().values

    match_data = pd.merge(match_data, minKills[nKills][['matchId','groupId',c]], how='left')

    match_data[c].fillna(0, inplace=True)

match_data = reduce_mem_usage(match_data)

del minKills, nKills



print(match_data.shape)
''' group mean, max, min

'''

all_data = pd.concat([

    group.size().to_frame('players'),

    group.mean(),

    group[agg_col].max().rename(columns=lambda s: 'max.' + s),

    group[agg_col].min().rename(columns=lambda s: 'min.' + s),

    ], axis=1).reset_index()

all_data = reduce_mem_usage(all_data)



print(all_data.shape)
numcols = all_data.select_dtypes(include='number').columns.values

numcols = numcols[numcols != 'winPlacePerc']
''' match summary, max

'''

all_data = pd.merge(all_data, match_data)

del match_data

gc.collect()



all_data['enemy.players'] = all_data['m.players'] - all_data['players']

for c in sum_col:

    all_data['enemy.' + c] = (all_data['m.sum.' + c] - all_data['sum.' + c]) / all_data['enemy.players']

    #all_data['p.sum_msum.' + c] = all_data['sum.' + c] / all_data['m.sum.' + c]

    #all_data['p.max_mmean.' + c] = all_data['max.' + c] / all_data['m.mean.' + c]

    all_data['p.max_msum.' + c] = all_data['max.' + c] / all_data['m.sum.' + c]

    all_data['p.max_mmax.' + c] = all_data['max.' + c] / all_data['m.max.' + c]

    all_data.drop(['m.sum.' + c, 'm.max.' + c], axis=1, inplace=True)

    

fillInf(all_data, 0)

print(all_data.shape)
''' match rank

'''

match = all_data.groupby('matchId')

matchRank = match[numcols].rank(pct=True).rename(columns=lambda s: 'rank.' + s)

all_data = reduce_mem_usage(pd.concat([all_data, matchRank], axis=1))

rank_col = matchRank.columns

del matchRank

gc.collect()



# instead of rank(pct=True, method='dense')

match = all_data.groupby('matchId')

matchRank = match[rank_col].max().rename(columns=lambda s: 'max.' + s).reset_index()

all_data = pd.merge(all_data, matchRank)

for c in numcols:

    all_data['rank.' + c] = all_data['rank.' + c] / all_data['max.rank.' + c]

    all_data.drop(['max.rank.' + c], axis=1, inplace=True)

del matchRank

gc.collect()



print(all_data.shape)
''' TODO: incomplete

''' 

killMinorRank = all_data[['matchId','min.kills','max.killPlace']].copy()

group = killMinorRank.groupby(['matchId','min.kills'])

killMinorRank['rank.minor.maxKillPlace'] = group.rank(pct=True).values

all_data = pd.merge(all_data, killMinorRank)



killMinorRank = all_data[['matchId','max.kills','min.killPlace']].copy()

group = killMinorRank.groupby(['matchId','max.kills'])

killMinorRank['rank.minor.minKillPlace'] = group.rank(pct=True).values

all_data = pd.merge(all_data, killMinorRank)



del killMinorRank

gc.collect()
# drop constant column

constant_column = [col for col in all_data.columns if all_data[col].nunique() == 1]

print('drop columns:', constant_column)

all_data.drop(constant_column, axis=1, inplace=True)
'''

solo  <-- solo,solo-fpp,normal-solo,normal-solo-fpp

duo   <-- duo,duo-fpp,normal-duo,normal-duo-fpp,crashfpp,crashtpp

squad <-- squad,squad-fpp,normal-squad,normal-squad-fpp,flarefpp,flaretpp

'''

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

all_data['matchType'] = all_data['matchType'].apply(mapper)



all_data = pd.concat([all_data, pd.get_dummies(all_data['matchType'], prefix='matchType')], axis=1)
cols = [col for col in all_data.columns if col not in ['Id','matchId','groupId']]

for i, t in all_data.loc[:, cols].dtypes.iteritems():

    if t == object:

        all_data[i] = pd.factorize(all_data[i])[0]



all_data = reduce_mem_usage(all_data)

all_data.head()
X_train = all_data[all_data['winPlacePerc'].notnull()].reset_index(drop=True)

X_test = all_data[all_data['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis=1).reset_index(drop=True)

del all_data

gc.collect()



Y_train = X_train.pop('winPlacePerc')

X_test_grp = X_test[['matchId','groupId']].copy()



# drop matchId,groupId

X_train.drop(['matchId','groupId'], axis=1, inplace=True)

X_test.drop(['matchId','groupId'], axis=1, inplace=True)



X_train_cols = X_train.columns



print(X_train.shape, X_test.shape)
#print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],

#                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])
from keras import optimizers, regularizers

from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.layers import Dense, Dropout, BatchNormalization, PReLU

from keras.models import load_model

from keras.models import Sequential



def createModel():

    model = Sequential()

    model.add(Dense(512, kernel_initializer='he_normal', input_dim=X_train.shape[1], activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))



    model.add(Dense(256, kernel_initializer='he_normal'))

    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))



    model.add(Dense(128, kernel_initializer='he_normal'))

    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))

    model.add(BatchNormalization())

    model.add(Dropout(0.1))



    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))



    optimizer = optimizers.Adam(lr=0.005)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    #model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mae'])

    

    return model
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10, verbose=0):

    ''' Wrapper function to create a LearningRateScheduler with step decay schedule. '''

    def schedule(epoch):

        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    

    return LearningRateScheduler(schedule, verbose)



lr_sched = step_decay_schedule(initial_lr=0.001, decay_factor=0.97, step_size=1, verbose=1)

early_stopping = EarlyStopping(monitor='val_mean_absolute_error', mode='min', patience=10, verbose=1)
from sklearn import preprocessing

from tensorflow import set_random_seed

np.random.seed(42)

set_random_seed(1234)



scaler = preprocessing.StandardScaler().fit(X_train.astype(float))

X_train = scaler.transform(X_train.astype(float))

X_test = scaler.transform(X_test.astype(float))



model = createModel()

history = model.fit(

        X_train, Y_train,

        epochs=200,

        batch_size=2**15,

        validation_split=0.2,

        callbacks=[lr_sched, early_stopping],

        verbose=2)

pred = model.predict(X_test).ravel()
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

plt.title('Mean Abosulte Error')

plt.ylabel('Mean absolute error')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
'''

from eli5.permutation_importance import get_score_importances



def score(X, y):

    y_pred = model.predict(X).ravel()

    return np.sum(np.abs(y - y_pred))



base_score, score_decreases = get_score_importances(score, X_train[:10000], Y_train[:10000])

feature_importances = np.mean(score_decreases, axis=0) * -1



feature_importances = 100.0 * (feature_importances / feature_importances.max())

sorted_idx = np.argsort(feature_importances)

sorted_idx = sorted_idx[len(feature_importances) - 30:]

pos = np.arange(sorted_idx.shape[0]) + .5



plt.figure(figsize=(12,8))

plt.barh(pos, feature_importances[sorted_idx], align='center')

plt.yticks(pos, X_train_cols[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()



X_train_cols[np.argsort(feature_importances)[::-1]].values

'''
X_test = pd.read_csv('../input/test_V2.csv')

X_test = X_test.groupby(['matchId','groupId','matchType']).first().reset_index()

X_test = X_test[['matchId','groupId','matchType','numGroups','maxPlace','kills','killPlace']]



group = X_test_grp.groupby(['matchId'])

X_test_grp['winPlacePerc'] = pred

X_test_grp['_rank.winPlacePerc'] = group['winPlacePerc'].rank(method='min')

X_test = pd.merge(X_test, X_test_grp)
fullgroup = (X_test['numGroups'] == X_test['maxPlace'])



# full group (201366) --> calculate from rank

subset = X_test.loc[fullgroup]

X_test.loc[fullgroup, 'winPlacePerc'] = (subset['_rank.winPlacePerc'].values - 1) / (subset['maxPlace'].values - 1)



# not full group (684872) --> align with maxPlace

subset = X_test.loc[~fullgroup]

gap = 1.0 / (subset['maxPlace'].values - 1)

new_perc = np.around(subset['winPlacePerc'].values / gap) * gap  # half&up

X_test.loc[~fullgroup, 'winPlacePerc'] = new_perc



X_test['winPlacePerc'] = X_test['winPlacePerc'].clip(lower=0,upper=1)
from tqdm import tqdm



# credit to https://www.kaggle.com/nagroda100/pubg-submission-postprocessor/code

print("Checking for anomalies in the winPlacePerc - players with same number of kills should have scores in order of killPlace")



do_correct = True

iteration_number = 1



while do_correct & (iteration_number <= 1000):

    X_test.sort_values(ascending=False, by=["matchId","kills","killPlace","winPlacePerc","groupId"], inplace=True)

    X_test["winPlacePerc_diff"] = X_test["winPlacePerc"].diff()

    X_test["kills_diff"] = X_test["kills"].diff()

    X_test["prev_matchId"] = X_test["matchId"].shift(1)

    X_test["prev_groupId"] = X_test["groupId"].shift(1)

    X_test["prev_winPlacePerc"] = X_test["winPlacePerc"].shift(1)



    df_sub2 = X_test[(X_test["winPlacePerc_diff"] < 0) 

                     & (X_test["kills_diff"] == 0) 

                     & (X_test["matchId"] == X_test["prev_matchId"])]

    anomalies_count = len(df_sub2)



    print("Iteration " + str(iteration_number) + " Anomalies count: " + str(anomalies_count))



    changed_groups = list()



    if anomalies_count > 0:

        print()

        print("Looking for pairs to change...")



        df_sub2["new_winPlacePerc"] = df_sub2["winPlacePerc"] 



        df_sub3 = pd.DataFrame()



        for i in tqdm(range(1, min(15001, max(anomalies_count, 2))), 

                      desc="Identifying unique groups", mininterval=10):

            row = df_sub2.iloc[i - 1]

            id_prev = str(row["prev_matchId"]) + "!" + str(row["prev_groupId"])

            id_cur = str(row["matchId"]) + "!" + str(row["groupId"])

            if (not id_prev in changed_groups) & (not id_cur in changed_groups):

                changed_groups.append(id_prev)

                changed_groups.append(id_cur)

                df_sub3 = df_sub3.append({"matchId": row["matchId"], "groupId": row["prev_groupId"], 

                                          "new_winPlacePerc": row["winPlacePerc"]}, 

                                         sort=False, ignore_index=True)

                df_sub3 = df_sub3.append({"matchId": row["matchId"], "groupId": row["groupId"], 

                                          "new_winPlacePerc": row["prev_winPlacePerc"]}, 

                                         sort=False, ignore_index=True)



        df_sub3.drop_duplicates(inplace=True)

        X_test = X_test.merge(df_sub3, on=["matchId", "groupId"], how="left")

        notna = X_test["new_winPlacePerc"].notna()

        X_test.loc[notna, "winPlacePerc"] = X_test.loc[notna]["new_winPlacePerc"]

        X_test.drop(labels="new_winPlacePerc", axis=1, inplace=True)

        del df_sub2

        del df_sub3

        df_sub2 = None

        df_sub3 = None

        gc.collect()

    else:

        do_correct = False



    iteration_number = iteration_number + 1



if do_correct:

    print("Limit of iterations reached...")



print("Finished correcting winPlacePerc")
# edge cases

X_test.loc[X_test['maxPlace'] == 0, 'winPlacePerc'] = 0

X_test.loc[X_test['maxPlace'] == 1, 'winPlacePerc'] = 1  # nothing

X_test.loc[(X_test['maxPlace'] > 1) & (X_test['numGroups'] == 1), 'winPlacePerc'] = 0

X_test['winPlacePerc'].describe()
test = pd.read_csv('../input/test_V2.csv')



submission = pd.merge(test, X_test[['matchId','groupId','winPlacePerc']])

submission = submission[['Id','winPlacePerc']]

submission.to_csv("submission.csv", index=False)