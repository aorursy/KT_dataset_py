!apt-get remove swig

!apt-get install swig3.0

!ln -s /usr/bin/swig3.0 /usr/bin/swig



!curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

!pip install auto-sklearn
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
all_data = pd.read_csv('../input/train_V2.csv')
all_data["_Kill_headshot_Ratio"] = all_data["kills"]/all_data["headshotKills"]

all_data['_killStreak_Kill_ratio'] = all_data['killStreaks']/all_data['kills']

all_data['_totalDistance'] = 0.25*all_data['rideDistance'] + all_data["walkDistance"] + all_data["swimDistance"]

all_data['_killPlace_MaxPlace_Ratio'] = all_data['killPlace'] / all_data['maxPlace']

all_data['_totalDistance_weaponsAcq_Ratio'] = all_data['_totalDistance'] / all_data['weaponsAcquired']

all_data['_walkDistance_heals_Ratio'] = all_data['walkDistance'] / all_data['heals']

all_data['_walkDistance_kills_Ratio'] = all_data['walkDistance'] / all_data['kills']

all_data['_kills_walkDistance_Ratio'] = all_data['kills'] / all_data['walkDistance']

all_data['_totalDistancePerDuration'] =  all_data["_totalDistance"]/all_data["matchDuration"]

all_data['_killPlace_kills_Ratio'] = all_data['killPlace']/all_data['kills']

all_data['_walkDistancePerDuration'] =  all_data["walkDistance"]/all_data["matchDuration"]

all_data['walkDistancePerc'] = all_data.groupby('matchId')['walkDistance'].rank(pct=True).values

all_data['killPerc'] = all_data.groupby('matchId')['kills'].rank(pct=True).values

all_data['killPlacePerc'] = all_data.groupby('matchId')['killPlace'].rank(pct=True).values

all_data['weaponsAcquired'] = all_data.groupby('matchId')['weaponsAcquired'].rank(pct=True).values

all_data['_walkDistance_kills_Ratio2'] = all_data['walkDistancePerc'] / all_data['killPerc']

all_data['_kill_kills_Ratio2'] = all_data['killPerc']/all_data['walkDistancePerc']

all_data['_killPlace_walkDistance_Ratio2'] = all_data['walkDistancePerc']/all_data['killPlacePerc']

all_data['_killPlace_kills_Ratio2'] = all_data['killPlacePerc']/all_data['killPerc']

all_data['_totalDistance'] = all_data.groupby('matchId')['_totalDistance'].rank(pct=True).values

all_data['_walkDistance_kills_Ratio3'] = all_data['walkDistancePerc'] / all_data['kills']

all_data['_walkDistance_kills_Ratio4'] = all_data['kills'] / all_data['walkDistancePerc']

all_data['_walkDistance_kills_Ratio5'] = all_data['killPerc'] / all_data['walkDistance']

all_data['_walkDistance_kills_Ratio6'] = all_data['walkDistance'] / all_data['killPerc']



all_data[all_data == np.Inf] = np.NaN

all_data[all_data == np.NINF] = np.NaN

all_data.fillna(0, inplace=True)
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

                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:

                    df[col] = df[col].astype(np.uint8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:

                    df[col] = df[col].astype(np.uint16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:

                    df[col] = df[col].astype(np.uint32)                    

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:

                    df[col] = df[col].astype(np.uint64)

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
data = reduce_mem_usage(all_data)

del all_data
#import seaborn as sns

#corr = data.corr()

#sns.heatmap(corr, 

#            xticklabels=corr.columns.values,

#            yticklabels=corr.columns.values)
#because IDs are not useful for predictions

columnsToBeRemoved = ['Id', 'groupId', 'matchId']
# Elo-like columns correlation

eloLikeColumns = ['killPoints', 'rankPoints', 'winPoints']

#elo_corr = data[eloLikeColumns].corr()

#sns.heatmap(elo_corr, 

#            xticklabels=elo_corr.columns.values,

#            yticklabels=elo_corr.columns.values)
#Decided to remove them

columnsToBeRemoved = columnsToBeRemoved + eloLikeColumns
def removeColumns(data):

    return data.drop(columns = columnsToBeRemoved)
from sklearn.preprocessing import LabelBinarizer



class MatchTypeEncoder:

    def __init__(self):

        self._columnName = 'matchType'

        self._encoder =  LabelBinarizer()

    

    def fit(self, data):

        column = data[[self._columnName]]

        self._encoder.fit(column)

    

    def transform(self, data):

        column = data[[self._columnName]]

        return self._encoder.transform(column)
from sklearn.preprocessing import MinMaxScaler



class NumericScaler:

    def __init__(self):

        self._columns = ['assists', 'boosts', 'damageDealt', 'DBNOs',

       'headshotKills', 'heals', 'killPlace', 'kills',

       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace',

       'numGroups', 'revives', 'rideDistance', 'roadKills',

       'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',

       'weaponsAcquired', '_Kill_headshot_Ratio',

       '_killStreak_Kill_ratio', '_totalDistance', '_killPlace_MaxPlace_Ratio',

       '_totalDistance_weaponsAcq_Ratio', '_walkDistance_heals_Ratio',

       '_walkDistance_kills_Ratio', '_kills_walkDistance_Ratio',

       '_totalDistancePerDuration', '_killPlace_kills_Ratio',

       '_walkDistancePerDuration', 'walkDistancePerc', 'killPerc',

       'killPlacePerc', '_walkDistance_kills_Ratio2', '_kill_kills_Ratio2',

       '_killPlace_walkDistance_Ratio2', '_killPlace_kills_Ratio2',

       '_walkDistance_kills_Ratio3', '_walkDistance_kills_Ratio4',

       '_walkDistance_kills_Ratio5', '_walkDistance_kills_Ratio6']

        self._scaler = MinMaxScaler()

        

    def fit(self, data):

        columns = data[self._columns]

        self._scaler.fit(columns)

        

    def transform(self, data):

        columns = data[self._columns]

        return self._scaler.transform(columns)
data = removeColumns(data)
categoricalEncoder = MatchTypeEncoder()

numericalScaler = NumericScaler()
categoricalEncoder.fit(data)

categoricalEncodedData = categoricalEncoder.transform(data)

categoricalEncodedData = pd.DataFrame(categoricalEncodedData, columns = data['matchType'].unique())
numericalScaler.fit(data)

numericalScaledData = numericalScaler.transform(data)

numericalScaledData = pd.DataFrame(numericalScaledData, columns = numericalScaler._columns)
processedData = pd.concat([categoricalEncodedData, numericalScaledData, data['winPlacePerc']], axis = 1)
from sklearn.utils import resample

#processedData = resample(processedData, n_samples = int(processedData.shape[0] * 0.4), random_state = 666)
from sklearn.model_selection import train_test_split

X = processedData.drop(columns= ['winPlacePerc'])

y = processedData[['winPlacePerc']]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 666, train_size = 0.6)
del numericalScaledData

del categoricalEncodedData

del data

del processedData
X_train.shape
from sklearn.decomposition import PCA

pca = PCA(n_components = 0.95)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
try:

    from autosklearn.metrics import mean_absolute_error

    from autosklearn.regression import AutoSklearnRegressor

except:

    import time

    time.sleep(5)

    

    from autosklearn.metrics import mean_absolute_error

    from autosklearn.regression import AutoSklearnRegressor
!wc -l ../input/train_V2.csv
automl = AutoSklearnRegressor(

        time_left_for_this_task = 60 * 10,

        per_run_time_limit = 60 * 2,

        tmp_folder = '/tmp/autosklearn_cv17_tmp',

        output_folder = '/tmp/autosklearn_cv17_out',

        delete_tmp_folder_after_terminate = True,

        ml_memory_limit = 1024 * 10,

        include_estimators=["random_forest", 'adaboost'],

        exclude_estimators=None,

        resampling_strategy = 'cv',

        resampling_strategy_arguments = {'folds': 5},

        n_jobs = 1

    )
X_train_copy = X_train.copy()

y_train_copy = y_train.copy()
automl.fit(X_train_copy, y_train_copy, dataset_name='PUBG', metric = mean_absolute_error)

automl.refit(X_train.copy(), y_train.copy())



print(automl.show_models())



predictions = automl.predict(X_test)

from sklearn.metrics import mean_absolute_error

print("Mean absolute error", mean_absolute_error(y_test, predictions))
test = pd.read_csv('../input/test_V2.csv')

ids = test['Id']





all_data = test



all_data["_Kill_headshot_Ratio"] = all_data["kills"]/all_data["headshotKills"]

all_data['_killStreak_Kill_ratio'] = all_data['killStreaks']/all_data['kills']

all_data['_totalDistance'] = 0.25*all_data['rideDistance'] + all_data["walkDistance"] + all_data["swimDistance"]

all_data['_killPlace_MaxPlace_Ratio'] = all_data['killPlace'] / all_data['maxPlace']

all_data['_totalDistance_weaponsAcq_Ratio'] = all_data['_totalDistance'] / all_data['weaponsAcquired']

all_data['_walkDistance_heals_Ratio'] = all_data['walkDistance'] / all_data['heals']

all_data['_walkDistance_kills_Ratio'] = all_data['walkDistance'] / all_data['kills']

all_data['_kills_walkDistance_Ratio'] = all_data['kills'] / all_data['walkDistance']

all_data['_totalDistancePerDuration'] =  all_data["_totalDistance"]/all_data["matchDuration"]

all_data['_killPlace_kills_Ratio'] = all_data['killPlace']/all_data['kills']

all_data['_walkDistancePerDuration'] =  all_data["walkDistance"]/all_data["matchDuration"]

all_data['walkDistancePerc'] = all_data.groupby('matchId')['walkDistance'].rank(pct=True).values

all_data['killPerc'] = all_data.groupby('matchId')['kills'].rank(pct=True).values

all_data['killPlacePerc'] = all_data.groupby('matchId')['killPlace'].rank(pct=True).values

all_data['weaponsAcquired'] = all_data.groupby('matchId')['weaponsAcquired'].rank(pct=True).values

all_data['_walkDistance_kills_Ratio2'] = all_data['walkDistancePerc'] / all_data['killPerc']

all_data['_kill_kills_Ratio2'] = all_data['killPerc']/all_data['walkDistancePerc']

all_data['_killPlace_walkDistance_Ratio2'] = all_data['walkDistancePerc']/all_data['killPlacePerc']

all_data['_killPlace_kills_Ratio2'] = all_data['killPlacePerc']/all_data['killPerc']

all_data['_totalDistance'] = all_data.groupby('matchId')['_totalDistance'].rank(pct=True).values

all_data['_walkDistance_kills_Ratio3'] = all_data['walkDistancePerc'] / all_data['kills']

all_data['_walkDistance_kills_Ratio4'] = all_data['kills'] / all_data['walkDistancePerc']

all_data['_walkDistance_kills_Ratio5'] = all_data['killPerc'] / all_data['walkDistance']

all_data['_walkDistance_kills_Ratio6'] = all_data['walkDistance'] / all_data['killPerc']



all_data[all_data == np.Inf] = np.NaN

all_data[all_data == np.NINF] = np.NaN

all_data.fillna(0, inplace=True)



data = reduce_mem_usage(all_data)

del all_data



test = removeColumns(data)



numericalScaler._columns = ['assists', 'boosts', 'damageDealt', 'DBNOs',

       'headshotKills', 'heals', 'killPlace', 'kills',

       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace',

       'numGroups', 'revives', 'rideDistance', 'roadKills',

       'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',

       'weaponsAcquired', '_Kill_headshot_Ratio',

       '_killStreak_Kill_ratio', '_totalDistance', '_killPlace_MaxPlace_Ratio',

       '_totalDistance_weaponsAcq_Ratio', '_walkDistance_heals_Ratio',

       '_walkDistance_kills_Ratio', '_kills_walkDistance_Ratio',

       '_totalDistancePerDuration', '_killPlace_kills_Ratio',

       '_walkDistancePerDuration', 'walkDistancePerc', 'killPerc',

       'killPlacePerc', '_walkDistance_kills_Ratio2', '_kill_kills_Ratio2',

       '_killPlace_walkDistance_Ratio2', '_killPlace_kills_Ratio2',

       '_walkDistance_kills_Ratio3', '_walkDistance_kills_Ratio4',

       '_walkDistance_kills_Ratio5', '_walkDistance_kills_Ratio6']



categoricalEncodedData = categoricalEncoder.transform(test)

categoricalEncodedData = pd.DataFrame(categoricalEncodedData, columns = test['matchType'].unique())



numericalScaledData = numericalScaler.transform(test)

numericalScaledData = pd.DataFrame(numericalScaledData, columns = numericalScaler._columns)



processedData = pd.concat([categoricalEncodedData, numericalScaledData], axis = 1)



processedData = pca.transform(processedData)
predictions = automl.predict(processedData)
output = pd.concat([

    pd.DataFrame(ids, columns = ['Id']),

    pd.DataFrame(predictions, columns = ['winPlacePerc'])

], axis = 1)

output.to_csv('submission.csv', index=False)
from sklearn.decomposition import PCA

pca = PCA(n_components = 0.95)

pca.fit_transform(X_train)
import numpy as np

np.sum(pca.explained_variance_ratio_)
pca.explained_variance_ratio_