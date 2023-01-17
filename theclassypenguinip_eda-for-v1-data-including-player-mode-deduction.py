# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import multiprocessing

try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2   # arbitrary default

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")
test_set = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")
dataset.head()
match_amount = len(dataset.matchId.unique())
print('Number of matches in the dataset: %d' % match_amount)
group = dataset.groupby("groupId")['matchId'].nunique()
group[group > 1]
def get_group_values(dataframe,groupby, values):
    
    grouped_set = dataframe.groupby(groupby)
        
    for value in values:
        
        dataframe = dataframe.join(grouped_set[value].rank(ascending=False, pct=True),on=groupby,rsuffix='_percentile')
        
        description = grouped_set[value].describe()
        description.columns = [value+'_count', value+'_mean', value+'_std', value+'_min', value+'_25%', value+'_50%', value+'_75%', value+'_max']
        dataframe = pd.concat([dataframe,description], axis=1)
        
    return dataframe
    
%timeit get_group_values(dataset.head(1000), 'matchId', dataset.columns)
def get_group_value(args):
    
    value, grouped_dataframe = args
    
    percentile = grouped_dataframe[value].rank(ascending=False, pct=True)
    description = grouped_dataframe[value].describe()
    variable_description = pd.concat([percentile,description], axis=1)
    variable_description.columns = [value+'_percentile',value+'_count', value+'_mean', value+'_std', value+'_min', value+'_25%', value+'_50%', value+'_75%', value+'_max']
    return variable_description

def get_group_values(dataframe,groupby, values):
    
    grouped_set = dataframe.groupby(groupby)
        
    pool = multiprocessing.Pool(processes=cpus)
    work = [[value, grouped_set] for value in values]
    return pd.concat([dataframe]+pool.map(get_group_value, work), axis=1)

%timeit get_group_values(dataset.head(1000), 'matchId', dataset.columns)

group_size = dataset.groupby(["matchId","groupId"])['Id'].nunique()
group_size.plot(kind='hist', logy = True, title='Group size distribution', bins=100)
matches = group_size.reset_index().matchId.unique()
not_solo_matches = group_size[group_size != 1].reset_index().matchId.unique()
solo_matches = np.setdiff1d(matches, not_solo_matches)
solo_matches_amount = len(solo_matches)
#not_solo_matches = pd.Series(not_solo_matches)
print('Number of matches in the dataset: %d' % match_amount)
print('Number of solo matches in the dataset: %d' % solo_matches_amount)
print('Percentage of solo matches in the dataset: %f%%' % ((solo_matches_amount/match_amount)*100))
not_duo_matches = group_size[group_size != 2].reset_index().matchId.unique()
duo_matches = np.setdiff1d(matches, not_duo_matches)
duo_matches_amount = len(duo_matches)
#not_duo_matches = pd.Series(not_duo_matches)
print('Number of matches in the dataset: %d' % match_amount)
print('Number of duo matches in the dataset: %d' % duo_matches_amount)
print('Percentage of duo matches in the dataset: %f%%' % ((duo_matches_amount/match_amount)*100))
not_squad_matches = group_size[group_size >= 5].reset_index().matchId.unique()
squad_matches = np.setdiff1d(np.setdiff1d(np.setdiff1d(matches, not_squad_matches), duo_matches),solo_matches)
squad_matches_amount = len(squad_matches)
print('Number of matches in the dataset: %d' % match_amount)
print('Number of squad matches in the dataset: %d' % squad_matches_amount)
print('Percentage of squad matches in the dataset: %f%%' % ((squad_matches_amount/match_amount)*100))
custom_matches = np.setdiff1d(np.setdiff1d(np.setdiff1d(matches, squad_matches), duo_matches),solo_matches)
#not_squad_matches = np.setdiff1d(groupsize.reset_index().matchID.unique(), custom_matches)
custom_matches_amount = len(custom_matches)
print('Number of matches in the dataset: %d' % match_amount)
print('Number of custom matches in the dataset: %d' % custom_matches_amount)
print('Percentage of custom matches in the dataset: %f%%' % ((custom_matches_amount/match_amount)*100))
labels = 'Duo', 'Squad', 'Solo' ,'Custom'
sizes = [duo_matches_amount, squad_matches_amount, solo_matches_amount, custom_matches_amount]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.05, 0.05, 0.05, 0.05)
plt.pie(sizes, labels=labels, explode=explode, colors=colors)
plt.axis('equal')
plt.show()
solo = set(solo_matches)
duo = set(duo_matches)
squad = set(squad_matches)
custom = set(custom_matches)

def one_hot_game_mode(Id, solo, duo, squad, custom):
    return int(Id in solo), int(Id in duo), int(Id in squad), int(Id in custom)

dataset['solo'] = 0
dataset['duo'] = 0
dataset['squad'] = 0
dataset['custom'] = 0
dataset[['solo','duo','squad','custom']] = pd.DataFrame(elem for elem in dataset.matchId.apply(one_hot_game_mode,args=(solo,duo,squad,custom)))

dataset.head()
dataset.damageDealt.plot(kind='hist', logy=True)
norm_list = ["winPoints","killPoints","damageDealt","rideDistance","walkDistance", "swimDistance", "longestKill"]
dataset[norm_list] =(dataset[norm_list] - dataset[norm_list].mean())/(dataset[norm_list].max() - dataset[norm_list].min())
test_set[norm_list] =(test_set[norm_list] - dataset[norm_list].mean())/(dataset[norm_list].max() - dataset[norm_list].min())
#train.killPoints =(train.winPoints-train.winPoints.mean())/(train.winPoints.max() - train.winPoints.min())
train.winPoints.plot(kind='hist', logy=True)
train.walkDistance.plot(kind='hist', logy=True)
correlation_matrix = dataset.corr()
correlation_matrix.winPlacePerc.sort_values(ascending=False)
aug_dataset = dataset = pd.read_csv("../input/pubgplayerstats/PUBG_Player_Statistics.csv")
print(str(aug_dataset.columns.values))

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

dataset_labels = dataset.pop('winPlacePerc')
dataset = dataset.drop(['Id','groupId','matchId'], axis=1)
test_set = test_set.drop(['Id','groupId','matchId'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(dataset, dataset_labels ,test_size=0.2)
#test = pd.read_csv("../input/test.csv")

xgb.fit(X_train,y_train, eval_metric='rmse', verbose=True)

predictions = xgb.predict(X_test)
print(explained_variance_score(predictions,y_test))
print(mean_squared_error(predictions,y_test))

pred = xgb.predict(test_set)
print(pred)
test = pd.read_csv("../input/sample_submission.csv")
test['winPlacePerc'] = pd.Series(pred)
test.to_csv("../submission.csv")