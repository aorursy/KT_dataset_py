# Imports
#from datetime import datetime
import datetime as dt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math
#import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff

'''For ML:'''
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

#plotly.__version__

# List of Datafiles
print(os.listdir("../input"))
pitstops = pd.read_csv('../input/pitStops.csv')
results = pd.read_csv('../input/results.csv')
races = pd.read_csv('../input/races.csv')
circuits = pd.read_csv('../input/circuits.csv', encoding='latin1')
drivers = pd.read_csv('../input/drivers.csv', encoding='latin1')

#identify yellow flag
laptimes = pd.read_csv('../input/lapTimes.csv')
laptimes.head()

# Time Behind Leader
laptimes.sort_values(by = ['raceId', 'driverId', 'lap'], inplace=True)

laptimes.head()
#calculating the "totalmilli" and creating apropriate column for it in the df
laptimes['totalmilli'] = laptimes.groupby(['raceId', 'driverId'])['milliseconds'].transform(pd.Series.cumsum)

# Creating the copies to mearge:
laptimes_2 = laptimes[['raceId', 'lap', 'position', 'totalmilli']].copy()
laptimes_3 = laptimes[['raceId', 'lap', 'position', 'totalmilli']].copy()

# Adding and subtractin "1" to each position, so than we can merge the "correct" position with the one in front of it:
laptimes_2['position'] = laptimes_2['position'] + 1
laptimes_2.rename(columns={'position': "position_plus_1", 'totalmilli' : 'totalmilli_plus_1'}, inplace=True)

laptimes_3['position'] = laptimes_3['position'] -1
laptimes_3.rename(columns={'position': "position_min_1", 'totalmilli' : 'totalmilli_min_1'}, inplace=True)

# Mearging two dataframes:
merged = pd.merge(laptimes, laptimes_2, how = 'left', left_on=['raceId', 'lap', 'position'],
                  right_on=['raceId', 'lap', 'position_plus_1'])

# Mearging two dataframes:
merged = pd.merge(merged, laptimes_3, how = 'left', left_on=['raceId', 'lap', 'position'],
                  right_on=['raceId', 'lap', 'position_min_1'])

# Calculating how far each car behind/in front:
merged['to_in_front'] = merged['totalmilli'] - merged['totalmilli_plus_1']
merged['to_behind'] = merged['totalmilli_min_1'] - merged['totalmilli']
#Checking Results of Time Between
# 'to_previous' has to be >= 0:
print("positive:", merged[merged['to_in_front']>0].shape)
print("equal zero:", merged[merged['to_in_front']==0].shape)
print('less than zero', merged[merged['to_in_front']<0].shape)
# Now we can delete 'position_plus_1' and 'totalmilli_plus_1' columns if needed.
merged.drop(['position_plus_1', 'totalmilli_plus_1', 'position_min_1', 'totalmilli_min_1'], axis=1, inplace = True)
# Puting merged df into laptimes
laptimes = merged.copy()
laptimes.head()
'''To calculate the average position at the finish we take only races that were finished by the driver'''
avg_position = results[results['milliseconds'].notnull()].groupby(['driverId'])['position'].mean()
avg_position = avg_position.to_frame()
avg_position.columns = ['avg_position']
'''Puting avg_position into seperate column in the results df'''
results = results.merge(avg_position, left_on='driverId', right_index=True)
laptimes = laptimes.merge(avg_position, left_on='driverId', right_index=True)
laptimes['relative_to_avg'] = laptimes['avg_position'] - laptimes['position']
laptimes.head()
pitstops.rename(columns = {'stop':'ps_order', 'time':'exact_ps_time', 'milliseconds':'ps_duration'}, inplace=True)

'''Take only races for which we have ps data (#841-988) and drop "duration" column from pitstops_df'''
laptimes = laptimes[laptimes['raceId']>=841].merge(pitstops.drop(['duration'], axis=1),
                                                    how='left',on =['raceId','driverId','lap'])
'''Getting the fastest laps times:'''
minLaps = laptimes.groupby(['raceId', 'lap'])['milliseconds'].min().reset_index()
minLaps.head()

# '''Getting the fastest laps in races times:'''
BestLapinRaces = laptimes.groupby(['raceId'])['milliseconds'].min().reset_index()
BestLapinRaces.head()

# '''Calculating yellowThreshold = fastest lap in race * 1.1'''
BestLapinRaces['yellowThreshold'] = BestLapinRaces['milliseconds'] * 1.1 
BestLapinRaces.head()

# '''flag if yellowThreshold > the best lap time'''
minLaps = minLaps.merge(BestLapinRaces[['raceId','yellowThreshold']], how = 'left', on='raceId')
minLaps['flag'] = (minLaps['yellowThreshold'] >  minLaps['milliseconds']).astype(int)
minLaps.drop(['yellowThreshold'], axis=1) #drop yellowThreshold
              
minLaps.rename(index=str, columns={"milliseconds": "minLap"}, inplace=True)
'''Merge laptimes_df with minLaps_df'''
laptimes = laptimes.merge(minLaps, how='left', on=['raceId', 'lap'])
'''Graph to better understand the number of pit stops that are ignored:'''
laptimes[(laptimes['ps_order'].notnull())&(laptimes['lap'] <= 10)]['lap'].hist(bins=10)
plt.title('Pitstops')
plt.xlabel('lap')
var = plt.ylabel('number of pit stops')
'''Creating df to get time_min_1 and put it into laptimes_df'''
lap_min_1 = laptimes.groupby(['raceId', 'driverId', 'lap'])['milliseconds'].first().to_frame()
lap_min_1.reset_index(inplace=True)
lap_min_1['lap'] = lap_min_1['lap'] + 2 # increased to 2 & 3 prior because entry lap show pit activity - ie slowing
lap_min_1.rename(columns={"milliseconds": "milli_for_min_1"}, inplace=True)

'''Creating df to get time_min_2 and put it into laptimes_df'''
lap_min_2 = laptimes.groupby(['raceId', 'driverId', 'lap'])['milliseconds'].first().to_frame()
lap_min_2.reset_index(inplace=True)
lap_min_2['lap'] = lap_min_2['lap'] + 3
lap_min_2.rename(columns={"milliseconds": "milli_for_min_2"}, inplace=True)

'''Merging 3 dataframes'''
laptimes = laptimes.merge(lap_min_1, how='left', on=['raceId', 'driverId', 'lap'])
laptimes = laptimes.merge(lap_min_2, how='left', on=['raceId', 'driverId', 'lap'])
laptimes['since_last_ps'] = np.nan
for index, row in laptimes[laptimes['ps_order'].notnull()].iterrows():
    if row['ps_order'] == 1: # and row['lap'] not in [1,2]:
        since_last_ps = row['totalmilli']
    elif row['ps_order'] != 1 and index != 115486: # 115486 because the data issue,see below
        since_last_ps = row['totalmilli'] - laptimes[(laptimes['ps_order'].notnull())&
                                                    (laptimes['raceId']==row['raceId'])&
                                                    (laptimes['driverId']==row['driverId'])&
                                                    (laptimes['ps_order']==row['ps_order']-1)
                                                    ]['totalmilli']
#     print(index, since_last_ps)
    laptimes.at[index, 'since_last_ps'] = since_last_ps

'''There is some data issue for (raceId, driverId) = (908,820)
The time for the 2nd lap is very big'''
# laptimes[(laptimes['raceId']==908)&(laptimes['driverId']==820)]
laptimes['since_last_ps'] = laptimes.apply(lambda x :x['milliseconds'] if math.isnan(x['since_last_ps'])
                                           else x['milliseconds'] + x['since_last_ps']*-1, axis=1)
# Sorting values
laptimes.sort_values(by = ['raceId', 'driverId', 'lap'], inplace=True)

'''Calculating since_last_ps (final)'''
laptimes['since_last_ps'] = laptimes.groupby(['raceId', 'driverId'])['since_last_ps'].transform(pd.Series.cumsum)
'''Identifying each lap according to its segment'''
laptimes['segment'] = laptimes['lap'].apply(lambda x: (x-1) // 10)

'''Calculating the min lap in each segment'''
laptimes['min_Lap_Segm'] = laptimes.groupby(['raceId', 'segment'])['milliseconds'].transform(pd.Series.min)

'''Calculating min lap for (race, driver, segment)'''
laptimes['min_Lap_Driv_Segm'] = laptimes.groupby(['raceId', 'driverId', 'segment'])['milliseconds'].transform(pd.Series.min)

'''Function to calculate ps_order: returns a list'''
def ps_position_fn(inp_list):
    ps_order_list = []
    ps_order = 0
    
    for ps in inp_list:
        if ~np.isnan(ps):
            ps_order += 1
        ps_order_list.append(ps_order)
    return ps_order_list
'''Calculating the fastest lap by driver and race since last pitstop'''

temp_df = laptimes.groupby(['raceId', 'driverId'])['ps_order'].apply(ps_position_fn).reset_index()
s = temp_df.apply(lambda x: pd.Series(x['ps_order']),axis=1).stack().reset_index(level=1, drop=True)
laptimes['temp_ps_order'] = s.values

'''Creating temp_df'''
temp_df = laptimes.groupby(['raceId', 'driverId', 'temp_ps_order'])['milliseconds'].min().reset_index()
temp_df['min_Lap_since_ps'] = 1

'''Merging'''
laptimes = laptimes.merge(temp_df, how='left', on = ['raceId', 'driverId', 'temp_ps_order', 'milliseconds'])
'''Creating new df and shifting the data for the previous lap to match the pit stop'''
temp_df = laptimes.copy()

min_1_df = temp_df.groupby(['raceId', 'driverId', 'lap'])['position', 'totalmilli', 'to_in_front',
                                                                    'to_behind', 'minLap', 'flag',
                                                                  'relative_to_avg', 'since_last_ps'].last()
min_1_df.reset_index(inplace=True)
min_1_df['lap'] = min_1_df['lap'] + 1
# min_1_df.rename(columns={"to_in_front": "to_in_front_min_1",
#                          'to_behind':'to_behind_min_1'}, inplace=True)
'''Merging'''
temp_df = temp_df.merge(min_1_df, how='left', on=['raceId', 'driverId', 'lap'])
'''We do not observe (race, driver, lap) for drivers who are at the position #1, because we do not have
to_in_front_y data for them.
Also we do not observe to_behind_y for some (race, driver, lap).
To better understand what data we lose, below are the codes to plot distributions
(uncommnet some of the lines):'''
# temp_df[(temp_df['milli_for_min_2'].notnull())
# #         &(temp_df['to_in_front_y'].isnull())
# #         &(temp_df['to_behind_y'].isnull())
#         ]['position_y'].hist(bins=100)
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# A function that gives a visual representation of the decision tree
def show_decision_tree(model):
    dot_data = tree.export_graphviz(ps_tree, out_file=None) 
    graph = graphviz.Source(dot_data) 
#     To save on a PDF file
#     graph.render("iris")
    return graph
'''transforming ps_order column into 1 (there was a pit stop) and 0 (there was not a pit stop)'''
temp_df['ps_order'] = temp_df['ps_order'].apply(lambda x: 0 if np.isnan(x) else 1)

'''getting X and y'''
X_y = temp_df[['ps_order', 'position_y', 'totalmilli_y', 'to_in_front_y', 'to_behind_y', 'minLap_y',
               'flag_y', 'relative_to_avg_y', 'milli_for_min_1','milli_for_min_2', 'since_last_ps_y']
             ].dropna()
X = X_y[['position_y', 'totalmilli_y', 'to_in_front_y', 'to_behind_y', 'minLap_y',
               'flag_y', 'relative_to_avg_y', 'milli_for_min_1','milli_for_min_2', 'since_last_ps_y'
           ]].values
y = X_y[['ps_order']].values

'''Visualization of the correlation between parameters:'''
plt.figure(figsize=(10,10))
plt.matshow(X_y.corr(), cmap="Blues", fignum = 1)
plt.colorbar(shrink=0.8)
plt.xticks(range(len(list(X_y))), list(X_y), rotation=90, size = 15)
plt.yticks(range(len(list(X_y))), list(X_y), size = 15)

print('Correlatino matrix:')
plt.show()

'''If you want to see the representation in numbers:'''
# try_df.corr()

'''Note:
We can also plot the distributions of the variables but I am not sure how informative it will be.'''
'''Splitting whole sample into train and test:'''
train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)
'''max-depth = 5'''
ps_tree = DecisionTreeClassifier(max_depth=5, criterion="gini")
ps_tree.fit(train_X, train_y)

pred_y = ps_tree.predict(test_X)
'''Checking on what is the optimal depth of te Decision Tree Classifier:'''
accuracies_train = []
accuracies_test = []
depths = range(1, 35)

for md in depths:
    model = DecisionTreeClassifier(max_depth=md)
    model.fit(train_X, train_y)
    
    accuracies_train.append(model.score(train_X, train_y))
    accuracies_test.append(model.score(test_X, test_y))

plt.plot(depths, accuracies_train, label="Train")
plt.plot(depths, accuracies_test, label="Test")
plt.title("Performance on train and test data")
plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.ylim([0.85, 1.05])
plt.xlim([1,35])
plt.legend()
plt.show()
# using the score function in each model class
print("accuracy on the test set", ps_tree.score(test_X, test_y))
print("accuracy on the training set", ps_tree.score(train_X, train_y))

# using single metric functions in the sklearn.metrics package 
print("accuracy on the test set", accuracy_score(pred_y, test_y))
'''Visualizing DecisionTree'''
# show_decision_tree(ps_tree)
'''Count number of pist stops (1) and number of laps without pit stops (0)'''
from collections import Counter
y_list = y.tolist()
y_list = [item for sublist in y_list for item in sublist]
Counter(y_list)
'''We can check how the feature_importances changes as we change the depth of the tree'''
max_depth = 1

ps_tree = DecisionTreeClassifier(max_depth=max_depth, criterion="gini")
ps_tree.fit(train_X, train_y)

ps_tree.feature_importances_
'''Indicating the circuitId
creating the column that will indicate the circuitId'''
laptimes = laptimes.merge(races[['raceId', 'circuitId']], how='left', on='raceId')
'''finding the circuitId with the most finished laps'''
# temp_df = laptimes.groupby(['circuitId', 'lap'])['raceId'].count().reset_index()
# temp_df['temp_col'] = temp_df['lap'] * temp_df['raceId']
# temp_df.sort_values('temp_col', ascending = False).head(2)

# '''So we will take cidrcuitId = 11'''
'''finding mid lap for each circuitId'''
laptimes = laptimes.merge(results[['raceId', 'driverId', 'position']].rename(columns={"position": "finish_position"}),
                            how='left', on=['raceId', 'driverId'])

'''take all the laps which were the part of the "finish" races
determine the mid lap for each circle'''
s_temp = laptimes[(laptimes['finish_position'].notnull())]
s_temp = s_temp.groupby(['circuitId'])['lap'].apply(lambda x: np.max(x) // 2)

'''taking only the laps forom the first half of the races'''
first_half_df = laptimes[laptimes[['lap','circuitId']].apply(lambda x: x[0] <= s_temp[x[1]], axis=1)]

'''Taking the laps which were the part of the finish races'''
first_half_df = first_half_df[first_half_df['finish_position'].notnull()]
'''Calculating time since last PS (before_prev_ps)'''
first_half_df['before_prev_ps'] = np.nan
for index, row in first_half_df[first_half_df['ps_order'].notnull()].iterrows():
    if row['ps_order'] == 1: # and row['lap'] not in [1,2]:
        before_prev_ps = row['totalmilli'] - row['ps_duration']
    elif row['ps_order'] != 1 and index not in [115486, 80831]: # 115486, 80831 because the data issue
        before_prev_ps = row['totalmilli'] - first_half_df[(first_half_df['ps_order'].notnull())&
                                                            (first_half_df['raceId']==row['raceId'])&
                                                            (first_half_df['driverId']==row['driverId'])&
                                                            (first_half_df['ps_order']==row['ps_order']-1)
                                                            ]['totalmilli'] - row['ps_duration']
#     print(index, before_prev_ps)
    first_half_df.at[index, 'before_prev_ps'] = before_prev_ps

'''Calculating the parameters'''
f = {'milliseconds':['min'],
     'ps_order':['count'],
     'before_prev_ps':['mean', 'min', 'max'],
     'position':['first', 'last'],
     'ps_duration':['sum'],
     'finish_position':['last'],
     'circuitId': ['last']}
X_y = first_half_df.groupby(['raceId', 'driverId']).agg(f).reset_index()
X_y.columns = ['raceId', 'driverId', 'milliseconds_min', 'ps_count', 'before_prev_ps_mean',
               'before_prev_ps_min', 'before_prev_ps_max','position_first', 'position_last',
               'ps_duration_sum', 'finish_position', 'circuitId']

'''Calculating avg of the fastest laps in the race'''
X_y['avg_milliseconds_min'] = X_y.groupby(['raceId'])['milliseconds_min'].transform(pd.Series.mean)
# X_y
'''Checking the size of our sample'''
print('Sample size:', X_y.shape)
'''Notes:
1)The sample size is porbably too small for the lightgbm.
    Accourding to this article (https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
    it can lead to overfitting.
2) How we sorted the data - we took only the races of the drivers who finished a particular race;
                          - for each circleId we take the average max number of laps and find the medium lap;
                          - we took all the laps that are <= medium lap
'''
'''Looking at our sample.
The sample is relatively small because we have pitstops data only for 841-988 raceId.
It would be good to have potstop data for previous races'''
try_races = races.copy()
try_races['date'] = try_races['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
try_races['date'].hist(bins = 73)
try_races[try_races['raceId'].isin(X_y['raceId'])]['date'].hist(bins = 10)
print('Whole sample:', len(try_races['date']))
print('The data that we have:', len(try_races[try_races['raceId'].isin(X_y['raceId'])]['date']))
# X_y.head()
'''Visualization of the correlation between parameters:'''
plt.figure(figsize=(10,10))
plt.matshow(X_y.corr(), cmap="Blues", fignum = 1)
plt.colorbar(shrink=0.8)
plt.xticks(range(len(list(X_y))), list(X_y), rotation=90, size = len(X_y.columns))
plt.yticks(range(len(list(X_y))), list(X_y), size = len(X_y.columns))

print('Correlatino matrix:')
plt.show()

'''If you want to see the representation in numbers:'''
# try_df.corr()

'''Note:
The variable we want to predict is "finish_position".
We can see that only the "position_first" and "position_last" have strong correlation with the "finish_position"
'''
import statsmodels.formula.api as sm

result = sm.ols(formula="finish_position ~ milliseconds_min + avg_milliseconds_min + ps_count +\
 + before_prev_ps_mean + before_prev_ps_min + before_prev_ps_max + position_first + position_last +\
 + ps_duration_sum", data=X_y).fit()

result.summary()
'''Creating additional parameters, which may increase the accuracye of the fixed effects model:'''
X_y_linearmodels = X_y.copy()
X_y_linearmodels['milliseconds_dif'] = X_y_linearmodels['milliseconds_min'] - X_y_linearmodels['avg_milliseconds_min']
X_y_linearmodels['laps_increase'] = X_y_linearmodels['position_last'] - X_y_linearmodels['position_first']

'''Splitting the dataset into the Training set and Test set'''

train, test = train_test_split(X_y_linearmodels.dropna(), 
                                train_size=0.8,
                                test_size=0.2,
                                random_state=123)
'''Preparing the dataframes to apply the fixed-effects'''
columns = ['milliseconds_min', 'ps_count',
           'before_prev_ps_mean', 'before_prev_ps_min', 'before_prev_ps_max',
           'position_first', 'position_last', 'ps_duration_sum', 'finish_position',
           'circuitId', 'avg_milliseconds_min', 'milliseconds_dif', 'laps_increase']

train = train.groupby(['driverId', 'raceId'])[columns].first()
test = test.groupby(['driverId', 'raceId'])[columns].first()
'''Fitting the model:'''
from linearmodels.panel import PanelOLS

'''
We use the fixed effects to controll for the driverId (different drivers have different abilities)
To use fixed effects - add "+ EntityEffects" in the equation below:
'''
equation = 'finish_position ~ 1 + milliseconds_min + avg_milliseconds_min + ps_count +\
 + before_prev_ps_mean + before_prev_ps_min + before_prev_ps_max + position_first + position_last +\
 + ps_duration_sum + EntityEffects'

mod = PanelOLS.from_formula(equation, train)
res = mod.fit(cov_type='clustered', cluster_entity=True)
# print(res)
'''To see the results of the model and coefficients (uncommend):'''
res
'''Predicting:'''
temp_columns = ['milliseconds_min', 'avg_milliseconds_min', 'ps_count', 'before_prev_ps_mean',
                'before_prev_ps_min', 'before_prev_ps_max', 'position_first', 'position_last',
                'ps_duration_sum', 'milliseconds_dif', 'laps_increase']
test_pred = res.predict(data= test[temp_columns])
# test['finish_position']
'''Calculating the accuracy:'''
difference_array = test_pred['predictions'].values - test['finish_position'].values

'''We assume that the prediction is currect if it within +-0.5 from the actual finish position'''
round_fun = lambda x: 1 if abs(x) <= 0.5 else 0
v_func = np.vectorize(round_fun)

round_accuracy_array = v_func(difference_array)
accuracy_score = dict(Counter(round_accuracy_array))
print('Accuracy score:', accuracy_score[1]/len(round_accuracy_array))
'''Notes:
The coef colum shows coefficients for each parameter.
For example, the coefficient next to the position_last = 0.6677, meaning that if the last position increases
by 1, the final position increases by 0.667.
The t and P>|t| columns indicates the statistical significance of each coefficient.

I would suggest to modify this model and use the fixed-effect to controll for the circuitId. Because
on different tracks drivers may use different pitstop strategy.
'''
'''Separating the X and y'''

X = X_y[['milliseconds_min', 'avg_milliseconds_min', 'ps_count',
       'before_prev_ps_mean', 'before_prev_ps_min', 'before_prev_ps_max',
       'position_first', 'position_last', 'ps_duration_sum']].values
y = X_y['finish_position'].values

'''Splitting the dataset into the Training set and Test set'''
x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

'''Feature Scaling'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
'''I do not think that LightGBM is good to answer our question: the sample size is not big enough,
the results are very sensitive to parameters we specify.'''
import lightgbm as lgb
d_train = lgb.Dataset(data = x_train, label=y_train)
params = {}
# params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mse' # mean squared error
# params['sub_feature'] = 0.5
# params['num_leaves'] = 10
# params['min_data'] = 50
# params['max_depth'] = 10
model = lgb.train(params, d_train)
lgb.plot_importance(model)
'''this is X:'''
X_y[['milliseconds_min', 'avg_milliseconds_min', 'ps_count',
       'before_prev_ps_mean', 'before_prev_ps_min', 'before_prev_ps_max',
       'position_first', 'position_last', 'ps_duration_sum']].head(3)
