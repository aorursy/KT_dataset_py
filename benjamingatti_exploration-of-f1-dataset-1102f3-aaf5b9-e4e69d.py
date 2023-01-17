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
# using the score function in each model class
print("accuracy on the test set", ps_tree.score(test_X, test_y))
print("accuracy on the training set", ps_tree.score(train_X, train_y))

# using single metric functions in the sklearn.metrics package 
print("accuracy on the test set", accuracy_score(pred_y, test_y))
'''Visualizing DecisionTree'''
show_decision_tree(ps_tree)
'''Count number of pist stops (1) and number of laps without pit stops (0)'''
from collections import Counter
y_list = y.tolist()
y_list = [item for sublist in y_list for item in sublist]
Counter(y_list)
