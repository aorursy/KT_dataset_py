# Importing libraries

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd 

from mpl_toolkits.basemap import Basemap

from collections import OrderedDict

import xgboost as xgb

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from termcolor import colored

import warnings

warnings.filterwarnings('ignore')
# Importing the datasets

OD_2014 = pd.read_csv('../input/OD_2014.csv', low_memory=False)

OD_2015 = pd.read_csv('../input/OD_2015.csv', low_memory=False)

OD_2016 = pd.read_csv('../input/OD_2016.csv', low_memory=False)

OD_2017 = pd.read_csv('../input/OD_2017.csv', low_memory=False)

Stations_2014 = pd.read_csv('../input/Stations_2014.csv', low_memory=False)

Stations_2015 = pd.read_csv('../input/Stations_2015.csv', low_memory=False)

Stations_2016 = pd.read_csv('../input/Stations_2016.csv', low_memory=False)

Stations_2017 = pd.read_csv('../input/Stations_2017.csv', sep = ';', low_memory=False)
OD_2017.head()
Stations_2017.head()
f, axes = plt.subplots(1, 2, figsize=(10,5))

Used_number = [len(OD_2014), len(OD_2015), len(OD_2016), len(OD_2017)]

Station_number = [len(Stations_2014), len(Stations_2015), len(Stations_2016), len(Stations_2017)]

labels = ['2014','2015','2016','2017']

x = range(4)

plt.sca(axes[0])

plt.plot(Used_number, color = 'r', label = 'Movements number')

plt.title('Numbers of Movements Grows in These Years')

plt.xticks(x, labels)

plt.ylabel('Number')

plt.xlabel('Year', fontsize=8)

plt.legend(loc=2, prop={'size': 10})

plt.sca(axes[1])

plt.plot(Station_number, color = 'g', label = 'Station number')

plt.title('Numbers of Stations Grows in These Years')

plt.xticks(x, labels)

plt.ylabel('Number')

plt.xlabel('Year', fontsize=8)

plt.legend(loc=2, prop={'size': 10})

plt.show()
OD_2017 = OD_2017[OD_2017['end_station_code']!='Tabletop (RMA)']

OD_2017.end_station_code = OD_2017.end_station_code.astype(int)

StartUsedSorted = pd.DataFrame(OD_2017.groupby(by=['start_station_code'])['duration_sec'].sum())

StartUsedSorted.columns = ['Total duration seconds']

StartUsedSorted = StartUsedSorted.sort_values(by = 'Total duration seconds', ascending=False)

StartUsedSorted.head()
StartSorted = OD_2017.groupby(by=['start_station_code'])['start_date'].agg({'Count': np.size})

StartSorted['Count'] = StartSorted.Count.astype(int)

StartSorted = StartSorted.sort_values(by = 'Count', ascending=False)

StartSorted.head()
f, axes = plt.subplots(2, 1, figsize=(20,10))

plt.sca(axes[0])

TopStartUsedStation = np.array(StartUsedSorted.head(20).index)

TopStartUsedStationData = OD_2017[OD_2017['start_station_code'].isin(TopStartUsedStation)]

sns.lvplot(data=TopStartUsedStationData,x='start_station_code', y='duration_sec',order=TopStartUsedStation)

plt.title('The longest useage duration of start station in 2017(top20)', fontsize = 18)



plt.sca(axes[1])

TopStartUsed = np.array(StartSorted.head(20).index)

TopStartUsedData = OD_2017[OD_2017['start_station_code'].isin(TopStartUsed)]

sns.countplot(data= TopStartUsedData, x ='start_station_code',order=TopStartUsed)

plt.title('The most useage times of start station in 2017(top20)', fontsize = 18)

plt.show()
StartUsedSorted.tail()
StartSorted.tail()
f, axes = plt.subplots(2, 1, figsize=(20,10))

plt.sca(axes[0])

LowStartUsedStation = np.array(StartUsedSorted.tail(20).index)

LowStartUsedStationData = OD_2017[OD_2017['start_station_code'].isin(LowStartUsedStation)]

vis2 = sns.lvplot(data=LowStartUsedStationData,x='start_station_code', y='duration_sec', order=LowStartUsedStation)

plt.title('The shortest useage duration of start station in 2017(Bottom20)', fontsize = 18)



plt.sca(axes[1])

LowStartUsed = np.array(StartSorted.tail(20).index)

LowStartUsedData = OD_2017[OD_2017['start_station_code'].isin(LowStartUsed)]

sns.countplot(data= LowStartUsedData, x ='start_station_code',order=LowStartUsed)

plt.title('The Least usage times of start station in 2017(Bottom20)', fontsize = 18)

plt.show()
EndUsedSorted = pd.DataFrame(OD_2017.groupby(by=['end_station_code'])['duration_sec'].sum())

EndUsedSorted.columns = ['Total duration seconds']

EndUsedSorted = EndUsedSorted.sort_values(by = 'Total duration seconds', ascending=False)

EndUsedSorted.head()
EndSorted = OD_2017.groupby(by=['end_station_code'])['start_date'].agg({'Count': np.size})

EndSorted['Count'] = EndSorted.Count.astype(int)

EndSorted = EndSorted.sort_values(by = 'Count', ascending=False)

EndSorted.head()
f, axes = plt.subplots(2, 1, figsize=(20,10))

plt.sca(axes[0])

TopEndUsedStation = np.array(EndUsedSorted.head(20).index)

TopEndUsedStationData = OD_2017[OD_2017['end_station_code'].isin(TopEndUsedStation)]

sns.lvplot(data=TopEndUsedStationData,x='end_station_code', y='duration_sec',order=TopEndUsedStation)

plt.title('The longest useage duration of end station in 2017(top20)', fontsize = 18)



plt.sca(axes[1])

TopEndUsed = np.array(EndSorted.head(20).index)

TopEndUsedData = OD_2017[OD_2017['end_station_code'].isin(TopEndUsed)]

sns.countplot(data= TopEndUsedData, x ='end_station_code',order=TopEndUsed)

plt.title('The most useage times of end station in 2017(top20)', fontsize = 18)

plt.show()
EndUsedSorted.tail()
EndSorted.tail()
f, axes = plt.subplots(2, 1, figsize=(20,10))

plt.sca(axes[0])

LowEndUsedStation = np.array(EndUsedSorted.tail(20).index)

LowEndUsedStationData = OD_2017[OD_2017['end_station_code'].isin(LowEndUsedStation)]

sns.lvplot(data=LowEndUsedStationData,x='end_station_code', y='duration_sec',order=LowEndUsedStation)

plt.title('The Shortest useage duration of end station in 2017(Bottom20)', fontsize = 18)



plt.sca(axes[1])

LowEndUsed = np.array(EndSorted.tail(20).index)

LowEndUsedData = OD_2017[OD_2017['end_station_code'].isin(LowEndUsed)]

sns.countplot(data= LowEndUsedData, x ='end_station_code',order=LowEndUsed)

plt.title('The least useage times of end station in 2017(Bottom20)', fontsize = 18)

plt.show()
d = pd.DataFrame(0, index=np.arange(10000), columns=np.array(StartUsedSorted.head(20).index))

for colname in np.array(StartUsedSorted.head(20).index):

    sample = TopStartUsedStationData.groupby('start_station_code').get_group(colname).sample(n=10000)

    sample = sample.drop( ['Unnamed: 0', 'start_date', 'start_station_code', 'end_date', 'end_station_code', 'is_member'] ,axis=1)

    sample.columns = [colname]

    d[colname] = sample.values

f, ax = plt.subplots(figsize=(20, 20))

# Draw the heatmap using seaborn

sns.heatmap(d.astype(float).corr(),linewidths=0.25,vmax=1.0, 

            square=True, cmap="cubehelix_r", linecolor='k', annot=True)

plt.title('The Useage Duration Relations Between Stations in 2017', fontsize = 30, color='g')

plt.show()
StartUsedSorted['start_station_code'] = StartUsedSorted.index

Full2017 = StartUsedSorted.merge(Stations_2017, left_on = 'start_station_code', right_on='code' )

station_useage = Full2017['Total duration seconds'].values



plt.figure(figsize=(11,11))



colors_choice = ['yellow', 'red', 'lightblue', 'purple', 'green', 'orange']

size_limits = [320000, 1000000, 5000000, 10000000, 20000000, 40000000]

labels = []

size_choice = [1, 2, 3, 5, 8, 12]

for i in range(len(size_limits)-1):

    labels.append("{} <.< {}".format(size_limits[i], size_limits[i+1])) 

map = Basemap(projection='lcc', 

            lat_0=45.5,

            lon_0=-73.55,

            resolution='f',

            llcrnrlon=-73.7, llcrnrlat=45.4,

            urcrnrlon=-73.4, urcrnrlat=45.6)

map.drawcoastlines()

map.drawcountries()

map.drawmapboundary()

map.drawrivers()

for index, (code, y,x) in Full2017[['start_station_code', 'latitude', 'longitude']].iterrows():

    x, y = map(x, y)

    isize = [i for i, val in enumerate(size_limits) if val < station_useage[index]]

    ind = isize[-1]

    map.plot(x, y, marker='o', markersize = size_choice[ind], alpha=0.6, markeredgewidth = 1, color = colors_choice[ind], markeredgecolor='k', label = labels[ind])

handles, labels = plt.gca().get_legend_handles_labels()

by_label = OrderedDict(zip(labels, handles))

key_order = ('320000 <.< 1000000', '1000000 <.< 5000000', '5000000 <.< 10000000',

             '10000000 <.< 20000000', '20000000 <.< 40000000')

new_label = OrderedDict()

for key in key_order:

    new_label[key] = by_label[key]

plt.legend(new_label.values(), new_label.keys(), loc = 1, prop= {'size':11},

           title='Station Total Usage Duration(Seconds) of 2017', frameon = True, framealpha = 1)

plt.show()
StartSorted['start_station_code'] = StartUsedSorted.index

FullSort2017 = StartSorted.merge(Stations_2017, left_on = 'start_station_code', right_on='code' )

station_useage = FullSort2017['Count']



plt.figure(figsize=(11,11))



colors_choice = ['yellow', 'red', 'lightblue', 'purple', 'green', 'orange']

size_limits = [250, 1000, 6000, 10000, 20000, 48000]

labels = []

size_choice = [1, 2, 3, 5, 8, 12]

for i in range(len(size_limits)-1):

    labels.append("{} <.< {}".format(size_limits[i], size_limits[i+1])) 

map = Basemap(projection='lcc', 

            lat_0=45.5,

            lon_0=-73.55,

            resolution='f',

            llcrnrlon=-73.7, llcrnrlat=45.4,

            urcrnrlon=-73.4, urcrnrlat=45.6)

map.drawcoastlines()

map.drawcountries()

map.drawmapboundary()

map.drawrivers()

for index, (code, y,x) in FullSort2017[['start_station_code', 'latitude', 'longitude']].iterrows():

    x, y = map(x, y)

    isize = [i for i, val in enumerate(size_limits) if val < station_useage[index]]

    ind = isize[-1]

    map.plot(x, y, marker='o', markersize = size_choice[ind], alpha=0.6, markeredgewidth = 1, color = colors_choice[ind], markeredgecolor='k', label = labels[ind])

handles, labels = plt.gca().get_legend_handles_labels()

by_label = OrderedDict(zip(labels, handles))

key_order = ('250 <.< 1000', '1000 <.< 6000', '6000 <.< 10000',

             '10000 <.< 20000', '20000 <.< 48000')

new_label = OrderedDict()

for key in key_order:

    new_label[key] = by_label[key]

plt.legend(new_label.values(), new_label.keys(), loc = 1, prop= {'size':11},

           title='Station Total Usage Times of 2017', frameon = True, framealpha = 1)

plt.show()
# Import the independent variables and dependent variables

Full2017 = OD_2017.merge(Stations_2017,left_on='start_station_code', right_on='code')

X = Full2017[['start_station_code', 'end_station_code', 'duration_sec','is_public', 'latitude','longitude' ]]

y = Full2017['is_member']

# Splitting the dataset into the Training set and Validation set

Xt, Xv, yt, yv = train_test_split(X, y, test_size = 0.25, random_state = 0)

dt = xgb.DMatrix(Xt.as_matrix(),label=yt.as_matrix())

dv = xgb.DMatrix(Xv.as_matrix(),label=yv.as_matrix())

#Build the model

params = {

    "eta": 0.2,

    "max_depth": 4,

    "objective": "binary:logistic",

    "silent": 1,

    "base_score": np.mean(yt),

    'n_estimators': 1000,

    "eval_metric": "logloss"

}

model = xgb.train(params, dt, 100, [(dt, "train"),(dv, "valid")], verbose_eval=20)
#Prediction on validation set

y_pred = model.predict(dv)



# Making the Confusion Matrix

cm = confusion_matrix(yv, (y_pred>0.5))

print(colored('The Confusion Matrix is: ', 'red'),'\n', cm)

# Calculate the accuracy on test set

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

print(colored('The Accuracy on Test Set is: ', 'blue'), colored(predict_accuracy_on_test_set, 'blue'))
HisData = pd.concat([OD_2014,OD_2015,OD_2016])

StartUsedSorted = pd.DataFrame(HisData.groupby(by=['start_station_code'])['duration_sec'].sum())

StartUsedSorted.columns = ['Total duration seconds']

StartUsedSorted = StartUsedSorted.sort_values(by = 'Total duration seconds', ascending=False)

StartUsedSorted.head()
StartSorted = HisData.groupby(by=['start_station_code'])['start_date'].agg({'Count': np.size})

StartSorted['Count'] = StartSorted.Count.astype(int)

StartSorted = StartSorted.sort_values(by = 'Count', ascending=False)

StartSorted.head()
f, axes = plt.subplots(2, 1, figsize=(20,10))

plt.sca(axes[0])

TopStartUsedStation = np.array(StartUsedSorted.head(20).index)

TopStartUsedStationData = HisData[HisData['start_station_code'].isin(TopStartUsedStation)]

sns.lvplot(data=TopStartUsedStationData,x='start_station_code', y='duration_sec',order=TopStartUsedStation)

plt.title('The longest useage duration of start station in Past 3 Years(Top20)', fontsize = 18)

plt.sca(axes[1])

TopStartUsed = np.array(StartSorted.head(20).index)

TopStartUsedData = HisData[HisData['start_station_code'].isin(TopStartUsed)]

sns.countplot(data= TopStartUsedData, x ='start_station_code',order=TopStartUsed)

plt.title('The most useage times of start station in Past 3 Years(Top20)', fontsize = 18)

plt.show()
StartUsedSorted.tail()
StartSorted.tail()
f, axes = plt.subplots(2, 1, figsize=(20,10))

plt.sca(axes[0])

LowStartUsedStation = np.array(StartUsedSorted.tail(20).index)

LowStartUsedStationData = HisData[HisData['start_station_code'].isin(LowStartUsedStation)]

vis2 = sns.lvplot(data=LowStartUsedStationData,x='start_station_code', y='duration_sec', order=LowStartUsedStation)

plt.title('The shortest useage duration of start station in Past 3 Years(Bottom20)', fontsize = 18)

plt.sca(axes[1])

LowStartUsed = np.array(StartSorted.tail(20).index)

LowStartUsedData = HisData[HisData['start_station_code'].isin(LowStartUsed)]

sns.countplot(data= LowStartUsedData, x ='start_station_code',order=LowStartUsed)

plt.title('The Least usage times of start station in Past 3 Years(Bottom20)', fontsize = 18)

plt.show()
EndUsedSorted = pd.DataFrame(HisData.groupby(by=['end_station_code'])['duration_sec'].sum())

EndUsedSorted.columns = ['Total duration seconds']

EndUsedSorted = EndUsedSorted.sort_values(by = 'Total duration seconds', ascending=False)

EndUsedSorted.head()
EndSorted = HisData.groupby(by=['end_station_code'])['start_date'].agg({'Count': np.size})

EndSorted['Count'] = EndSorted.Count.astype(int)

EndSorted = EndSorted.sort_values(by = 'Count', ascending=False)

EndSorted.head()
f, axes = plt.subplots(2, 1, figsize=(20,10))

plt.sca(axes[0])

TopEndUsedStation = np.array(EndUsedSorted.head(20).index)

TopEndUsedStationData = HisData[HisData['end_station_code'].isin(TopEndUsedStation)]

sns.lvplot(data=TopEndUsedStationData,x='end_station_code', y='duration_sec',order=TopEndUsedStation)

plt.title('The longest useage duration of end station in Past 3 Years(Top20)', fontsize = 18)

plt.sca(axes[1])

TopEndUsed = np.array(EndSorted.head(20).index)

TopEndUsedData = HisData[HisData['end_station_code'].isin(TopEndUsed)]

sns.countplot(data= TopEndUsedData, x ='end_station_code',order=TopEndUsed)

plt.title('The most useage times of end station in Past 3 Years(Top20)', fontsize = 18)

plt.show()
EndUsedSorted.tail()
EndSorted.tail()
f, axes = plt.subplots(2, 1, figsize=(20,10))

plt.sca(axes[0])

LowEndUsedStation = np.array(EndUsedSorted.tail(20).index)

LowEndUsedStationData = HisData[HisData['end_station_code'].isin(LowEndUsedStation)]

sns.lvplot(data=LowEndUsedStationData,x='end_station_code', y='duration_sec',order=LowEndUsedStation)

plt.title('The Shortest useage duration of end station in Past 3 Years(Bottom20)', fontsize = 18)

plt.sca(axes[1])

LowEndUsed = np.array(EndSorted.tail(20).index)

LowEndUsedData = HisData[HisData['end_station_code'].isin(LowEndUsed)]

sns.countplot(data= LowEndUsedData, x ='end_station_code',order=LowEndUsed)

plt.title('The least useage times of end station in Past 3 Years(Bottom20)', fontsize = 18)

plt.show()
d = pd.DataFrame(0, index=np.arange(10000), columns=np.array(StartUsedSorted.head(20).index))

for colname in np.array(StartUsedSorted.head(20).index):

    sample = TopStartUsedStationData.groupby('start_station_code').get_group(colname).sample(n=10000)

    sample = sample.drop( ['Unnamed: 0', 'start_date', 'start_station_code', 'end_date', 'end_station_code', 'is_member'] ,axis=1)

    sample.columns = [colname]

    d[colname] = sample.values

f, ax = plt.subplots(figsize=(20, 20))

# Draw the heatmap using seaborn

sns.heatmap(d.astype(float).corr(),linewidths=0.25,vmax=1.0, 

            square=True, cmap="cubehelix_r", linecolor='k', annot=True)

plt.title('The Relations Useage Duration Between Stations in Past 3 Years', fontsize = 30, color='g')

plt.show()
StartUsedSorted['start_station_code'] = StartUsedSorted.index

Full2017 = StartUsedSorted.merge(Stations_2016, left_on = 'start_station_code', right_on='code' )

station_useage = Full2017['Total duration seconds'].values



plt.figure(figsize=(11,11))



colors_choice = ['yellow', 'red', 'lightblue', 'purple', 'green', 'orange']

size_limits = [710000, 1200000, 10000000, 40000000, 70000000, 92000000]

labels = []

size_choice = [1, 2, 3, 5, 8, 12]

for i in range(len(size_limits)-1):

    labels.append("{} <.< {}".format(size_limits[i], size_limits[i+1])) 

map = Basemap(projection='lcc', 

            lat_0=45.5,

            lon_0=-73.55,

            resolution='f',

            llcrnrlon=-73.7, llcrnrlat=45.4,

            urcrnrlon=-73.4, urcrnrlat=45.6)

map.drawcoastlines()

map.drawcountries()

map.drawmapboundary()

map.drawrivers()

for index, (code, y,x) in Full2017[['start_station_code', 'latitude', 'longitude']].iterrows():

    x, y = map(x, y)

    isize = [i for i, val in enumerate(size_limits) if val < station_useage[index]]

    ind = isize[-1]

    map.plot(x, y, marker='o', markersize = size_choice[ind], alpha=0.6, markeredgewidth = 1, color = colors_choice[ind], markeredgecolor='k', label = labels[ind])

handles, labels = plt.gca().get_legend_handles_labels()

by_label = OrderedDict(zip(labels, handles))

key_order = ('710000 <.< 1200000', '1200000 <.< 10000000', '10000000 <.< 40000000',

             '40000000 <.< 70000000', '70000000 <.< 92000000')

new_label = OrderedDict()

for key in key_order:

    new_label[key] = by_label[key]

plt.legend(new_label.values(), new_label.keys(), loc = 1, prop= {'size':11},

           title='Station Total Usage Duration(Seconds) of Past 3 Years', frameon = True, framealpha = 1)

plt.show()
StartSorted['start_station_code'] = StartUsedSorted.index

FullSort2017 = StartSorted.merge(Stations_2016, left_on = 'start_station_code', right_on='code' )

station_useage = FullSort2017['Count']



plt.figure(figsize=(11,11))



colors_choice = ['yellow', 'red', 'lightblue', 'purple', 'green', 'orange']

size_limits = [680, 5000, 30000, 60000, 100000, 130000]

labels = []

size_choice = [1, 2, 3, 5, 8, 12]

for i in range(len(size_limits)-1):

    labels.append("{} <.< {}".format(size_limits[i], size_limits[i+1])) 

map = Basemap(projection='lcc', 

            lat_0=45.5,

            lon_0=-73.55,

            resolution='f',

            llcrnrlon=-73.7, llcrnrlat=45.4,

            urcrnrlon=-73.4, urcrnrlat=45.6)

map.drawcoastlines()

map.drawcountries()

map.drawmapboundary()

map.drawrivers()

for index, (code, y,x) in FullSort2017[['start_station_code', 'latitude', 'longitude']].iterrows():

    x, y = map(x, y)

    isize = [i for i, val in enumerate(size_limits) if val < station_useage[index]]

    ind = isize[-1]

    map.plot(x, y, marker='o', markersize = size_choice[ind], alpha=0.6, markeredgewidth = 1, color = colors_choice[ind], markeredgecolor='k', label = labels[ind])

handles, labels = plt.gca().get_legend_handles_labels()

by_label = OrderedDict(zip(labels, handles))

key_order = ('680 <.< 5000', '5000 <.< 30000', '30000 <.< 60000',

             '60000 <.< 100000', '100000 <.< 130000')

new_label = OrderedDict()

for key in key_order:

    new_label[key] = by_label[key]

plt.legend(new_label.values(), new_label.keys(), loc = 1, prop= {'size':11},

           title='Station Total Usage Times of Past 3 Years', frameon = True, framealpha = 1)

plt.show()
FullHis = HisData.merge(Stations_2016,left_on='start_station_code', right_on='code')

X = FullHis[['start_station_code', 'end_station_code', 'duration_sec', 'latitude','longitude' ]]

y = FullHis['is_member']

# Splitting the dataset into the Training set and Validation set

Xt, Xv, yt, yv = train_test_split(X, y, test_size = 0.25, random_state = 0)

dt = xgb.DMatrix(Xt.as_matrix(),label=yt.as_matrix())

dv = xgb.DMatrix(Xv.as_matrix(),label=yv.as_matrix())

#Build the model

params = {

    "eta": 0.2,

    "max_depth": 4,

    "objective": "binary:logistic",

    "silent": 1,

    "base_score": np.mean(yt),

    'n_estimators': 200,

    "eval_metric": "logloss"

}

model = xgb.train(params, dt, 25, [(dt, "train"),(dv, "valid")], verbose_eval=5)
#Prediction on validation set

y_pred = model.predict(dv)



# Making the Confusion Matrix

cm = confusion_matrix(yv, (y_pred>0.5))

print(colored('The Confusion Matrix is: ', 'red'),'\n', cm)

# Calculate the accuracy on test set

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

print(colored('The Accuracy on Test Set is: ', 'blue'), colored(predict_accuracy_on_test_set, 'blue'))