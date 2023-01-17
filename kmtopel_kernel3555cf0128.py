#data analysis/linear algebra

import numpy as np

import pandas as pd



#machine learning

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import zscore



#data visualization

import seaborn as sns

import matplotlib.pyplot as plt



#display parameters

import ipywidgets as widgets

from IPython.display import display, Math

%matplotlib inline



# data files and filepaths

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2020/train.csv",low_memory=False)

print('The shape of the training data set is '+str(data.shape[0])+' rows by '+str(data.shape[1])+' columns.')
features = data.columns.drop('Yards')

target = ['Yards']
home = data[(data['GameId']==2017090700) & (data['Team']=='home')]

away = data[(data['GameId']==2017090700) & (data['Team']=='away')]

plt.scatter(x=home['X'],y=home['Y'],s=8)

plt.scatter(x=away['X'],y=away['Y'],s=8)

plt.axvline(10)

plt.text(x=10,y=0,s='end zone',rotation=90)

plt.text(x=110,y=0,s='end zone',rotation=90)

plt.text(x=60,y=0,s='50 Yardline',rotation=90)

plt.axvline(110)

plt.axvline(60)

plt.yticks([])

plt.xticks([])

print('The following plot shows the starting positions of every rushing player in one game:')

plt.show()
data.corr()['Yards'].round(2).abs().sort_values(ascending=False).drop('Yards')
defender_play_counts = {}

for i in range(1,12):

    num = data[data['DefendersInTheBox']==i].shape[0]

    defender_play_counts[i] = num

    

avg_yards_per_num_defenders = {}

for i,val in defender_play_counts.items():

    avg_yards_per_num_defenders[i]=data[data['DefendersInTheBox']==i]['Yards'].sum()/defender_play_counts[i]
plt.bar(x=avg_yards_per_num_defenders.keys(),height=avg_yards_per_num_defenders.values())

plt.xticks(range(1,12))

plt.xlabel('Number of Defenders in the Box')

plt.ylabel('Avg Yards Per Play')

plt.show()
plt.figure(figsize=(7,5))

plt.scatter(x=data['DefendersInTheBox'],y=data['Yards'])

plt.xticks(range(1,12))

plt.ylabel('Rushing Yards')

plt.xlabel('Number of Defenders in the Box')

plt.show()
data['yards_cats'] = pd.cut(data['Yards'],bins=5)

data['yards_cats'].value_counts()
train,test = train_test_split(data)
data.select_dtypes(np.number).columns
features_1 = ['X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir']

target = ['Yards']



scaler = MinMaxScaler()

scaled = scaler.fit_transform(train[features_1])

scaled_df = pd.DataFrame(scaled,columns=features_1)



def scale_data(x):

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(x)

    scaled_df = pd.DataFrame(scaled,columns=x.columns)

    

    return scaled_df,