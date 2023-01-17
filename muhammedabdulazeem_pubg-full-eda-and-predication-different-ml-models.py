# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

pd.set_option('display.max_columns', 500)

from scipy import stats

from scipy.stats import norm,skew
train = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
#test = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
train.shape #,test.shape
train.isnull().sum()
train.dropna(inplace=True)
train.head()
train['winPlacePerc'].value_counts()
plt.figure(figsize=(15,8))

sns.distplot(train['winPlacePerc'],fit=norm)
train['assists'].value_counts()
plt.figure(figsize=(15,8))

sns.countplot(train['kills'].sort_values())
plt.figure(figsize=(15,8))

sns.countplot(train['assists'].sort_values())
for i in range(0,60):

    percentage = train[train['kills'] == i].shape[0]/len(train['kills']) * 100

    print("Number of players having",i,"kills in their matches:",percentage)
plt.figure(figsize=(15,8))

sns.countplot(train['boosts'].sort_values())
plt.figure(figsize=(15,8))

sns.countplot(train['assists'].sort_values())
plt.figure(figsize=(10,8))

sns.jointplot(x=train['winPlacePerc'],y=train['kills'])
plt.figure(figsize=(10,8))

sns.jointplot(x=train['headshotKills'],y=train['kills'])
plt.figure(figsize=(10,8))

sns.jointplot(x=train['killPlace'],y=train['kills'])
plt.figure(figsize=(10,8))

sns.jointplot(x=train['killPoints'],y=train['kills'])
plt.figure(figsize=(10,8))

sns.countplot(train['killStreaks'])
plt.figure(figsize=(10,8))

sns.distplot(train['walkDistance'],fit=norm)
work_dist_zero = train[train['walkDistance'] <= 0].shape[0]

print('Number of people how walked zero distance that is they are killed or they exit the game before even stepping few steps',work_dist_zero)
avg_work = np.average(train['walkDistance'])

print('Average distance worked by the player is',avg_work)
plt.figure(figsize=(10,8))

sns.jointplot(x=train['winPlacePerc'],y=train['walkDistance'])
plt.figure(figsize=(10,8))

sns.jointplot(x=train['winPlacePerc'],y=train['rideDistance'])
avg_drive = np.average(train['rideDistance'])

print('Average distance worked by the player is',avg_drive)
ride_dist_zero = train[train['rideDistance'] <= 0].shape[0]

print("Players how don't drive:",ride_dist_zero)
plt.figure(figsize=(10,8))

sns.pointplot(x=train['vehicleDestroys'],y=train['winPlacePerc'])
data = train.copy()
data = data.groupby('groupId')['kills'].sum().reset_index()
data = data.sort_values(by='kills',ascending=False)
data.head(50)
plt.figure(figsize=(15,8))

sns.jointplot(x=train['kills'],y=train['headshotKills'])
plt.figure(figsize=(10,8))

sns.jointplot(x=train['winPlacePerc'],y=train['heals'])
plt.figure(figsize=(10,8))

sns.jointplot(x=train['winPlacePerc'],y=train['revives'])
solos = train[train['numGroups']>50]

duos = train[(train['numGroups']>25) & (train['numGroups']<=50)]

squads = train[train['numGroups']<=25]
print('they were around',(solos.shape[0]/train.shape[0]) * 100,'% of the solo games')

print('they were around',(duos.shape[0]/train.shape[0]) * 100,'% of the duos games')

print('they were around',(squads.shape[0]/train.shape[0]) * 100,'% of the squads games')
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='kills',y='winPlacePerc',data=solos,color='black',alpha=0.8)

sns.pointplot(x='kills',y='winPlacePerc',data=duos,color='#CC0000',alpha=0.8)

sns.pointplot(x='kills',y='winPlacePerc',data=squads,color='#3399FF',alpha=0.8)

plt.text(37,0.6,'Solos',color='black',fontsize = 17,style = 'italic')

plt.text(37,0.55,'Duos',color='#CC0000',fontsize = 17,style = 'italic')

plt.text(37,0.5,'Squads',color='#3399FF',fontsize = 17,style = 'italic')

plt.xlabel('Number of kills',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.title('Solo vs Duo vs Squad Kills',fontsize = 20,color='blue')

plt.grid()

plt.show()
f,ax = plt.subplots(figsize=(15, 15))

train_matrix = train.corr()



sns.heatmap(train_matrix,annot=True)
def heatmap(x, y, size):

    fig, ax = plt.subplots()

    

    # Mapping from column names to integer coordinates

    x_labels = [v for v in sorted(x.unique())]

    y_labels = [v for v in sorted(y.unique())]

    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 

    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 

    

    size_scale = 500

    ax.scatter(

        x=x.map(x_to_num), # Use mapping for x

        y=y.map(y_to_num), # Use mapping for y

        s=size * size_scale, # Vector of square sizes, proportional to size parameter

        marker='s' # Use square as scatterplot marker

    )

    

    # Show column labels on the axes

    ax.set_xticks([x_to_num[v] for v in x_labels])

    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')

    ax.set_yticks([y_to_num[v] for v in y_labels])

    ax.set_yticklabels(y_labels)
train_matrix['winPlacePerc'].sort_values(ascending=False)
columns = ['walkDistance','boosts','weaponsAcquired','damageDealt','heals','kills','longestKill']



corr = train[columns].corr()

corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y

corr.columns = ['x', 'y', 'value']

heatmap(

    x=corr['x'],

    y=corr['y'],

    size=corr['value'].abs()

)
train.columns
train.drop(['Id','groupId','matchId'],axis=1,inplace=True)
train.head()
train['matchType'].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['matchType'] = le.fit_transform(train['matchType'])
train['matchType'].astype('int32')
from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.ensemble import RandomForestRegressor

import xgboost

from xgboost import XGBRegressor
features = train.drop('winPlacePerc',axis=1)

labels = train['winPlacePerc']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.1,random_state=42,shuffle=True)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)

X_test_scaled = ss.transform(X_test)
X_train_scaled
X_test_scaled
np.max(X_test_scaled),np.max(X_train_scaled)
lr = LinearRegression()
lr.fit(X_train_scaled,y_train)
ypred_lr = lr.predict(X_test_scaled)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(ypred_lr,y_test)
xgb = XGBRegressor()
lasso = Lasso()
lasso.fit(X_train_scaled,y_train)
ypred_lasso = lasso.predict(X_test_scaled)
mean_absolute_error(ypred_lasso,y_test)