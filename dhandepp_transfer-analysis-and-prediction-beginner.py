# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/top250-00-19.csv')

data.info()
data.head()
def modify_season(row):

    row.Season = row.Season.split('-')[0]

    return row

data = data.apply(modify_season,axis = 'columns')
data.loc[(data.Season >= '2013') & (data.Season <='2018')].groupby('Team_to').Transfer_fee.agg('sum').sort_values(ascending = False).head(5)

#data.groupby('Team_to').apply(lambda l:l.Season == '2017-2018')



import seaborn as sns

data.loc[(data.Season >= '2013') & (data.Season <='2018')].groupby('Team_to').Transfer_fee.agg('sum').sort_values(ascending = False).head(5).plot.bar()
data.groupby('Position').Transfer_fee.sum().sort_values(ascending = False).head(5)

position_map = {'Right Winger': 'F','Centre-Forward':'F','Left Winger':'F','Centre-Back':'D','Central Midfield':

               'M','Attacking Midfield': 'M', 'Defensive Midfield': 'M','Second Striker': 'F', 'Goalkeeper': 'G',

               'Right-Back':'D','Left Midfield': 'M', 'Left-Back':'D','Right Midfield':'M','Forward':'F','Sweeper':'M',

               'Defender':'D','Midfielter':'M'}

data['New_position'] = pd.Series(data.Position.map(position_map), index = data.index)

data.head(10)



sns.lmplot(x='Transfer_fee',y='Age',hue='New_position',data = data,fit_reg = False)
nt = data.loc[(data.Age >=33) & (data.New_position == 'F')].Team_to.unique()

print(nt)

nl = data.loc[(data.Age >=33) & (data.New_position == 'F')].League_to.unique()

print(nl)
nt = data.loc[(data.Age >=35) & (data.New_position == 'M')].Team_to.unique()

print(nt)

nl = data.loc[(data.Age >=35) & (data.New_position == 'M')].League_to.unique()

print(nl)
nt = data.loc[(data.Age >=34) & (data.New_position == 'D')].Team_to.unique()

print(nt)

nl = data.loc[(data.Age >=34) & (data.New_position == 'D')].League_to.unique()

print(nl)
nt = data.loc[(data.Age >=34) & (data.New_position == 'G')].Team_to.unique()

print(nt)

nl = data.loc[(data.Age >=34) & (data.New_position == 'G')].League_to.unique()

print(nl)
def map_position(act,exp):

    if(act == exp):

        return 1

    else:

        return 0



series = data.New_position.map(lambda p: map_position(p,'F'))

data['F'] = pd.Series(series, index = data.index)



series = data.New_position.map(lambda p: map_position(p,'M'))

data['M'] = pd.Series(series, index = data.index)



series = data.New_position.map(lambda p: map_position(p,'D'))

data['D'] = pd.Series(series, index = data.index)



series = data.New_position.map(lambda p: map_position(p,'G'))

data['G'] = pd.Series(series, index = data.index)

data.head(10)
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression



features = ['F','M','D','G','Age','Season']

test_data1 = pd.DataFrame({'F':[1,0,0,0],'M':[0,0,0,1],'D':[0,0,1,0],

                           'G':[0,1,0,0],'Age': [29,39,21,26],'Season':[2013,2011,2018,2006]})

X = data[features]

y = data.Transfer_fee

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)



# Using Linear Regression model



rf_model = LinearRegression()



# train your model

rf_model.fit(train_X, train_y)



# predict for test data

predictions = rf_model.predict(test_X)



print(mean_absolute_error(test_y, predictions))

preds = rf_model.predict(test_data1)

print('Predictions for test_data1 ')

preds