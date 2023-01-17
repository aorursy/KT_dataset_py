# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import OrdinalEncoder 

from sklearn.preprocessing import StandardScaler 

from scipy import stats as st

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_mov = pd.read_csv('/kaggle/input/movietweetings/movies.dat', delimiter='::', engine='python', header=None, names = ['Movie ID', 'Movie Title', 'Genre'])

data_us = pd.read_csv('/kaggle/input/movietweetings/users.dat', delimiter='::', engine='python', header=None, names = ['User ID', 'Twitter ID'])

data_rat = pd.read_csv('/kaggle/input/movietweetings/ratings.dat', delimiter='::', engine='python', header=None, names = ['User ID', 'Movie ID', 'Rating', 'Rating Timestamp'])
print(data_mov.info())

print(data_us.info())

print(data_rat.info())
data_mov = data_mov.dropna()

data_mov = data_mov.drop_duplicates().reset_index(drop=True)
print(data_mov.info())
data_mov.head()
year = data_mov['Movie Title'].str.split('(',expand=True)[1]

data_mov['Year'] = year.str.split(')',expand=True)[0]

data_mov['Year'] =data_mov['Year'].astype('int')

data_mov['Movie Title'] = data_mov['Movie Title'].str.split('(',expand=True)[0]
data_mov.head(10)
data_rat['Rating'].hist(color='red',alpha=0.3)
data.boxplot('Rating')
data_mov
name_len=[]

for i in range(len(data_mov['Movie Title'])):

     name_len.append(len(data_mov['Movie Title'][i]))

data_mov['name_length']  = name_len        
data_mov.pivot_table(index = 'Year',values = 

                     'name_length').reset_index().plot(grid=True,

                    x= 'Year',y = 'name_length',kind='scatter',

                    figsize=[12,3] , alpha= 0.4,color = 'red')
data = data_mov.merge(data_rat,on ='Movie ID',how='left').merge(data_us,on='User ID',how='left')

data['Rating'] = data['Rating']/2
data
data.pivot_table(index =['Year','Movie Title'],

values=['User ID','Rating'],aggfunc ={'User ID':'count',

        'Rating':'mean'}).reset_index().sort_values(['User ID','Rating'],ascending=False).head(20)
data.info()
data['Genre'] = data['Genre'].str.split('|')

genres = data['Genre'].str.join('|').str.get_dummies()

genres = genres.reset_index()

genres = genres.drop(['index'],axis=1)
data= pd.concat([data,genres],axis = 1)

data = data.drop('Genre',axis = 1)
target = data['Rating']

features = data.drop(['Rating','Movie Title'], axis=1)



target_train, target_valid, features_train, features_valid = train_test_split(target,features, test_size=0.25, random_state=42)
numeric = ['Movie ID', 'Year', 'name_length', 'User ID',

       'Rating Timestamp', 'Twitter ID', 'Action', 'Adult', 'Adventure',

       'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',

       'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',

       'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance',

       'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']

pd.options.mode.chained_assignment = None

# ... (загрузка и деление данных) ...

scaler = StandardScaler()

scaler.fit(features_train[numeric])

features_train[numeric]= scaler.transform(features_train[numeric])

features_valid[numeric]= scaler.transform(features_valid[numeric])
model = RandomForestRegressor(n_estimators=50, max_depth=19, random_state=12345)

model.fit(features_train, target_train)

predicted_valid = model.predict(features_valid)

print("R2 =", r2_score(target_valid,predicted_valid))