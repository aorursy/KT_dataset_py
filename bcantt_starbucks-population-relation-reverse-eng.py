# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

world_pop = pd.read_csv("../input/country-wise-population-data/world_pop.csv")

stores = pd.read_csv('/kaggle/input/store-locations/directory.csv')

cities15000 = pd.read_csv("../input/cities-of-the-world/cities15000.csv", encoding='latin-1')

cost_of_living = pd.read_csv("../input/cost-of-living/cost-of-living.csv")

Cost_of_living_index = pd.read_csv("../input/cost-of-living-index-by-country/Cost_of_living_index.csv")

cities15000 = cities15000.rename(columns = {'name':'City'})
cities15000
stores
cities15000 = cities15000.fillna(0)
data_1 = stores.groupby('City').count().reset_index()

data_2 = cities15000[['population','City']]
data = data_1.merge(data_2,on = 'City')
data
data = data[['City','population','Brand']]
data = data.rename(columns = {'Brand':'Count'})
data['population'] = pd.to_numeric(data['population'] )

data['Count'] = pd.to_numeric(data['Count'] )
data = data.sort_values('population')
corr_list =[]

for i in range(100):

    corr_list.append(data.iloc[int(i*(data.shape[0]/100)):int((i+1)*(data.shape[0]/100))].corr().iloc[1][0])
import matplotlib.pyplot as plt

plt.plot(corr_list)

plt.xlabel('100 samples and their correlation rate')

plt.ylabel('Correlation ratio of population and starbucks store number')

plt.show()
cost_of_living
df = cost_of_living.T.drop(['Unnamed: 0'])
cost_of_living = cost_of_living.rename(columns={"Unnamed: 0": "food"})
def student_spending_weekly(data,place):

    spending = 0

    food_dic = {}

    k = -1

    x = ""

    for name in data.food:

        k += 1

        if k == 21:

            x = random.choice(['Apartment (1 bedroom) in City Centre','Apartment (1 bedroom) Outside of Centre','Apartment (3 bedrooms) in City Centre','Apartment (3 bedrooms) Outside of Centre'])

        if x == name:

            food_dic[x] = 1 

        if k > 51:

            food_dic[name] = 1 

        food_dic[name] = random.randint(0,10)

    for index,row in data.iterrows():

        spending += row[place] * food_dic[row['food']]

        

    return spending
student_spending_weekly(cost_of_living,'Helsinki, Finland')
cost_of_living.columns
set(data['City'].values) & set(cost_of_living.columns)
student_spending_weekly(cost_of_living,'Helsinki, Finland')
city_names = [name.split(",")[0] for name in cost_of_living.columns.drop('food')]
cost_of_living.columns =  ['food'] + city_names
for index, row in data.iterrows():

    if row['City'] in cost_of_living.columns:

        data.loc[index,'cost'] = student_spending_weekly(cost_of_living,row['City'])

        print(student_spending_weekly(cost_of_living,row['City']))

        print(row['City'])
for name in data.City.values:

    if name in cost_of_living.columns:

        print(name)
corr_list_cost = []

for i in range(100):

    corr_list_cost.append(data.iloc[int(i*(data.shape[0]/100)):int((i+1)*(data.shape[0]/100))].corr().iloc[2][0])
data.corr().iloc[2][0]




import matplotlib.pyplot

import pylab





matplotlib.pyplot.scatter(corr_list_cost,[i for i in range(len(corr_list_cost))])



matplotlib.pyplot.show()
import matplotlib.pyplot as plt

plt.plot(corr_list_cost)

plt.xlabel('100 samples and their correlation rate')

plt.ylabel('Correlation ratio of population and starbucks store number')

plt.show()
data
import math
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import linear_model

from sklearn import svm

from sklearn import tree

import xgboost as xgb

from sklearn.ensemble import BaggingRegressor

import numpy as np 

import pandas as pd 

import random
df = df.reset_index()
df = df.rename(columns = {'index':'City'})
df['City'] = [name.split(",")[0] for name in cost_of_living.columns.drop('food')]
data = data.merge(df)
Cost_of_living_index['City'] = [name.split(",")[0] for name in Cost_of_living_index['City'].values]
data = data.merge(Cost_of_living_index)
data = data.fillna(data.median())
X = data.drop(['City','Count'],axis = 1)

y = data['Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
regr = RandomForestRegressor()

regr.fit(X_train, y_train)



predictions = regr.predict(X_test)
mean_squared_error(predictions.round(), y_test)

from sklearn.metrics import r2_score

r2_score(predictions.round(), y_test,multioutput='variance_weighted')
results = pd.DataFrame(predictions.round(),y_test).reset_index()
results.corr()
from pandas import DataFrame

import seaborn as sns



corrMatrix = results.corr()

sns.heatmap(corrMatrix, annot=True)
results.plot()