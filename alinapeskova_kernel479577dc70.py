# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import ast

import re

import datetime



data = pd.read_csv('main_task.csv')



df = data.drop(['Restaurant_id', 'City', 'Cuisine Style', 'Price Range',

               'Reviews', 'URL_TA', 'ID_TA'], axis=1)



df.loc[df['Number of Reviews'].isnull(), 'Number of Reviews'] = data['Number of Reviews'].fillna(value=data['Number of Reviews'].mean()) 



def columnsFilling(inDataFrame, outDataFrame, columnName, string): 

    fixedColumn = inDataFrame[columnName].fillna(string)

    numbers = []

    for text in fixedColumn:

        wordNum = len(text.split(','))

        numbers.append(wordNum)

    outDataFrame[columnName] = pd.DataFrame(numbers, columns = [columnName])





def replaceCostWithNumbers(inDataFrame, outDataFrame):   

    fixedColumn = data['Price Range'].fillna('0')

    numbers = []

    for price in fixedColumn:

        if price == '$':

            numbers.append(3)

        elif price == '$$ - $$$':

            numbers.append(2)

        elif price == '$$$$':

            numbers.append(1)

        else:

            numbers.append(2)

        

    outDataFrame['Price Range'] = pd.DataFrame(numbers, columns = ['Price Range'])



columnsFilling(data, df, 'Cuisine Style', 'European')    



replaceCostWithNumbers(data, df)



City_dummy = pd.get_dummies(data['City'])

df = pd.concat([df, City_dummy], axis=1)





maxRankingDict = dict(data.City.value_counts())

for key in maxRankingDict:

    maxRankingDict[key] = 0



for index, row in data.iterrows():

    maxRankingDict[row.City] = max(maxRankingDict[row.City], row.Ranking)



rankingColumn = []

for index, row in data.iterrows():

    ranking = row.Ranking

    rankingMax = maxRankingDict[row.City]

    rankingColumn.append(1.0 - ranking / rankingMax)

Ranking = pd.DataFrame(rankingColumn, columns = ['Ranking'])



df['Ranking'] = Ranking





X = df.drop(['Rating'], axis = 1)

y = df['Rating']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)



from sklearn.ensemble import RandomForestRegressor



from sklearn import metrics



regr = RandomForestRegressor(n_estimators=100)



regr.fit(X_train, y_train)



data_new = pd.read_csv('kaggle_task.csv')



df_new = data_new.drop(['Restaurant_id', 'Name', 'City', 'Cuisine Style', 'Price Range',

               'Reviews', 'URL_TA', 'ID_TA'], axis=1)



df_new.loc[df_new['Number of Reviews'].isnull(), 'Number of Reviews'] = data_new['Number of Reviews'].fillna(value=data_new['Number of Reviews'].mean()) 



columnsFilling(data_new, df_new, 'Cuisine Style', 'European')    



replaceCostWithNumbers(data_new, df_new)



City_dummy = pd.get_dummies(data_new['City'])

df_new = pd.concat([df_new, City_dummy], axis=1)





maxRankingDict = dict(data.City.value_counts())

for key in maxRankingDict:

    maxRankingDict[key] = 0



for index, row in data_new.iterrows():

    maxRankingDict[row.City] = max(maxRankingDict[row.City], row.Ranking)



rankingColumn = []

for index, row in data_new.iterrows():

    ranking = row.Ranking

    rankingMax = maxRankingDict[row.City]

    rankingColumn.append(1.0 - ranking / rankingMax)

Ranking = pd.DataFrame(rankingColumn, columns = ['Ranking'])



df_new['Ranking'] = Ranking



y_pred = regr.predict(df_new)



submission = pd.DataFrame(data_new['Restaurant_id'], columns = ['Restaurant_id'])

Rating = pd.DataFrame(y_pred, columns = ['Rating'])

submission['Rating'] = Rating

submission.to_csv('solution.csv', index = False)
