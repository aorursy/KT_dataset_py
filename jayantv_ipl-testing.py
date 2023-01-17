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
IPL=pd.read_csv("/kaggle/input/IPL Match Data.csv")



IPL.head()
print('DataType in Dataset')

print(IPL.dtypes)

print('Number of Columns containing Null Value')

print(IPL.isnull().any().sum(), ' / ', len(IPL.columns))

print('Number of rows containing null in either column')

print(IPL.isnull().any(axis=1).sum(), ' / ', len(IPL))

print('Checking colinearity with Winner')

#print(IPL.corr().abs().unstack().sort_values()['winner'])
#check if dataset contains null or not



print(IPL.isnull().values.any())

print(len(IPL))



UnqiueCountDF = pd.DataFrame(columns=['feature', 'UniqueValues'])

for column in IPL.columns:

    UnqiueCountDF = UnqiueCountDF.append({'feature': column, 'UniqueValues': len(IPL[column].unique())},ignore_index=True)



UnqiueCountDF
IPL.drop(columns = ['umpire3'], inplace = True) #Deleted umpire 3 column as it had all null values
## Fill null value in column "winner" to Draw

IPL['winner'].fillna('Draw',inplace=True)
IPL.columns
feature_cols = ['id', 'season', 'city', 'date', 'team1', 'team2', 'toss_winner',

       'toss_decision', 'result', 'dl_applied', 'winner', 'win_by_runs',

       'win_by_wickets', 'player_of_match', 'venue', 'umpire1', 'umpire2']



x=IPL[feature_cols]
x = pd.get_dummies(x, prefix='Category_', columns=['team1','team2','toss_winner','winner'])

print(x.columns)
