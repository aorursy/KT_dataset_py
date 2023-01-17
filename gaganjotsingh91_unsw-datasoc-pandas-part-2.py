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
'''

This tutorial will work on the Fifa Data Set. 

We will create a DataFrame names fifa and keep using it throughout the Tutorial

'''

fifa = pd.read_csv('../input/data.csv' , index_col=0)
#printing the first 5 rows

fifa.head()
#printing first 10 rows

fifa.head(10)
#getting information about the DataFrame

fifa.info()
#list of columns

fifa.columns
#iterating through all columns

for column in fifa.columns:

    print(column)
#iterating through rows

for index,row in fifa.iterrows():

    print(f'Player {row["Name"]} belongs to {row["Nationality"]}')

    if index> 5:

        break
#Accessing data for a single column

fifa['Name']
#alternate way

fifa.Name
name = fifa.Name

print(type(name))
fifa.iloc[0:2 ,0:10 ]
fifa.loc[0:2,['Name','Age','CB']]
#using the iloc operator on an attribute of DataFrame

fifa.Club.iloc[0:10]
fifa.Club =='Juventus'
fifa[fifa.Club =='Juventus']
fifa[(fifa.Club =='Juventus') & (fifa.Nationality =='Portugal')]
fifa.loc[fifa.LF.isnull()]
fifa[fifa.LF.isnull()]
fifa['goodPlayer'] = True
fifa.head(2)
fifa.goodPlayer = fifa.apply(lambda x: True if x.Potential >90 else False, axis='columns' )
fifa.loc[:20,['Potential','goodPlayer']]
fifa_copy = fifa

#fifa_copy.drop(['CM','CB'], axis=1)

#replacing values with Null Values by some other value

fifa.LF.fillna('80+5')
#replacing a value using apply function

#apply lambda function
fifa.Nationality.describe()
fifa[fifa.Nationality=='England']
fifa.CM.describe()
fifa.Potential.describe()
fifa.dtypes
#Selects Unique Nationalities of players belonging to Juventus

fifa[fifa.Club=='Juventus'].Nationality.unique()
#Selects the number of players per Country belonging to Juventus

fifa[fifa.Club=='Juventus'].Nationality.value_counts()
#Selects the number of players per Country belonging to Juventus

fifa[fifa.Club=='Juventus'].groupby(fifa.Nationality).Nationality.count()
#Lists the maximum Potential per Nationality

fifa.groupby(fifa.Nationality).Potential.max()
#apply Lambda Function per Nationality. the Lambda function simply sums up the Potential

fifa.groupby(fifa.Nationality).apply(lambda x:sum(x.Potential))
fifa.sort_values(by ="Nationality", ascending =True)
#placeholder for q1
#placeholder for q2
#placeholder for q3
#placeholder for q4
#placeholder for q5
#placeholder for q6
#placeholder for q7