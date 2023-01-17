import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data=pd.read_csv('../input/Video_Games_Sales_as_at_30_Nov_2016.csv')

#data.head(10)



#Critic_score - Aggregate score compiled by Metacritic staff

#Critic_count - The number of critics oused in coming up with the Critic_score

#User_score - Score by Metacritic's subscribers

#User_count - Number of users who gave the user_score

#Developer - Party responsible for creating the game

#Rating - The ESRB ratings
# First Assignment. Predict the Genre from the other data

colNames=list(data.columns.values)

colNames.remove('Name')

colNames.remove('Genre')



# We will impute data points

from sklearn.preprocessing import Imputer

data.head(100)



# Numerical Columns

numCol=['User_Count','User_Score','Critic_Count','Critic_Score','Global_Sales','Other_Sales','JP_Sales','NA_Sales','EU_Sales','Year_of_Release']

catCol=['Name','Platform','Genre','Publisher','Developer','Rating']



def checkForIncorrectData(x):

    try:

        float(x)

        return x

    except ValueError:

        return 0



def colwise(colName,data):

    data[colName]=data[colName].apply(checkForIncorrectData)



[colwise(x,data) for x in numCol]



# Imputing the values

from sklearn.preprocessing import Imputer

numImputer=Imputer(strategy='mean',axis=0,verbose=1,copy=False)

results=numImputer.fit_transform(data[numCol])
data[numCol]