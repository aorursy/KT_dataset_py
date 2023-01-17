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
#Load the data



import pandas as pd



data=pd.read_csv('../input/USA_Housing.csv')

data.head()
#last five row

data.tail()


#random five row

data.sample(5)
#information about data #datatype



data.info()
#column name



data.columns
# no of row n column



data.shape


#row count



data.shape[0]
#column count



data.shape[1]
#mathematical description of data



data.describe()
#to check any null value in data



data.isnull().values.sum()
# to check any column has nullvalue

data.isnull().any()
# to check any column has nullvalue

data.isnull().sum()
#data visualisation





import matplotlib.pyplot as plot

import seaborn as sns

df_corr=data.corr()

plot.figure(figsize=(5,2))

sns.heatmap(df_corr,cmap='viridis')

#normalization



new_data=data.drop(['Address'],axis=1)



def meannormalise(x):

    return(x-x.mean())/x.std()



new_data.apply(meannormalise)

corr_df=new_data.corr()

sns.heatmap(corr_df,cmap='viridis')

#add value heatmap



sns.heatmap(df_corr,annot=True,cmap='viridis')



#Price and  Avg. Area Income are highly corelated
#add line between row n colummn

sns.heatmap(df_corr,annot=True,cmap='viridis',linewidth=1)

sns.pairplot(data,kind='scatter')

# we can see price is linearly coorelated with 4 features .
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error ,r2_score

from sklearn.model_selection import train_test_split,KFold,cross_val_score

data.columns

x=data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

       'Avg. Area Number of Bedrooms', 'Area Population']]

y=data['Price']

x_train ,x_test ,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=5)

regr=LinearRegression()

regr.fit(x_train,y_train)

df_y_train_predict=regr.predict(x_train)

print("Mean square Error",mean_squared_error(y_train,df_y_train_predict))

print("R square variance",r2_score(y_train,df_y_train_predict))



# now test data set



df_y_test_predict=regr.predict(x_test)

print("Mean square Error",mean_squared_error(y_test,df_y_test_predict))

print("R square variance",r2_score(y_test,df_y_test_predict))



#adjusted r2



N = y_test.size

p = x_train.shape[1]

adjr2score = 1 - ((1-r2_score(y_test, df_y_test_predict))*(N - 1))/ (N - p - 1)

print("Adjusted R^2 Score %.2f" % adjr2score)