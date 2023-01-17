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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
df = pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")
df.head()
df['Notes'].value_counts()
df['Year']
df.columns
cols_to_Encode = ['Gender','Race/ Ethnicity','Indicator Category']

continuous_cols = ['Value']
encoded_cols = pd.get_dummies(df[cols_to_Encode])
df_final = pd.concat([encoded_cols,df[continuous_cols]], axis = 1)
y = df_final['Value']

x = df_final.drop(columns = 'Value')
df_final.shape
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error
model = LinearRegression()
train_X, test_X, train_Y, test_Y = train_test_split(x,y,test_size = 0.3)

df_final.columns.dtype
df_column_category = df_final.select_dtypes(exclude=np.number).columns

df_column_category
df_final['Year'].value_counts()
df_final.isna().sum()
df_final['Value'].value_counts()
%matplotlib inline

df_final.Value.plot(kind="box")
df_final.Value.fillna(df.Value.median(),inplace = True)
model.fit(train_X,train_Y)
model.intercept_
model.coef_
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)
##MAE

print(mean_absolute_error(train_Y,train_predict))

##MAE

print(mean_absolute_error(test_Y,test_predict))
##MSE

print(mean_squared_error(train_Y,train_predict))

##MSE

print(mean_squared_error(test_Y,test_predict))