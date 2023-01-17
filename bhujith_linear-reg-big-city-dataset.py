# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

df=pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")

df.head(10)

# Any results you write to the current directory are saved as output.
df.columns
df.drop(columns=['BCHC Requested Methodology', 'Source', 'Methods',

       'Notes'],inplace=True)
df.columns
df['Race/ Ethnicity'].value_counts()
df['Race/ Ethnicity'].replace('Native American','Native American/Black',inplace=True)
#There is only 1 duplicate and dropping it

df.duplicated().sum()

df.drop_duplicates(inplace = True)
df.duplicated().sum()
df['Year'].value_counts()
df['Year'].replace(['2007-2012','2003-2012','2003-2013','2004-2013','2011-2013'],'NA',inplace=True)
df.drop(df[df['Year'] =='NA'].index, inplace = True)
df.columns

#No useful information can be derived.Hence deleting this columns
df.Place.value_counts()
df['State'] = df['Place'].str[-2:]

df.drop(columns='Indicator',inplace=True)
df.State.value_counts()
df.columns
df['Indicator Category'].value_counts()

df['Indicator Category'].replace(['Injury and Violence','Nutrition, Physical Activity, & Obesity','Infectious Disease','Maternal and Child Health'],'Non Life Threatening',inplace=True)
categorical_cols = df.select_dtypes(exclude =np.number).columns
categorical_cols
one_hot = pd.get_dummies(df[categorical_cols])
df_big = pd.concat([one_hot,df["Value"]],axis=1)
df_big.info()
df_big.corr()['Value']
df_big['Value'].isna().sum()

df_big['Value'].fillna((df_big['Value'].mean()), inplace=True)
y = df_big['Value']

x = df_big.drop(columns=['Value'])

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.3, random_state = 42)

model = LinearRegression()

model.fit(train_x,train_y)

train_predict = model.predict(train_x)

test_predict = model.predict(test_x)
print("MSE - Train :" ,mean_squared_error(train_y,train_predict))

print("MSE - Test :" ,mean_squared_error(test_y,test_predict))

print("MAE - Train :" ,mean_absolute_error(train_y,train_predict))

print("MAE - Test :" ,mean_absolute_error(test_y,test_predict))

print("R2 - Train :" ,r2_score(train_y,train_predict))

print("R2 - Test :" ,r2_score(test_y,test_predict))

print("Mape - Train:" , np.mean(np.abs((train_y,train_predict))))

print("Mape - Test:" ,np.mean(np.abs((test_y,test_predict))))