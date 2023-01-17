# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
hd_df = pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")
hd_df[20:30]
hd_df.isna().sum()
hd_df.count()
hd_df['Indicator Category'].unique()
hd_df['Race/ Ethnicity'].unique()
hd_df['Race/ Ethnicity'].value_counts()
hd_df['Indicator'].unique()
ind = hd_df["Indicator"]

val = {""}

type(val)

for i in ind:

     if 'hiv' in i or 'HIV' in i or 'AIDS' in i or 'aids' in i:

         val.add(i)



val
ind = hd_df["Indicator"]

val = []

for i in ind:

    if 'hiv' in i or 'HIV' in i or 'AIDS':

        val.append(i)



val
hd_df['Year'].value_counts()
hd_df['Race/ Ethnicity'].value_counts()
hd_df['Place'].value_counts()
hd_df['BCHC Requested Methodology'].value_counts()
hd_df['Methods'].value_counts()
hd_df['Source'].value_counts()
hd_df['Notes'].value_counts()
hd_df.duplicated().sum()
hd_df.tail()
hd_df.shape
hd_df = hd_df.drop_duplicates()
hd_df.head()
hd_df.shape
hd_df.isna().sum()
hd_df.Value.mean()
hd_df.Value.fillna(285.7091791376513,inplace=True)
hd_df.isna().sum()
hd_df.drop(columns=['BCHC Requested Methodology','Source','Methods','Notes','Indicator'], inplace=True)
hd_df.info()
for val in hd_df['Year']:

    if '-' in val:

        temp = val[5:len(val)]

        hd_df['Year'].replace(val, temp, inplace=True)
hd_df['Year'].value_counts()
hd_df.head()

#this is the finalised table with meaning full data
num_col = hd_df.select_dtypes(include=np.number).columns

num_col

catColmns = hd_df.select_dtypes(exclude=np.number).columns

catColmns
encodedColmns = pd.get_dummies(hd_df[catColmns])

encodedColmns.head()
finalDF = pd.concat([hd_df[num_col],encodedColmns], axis = 1)

finalDF.head()
finalDF.corr()
yAxis = finalDF['Value']

xAxis = finalDF.drop(columns='Value')
train_x,test_x,train_y,test_y = train_test_split(xAxis,yAxis,test_size=0.3)
from sklearn.linear_model import LinearRegression

model= LinearRegression()
model.fit(train_x,train_y)
trainPredict = model.predict(train_x)

testPredict = model.predict(test_x)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
MAE_train = mean_absolute_error(train_y,trainPredict)

MAE_test = mean_absolute_error(test_y,testPredict)



MSE_train = mean_squared_error(train_y,trainPredict)

MSE_test = mean_squared_error(test_y,testPredict)



RMSE_train = np.sqrt(MSE_train)

RMSE_test = np.sqrt(MSE_test)



Mape_train = np.mean(np.abs((train_y,trainPredict)))

Mape_test = np.mean(np.abs((test_y,testPredict)))



R2_train = r2_score(train_y, trainPredict)



R2_test = r2_score(test_y, testPredict)
print("MAE of Trained data : ",MAE_train)

print("MAE of Test data    : ", MAE_test)



print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



print("MSE of Trained Data", MSE_train)

print("MSE of Test Data", MSE_test)



print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



print("RMSE of Trained Data", RMSE_train)

print("RMSE of Test Data", RMSE_test)



print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

print("Mape of train :, ",Mape_train)

print("Mape of test :, ",Mape_test)



print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

print("R2 of train: ", R2_train)

print("R2 of test: ", R2_test)