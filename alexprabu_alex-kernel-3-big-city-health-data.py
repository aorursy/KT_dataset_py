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
df = pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")
df.head()
df.tail()
df.columns
df.shape
df.describe().T
df["Indicator Category"].value_counts()
ind = df["Indicator"]

val = []

n= 0

for i in ind:

    if 'hiv' in i or 'HIV' in i or 'aids' in i or 'AIDS' in i:

        val.append(i)

        n+=1

n
df["Year"].value_counts()
df['Gender'].value_counts()
df['Race/ Ethnicity'].value_counts()
df['Place'].value_counts()
df['BCHC Requested Methodology'].value_counts()
df['Source'].value_counts()
df['Methods'].value_counts()
df['Notes'].value_counts()
df.duplicated().sum()
df.shape
df = df.drop_duplicates()
df.shape
df.isna().sum()
df['Value'].mean()
df['Value'].fillna(df['Value'].mean(),inplace=True)
df.isna().sum()
df.columns
df.drop(columns=['Indicator','BCHC Requested Methodology', 'Source', 'Methods', 'Notes'],inplace=True)
df.columns
df["Year"].value_counts()
for val in df['Year']:

    if '-' in val:

        temp = val[5:len(val)]

        df['Year'].replace(val, temp, inplace=True)
df["Year"].value_counts()
df.columns
cat_cols =  df.select_dtypes(exclude=np.number)
num_cols = df.select_dtypes(include=np.number)
encoded_cat_cols = pd.get_dummies(cat_cols)
preprocessed_df = pd.concat([encoded_cat_cols, num_cols], axis=1)
preprocessed_df.head()
x = preprocessed_df.drop(columns='Value')
y = preprocessed_df[['Value']]
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=12)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_x, train_y)
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
MAE_train = mean_absolute_error(train_y,train_predict)

MAE_test = mean_absolute_error(test_y,test_predict)



MSE_train = mean_squared_error(train_y,train_predict)

MSE_test = mean_squared_error(test_y,test_predict)



RMSE_train = np.sqrt(MSE_train)

RMSE_test = np.sqrt(MSE_test)



R2_train = r2_score(train_y, train_predict)

R2_test = r2_score(test_y, test_predict)
print("MAE of Trained data : ",MAE_train)

print("MAE of Test data : ", MAE_test)



print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



print("MSE of Trained Data", MSE_train)

print("MSE of Test Data", MSE_test)



print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



print("RMSE of Trained Data", RMSE_train)

print("RMSE of Test Data", RMSE_test)



print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

print("R2 of train: ", R2_train)

print("R2 of test: ", R2_test)