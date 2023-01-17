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
#importing required libraires

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#Read data file

df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.info()
#Data Preprocessing

df.describe(percentiles= [.5,.75,.9,.95,.99])
#finding impact of columns

sns.boxplot(df['price'])
df.head()
#generic prediction without model:

df['sqft_living'] - df['sqft_above'] + df['sqft_basement']
for col in df.columns:

    print('Unique Values for Column:',col, df[col].nunique())
# Data Preprocessing

# Remove column having unique values

df.drop(['id','zipcode'],axis=1, inplace = True)
df.drop(['yr_built','yr_renovated','lat','long'],axis=1, inplace = True)
from datetime import datetime 

df['date'] = df['date'].apply(lambda x: datetime.strptime(x[:8],'%Y%m%d'))
df.head()
# As we are not doing time series analysis I can exclude the date and sqft_living has nothing to do with date. 

# So we definitely can exclude this from model building.



df.drop('date',axis=1,inplace = True)
df['waterfront'].value_counts(normalize=True)
print(df['waterfront'].value_counts(normalize=True))

sns.boxplot(y='sqft_living',x='waterfront',data=df)
print(df['floors'].value_counts(normalize=True))

sns.boxplot(y='sqft_living',x='floors',data=df)
print(df['view'].value_counts(normalize=True))

sns.boxplot(y='sqft_living',x='view',data=df)
print(df['grade'].value_counts(normalize=True))

sns.boxplot(y='sqft_living',x='grade',data=df)
# Dealing with categorical data

cat_cols = ['floors','waterfront','view','condition','grade']



df = pd.get_dummies(df,columns=cat_cols,drop_first=True)
df.head()
print(df['bedrooms'].value_counts(normalize=True))

sns.boxplot(y='sqft_living',x='bedrooms',data=df)
print(df['bathrooms'].value_counts(normalize=True))

sns.boxplot(y='sqft_living',x='bathrooms',data=df)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import sklearn

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
# Train Test Split

df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 42)
# Scaling

scaler = MinMaxScaler()

all_cols = list(df.columns)

# scale Train data using min max scalar

df_train[all_cols] = scaler.fit_transform(df_train[all_cols])

y_train = df_train.pop('sqft_living')

X_train = df_train



# Scale Test Set

df_test[all_cols] = scaler.transform(df_test[all_cols])

y_test = df_test.pop('sqft_living')

X_test = df_test
X_train.head()
# Model Building



lr = LinearRegression()



# fit the model to the training data

lr.fit(X_train,y_train)





#Make Prediction Using the test set

y_hat = lr.predict(X_train)

mse = mean_squared_error(y_train, y_hat)

r_squared = r2_score(y_train, y_hat)



print('Model 1 Evaluation:')



print('\tTrain Mean_Squared_Error :' ,mse)

print('\tTrain R_square_value :',r_squared)



'''

Step 3: Predict and Evaluate the training MSE and R-Square

'''

y_hat_test = lr.predict(X_test)

mse = mean_squared_error(y_test, y_hat_test)

r_squared = r2_score(y_test, y_hat_test) 



print('\tTest Mean_Squared_Error :' ,round(mse,3))

print('\tTest R_square_value :',round(r_squared,3))
y_test.head()
y_hat_test