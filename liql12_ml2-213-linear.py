

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score

%matplotlib inline



# get data

df=pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

data=df.iloc[:,2:]



# correlation analysis

k=10

plt.figure(figsize=(20,10))

internal_chars=[data.columns.tolist()]

corrmat=data.corr()   # calculate correlation coefficients

sns.heatmap(corrmat,square=False,linewidths=.5,annot=True)



print(corrmat['sqft_living'].sort_values(ascending=False)[0:11])  # sort and select top 10

final_data=pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv',usecols=['sqft_living','sqft_above','grade','sqft_living15','bathrooms','price','bedrooms','sqft_basement','floors','yr_built','view'])



# set and train model

feature_data=final_data.drop(['sqft_living'],axis=1)

target_data=final_data['sqft_living']



X_train,X_test,y_train,y_test=train_test_split(feature_data,target_data,test_size=0.2)  # divide train group and test group



LR=LinearRegression()

LR.fit(X_train,y_train)



# predict

y_predict=LR.predict(X_test)



# evaluate model

print(mean_squared_error(y_test, y_predict))