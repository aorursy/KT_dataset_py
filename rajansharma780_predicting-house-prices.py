import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

import os

print(os.listdir("../input"))
#load the csv file and make the data frame

house_df = pd.read_csv('../input/kc_house_data.csv')
#display the first 5 rows of dataframe 

house_df.head()
print("the dataframe has {} rows and {} columns".format(house_df.shape[0],house_df.shape[1]))
#check null values are there or not

house_df.apply(lambda x : sum(x.isnull()))
#info of dataframe

house_df.info()
#5 point summary of numeric columns

house_df.describe()
#multivariate plot

sns.pairplot(data=house_df,diag_kind='kde')

plt.show()
#removing Date column as it is not useful in linear regression model

house_df.drop('date',axis=1,inplace=True)
for i in house_df.columns:

    print("the r and p value for "+i+" and price respectively is {}".format(stats.pearsonr(house_df[i],house_df['price'])))
plt.figure(figsize=(15,10))

sns.heatmap(house_df.corr(),annot=True)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

X = house_df[['sqft_living']]#taking only sqft_living column as an explanatory variable or independent variable 

Y = house_df[['price']]#taking price column as an response variable or dependent variable or target variable

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.30,random_state=1)#splitting the dataset into training and test dataset in(70:30 ratio)
lp = LinearRegression()#instantiate the object

lp.fit(X_train,Y_train)
lp.score(X_test,Y_test)
print("Intercept::{}".format(lp.intercept_))

print("slope::{}".format(lp.coef_))
plt.figure(figsize=(10,5))

plt.scatter(X,Y,color='darkblue',label="Data", alpha=.1)

plt.plot(X,(273.80839877*X-31356.78457486),label="predicted regression line")

plt.xlabel("sqft_living")

plt.ylabel("price")

plt.legend()

plt.show()
X1 = house_df[['sqft_living','bathrooms','grade','sqft_above','sqft_living15']]#taking column whose r_value>0.5 as an explanatory variable or independent variable 

Y1 = house_df[['price']]#taking price column as an response variable or dependent variable or target variable

X1_train,X1_test,Y1_train,Y1_test = train_test_split(X1,Y1,test_size = 0.30,random_state=1)#splitting the dataset into training and test dataset in(70:30 ratio)
lp1 = LinearRegression()#instantiate the object

lp1.fit(X1_train,Y1_train)
lp1.score(X1_test,Y1_test)
print("Intercept::{}".format(lp1.intercept_))

print("slope::{}".format(lp1.coef_))