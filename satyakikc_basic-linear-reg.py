import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

%matplotlib inline
df=pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/car data.csv")
df.head()
df.shape
df.describe()
df.isnull().sum()
df.head()
sns.pairplot(df)#checking the relations amongst the variables
fig=plt.figure(figsize=(8,4))

sns.heatmap(df.corr(),annot=True)
sns.regplot('Selling_Price','Present_Price',data=df)
#from the above heatmap and regplot we can see that sellingprice and presentprice are linearly related.

#so we will form our model using these two parameters
sns.distplot(df.Year)
#we can see that most of the cars are from the year 2015
df.head()
sns.countplot(df.Fuel_Type)#checking fuel type
sns.countplot(df.Transmission)#checking transmission type
X=df.drop(["Car_Name","Year","Kms_Driven","Fuel_Type","Seller_Type","Transmission","Owner","Selling_Price"],axis="columns")

Y=df.Selling_Price
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
reg=LinearRegression()

reg.fit(X_train,y_train)
reg.score(X_train,y_train)#accuracy of our model is 78%
reg.predict([[8.59]])#predicting the selling price(testing data)