import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

import random
df = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")
df.info()
df.describe()  
df.head()
num_col = df.select_dtypes(include=np.number).columns

print("Numerical columns: \n",num_col)



cat_col = df.select_dtypes(exclude=np.number).columns

print("Categorical columns: \n",cat_col)
# Remove the extra space from column names

df = df.rename(columns=lambda x: x.strip())
label_encoder = preprocessing.LabelEncoder() 

  

df['Status']= label_encoder.fit_transform(df['Status'])

  

df.head()
print(df.isna().sum())

print(df.shape)
# Replace using mean 

for i in df.columns.drop('Country'):

    df[i].fillna(df[i].mean(), inplace = True)
df.head()
print(df.isna().sum())
# Let's check the distribution of y variable (Life Expectancy)

plt.figure(figsize=(8,8), dpi= 80)

sns.boxplot(df['Life expectancy'])

plt.title('Life expectancy Box Plot')

plt.show()
plt.figure(figsize=(8,8))

plt.title('Life expectancy Distribution Plot')

sns.distplot(df['Life expectancy'])
num_col = df.select_dtypes(include=np.number).columns

print("Numerical columns: \n",num_col)



cat_col = df.select_dtypes(exclude=np.number).columns

print("Categorical columns: \n",cat_col)
# Let's check the multicollinearity of features by checking the correlation matric



plt.figure(figsize=(15,15))

p=sns.heatmap(df[num_col].corr(), annot=True,cmap='RdYlGn',center=0) 
# Pair Plots to know the relation between different features

ax = sns.pairplot(df[num_col])
# Train test split

X=df.drop(columns=['Life expectancy','Country'])

y=df[['Life expectancy']]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
model = LinearRegression()
model.fit(X, y)
r_sq = model.score(X, y)

print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)