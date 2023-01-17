import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,PolynomialFeatures

from sklearn.linear_model import LinearRegression

%matplotlib inline
file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'

df=pd.read_csv(file_name)
df.head()
df.dtypes
df.describe()
df.drop(['id', 'Unnamed: 0'], axis = 1, inplace = True)

df.describe()
df.bedrooms.isnull().sum()
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())

print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

mean=df['bedrooms'].mean()

df['bedrooms'].replace(np.nan,mean, inplace=True)
df.bedrooms.isnull().sum()
mean=df['bathrooms'].mean()

df['bathrooms'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())

print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
df.floors.value_counts().to_frame()
sns.boxplot(x=df["waterfront"], y=df["price"], data=df)
sns.regplot(x=df["sqft_above"],y=df["price"],data=df)

plt.ylim(0,)
df.corr()['price'].sort_values()
X = df[['long']]

Y = df['price']

lm = LinearRegression()

lm.fit(X,Y)

lm.score(X, Y)
x = df[['sqft_living']]

y = df.price

lr = LinearRegression()

lr.fit(x, y)

lr.score(x, y)
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     
X = df[features]

y = df.price

lr.fit(X, y)

lr.score(X, y)
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
X = df[features]

y = df.price

pipe = Pipeline(Input)

pipe.fit(X, y)

pipe.score(X, y)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

print("done")
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    

X = df[features]

Y = df['price']



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)





print("number of test samples:", x_test.shape[0])

print("number of training samples:",x_train.shape[0])
from sklearn.linear_model import Ridge
rm = Ridge(alpha=0.1)

rm.fit(x_train, y_train)

rm.score(x_test, y_test)
pr = PolynomialFeatures(degree = 2)

X_train_pr = pr.fit_transform(x_train)

X_test_pr = pr.fit_transform(x_test)



rr = Ridge(alpha = 0.1)

rr.fit(X_train_pr, y_train)

rr.score(X_test_pr, y_test)