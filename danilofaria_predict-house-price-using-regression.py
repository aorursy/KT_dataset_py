import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
%matplotlib inline 
ds = pd.read_csv("../input/kc_house_data.csv")
ds.head()
ds.describe()
ds.dtypes
ds = ds.drop(['id', 'date'], axis = 1)
f, ax = plt.subplots(figsize=(12,5))
sns.distplot(ds.price, ax = ax, fit = stats.gausshyper)
plt.show()
f, ax = plt.subplots(figsize=(12,5))
sns.boxplot(x = 'price', data = ds, ax=ax, showmeans=True, fliersize=3, orient="h", color = "silver")
plt.show()

print('Min: ' + str(ds['price'].min()))
print('1 Q: ' + str(np.percentile(ds['price'], 25)))
print('Median:' + str(ds.price.median()))
print('3 Q: ' + str(np.percentile(ds['price'], 75)))
print('Max: ' + str(ds['price'].max()))
# add a new variable to analyse if the house is renovated
ds['is_renovated'] = ds['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
sns.countplot(x = ds.is_renovated, data = ds)
print(ds['is_renovated'].value_counts())
# Continous and Categorical variables
# To biserial variables (i.e. is_renovated and waterfront) we could use stats.biserial (https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pointbiserialr.html)
ds.drop(['view','grade','floors','bedrooms','bathrooms','condition'], axis = 1).corr(method = 'pearson')
%%javascript
IPython.OutputArea.auto_scroll_threshold = 9999;
sns.jointplot(x = 'sqft_living', y = 'price', data = ds, kind = 'reg')
sns.jointplot(x = 'sqft_above', y = 'price', data = ds, kind = 'reg')
sns.jointplot(x = 'sqft_living15', y = 'price', data = ds, kind = 'reg')
sns.jointplot(x = 'sqft_basement', y = 'price', data = ds, kind = 'reg')
sns.jointplot(x = 'lat', y = 'price', data = ds, kind = 'reg')
plt.show()
#Price by waterfront
f, ax = plt.subplots(figsize=(12,5))
sns.boxplot(x="waterfront", y="price" , hue="waterfront", ax=ax, data=ds, dodge = False)
plt.show()
#Price by is_renovated
f, ax = plt.subplots(figsize=(12,5))
sns.boxplot(x="is_renovated", y="price" , hue="is_renovated", ax=ax, data=ds, dodge = False)
plt.show()
#Ordinal variables
ds[['price','view','grade','floors','bedrooms','bathrooms','condition']].corr(method = 'spearman')
#Price by grade
f, ax = plt.subplots(figsize=(12,5))
sns.boxplot(x="grade", y="price" , hue="grade", ax=ax, data=ds, dodge = False);
plt.show()
f, ax = plt.subplots(figsize=(12,5))
sns.countplot(x = ds.grade, data = ds)
plt.show()
ds.groupby(["grade"])["grade"].count()

ds[['grade', 'price']].groupby('grade')['price'].sum().map('{:,.2f}'.format)
#Price by view
f, ax = plt.subplots(figsize=(12,5))
sns.boxplot(x="view", y="price" , hue="view", ax=ax, data=ds, dodge = False);
plt.show()
#Price by bedrooms
f, ax = plt.subplots(figsize=(12,5))
sns.boxplot(x="bedrooms", y="price" , hue="bedrooms", ax=ax, data=ds, dodge = False);
plt.show()
#Price by bathrooms
f, ax = plt.subplots(figsize=(12,10))
sns.boxplot(x="bathrooms", y="price" , hue="bathrooms", ax=ax, data=ds, dodge = False)
plt.show()
#Define X and Y
x = ds.drop(['price'], axis = 1)
y = ds['price'].values

#Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

#create linear model
model_1 = linear_model.LinearRegression()

#train model
model_1_fit = model_1.fit(x_train, y_train)

#evaluating error
mean_squared_error(model_1_fit.predict(x_test), y_test)
#Define X and Y
x = ds.drop(['price'], axis = 1)
scaler = MinMaxScaler(feature_range=(0,1))
x = scaler.fit_transform(x)
y = ds['price'].values

#Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

#create Linear Model
model_2 = linear_model.LinearRegression()

#train model
model_2_fit = model_2.fit(x_train, y_train)

#evaluating error
mean_squared_error(model_2_fit.predict(x_test), y_test)
#Define x and y
x = ds.drop(['price'], axis = 1)
y = ds['price'].values

#Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

#train
rf = RandomForestRegressor()
rf.fit(x_train, y_train)

#evaluating error
mean_squared_error(rf.predict(x_test), y_test)
#Feature importance 
sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), ds.drop(['price'], axis = 1).columns), reverse=True)