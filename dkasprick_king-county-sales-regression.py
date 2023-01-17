# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, KFold 
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, metrics
from sklearn.metrics import mean_squared_error, r2_score
import xgboost
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/kc_house_data.csv")
data.head()

x=["date","price","bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated","zipcode","lat","long","sqft_living15","sqft_lot15"]
# Cleaning Data
df = pd.DataFrame(data, columns=x)
df['year'] = pd.DatetimeIndex(df['date']).year
df=df.drop(['date'],axis=1)
my_imputer = SimpleImputer()
df = pd.DataFrame(my_imputer.fit_transform(df))
df.columns=["price","bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated","zipcode","lat","long","sqft_living15","sqft_lot15","year"]
df.describe()
df['nprice'] = df['price']/1000
df.plot.scatter(x='nprice',y='grade',c='DarkBlue').set_xlabel('Price in Thousands')
df.plot.scatter(x='nprice',y='bedrooms',c='DarkBlue').set_xlabel('Price in Thousands')
df['bedrooms'].value_counts().sort_index().plot.bar()
df = df[df.bedrooms != 33]
df.plot.scatter(x='nprice',y='bedrooms',c='DarkBlue').set_xlabel('Price in Thousands')
sns.set(style="white")
corr=df.corr()
corr
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
y=df.price
df=df.drop(['price'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)
df['view'].value_counts()
df['floors'].value_counts()
df['bathrooms'].value_counts()
plt.scatter(df['bathrooms'],y)

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean Absolute Error',metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error',metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
residuals = y_test - y_pred 
plt.scatter(residuals,y_test)
from sklearn.tree import DecisionTreeRegressor
regr_1 = DecisionTreeRegressor(max_depth=df.shape[0]-1)
regr_1.fit(X_train, y_train)
y_1 = regr_1.predict(X_test)

print('Mean Absolute Error',metrics.mean_absolute_error(y_test, y_1))
print('Mean Squared Error',metrics.mean_squared_error(y_test, y_1))
print('Root Mean Squared Error',np.sqrt(metrics.mean_squared_error(y_test, y_1)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_1))
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7)
model=xgb.fit(X_train,y_train)
predicted = model.predict(X_test)
print('Mean Absolute Error',metrics.mean_absolute_error(y_test, predicted))
print('Mean Squared Error',metrics.mean_squared_error(y_test, predicted))
print('Root Mean Squared Error',np.sqrt(metrics.mean_squared_error(y_test, predicted)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, predicted))
scores = cross_val_score(model,df, y, cv=10)
scores.mean()
xgb1 = XGBRegressor()
parameters = {'nthread':[4],
              'objective':['reg:linear'],
              'learning_rate': [.07], 
              'max_depth': [5],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)
xgb_grid.fit(X_train,y_train)

predicted = xgb_grid.predict(X_test)
print('Mean Absolute Error',metrics.mean_absolute_error(y_test, predicted))
print('Mean Squared Error',metrics.mean_squared_error(y_test, predicted))
print('Root Mean Squared Error',np.sqrt(metrics.mean_squared_error(y_test, predicted)))
print('Variance score: %.2f' % r2_score(y_test, predicted))
scores = cross_val_score(xgb_grid,df, y, cv=10)
scores.mean()