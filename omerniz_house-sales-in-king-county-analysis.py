# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectPercentile

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/kc_house_data.csv')
df = pd.get_dummies(df, 'zipcode')
df.head(5)
def plot_cats(cat):
    baths = df[cat].unique()
    baths.sort()
    total_baths = [df[df[cat] == b][cat].count() for b in baths]
    plt.bar(baths, total_baths)
plot_cats('grade')
plot_cats('yr_built')
df.corr()
df['yr_renovated'] = df.loc[df.yr_renovated > 2007 ,'yr_renovated'] = 1
df['yr_renovated'] = df.loc[df.yr_renovated <= 2007 ,'yr_renovated'] = 0
df = df.rename(columns = {"yr_renovated" : "is_renovated_in_last_10_years"})
df.to_csv('ready.csv')
# features_df = df[['sqft_living','bathrooms', 'sqft_living15', 'grade', 'bedrooms', 'floors', 'waterfront', \
#                   'view', 'sqft_above', 'sqft_basement', 'sqft_lot15', 'lat', 'is_renovated_in_last_10_years']]
features_df = df.drop('price', axis=1)
features_df = SelectPercentile(percentile = 75).fit(features_df,df.price).transform(features_df)
features_df = StandardScaler().fit(features_df).transform(features_df)
x_train, x_test, y_train, y_test = train_test_split(features_df, df.price)
linear_regr =  linear_model.ElasticNet(alpha=0.001, max_iter = 5000) # RandomForestRegressor(n_estimators = 75) #
model = linear_regr.fit(x_train, y_train)
predictions = model.predict(x_test)
plt.scatter(y_test,predictions)
plt.rcParams["figure.figsize"] =(15,12)
plt.show()
print("Mean squared error: %.3f"% mean_squared_error(y_test, predictions))
print("Mean absolute error: %.3f"% mean_absolute_error(y_test, predictions))
print('Variance score: %.3f' % r2_score(y_test, predictions))
approx = linear_model.LinearRegression().fit(pd.DataFrame(y_test),predictions).predict(pd.DataFrame(y_test))
plt.plot(y_test,approx)
plt.plot(np.arange(8000000), np.arange(8000000))
plt.show()
