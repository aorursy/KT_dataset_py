# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/autompg/auto-mpg.csv')
df.head()
df.describe()
df.hist(bins=20, figsize=(15,15))
plt.show()
from pandas.plotting import scatter_matrix

attribs=['horsepower', 'model year', 'mpg']
scatter_matrix(df[attribs], figsize=(12,12))
plt.show()
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

sbs = train_set.copy()
corr_matrix = sbs.corr()
corr_matrix['mpg'].sort_values(ascending=False)
sbs = train_set.drop('mpg', axis=1)
sbs_labels = train_set['mpg'].copy()
cat_attribs = ['car name']
sbs_num = sbs.drop(labels=cat_attribs, axis=1)
sbs_cat = sbs[cat_attribs]
# Let's see some of the categorical data...
sbs_num.head()
#Adapt all features to numeric processing, but... we don't need to so nevermind
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder() #This thing is pretty cool not gonna lie
#sbs_cat_1hot = encoder.fit_transform(sbs_cat)
#I think I'm gonna not mess with this, as I'm not using the categorical data
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Note that we set the values of do_temp_hour and do_temp_per_humidity
# to indicate whether or not we want to include these extra features
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler()),
    ])
# Deal with missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
imputer.fit(sbs_num)
X = imputer.transform(sbs_num)

sbs_tr = pd.DataFrame(X, columns=sbs_num.columns, index=sbs_num.index)
sbs_tr.info()
from sklearn.linear_model import LinearRegression

lr_clf = LinearRegression()
lr_clf.fit(sbs_tr, sbs_labels)
some_data = sbs_tr.iloc[:5]
some_labels = sbs_labels.iloc[:5]
#some_data_prepared = full_pipeline.transform(some_data) #Already prepped
print('Predictions:', np.round(lr_clf.predict(some_data)))
print('Labels:     ', list(some_labels))
from sklearn.metrics import mean_squared_error

sbs_predictions = lr_clf.predict(sbs_tr)
lr_mse = mean_squared_error(sbs_labels, sbs_predictions)
lr_rmse = np.sqrt(lr_mse)
print(f'Root mean squared error = {lr_rmse}')
from sklearn.tree import DecisionTreeRegressor

dt_clf = DecisionTreeRegressor()
dt_clf.fit(sbs_tr, sbs_labels)
print('Predictions:', np.round(dt_clf.predict(some_data)))
print('Labels:     ', list(some_labels))
sbs_predictions = dt_clf.predict(sbs_tr)
dt_mse = mean_squared_error(sbs_labels, sbs_predictions)
dt_rmse = np.sqrt(dt_mse)
print(f'Root mean squared error = {dt_rmse}')
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt_clf, sbs_tr, sbs_labels, scoring='neg_mean_squared_error', cv=10)
dt_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print('Scores;', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())

display_scores(dt_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

rf_clf = RandomForestRegressor()
rf_clf.fit(sbs_tr, sbs_labels)
print('Predictions:', np.round(rf_clf.predict(some_data)))
print('Labels:     ', list(some_labels))
sbs_predictions = rf_clf.predict(sbs_tr)
rf_mse = mean_squared_error(sbs_labels, sbs_predictions)
rf_rmse = np.sqrt(rf_mse)
print(f'Root mean squared error = {rf_rmse}')
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_clf, sbs_tr, sbs_labels, scoring='neg_mean_squared_error', cv=10)
rf_rmse_scores = np.sqrt(-scores)
display_scores(rf_rmse_scores)