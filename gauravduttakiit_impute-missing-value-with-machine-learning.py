import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



movie=pd.read_csv(r'/kaggle/input/movie-for-forever/Movie_regression.csv')

movie.head()
movie.info()
movie.shape
movie.describe()
round((movie.isnull().sum() * 100 / len(movie)),2)


sns.distplot(movie['Time_taken'])

plt.show()
#Encode categorical data

dummy = pd.get_dummies(movie[["Genre","3D_available"]]).iloc[:,:-1]

movie = pd.concat([movie,dummy], axis=1)

movie = movie.drop(["Genre","3D_available"], axis=1)

movie.shape
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

it = IterativeImputer(estimator=LinearRegression())

newdata_lr = pd.DataFrame(it.fit_transform(movie))

newdata_lr.columns = movie.columns

newdata_lr.head()
import scipy.stats as stats

stats.ttest_ind(newdata_lr.Time_taken,movie.Time_taken,nan_policy='omit')
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

it = IterativeImputer(estimator=RandomForestRegressor(random_state=42))

newdata_rfr = pd.DataFrame(it.fit_transform(movie))

newdata_rfr.columns = movie.columns

newdata_rfr.head()
import scipy.stats as stats

stats.ttest_ind(newdata_rfr.Time_taken,movie.Time_taken,nan_policy='omit')
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn import tree

it = IterativeImputer(estimator=tree.DecisionTreeRegressor(random_state=42))

newdata_bg = pd.DataFrame(it.fit_transform(movie))

newdata_bg.columns = movie.columns

newdata_bg.head()
import scipy.stats as stats

stats.ttest_ind(newdata_bg.Time_taken,movie.Time_taken,nan_policy='omit')
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn import tree

it = IterativeImputer(estimator=AdaBoostRegressor(random_state=42))

newdata_abc = pd.DataFrame(it.fit_transform(movie))

newdata_abc.columns = movie.columns

newdata_abc.head()
import scipy.stats as stats

stats.ttest_ind(newdata_abc.Time_taken,movie.Time_taken,nan_policy='omit')
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import tree

it = IterativeImputer(estimator=GradientBoostingRegressor(random_state=42))

newdata_gbr = pd.DataFrame(it.fit_transform(movie))

newdata_gbr.columns = movie.columns

newdata_gbr.head()
import scipy.stats as stats

stats.ttest_ind(newdata_gbr.Time_taken,movie.Time_taken,nan_policy='omit')
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb

from sklearn import tree

it = IterativeImputer(estimator=xgb.XGBRegressor(random_state=42))

newdata_xgb = pd.DataFrame(it.fit_transform(movie))

newdata_xgb.columns = movie.columns

newdata_xgb.head()
import scipy.stats as stats

stats.ttest_ind(newdata_xgb.Time_taken,movie.Time_taken,nan_policy='omit')
# Compare with original v/s modified 
movie.Time_taken.describe()
newdata_lr.Time_taken.describe()
newdata_rfr.Time_taken.describe()
newdata_bg.Time_taken.describe()
newdata_abc.Time_taken.describe()
newdata_gbr.Time_taken.describe()
newdata_xgb.Time_taken.describe()
print('Linear Regression         :',stats.ttest_ind(newdata_lr.Time_taken,movie.Time_taken,nan_policy='omit'))

print('RandomForest Regressor    :',stats.ttest_ind(newdata_rfr.Time_taken,movie.Time_taken,nan_policy='omit'))

print('Bagging Regressor         :',stats.ttest_ind(newdata_bg.Time_taken,movie.Time_taken,nan_policy='omit'))

print('AdaBoost Regressor        :',stats.ttest_ind(newdata_abc.Time_taken,movie.Time_taken,nan_policy='omit'))

print('GradientBoosting Regressor:',stats.ttest_ind(newdata_gbr.Time_taken,movie.Time_taken,nan_policy='omit'))

print('XGB Regressor             :',stats.ttest_ind(newdata_xgb.Time_taken,movie.Time_taken,nan_policy='omit'))