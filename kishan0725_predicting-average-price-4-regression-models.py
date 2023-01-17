import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
# import our dataset

data = pd.read_csv('../input/avocado-prices/avocado.csv')
# first 10 observations of our dataset

data.head(10)
# renaming column names into meaningful names (refer kaggle's avacado dataset description)

data = data.rename(columns={'4046':'PLU_4046','4225':'PLU_4225','4770':'PLU_4770'})
# removing unnecessary column

data = data.drop(['Unnamed: 0'],axis = 1)

data.head(10)
# convert the type of Date feature from obj to datetime type

data['Date'] = pd.to_datetime(data['Date'])
# categorizing into several seasons

def season_of_date(date):

    year = str(date.year)

    seasons = {'spring': pd.date_range(start='21/03/'+year, end='20/06/'+year),

               'summer': pd.date_range(start='21/06/'+year, end='22/09/'+year),

               'autumn': pd.date_range(start='23/09/'+year, end='20/12/'+year)}

    if date in seasons['spring']:

        return 'spring'

    if date in seasons['summer']:

        return 'summer'

    if date in seasons['autumn']:

        return 'autumn'

    else:

        return 'winter'
# creating a new feature 'season' and assign the corresponding season for the Date using map function over our season_of_date function

data['season'] = data.Date.map(season_of_date)
# now, we can see the season feature appended at the last

data.head(10)
# no of observations for each seasons

data.season.value_counts()
# droping date feature

data = data.drop(['Date'],axis = 1)
# converting categorical features of text data into model-understandable numerical data

label_cols = ['type','region','season']

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

data[label_cols] = data[label_cols].apply(lambda x : label.fit_transform(x))
# Scaling the features and 

# spliting the label encoded features into distinct features inorder to prevent our model to think that columns have data with some kind of order or hierarchy

# column_tranformer allows us to combine several feature extraction or transformation methods into a single transformer

from sklearn.compose import make_column_transformer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

scale_cols = data.drop(['AveragePrice','type','year','region','season'],axis=1)

col_trans = make_column_transformer(

            (OneHotEncoder(), data[label_cols].columns),

            (StandardScaler(), scale_cols.columns),

            remainder = 'passthrough')
# splitting our dataset into train and test set such that 20% of observations are considered as test set

X = data.drop(['AveragePrice'],axis=1)

y = data.AveragePrice

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

pipe = make_pipeline(col_trans,linreg)

pipe.fit(X_train,y_train)
y_pred_test = pipe.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error

print('MAE for testing set: {}'.format(mean_absolute_error(y_pred_test,y_test)))

print('MSE for testing set: {}'.format(mean_squared_error(y_pred_test,y_test)))

print('RMSE for testing set: {}'.format(np.sqrt(mean_squared_error(y_pred_test,y_test))))
from sklearn.svm import SVR

svr = SVR()

pipe = make_pipeline(col_trans,svr)

pipe.fit(X_train,y_train)
y_pred_test = pipe.predict(X_test)
print('MAE for testing set: {}'.format(mean_absolute_error(y_pred_test,y_test)))

print('MSE for testing set: {}'.format(mean_squared_error(y_pred_test,y_test)))

print('RMSE for testing set: {}'.format(np.sqrt(mean_squared_error(y_pred_test,y_test))))
from sklearn.tree import DecisionTreeRegressor

dr=DecisionTreeRegressor()

pipe = make_pipeline(col_trans,dr)

pipe.fit(X_train,y_train)
y_pred_test = pipe.predict(X_test)
print('MAE for testing set: {}'.format(mean_absolute_error(y_pred_test,y_test)))

print('MSE for testing set: {}'.format(mean_squared_error(y_pred_test,y_test)))

print('RMSE for testing set: {}'.format(np.sqrt(mean_squared_error(y_pred_test,y_test))))
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor()

pipe = make_pipeline(col_trans,forest_model)

pipe.fit(X_train,y_train)
y_pred_test = pipe.predict(X_test)
print('MAE for testing set: {}'.format(mean_absolute_error(y_pred_test,y_test)))

print('MSE for testing set: {}'.format(mean_squared_error(y_pred_test,y_test)))

print('RMSE for testing set: {}'.format(np.sqrt(mean_squared_error(y_pred_test,y_test))))
sns.distplot((y_test-y_pred_test),bins=50)