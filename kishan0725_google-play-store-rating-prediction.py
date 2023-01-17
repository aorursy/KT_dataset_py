import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
data.head(10)
# Removing irrelevant features

data = data.drop(['App','Last Updated','Current Ver','Android Ver'],axis='columns')
data.head(10)
# checking for null values

data.isna().sum()
# drop the entire record if null value is present in 'any' of the feature

data.dropna(how='any',inplace=True)
data.shape
data.isna().sum()
data.dtypes
# changing the datatype of Review column from integer from object

data = data.astype({'Reviews':'int'})
data.Size.value_counts().head()
data.Size.value_counts().tail()
# Replacing 'Varies with device' value with Nan values

data['Size'].replace('Varies with device', np.nan, inplace = True ) 
# Removing the suffixes (k and M) and representing all the data as bytes 

# (i.e)for k, value is multiplied by 100 and for M, the value is multiplied by 1000000 

data.Size = (data.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \

             data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1)

            .replace(['k','M'], [10**3, 10**6]).astype(int))
# filling "Varies with device" with mean of size in each category

data['Size'].fillna(data.groupby('Category')['Size'].transform('mean'),inplace = True)
# Removing comma(,) and plus(+) signs

data.Installs = data.Installs.apply(lambda x: x.replace(',',''))

data.Installs = data.Installs.apply(lambda x: x.replace('+',''))
# changing the datatype from object to integer

data = data.astype({'Installs':'int'})
data.Price.value_counts()
# Removing dollar($) sign and changing the type to float

data.Price = data.Price.apply(lambda x: x.replace('$',''))

data['Price'] = data['Price'].apply(lambda x: float(x))
data.Genres.value_counts().tail()
data['Genres'] = data.Genres.str.split(';').str[0]
data.Genres.value_counts()
# Group Music & Audio as Music

data['Genres'].replace('Music & Audio', 'Music',inplace = True)
data['Content Rating'].value_counts()
# Removing the entire row from the data where content rating is unrated as there is only one row

data = data[data['Content Rating'] != 'Unrated']
data.dtypes
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.compose import make_column_transformer

from sklearn.preprocessing import StandardScaler

column_trans = make_column_transformer(

                (OneHotEncoder(),['Category','Installs','Type','Content Rating','Genres']),

                (StandardScaler(),['Reviews','Size','Price']),

                remainder = 'passthrough')
# Choosing X and y value

X = data.drop('Rating',axis='columns')

y = data.Rating
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
column_trans.fit_transform(X_train)
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline

linreg = LinearRegression()

pipe = make_pipeline(column_trans,linreg)
from sklearn.model_selection import cross_validate

linreg_score = cross_validate(pipe, X_train, y_train, cv=10, scoring=['neg_mean_squared_error','neg_mean_absolute_error'],return_train_score=False)

print('Mean Absolute Error: {}'.format(linreg_score['test_neg_mean_absolute_error'].mean()))

print('Mean Squared Error: {}'.format(linreg_score['test_neg_mean_squared_error'].mean()))

print('Root Mean Squared Error: {}'.format(np.sqrt(-linreg_score['test_neg_mean_squared_error'].mean())))
from sklearn.svm import SVR

svr = SVR()

pipe = make_pipeline(column_trans,svr)

svr_score = cross_validate(pipe, X_train, y_train, cv=10, scoring=['neg_mean_squared_error','neg_mean_absolute_error'],return_train_score=False)

print('Mean Absolute Error: {}'.format(svr_score['test_neg_mean_absolute_error'].mean()))

print('Mean Squared Error: {}'.format(svr_score['test_neg_mean_squared_error'].mean()))

print('Root Mean Squared Error: {}'.format(np.sqrt(-svr_score['test_neg_mean_squared_error'].mean())))
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(n_estimators=100, max_features=3, min_samples_leaf=10)

pipe = make_pipeline(column_trans,forest_model)

rfr_score = cross_validate(pipe, X_train, y_train, cv=10, scoring=['neg_mean_squared_error','neg_mean_absolute_error'],return_train_score=False)

print('Mean Absolute Error: {}'.format(rfr_score['test_neg_mean_absolute_error'].mean()))

print('Mean Squared Error: {}'.format(rfr_score['test_neg_mean_squared_error'].mean()))

print('Root Mean Squared Error: {}'.format(np.sqrt(-rfr_score['test_neg_mean_squared_error'].mean())))
pipe = make_pipeline(column_trans,linreg)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error

print('Mean Absolute Error: {}'.format(mean_absolute_error(y_pred,y_test)))

print('Mean Squared Error: {}'.format(mean_squared_error(y_pred,y_test)))

print('Root Mean Squared Error: {}'.format(np.sqrt(mean_absolute_error(y_pred,y_test))))
pipe = make_pipeline(column_trans,svr)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print('Mean Absolute Error: {}'.format(mean_absolute_error(y_pred,y_test)))

print('Mean Squared Error: {}'.format(mean_squared_error(y_pred,y_test)))

print('Root Mean Squared Error: {}'.format(np.sqrt(mean_absolute_error(y_pred,y_test))))
pipe = make_pipeline(column_trans,forest_model)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print('Mean Absolute Error: {}'.format(mean_absolute_error(y_pred,y_test)))

print('Mean Squared Error: {}'.format(mean_squared_error(y_pred,y_test)))

print('Root Mean Squared Error: {}'.format(np.sqrt(mean_absolute_error(y_pred,y_test))))