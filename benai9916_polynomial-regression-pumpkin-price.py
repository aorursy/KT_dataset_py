import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

import sklearn.pipeline as pp

# allow plot to appear directly in the notebook
%matplotlib inline

# 
pd.set_option('max_columns', 24)
pumpkin_nw = pd.read_csv('../input/a-year-of-pumpkin-prices/new-york_9-24-2016_9-30-2017.csv')
pumpkin_la = pd.read_csv('../input/a-year-of-pumpkin-prices/los-angeles_9-24-2016_9-30-2017.csv')

# checking the first 5 row of new_york data
pumpkin_la.head()
# checking the first 5 row of los_angeles data 
pumpkin_nw.head()
# check both the dataframe columns are same

pumpkin_la.columns.values in pumpkin_nw.columns.values
# print now of row and columsn
pumpkin_la.shape, pumpkin_nw.shape
# append both the data frame

pumpkin_la_nw = pumpkin_la.append(pumpkin_nw)
# check the new combine data frame
pumpkin_la_nw.shape
# checkin for missing values
pumpkin_la_nw.isnull().sum()
# check both the low and mostly low an high and mostly high price are same or not

cols = ['Low Price', 'High Price']
cols2 = ['Mostly Low', 'Mostly High']
def check_price(cols, cols2):
    for j, no in enumerate(cols):
        col1_value = pumpkin_la_nw[cols[j]].values
        col2_value = pumpkin_la_nw[cols2[j]].values

        if col1_value not in col2_value:

            print(col1_value ,'---------'  ,col2_value)

check_price(cols, cols2)
# keep only the below columns and drop all
 
new_cols_name = ['Item Size', 'price']
# take the average of the price
pumpkin_la_nw['price'] = pumpkin_la_nw['Low Price'] + pumpkin_la_nw['High Price'] / 2
pumpkin_la_nw.head()
# keep only the selected columns

pumpkin_la_nw = pumpkin_la_nw.drop([c for c in pumpkin_la_nw.columns if c not in new_cols_name], axis='columns')
pumpkin_la_nw.head()
# check missing values 
pumpkin_la_nw.isnull().sum()
# there are 24 missing values in 'item size' column
# will fill those missing value with the most common value
pumpkin_la_nw['Item Size'].value_counts()
# will fill those missing value with the most common value
pumpkin_la_nw['Item Size'].fillna('lge', inplace=True)
# verify if there is any missing vaues
pumpkin_la_nw.isnull().sum()
# transform all the categorical value to numeric value

label_enode = LabelEncoder()

pumpkin_la_nw.iloc[:, 0:-1] = pumpkin_la_nw.iloc[:, 0:-1].apply(LabelEncoder().fit_transform)
pumpkin_la_nw
# checking for outliers in price

sns.boxplot(pumpkin_la_nw['price'])
# splitting into dependent and independent variable
x = pumpkin_la_nw.iloc[:,0:-1]
y = pumpkin_la_nw.iloc[:, -1]
# set the degree of the polyniomial
y_poly = PolynomialFeatures(degree=4).fit_transform(x)

# add a const of
x2 = sm.add_constant(x)

# use old for prediction
pg_stats_model = sm.OLS(y, x2)

# fit the model
results = pg_stats_model.fit()

print(results.summary())
# train and test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# create an instance of logistic regression
logistic_model = LinearRegression()

# fit the data
logistic_model.fit(x_train, y_train)

# predict with the test set
y_predict = logistic_model.predict(x_test)

# predict with the training set
y_predict_train = logistic_model.predict(x_train)
print('The Accuracy  on the training dataset is: ', logistic_model.score(x_train, y_train) )
print('The Accuracy r2  on the training dataset is: ',r2_score(y_train,y_predict_train) )   

print("")
# Model Accuracy on testing dataset
print('The Accuracy  on the testing dataset is: ', logistic_model.score(x_test, y_test) )

print("")
# The Root Mean Squared Error (RMSE)
print('The RMSE  on the training dataset is: ', np.sqrt(mean_squared_error(y_train,y_predict_train)))
print('The RMSE  on the testing dataset is: ',np.sqrt(mean_squared_error(y_test,logistic_model.predict(x_test))))

print("")
# The Mean Absolute Error (MAE)
print('The MAE  on the training dataset is: ',mean_absolute_error(y_train,y_predict_train))
print('The MAE  on the testing dataset is: ',mean_absolute_error(y_test,logistic_model.predict(x_test)))


print("")
# Coefficients
print('Coefficients: ', logistic_model.coef_ )

print("")
# The Intercept
print('Intercept: ', logistic_model.intercept_)

plt.style.use('fivethirtyeight')

pd.Series((y_test - y_predict)).value_counts().sort_index().plot.bar(
    title='$y - \hat{y}$',
    figsize=(16, 5)
)


