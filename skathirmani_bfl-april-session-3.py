# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
store_data = pd.read_csv('/kaggle/input/trainings/sales_store_data.csv')

sales_data = pd.read_csv('/kaggle/input/trainings/sales_data.csv')



print(store_data.shape)

print(sales_data.shape)
sales_data.head()
sales_data.describe()
sales_data.info()
sales_data.head()
sales_data['Store'].nunique()
sales_data['Store'].unique()
sales_data['DayOfWeek'].unique()
sales_data['DayOfWeek'].value_counts()
# pd.Series([1, 2, None, 4, 5, 6, 9, 4,4,4]).unique()
store_data['StoreType'].value_counts()
sales_data.head()
sales_data['Sales'].mean(), sales_data['Sales'].median()
sales_data['Sales'].quantile([0.25, 0.5, 0.75])
sales_data['Sales'].plot.hist()
## Filtering & Date columns

sales_data.head()
## Filter for store id 1

store_id = 1

store_ids = [1, 2, 3]

sales_subset = sales_data[sales_data['Store'] == store_id]

sales_subset.head()
store_ids = [1, 2, 3]

sales_subset = sales_data[sales_data['Store'].isin(store_ids)]
sales_subset = sales_data[(sales_data['Store'] == 1) & (sales_data['DayOfWeek'] == 1)]

sales_subset.shape
store_id = 1

sales_data.head()
sales_data['Date'] = pd.to_datetime(sales_data['Date'], format='%Y-%m-%d')

sales_data.dtypes
store_id = 1

sales_subset = sales_data[sales_data['Store'] == store_id]

sales_subset.head()
sales_subset['Date'].min(), sales_subset['Date'].max()
sales_subset.resample(on='Date', rule='1m')['Sales'].mean().plot.line(figsize=(20,5))

## Get all the values displayed in x axis
### Read the sales and store data

store_data = pd.read_csv('/kaggle/input/trainings/sales_store_data.csv')

sales_data = pd.read_csv('/kaggle/input/trainings/sales_data.csv')
sales_data.head()
#Using sales data, idendify avg sales across all the store when they are closed (when the Open column is having value zero)

store_closed_data = sales_data[sales_data['Open'] == 0]

store_closed_data['Sales'].mean()
## Identify store wise avg sales and plot top 10 stores avg sales using bar chart (use groupby)

avg_sales_stores = sales_data.groupby(['Store']).agg({'Sales': 'mean', 'Customers': 'mean'}).rename(columns={'Sales': 'Avg Sales'})

#avg_sales_stores.sort_values(by='Avg Sales', ascending=False).head(10)['Avg Sales'].plot.bar()
# Filter and groupby

# sales_data[sales_data['Open'] == 1].groupby(['Store']).agg({'Sales': 'mean'})
sales_data['day'] = sales_data['Date'].dt.day

sales_data['month'] = sales_data['Date'].dt.month

sales_data['year'] = sales_data['Date'].dt.year

sales_data[['Date', 'day', 'month', 'year']].head()
sales_data['is_december'] = sales_data['month'].apply(lambda v: 1 if v == 12 else 0)

sales_data[['month', 'is_december']].drop_duplicates()
store_data.isna().sum()
insurance = pd.read_csv('/kaggle/input/trainings/insurance.csv')

insurance.head()
insurance.describe().columns
import seaborn as sns
sns.pairplot(insurance)
np.log(insurance['expenses']).plot.hist()
insurance.shape
target_col = 'expenses'
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
insurance.isna().sum()
df = pd.DataFrame({

    'x1': [10, 20, None],

    'x2': [20, None, 40],

    'x3': ['a', 'b', None]

})

df
df.dropna()
# Column wise percentage of missing values

df.isna().sum() / df.shape[0] * 100
insurance.isna().sum() / insurance.shape[0] * 100
store_data.isna().sum() / store_data.shape[0] * 100
insurance.head()
pd.get_dummies(insurance[['region']], drop_first=True).drop_duplicates()
insurance_dummies = pd.get_dummies(insurance, drop_first=True)

insurance.shape, insurance_dummies.shape
insurance[['sex']]
store_data = pd.read_csv('/kaggle/input/trainings/sales_store_data.csv')

sales_data = pd.read_csv('/kaggle/input/trainings/sales_data.csv')

## Identify percentage of missing values in sales_data for each column

## Identify no.of columns in store_data after converting it to dummy variables

#sales_data.isna().sum() / sales_data.shape[0] * 100
print(store_data.shape)

print(pd.get_dummies(store_data, drop_first=True).shape)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
import seaborn as sns

sns.pairplot(insurance, hue='smoker')
target_col = 'expenses'

input_cols = insurance_dummies.columns.drop(['expenses'])

train_x, test_x, train_y, test_y = train_test_split(insurance_dummies[input_cols],

                                                    insurance_dummies[target_col],

                                                   test_size=0.2,

                                                    random_state=1

                                                   )

train_x.shape, test_x.shape, train_y.shape, test_y.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train_x) # identify mean and standard deviation for each column

train_x_scaled = scaler.transform(train_x)

test_x_scaled = scaler.transform(test_x)
train_x.shape
model = LinearRegression().fit(train_x_scaled, train_y)

df = pd.DataFrame(model.coef_, index=train_x.columns, columns=['slope'])

df.sort_values(by='slope', ascending=False)#.plot.bar()
# expenses = 257.46age + 321.7*bmi + ...... -902.04*region_southwest -11300
model.intercept_
### Testing phase
test_y_pred = model.predict(test_x_scaled)

df_pred = pd.DataFrame({

    'actual': test_y,

    'predicted': test_y_pred

})

df_pred['error'] = df_pred['actual'] - df_pred['predicted']

#df_pred['square_error'] = df_pred['error'].apply(lambda v: v * v)

df_pred['square_error'] = np.square(df_pred['error'])

sse = df_pred['square_error'].sum()

mse = sse / df_pred.shape[0]

mse
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(test_y, test_y_pred)

rmse = np.sqrt(mse)

rmse # how bad is your model
rsquare = r2_score(test_y, test_y_pred)

rsquare # how good is your model

## percentage of information/variance in target variable explained/understood using the input variable
### Exercise

sales_data.head()

## ignore date column
insurance.isna().sum() / insurance.shape[0] * 100
insurance.columns.drop('region')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, r2_score





insurance_dummies = pd.get_dummies(insurance.drop(['sex', 'region'], axis=1), drop_first=True) # Dummies creation

insurance_dummies['bmi_above_28'] = insurance_dummies['bmi'].apply(lambda v: 1 if v > 28 else 0)

insurance_dummies['high_bmi_and_smoker'] = insurance_dummies.apply(lambda row:1 if (row['smoker_yes']==1) & (row['bmi']>28) else 0, axis=1)

insurance_dummies['age_square'] = np.square(insurance['age'])

insurance_dummies['age_log'] = np.log(insurance['age'])



target_col = 'expenses'

input_cols = insurance_dummies.columns.drop(['expenses'])



# train vs test split

train_x, test_x, train_y, test_y = train_test_split(insurance_dummies[input_cols], 

                                                    insurance_dummies[target_col],

                                                   test_size=0.2,

                                                    random_state=1

                                                   )



# Standardization

scaler = StandardScaler().fit(train_x) # identify mean and standard deviation for each column

train_x_scaled = scaler.transform(train_x) # Standardization

test_x_scaled = scaler.transform(test_x)



# Model building & prediction

model = LinearRegression().fit(train_x_scaled, train_y)

test_y_pred = model.predict(test_x_scaled) # Prediction



# Model Evaluation

mse = mean_squared_error(test_y, test_y_pred)

rmse = np.sqrt(mse)

rsquare = r2_score(test_y, test_y_pred)

print(rmse)

print(rsquare)
df = pd.DataFrame(model.coef_, index=train_x.columns, columns=['slope'])

df.sort_values(by='slope', ascending=False)#.plot.bar()
## P- Value of model coefficients
import statsmodels.api as sm

X = pd.DataFrame(train_x_scaled, columns=train_x.columns, index=train_x.index)



X2 = sm.add_constant(X) # to ask statsmodel to calculate intercept

model = sm.OLS(train_y, X2).fit()

model.summary()