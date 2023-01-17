import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api

from sklearn.linear_model import LinearRegression
row_data = pd.read_csv('../input/car-price/cars.csv')

row_data.head()
data = row_data.drop(['Model'],axis=1)

data.head()
data.isna().sum()
data = data.dropna(axis=0)
data.describe(include='all')
sns.distplot(data['Price'])
q = data['Price'].quantile(0.99)
data_1 = data[data['Price']<q]
sns.distplot(data_1['Price'])
sns.distplot(data_1['Mileage'])
q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]
sns.distplot(data_2['EngineV'])
data_3 = data_2[data_2['EngineV']<6.5]
sns.distplot(data_3["EngineV"])
q = data_3['Year'].quantile(0.99)

data_4 = data_3[data_3['Year']>0.01]
sns.distplot(data_4['Year'])
data_cleaned = data_4.reset_index(drop=True)
data_cleaned.describe(include='all')
sns.scatterplot(data_cleaned['Year'],data_cleaned['Price'])
sns.scatterplot(data_cleaned['EngineV'],data_cleaned['Price'])
sns.scatterplot(data_cleaned['Mileage'],data_cleaned['Price'])
log_price = np.log(data_cleaned['Price'])

data_cleaned['log_price']=log_price

data_cleaned.drop(['Price'],axis=1)
sns.scatterplot(data_cleaned['Mileage'],data_cleaned['log_price'])
sns.scatterplot(data_cleaned['Year'],data_cleaned['log_price'])
sns.scatterplot(data_cleaned['EngineV'],data_cleaned['log_price'])
from statsmodels.stats.outliers_influence import variance_inflation_factor

variable=data_cleaned[['Mileage','EngineV','Year']]

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(variable.values,i) for i in range(variable.shape[1])]

vif['features'] = variable.columns



vif
data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)
data_with_dummies = pd.get_dummies(data_no_multicollinearity,drop_first=True)
data_with_dummies.head()
data_with_dummies.columns.values
cols = ['log_price','Mileage', 'EngineV','Brand_BMW',

       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',

       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',

       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',

       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()
target = data_preprocessed['log_price']

inputs = data_preprocessed.drop(['log_price'],axis=1)
#now we are going to scale the data using sklearn standard scaler and then use it

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(inputs)
input_scaled = scaler.transform(inputs)
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(input_scaled,target,test_size=0.8,random_state=171)
reg = LinearRegression()

reg.fit(x_train,y_train)

y_hat = reg.predict(x_train)
sns.distplot(y_hat-y_train)
# now create soe regression summary
reg.intercept_
reg_summary = pd.DataFrame(inputs.columns.values,columns=['features'])

reg_summary['weights'] = reg.coef_

reg_summary
y_hat_test = reg.predict(x_test)

sns.scatterplot(y_test,y_hat_test)
predictions = pd.DataFrame(x_test)

df_predictions.head()