import pandas as pd

import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import seaborn as sns

sns.set()
raw_data = pd.read_csv("../input/car-price/cars.csv")

raw_data.head()
raw_data.describe(include = 'all') 

"""By default only numerical variables are included in the description. To include categorical variables, we 

pass the include argument and set the value to all"""

data = raw_data.drop(['Model'], axis =1) # We take out model 

data.describe(include = 'all') 
data.isnull().sum()
"""Having discovered the missing values in the dataset, since the sum of these obervations are less than 5% of 

the total observations, we eliminate all obervations with null values"""

data_no_mv = data.dropna(axis = 0)
data_no_mv.describe(include = 'all')
sns.distplot(data_no_mv['Price'])
q = data_no_mv['Price'].quantile(0.99)

data_1 = data_no_mv[data_no_mv['Price'] < q]

data_1.describe(include = 'all')
sns.distplot(data_1['Price'])
p = data_1['Mileage'].quantile(0.99)

data_2 = data_1[data_1['Mileage'] < p]

sns.distplot(data_2['Mileage'])
data_3 = data_2[data_2['EngineV'] < 6.5]

sns.distplot(data_3['EngineV'])
r = data_3['Year'].quantile(0.01)

data_4 = data_3[data_3['Year'] > r]

sns.distplot(data_4['Year'])
"""In order to let our indexes represent the filtered/cleaned data, we reset the indexes and drop the indexes of the 

raw data"""

data_cleaned = data_4.reset_index(drop = True) 

data_cleaned.describe(include = 'all')
f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey = True, figsize=(15,3))

ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])

ax1.set_title("Price and Year")



ax2.scatter(data_cleaned['Mileage'], data_cleaned['Price'])

ax2.set_title("Price and Mileage")



ax3.scatter(data_cleaned['EngineV'], data_cleaned['Price'])

ax3.set_title("Price and Engine Volume")



plt.show()
sns.distplot(data_cleaned['Price'])
log_price = np.log(data_cleaned['Price'])

data_cleaned['log_price'] = log_price
f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey = True, figsize=(15,3))

ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])

ax1.set_title("Log Price and Year")



ax2.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])

ax2.set_title("Log Price and Mileage")



ax3.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])

ax3.set_title("Log Price and Engine Volume")



plt.show()
data_cleaned = data_cleaned.drop(['Price'],axis =1)

data_cleaned.describe(include = 'all')
from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = data_cleaned[['Mileage','Year','EngineV']]

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]

vif['Features'] = variables.columns
vif
data_no_multicollinearity = data_cleaned.drop(['Year'],axis = 1)
data_no_multicollinearity
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first = True)#The drop_first = True ensures 

#that no dummy is created for all features of the categories
data_with_dummies.head()
data_with_dummies.columns.values
cols = ['log_price','Mileage', 'EngineV','Brand_BMW',

       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',

       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',

       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',

       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data_preprocessed = data_with_dummies[cols]

data_preprocessed.head()
inputs = data_preprocessed.drop(['log_price'],axis = 1)

targets = data_preprocessed['log_price']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(inputs)

inputs_scaled = scaler.transform(inputs)# It's not recommended to standardize dummy variables as we did here
inputs_scaled
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size = 0.2, random_state = 365)
reg = LinearRegression()

reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)

plt.scatter(y_hat, y_train)

plt.xlabel('Predictions(y_hat)',size = 18)

plt.ylabel('Targets(y_train)',size = 18)

plt.xlim(6,13)

plt.ylim(6,13)

plt.show()
sns.distplot(y_train - y_hat)

plt.title('Residuals PDF', size = 18)
reg.score(x_train,y_train)
reg.intercept_
reg.coef_
reg_summary = pd.DataFrame(data = inputs.columns.values, columns = ['Features'])

reg_summary['Weights'] = reg.coef_

reg_summary
data_cleaned['Brand'].unique()
y_hat_test = reg.predict(x_test)

plt.scatter(y_hat_test, y_test, alpha = 0.2) # The alpha parameter allows us to see how concentrated the values are around the 45-degree line

plt.xlabel('Predictions(y_hat_test)',size = 18)

plt.ylabel('Targets(y_test)',size = 18)

plt.xlim(6,13)

plt.ylim(6,13)

plt.show()
df_perf = pd.DataFrame(data = np.exp(y_hat_test),columns = ['Predictions'])# We take the exponent(opposite) of log_price 

df_perf.head()
df_perf['Targets'] = np.exp(y_test)

df_perf
y_test
y_test = y_test.reset_index(drop = True)
df_perf['Targets'] = np.exp(y_test)

df_perf
df_perf['Residuals'] = df_perf['Targets'] - df_perf['Predictions']

df_perf['Residuals%'] = abs(df_perf['Targets'] - df_perf['Predictions'])/100

df_perf
df_perf.describe()
pd.options.display.max_rows = 999

pd.set_option('display.float_format', lambda x: '%.2f'%x)

df_perf.sort_values(by = ['Residuals%'])