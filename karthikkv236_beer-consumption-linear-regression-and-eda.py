import numpy as np

import pandas as pd

import datetime

import seaborn as sns

import matplotlib.pyplot as plt
beer_df = pd.read_csv("../input/beer-consumption-sao-paulo/Consumo_cerveja.csv")
beer_df
beer_df.info()
# Dropping rows with all NAN Values

beer_df.dropna(how = 'all', inplace = True)
beer_df.info()
# Replacing commas with period

beer_df.replace({',':'.'}, regex = True, inplace = True)
beer_df
beer_df.info()
# Converting the type of Data to Date time

beer_df['Data'] = pd.to_datetime(beer_df['Data'])
beer_df.info()

days = ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
beer_df['Day'] = beer_df['Data'].apply(lambda a: days[a.weekday()])
beer_df['Day']
plt.figure(figsize=(10,5))

ax = sns.barplot(x="Day", y="Consumo de cerveja (litros)", data=beer_df)
beer_df.drop(['Data','Day'], axis = 1, inplace = True)
# Converting temperature and rainfall columns into float type

beer_df  = beer_df.apply(pd.to_numeric)
beer_df.info()
ax = sns.pairplot(beer_df)
import statsmodels.api  as sm
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(beer_df, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' variables

num_vars = ['Temperatura Media (C)',

'Temperatura Minima (C)',

'Temperatura Maxima (C)',

'Precipitacao (mm)',

'Consumo de cerveja (litros)']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
# Dividing into X and Y sets for model building

y_train = df_train.pop('Consumo de cerveja (litros)')

X_train = df_train
# Add a constant because for stats model we need to explicitely add a constant or the line passes through origin by default

X_train_lm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_lm).fit()
# Print a summary of the linear regression model obtained

print(lr.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X = X_train.drop('Temperatura Media (C)', 1,)
# Building another model

X_train_lm = sm.add_constant(X)



lr_2 = sm.OLS(y_train, X_train_lm).fit()
# Print a summary of the linear regression model obtained

print(lr_2.summary())
# Calculate the VIFs again for the new model

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X = X.drop('Temperatura Minima (C)', 1,)
# Building another model

X_train_lm = sm.add_constant(X)



lr_3 = sm.OLS(y_train, X_train_lm).fit()
# Printing the summary of the linear regression model obtained

print(lr_3.summary())
y_train_pred = lr_3.predict(X_train_lm)
fig = plt.figure()

sns.distplot((y_train - y_train_pred), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)               
df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('Consumo de cerveja (litros)')

X_test = df_test
X_test = sm.add_constant(X_test)
X_test = X_test.drop(['Temperatura Minima (C)', 'Temperatura Media (C)'], axis = 1)
y_pred = lr_3.predict(X_test)
# Plotting y_test and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y_test, y_pred)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)     