# Importing all required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
estate = pd.DataFrame(pd.read_csv("../input/real-estate-price/real_estate_price_size_year_view.csv"))
estate.head()
estate.describe()
import matplotlib.pyplot as plt

import seaborn as sns
sns.pairplot(estate)

plt.show()
plt.figure(figsize=(20, 12))

sns.regplot(x = 'size', y = 'price', data = estate)

plt.show()


plt.figure(figsize=(20, 12))

plt.subplot(2,3,2)

sns.boxplot(x = 'year', y = 'price', data = estate)

plt.subplot(2,3,3)

sns.boxplot(x = 'view', y = 'price', data = estate)

plt.show()
estate['view']=estate['view'].map({'No sea view':1,'Sea view':0})
estate.head()
estate['year'].value_counts()
year = pd.get_dummies(estate['year'])
year.head()
year = pd.get_dummies(estate['year'], drop_first = True)
estate = pd.concat([estate, year], axis = 1)
estate.head()
estate.drop(['year'], axis = 1, inplace = True)
estate.head()
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(estate, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_vars = ['size','price']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
plt.figure(figsize=[10,10])

sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
plt.figure(figsize=[10,10])

sns.regplot(x = 'size', y = 'price', data = df_train)

plt.show()
y_train = df_train.pop('price')

X_train = df_train
estate.columns
import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train)



lr= sm.OLS(y_train, X_train_lm).fit()



lr.params
print(lr.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X = X_train.drop(2009, 1,)
X_train_lm = sm.add_constant(X)



lr_2 = sm.OLS(y_train, X_train_lm).fit()
print(lr_2.summary())
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_price = lr_2.predict(X_train_lm)
fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  

plt.xlabel('Errors', fontsize = 18)

plt.show()
num_vars = ['size','price']



df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.head()
df_test.describe()
y_test = df_test.pop('price')

X_test = df_test
X_test_m2 = sm.add_constant(X_test)
y_pred_m2 = lr.predict(X_test_m2)
fig = plt.figure()

sns.regplot(y_test, y_pred_m2)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)   

plt.show()
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_m2))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_m2))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_m2)))
df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred_m2.values.flatten()})

df
df1 = df.head(30)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()