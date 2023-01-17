import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_columns = None

sns.set(style="whitegrid", color_codes=True)
df = pd.read_csv("../input/CarPrice_Assignment.csv")
df.info()
df.head(5)
df['doornumber'] = df['doornumber'].map({'two': 2, 'four': 4})
df['cylindernumber'] = df['cylindernumber'].map({'two': 3, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12})
df.drop('car_ID', axis = 1, inplace = True)
df.info()
# Checking for any duplicate values

df.duplicated().sum()
# Selecting only numeric columns

df_int = df.select_dtypes(include=['int64', 'float'])
df_int.head()
df_int.info()
# Visualizing numeric data and variation with target variable

plt.figure(figsize=(15, 30))

for i, x_var in enumerate(df_int.drop('price', axis=1).columns):

    plt.subplot(8,2,i+1)

    sns.scatterplot(x = x_var, y = 'price', data = df_int)
# Finding correlated features

plt.figure(figsize = (16, 10))

sns.heatmap(df_int.corr(), annot = True, cmap="YlGnBu")
df_cat = df.select_dtypes(exclude=['int64', 'float'])
df_cat.info()
df_cat = pd.concat([df_cat, df.price], axis = 1)
# Visualizing categorical features

plt.figure(figsize=(20, 30))

for i, x_var in enumerate(df_cat.drop('price', axis =1).columns):

    plt.subplot(4,2,i+1)

    sns.boxplot(x = x_var, y = 'price', data = df_cat)
df['car_company'] = df['CarName'].apply(lambda x: x.split(' ')[0])
df.car_company.value_counts()
Top_companies = ['toyota', 'nissan', 'mazda', 'honda', 'mitsubishi', 'subaru',

                 'volvo', 'peugeot', 'volkswagen', 'dodge', 'buick', 'bmw']
df['car_company'] = df['car_company'].apply(lambda x: x if x in Top_companies else 'Other')
df.car_company.value_counts()
df.drop('CarName', axis = 1, inplace = True)
plt.figure(figsize = (10,6))

sns.boxplot(x='car_company', y = 'price', data = df)
# Separating out the list of numeric features for future use

df_int = df.select_dtypes(include=['int64', 'float'])

df_int = df_int.drop('price', axis = 1)

df_int.columns
df_cat = df.select_dtypes(exclude=['int64', 'float'])
df_cat.columns
df = pd.get_dummies(df, columns = df_cat.columns, drop_first = True)
df.head()
df.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('price', axis=1), df['price'], test_size=0.3, random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[df_int.columns] = scaler.fit_transform(X_train[df_int.columns])
X_train[df_int.columns].head()
X_train[df_int.columns].describe()
X_train_int = X_train[df_int.columns]
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
lm = LinearRegression()
# Using all numeric features

lm.fit(X_train_int, y_train)
y_pred = lm.predict(X_train_int)

print('R2 score with all numeric columns:', round(r2_score(y_train, y_pred),2))
# Eliminating non-relevant features

from sklearn.feature_selection import RFE
rfe = RFE(lm, 6)           # running RFE

rfe = rfe.fit(X_train_int, y_train)
pd.DataFrame(list(zip(X_train_int.columns, rfe.support_, rfe.ranking_))).sort_values(by = 2)
col = X_train_int.columns[rfe.support_]

col
X_train_int_rfe = X_train_int[col]
import statsmodels.api as sm  

X_train_int_rfe = sm.add_constant(X_train_int_rfe)
lm_rfe = sm.OLS(y_train,X_train_int_rfe).fit()
print(lm_rfe.summary())
y_pred = lm_rfe.predict(X_train_int_rfe)

print('R2 score (train data) with relevant numeric columns:', round(r2_score(y_train, y_pred),2))
lm.fit(X_train, y_train)

y_pred = lm.predict(X_train)

print('R2 score with all numeric & categorical columns:', round(r2_score(y_train, y_pred),2))
R2_feat = pd.DataFrame()

for i in range(X_train.shape[1]):

    rfe = RFE(lm, 50-i)           # running RFE

    rfe = rfe.fit(X_train, y_train)

    col = X_train.columns[rfe.support_]

    lm.fit(X_train[col], y_train)

    y_pred = lm.predict(X_train[col])

    R2_feat = R2_feat.append({'# of features': 50-i,'R2 score': round(r2_score(y_train, y_pred),2)}, ignore_index= True)
plt.figure(figsize = (14,4))

ax = sns.barplot(x = '# of features', y = 'R2 score', data = R2_feat, palette="GnBu_d")

ax.set_xticklabels(51-R2_feat['# of features'].astype(int)) # This line is just to remove decimal points in X_ticks

plt.tight_layout()
rfe = RFE(lm, 7)           # running RFE

rfe = rfe.fit(X_train, y_train)
pd.DataFrame(list(zip(X_train.columns, rfe.support_, rfe.ranking_))).sort_values(by = 2)
col = X_train.columns[rfe.support_]

col
X_train_rfe = X_train[col]

X_train_rfe = sm.add_constant(X_train_rfe)
lm_rfe = sm.OLS(y_train,X_train_rfe).fit()

print(lm_rfe.summary())
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
X_train_vif = X_train_rfe.drop(['const'], axis=1)
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_vif.columns

vif['VIF'] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping carlength because of high VIF

X_train_vif = X_train_rfe.drop(['curbweight', 'const'], axis=1)
# Checking VIF again

vif = pd.DataFrame()

vif['Features'] = X_train_vif.columns

vif['VIF'] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Not dropping further columns since p value is < 0.05 for remaining
col = X_train_vif.columns

col
lm.fit(X_train[col], y_train)

y_pred = lm.predict(X_train[col])

print('R2 score with all relevant numeric & categorical columns:', round(r2_score(y_train, y_pred),2))
fig = plt.figure()

sns.distplot((y_train - y_pred), bins = 25)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
X_test[df_int.columns] = scaler.transform(X_test[df_int.columns])
X_test.describe()
y_pred = lm.predict(X_test[col])

print('R2 score with all numeric & categorical columns:', round(r2_score(y_test, y_pred),2))

print('Relevant features:', col)
sns.scatterplot(y_test, y_pred)
coeffecients = pd.DataFrame(lm.coef_,X_train[col].columns)

coeffecients.columns = ['Coeffecient']

coeffecients