# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
import numpy as np 

import pandas as pd
df = pd.read_csv('../input/boombikes/day.csv')
df.head()
df.info()
df.isnull().sum()
df.head()
# The below variables are not useful in the analysis

# We have temp and atemp which are highly correlated. So, removing temp variable from the analysis



# Also, we should not have casual and registered since they are directly used in the cnt calculation

# cnt = casual + registered, so both these values will directly affect the calculation

redundant_variables = ['instant', 'dteday', 'temp', 'casual', 'registered']



df = df.drop(redundant_variables, axis=1)
df.head()
import matplotlib.pyplot as plt

import seaborn as sns
sns.pairplot(df)

plt.show()
sns.regplot(x='atemp', y='cnt', data=df)

plt.show()
plt.figure(figsize=(20,20))

plt.subplot(3,3,1)

sns.boxplot(x='season', y='cnt', data=df)

plt.subplot(3,3,2)

sns.boxplot(x='yr', y='cnt', data=df)

plt.subplot(3,3,3)

sns.boxplot(x='mnth', y='cnt', data=df)

plt.subplot(3,3,4)

sns.boxplot(x='holiday', y='cnt', data=df)

plt.subplot(3,3,5)

sns.boxplot(x='weekday', y='cnt', data=df)

plt.subplot(3,3,6)

sns.boxplot(x='workingday', y='cnt', data=df)

plt.subplot(3,3,7)

sns.boxplot(x='weathersit', y='cnt', data=df)

plt.show()
plt.figure(figsize=(20,20))

sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")

plt.show()
# We have to first create dummy columns for our analysis

#season : season (1:spring, 2:summer, 3:fall, 4:winter)

season_dict = { 

    1: 'Spring',

    2: 'Summer',

    3: 'Fall',

    4: 'Winter'

}

# Months

month_dict = {

    1: 'Jan',

    2: 'Feb',

    3: 'Mar',

    4: 'Apr',

    5: 'May',

    6: 'Jun',

    7: 'Jul',

    8: 'Aug',

    9: 'Sep',

    10: 'Oct',

    11: 'Nov',

    12: 'Dec'

}

# weathersit : 

# 1: Clear, Few clouds, Partly cloudy, Partly cloudy

# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist

# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds

# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

weathersit_dict = {

    1: 'Clear',

    2: 'Mist',

    3: 'Light_Snow',

    4: 'Heavy_Rain'

}

weekday_dict = {

    0: 'Sun',

    1: 'Mon',

    2: 'Tue',

    3: 'Wed',

    4: 'Thu',

    5: 'Fri',

    6: 'Sat',

}
df.weathersit = df.weathersit.apply(lambda x: weathersit_dict[x])

df.weathersit.value_counts()
df.mnth = df.mnth.apply(lambda x: month_dict[x])

df.season = df.season.apply(lambda x: season_dict[x])

df.weekday = df.weekday.apply(lambda x: weekday_dict[x])
plt.figure(figsize=(10,12))

plt.subplot(4,1,1)

sns.boxplot(x='season', y='cnt', data=df)

plt.subplot(4,1,2)

sns.boxplot(x='weathersit', y='cnt', data=df)

plt.subplot(4,1,3)

sns.boxplot(x='mnth', y='cnt', data=df)

plt.show()

sns.boxplot(x='weekday', y='cnt', data=df)

plt.show()
month_dummies = pd.get_dummies(df.mnth, drop_first = True)

season_dummies = pd.get_dummies(df.season, drop_first = True)

weathersit_dummies = pd.get_dummies(df.weathersit, drop_first = True)

weekday_dummies = pd.get_dummies(df.weekday, drop_first = True)
df = pd.concat([df, month_dummies, season_dummies, weathersit_dummies, weekday_dummies], axis=1)
df.head()
df.drop(['mnth', 'weekday', 'weathersit', 'season'], axis=1, inplace=True)

df.head()
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
df.describe()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_vars = ['atemp', 'hum', 'windspeed']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train.head()
y_train = df_train.pop('cnt')

X_train = df_train
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 15)             # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# Getting top15 columns

top15_cols = X_train.columns[rfe.support_]



top15_cols
# following columns are dropped from analysis

X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[top15_cols]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()
print(lm.summary())
X_train_rfe = X_train_rfe.drop(['const'], axis = 1)
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping hum since it's highly collinear VIF value is very high.

X_train_new = X_train_rfe.drop(['hum'], axis = 1)
# Rebuilding the model

X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()

print(lm.summary())
#Let's check for collinearity again in the new model

vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop(['atemp'], axis = 1)

# Rebuilding the model

X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()

print(lm.summary())
X_train_new = X_train_new.drop(['Jul'], axis = 1)

# Rebuilding the model

X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()

print(lm.summary())
X_train_new = X_train_new.drop(['Winter'], axis = 1)

# Rebuilding the model

X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()

print(lm.summary())
#Let's check for collinearity again in the new model

vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_cnt = lm.predict(X_train_lm)
fig = plt.figure()

sns.distplot((y_train - y_train_cnt), bins = 50)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('cnt')

X_test = df_test
# Using only the filtered columns present in X_train_new

X_test_new = X_test[X_train_new.columns]
# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
print(lm.summary())

# Total variables considered - 11