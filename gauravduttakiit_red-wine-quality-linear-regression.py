import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
wine = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
wine.head()
wine.info()
wine.describe()
wine.shape
round(100*(wine.isnull().sum()/len(wine)),2).sort_values(ascending=False)
round(100*(wine.isnull().sum(axis=1)/len(wine)),2).sort_values(ascending=False)
dub_wine=wine.copy()

dub_wine.drop_duplicates(subset=None,inplace=True)
dub_wine.shape
wine.shape
wine=dub_wine
for col in wine:

    print(wine[col].value_counts(ascending=False), '\n\n\n')
wine.shape
wine.info()
from sklearn.model_selection import train_test_split

np.random.seed(0)

df_train,df_test=train_test_split(wine,train_size=0.7,test_size=0.3,random_state=100)
df_train.info()
df_train.shape
df_test.info()
df_train.shape
df_train.info()
df_train.columns
sns.pairplot(df_train) 

plt.show()
plt.figure(figsize=(20,25))

sns.heatmap(wine.corr(), annot=True,cmap='RdBu')

plt.show()
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df_train.head()
df_train.columns
df_train[:]=scaler.fit_transform(df_train[:])
df_train.head()
y_train=df_train.pop('quality')

X_train=df_train
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train, y_train)

rfe = RFE(lm,9)             

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
X_train_rfe = X_train[col]
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train_rfe.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
import statsmodels.api as sm

X_train_lm1 = sm.add_constant(X_train_rfe)

lr1 = sm.OLS(y_train, X_train_lm1).fit()
lr1.params
print(lr1.summary())
X_train_new = X_train_rfe.drop(["residual sugar"], axis = 1)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_lm2 = sm.add_constant(X_train_new)

lr2 = sm.OLS(y_train, X_train_lm2).fit()
lr2.params
print(lr2.summary())
X_train_new = X_train_new.drop(["density"], axis = 1)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_lm3 = sm.add_constant(X_train_new)

lr3 = sm.OLS(y_train, X_train_lm3).fit()
lr3.params
print(lr3.summary())
X_train_new = X_train_new.drop(["free sulfur dioxide"], axis = 1)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_lm4 = sm.add_constant(X_train_new)

lr4 = sm.OLS(y_train, X_train_lm4).fit()
lr4.params
print(lr4.summary())
X_train_new = X_train_new.drop(["pH"], axis = 1)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_lm5 = sm.add_constant(X_train_new)

lr5 = sm.OLS(y_train, X_train_lm5).fit()
lr5.params
print(lr5.summary())
X_train_new = X_train_new.drop(["sulphates"], axis = 1)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_lm6 = sm.add_constant(X_train_new)

lr6 = sm.OLS(y_train, X_train_lm6).fit()
lr6.params
print(lr6.summary())
X_train_new = X_train_new.drop(["chlorides"], axis = 1)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_lm7 = sm.add_constant(X_train_new)

lr7 = sm.OLS(y_train, X_train_lm7).fit()
lr7.params
print(lr7.summary())
X_train_new = X_train_new.drop(["total sulfur dioxide"], axis = 1)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif



X_train_lm8 = sm.add_constant(X_train_new)

lr8 = sm.OLS(y_train, X_train_lm8).fit()



lr8.params



print(lr8.summary())
y_train_pred = lr8.predict(X_train_lm8)
res = y_train-y_train_pred

# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((res), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)
wine_num=wine[[ 'volatile acidity', 'alcohol', 'quality']]



sns.pairplot(wine_num)

plt.show()
vif = pd.DataFrame()

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
df_test[:]=scaler.fit_transform(df_test[:])
df_test.head()
df_test.describe()
y_test = df_test.pop('quality')

X_test = df_test

X_test.info()
#Selecting the variables that were part of final model.

col1=X_train_new.columns

X_test=X_test[col1]

# Adding constant variable to test dataframe

X_test_lm8 = sm.add_constant(X_test)

X_test_lm8.info()
y_pred = lr8.predict(X_test_lm8)
fig = plt.figure()

plt.scatter(y_test, y_pred, alpha=.5)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16) 

plt.show()
df= pd.DataFrame({'Actual':y_test,'Predictions':y_pred})

df['Predictions']= round(df['Predictions'],2)

df.head()
sns.regplot('Actual','Predictions',data=df)
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
r2=0.32264089150785114

X_test.shape
n = X_test.shape[0]





# Number of features (predictors, p) is the shape along axis 1

p = X_test.shape[1]



# We find the Adjusted R-squared using the formula



adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

adjusted_r2
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))