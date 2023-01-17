# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.patches as mpatches



from sklearn.preprocessing import LabelEncoder

from scipy import stats

from scipy.stats import skew



from sklearn.model_selection import train_test_split 

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LinearRegression 

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import learning_curve
train=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip")

test=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip")

stores=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv")

features=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip")
train['Date'] =pd.to_datetime(train['Date'], format="%Y-%m-%d")

features['Date'] =pd.to_datetime(features['Date'], format="%Y-%m-%d")

test['Date'] = pd.to_datetime(test['Date'], format="%Y-%m-%d")
stores.head()
stores.describe()
stores.isnull().sum()
grouped=stores.groupby('Type')

grouped.describe()['Size'].round(2)
fig, ax = plt.subplots(1, figsize = (8,6))

ax.bar(stores['Type'].unique(), stores['Size'].groupby(stores['Type']).count())

ax.set_ylabel('No of Stores')

ax.set_xlabel('Type')

ax.yaxis.grid(True, linewidth=0.3)
data = pd.concat([stores['Type'], stores['Size']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='Type', y='Size', data=data)
stores[(stores['Size'] < 50000) & ((stores['Type'].isin(['A'])) | (stores['Type'].isin(['B'])))]
i = stores[(stores['Size'] < 50000) & ((stores['Type'].isin(['A'])) | (stores['Type'].isin(['B'])))].index

stores = stores.drop(i)
features.head()
features.describe()
features.info()
total = features.isnull().sum().sort_values(ascending=False)

percent = (features.isnull().sum()/features.isnull().count()).sort_values(ascending=False)

pd.concat([total, percent], axis=1, keys=['Total', '%'])
train = pd.merge(train,stores,how='left',on='Store')

test = pd.merge(test,stores,how='left',on='Store')
train = pd.merge(train, features, how = "inner", on=["Store","Date",'IsHoliday'])

test = pd.merge(test, features, how = "inner", on=["Store","Date",'IsHoliday'])
train.head()
train.info()
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

pd.concat([total, percent], axis=1, keys=['Total', '%'])
corr = train.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(11, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, 

            square=True, linewidths=.5, annot=True, cmap=cmap)

plt.yticks(rotation=0)

plt.title('Correlation Matrix')

plt.show()
plt.figure(figsize=(15,6))



sns.boxplot(x='Weekly_Sales', data=train)

plt.ticklabel_format(style='plain', axis='x')

plt.show()
train = train[(train['Weekly_Sales'] > 0) & (train['Weekly_Sales'] < 500000)]
test.head()
test.info()
train['MarkDown1'] = train[(train['MarkDown1'] > 0)]

train['MarkDown2'] = train[(train['MarkDown2'] > 0)]

train['MarkDown3'] = train[(train['MarkDown3'] > 0)]

train['MarkDown4'] = train[(train['MarkDown4'] > 0)]

train['MarkDown5'] = train[(train['MarkDown5'] > 0)]
train = train[train['Type'].notna()]
le = LabelEncoder()

le.fit(train.Type)

train['Type'] = le.transform(train.Type)
train = pd.get_dummies(train, columns=['IsHoliday'])

test = pd.get_dummies(test, columns=['IsHoliday'])
train['Month'] = train['Date'].dt.month

train['Week'] = train['Date'].dt.week

train = train.drop(columns=["Date"])
test['Month'] = test['Date'].dt.month

test['Week'] = test['Date'].dt.week

test = test.drop(columns=["Date"])
train = train.fillna(0)
print("Skewness MarkDown1: {0}".format(skew(train['MarkDown1'])))

print("Skewness MarkDown2: {0}".format(skew(train['MarkDown2'])))

print("Skewness MarkDown3: {0}".format(skew(train['MarkDown3'])))

print("Skewness MarkDown4: {0}".format(skew(train['MarkDown4'])))

print("Skewness MarkDown5: {0}".format(skew(train['MarkDown5'])))
skewed = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']

train[skewed] = train[skewed].apply(lambda x: np.log(x + 1))

test[skewed] = test[skewed].apply(lambda x: np.log(x + 1))
sns.set_style("whitegrid")

plt.figure(figsize=(15,8))

plotd = sns.distplot(train['Weekly_Sales'], kde=True, bins=350)



tick_spacing=250000 

plotd.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

# plotd.set_xlim([-100000, 5000000]) 

plt.xticks(rotation=30) 

plt.axvline(train['Weekly_Sales'].mean(), c='red')

plt.axvline(train['Weekly_Sales'].median(), c='blue')



print("Skewness : {0}".format(skew(train['Weekly_Sales'])))

plt.show()
train['Weekly_Sales'] = train['Weekly_Sales'].apply(lambda x: np.log(x + 1))
sns.set_style("whitegrid")

plt.figure(figsize=(15,8))

plotd = sns.distplot(train['Weekly_Sales'], kde=True, bins=350)



tick_spacing=250000 

plotd.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

# plotd.set_xlim([-100000, 5000000]) 

plt.xticks(rotation=30) 

plt.axvline(train['Weekly_Sales'].mean(), c='red')

plt.axvline(train['Weekly_Sales'].median(), c='blue')



print("Skewness : {0}".format(skew(train['Weekly_Sales'])))

plt.show()
Y = train['Weekly_Sales']

X = train.drop(['Weekly_Sales'], axis=1)



X.shape , Y.shape
X_train ,X_test, Y_train , Y_test = train_test_split(X , Y , test_size = 0.3 , random_state =75)
linreg = LinearRegression()

linreg.fit(X_train, Y_train)

Y_pred_lin = linreg.predict(X_test)

np.sqrt(mean_squared_error(Y_test,Y_pred_lin))
alpha=0.00099

lasso_regr=Lasso(alpha=alpha,max_iter=50000)

lasso_regr.fit(X_train, Y_train)

Y_pred_lasso=lasso_regr.predict(X_test)

np.sqrt(mean_squared_error(Y_test,Y_pred_lasso))
ridge = Ridge(alpha=0.01, normalize=True)

ridge.fit(X_train, Y_train)

Y_pred_ridge = ridge.predict(X_test)

np.sqrt(mean_squared_error(Y_test,Y_pred_ridge))
rf_regr = RandomForestRegressor()

rf_regr.fit(X_train, Y_train)

Y_pred_rf = rf_regr.predict(X_test)

np.sqrt(mean_squared_error(Y_test,Y_pred_rf))