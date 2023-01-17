# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

sns.set(rc=({'figure.figsize':(11,15)}))

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#-- importing files
db_features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
db_sampleSubmission = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')
db_stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
db_test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
db_train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
#-- printing head
db_features.head()
#-- checking db dimension
print(f'Rows: {db_features.shape[0]}')
print(f'\nColumns: {db_features.shape[1]}')
#-- checking features type
db_features.info()
#-- converting date field
db_features['Date'] = db_features['Date'].apply(pd.to_datetime)
#-- checking convertion
db_features.info()
#-- checking missing values
db_features.isnull().sum().sort_values(ascending=False).to_frame() / len(db_features)
#-- checking data
db_features.describe()
#-- printing date values
print(f"Min Date value: {min(db_features['Date'])}")
print(f"\nMax Date value: {max(db_features['Date'])}")
#-- checking % of registers before Nov 2011
print("Values which will be removed, considering markdown availability: " +
      f"{len(db_features[db_features['Date'] < '2011-12-01']) / len(db_features):.2f}")
#-- printing head
db_stores.head()
#-- checking db dimension
print(f'Rows: {db_stores.shape[0]}')
print(f'\nColumns: {db_stores.shape[1]}')
#-- checking features type
db_stores.info()
#-- checking missing values
db_stores.isnull().sum().sort_values(ascending=False).to_frame() / len(db_stores)
#-- checking data
db_stores.describe()
#-- counting stores types
db_stores['Type'].value_counts()
#-- printing head
db_train.head()
#-- checking db dimension
print(f'Rows: {db_train.shape[0]}')
print(f'\nColumns: {db_train.shape[1]}')
#-- checking features type
db_train.info()
#-- converting date field
db_train['Date'] = db_train['Date'].apply(pd.to_datetime)
#-- checking convertion
db_train.info()
#-- checking missing values
db_train.isnull().sum().sort_values(ascending=False).to_frame() / len(db_train)
#-- checking data
db_train.describe()
#-- grouping stores
db_train_g = db_train.groupby(['Store', 'Dept'])['Dept'].count().to_frame().rename(columns={'Dept':'count'})
db_train_g.reset_index(inplace=True)
#-- checking number of departments by Store
db_train_g['Store'].value_counts().to_frame().sort_index()
#-- merging db_train + db_store
db_train_store = pd.merge(left=db_train, right=db_stores, on='Store', how='left')
db_train_store.head()
#-- grouping weekly sales by store type
db_train_store_g = db_train_store.groupby(['Date', 'Type'])['Weekly_Sales'].sum().reset_index()
db_train_store_g.index = db_train_store_g['Date']
#-- ploting seasonal for stores A
result_a = seasonal_decompose(db_train_store_g[db_train_store_g['Type'] == 'A']['Weekly_Sales'], model='additive')

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1, figsize=(12,8))
result_a.observed.plot(ax=ax1)
result_a.trend.plot(ax=ax2)
result_a.seasonal.plot(ax=ax3)
result_a.resid.plot(ax=ax4)
plt.tight_layout()
#-- ploting seasonal for stores B
result_b = seasonal_decompose(db_train_store_g[db_train_store_g['Type'] == 'B']['Weekly_Sales'], model='additive')

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1, figsize=(12,8))
result_b.observed.plot(ax=ax1)
result_b.trend.plot(ax=ax2)
result_b.seasonal.plot(ax=ax3)
result_b.resid.plot(ax=ax4)
plt.tight_layout()
#-- ploting seasonal for stores C
result_c = seasonal_decompose(db_train_store_g[db_train_store_g['Type'] == 'C']['Weekly_Sales'], model='additive')

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1, figsize=(12,8))
result_c.observed.plot(ax=ax1)
result_c.trend.plot(ax=ax2)
result_c.seasonal.plot(ax=ax3)
result_c.resid.plot(ax=ax4)
plt.tight_layout()
#-- function to define stationarity
def estacionario(x):
    """Função para avaliar se os dados são estacionários"""
    
    if adfuller(x)[1] <= 0.05:
        print('Conjunto de dados é estacionário')
        print(f'Número de lags utilizado: {adfuller(x)[2]}')
    else:
        print('Conjuntos de dados não é estacionário')
#-- function to define normality
def normal(x):
    """Função para avaliar normalidade"""
    p_normal = stats.shapiro(x)[1]
    
    if p_normal >= 0.05:
        print(f'Dados seguem uma distribuição normal - p_value = {p_normal:.2}')
    else:
        print(f'Dados não seguem uma distribuição normal - p_value = {p_normal:.2}')
#-- stationarity test
estacionario(db_train_store_g[db_train_store_g['Type'] == 'A']['Weekly_Sales'])
#-- stationarity test
estacionario(db_train_store_g[db_train_store_g['Type'] == 'B']['Weekly_Sales'])
#-- stationarity test
estacionario(db_train_store_g[db_train_store_g['Type'] == 'C']['Weekly_Sales'])
#-- normality teste
normal(db_train_store_g[db_train_store_g['Type'] == 'A']['Weekly_Sales'])
#-- normality teste
normal(db_train_store_g[db_train_store_g['Type'] == 'B']['Weekly_Sales'])
#-- normality teste
normal(db_train_store_g[db_train_store_g['Type'] == 'C']['Weekly_Sales'])
#-- merging train_stores + features
db_train_store_features = pd.merge(left=db_train_store, right=db_features, on=['Store', 'Date'], how='left', suffixes=('_train', '_features'))
db_train_store_features.head()
db_train_store_features.info()
#-- creating db bi store type
type_a = db_train_store_features.loc[db_train_store_features['Type'] == 'A']
type_b = db_train_store_features.loc[db_train_store_features['Type'] == 'B']
type_c = db_train_store_features.loc[db_train_store_features['Type'] == 'C']
#-- EDA
_ = sns.pairplot(db_train_store_features_n)
#-- checking correlation between features
_ = plt.clf()
_ = plt.style.use('fivethirtyeight')
_ = font_opts = {'fontsize':15, 'fontweight':'bold'}
_ = plt.figure(figsize=(20,10))

x = sns.heatmap(
    db_train_store_features.corr(), 
    annot=db_train_store_features.corr(), 
    fmt='.2f', 
    annot_kws={'fontsize':10, 'fontweight':'bold'},
    cmap='RdPu'
)

_ = plt.title("Correlation Matrix\n", **font_opts)
_ = plt.xticks(**font_opts)
_ = plt.yticks(**font_opts)


_ = plt.tight_layout();
_ = plt.plot();
#-- converting strings to number
db_train_store_features.Type = db_train_store_features.Type.map({'A':1, 'B':2, 'C':3})
#-- selecting features
X_train = db_train_store_features[['Store','Dept','IsHoliday_train','Size', 'Type']]
Y_train = db_train_store_features['Weekly_Sales']
#-- creating function to ml
def random_forest(n_estimators, max_depth):
    resultado = []
    for estimator in n_estimators:
        for depth in max_depth:
            wmaes = []
            for i in range(1,5):
                print('k:', i, ', n_estimators:', estimator, ', max_depth:', depth)
                x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)
                ml_rf = RandomForestRegressor(n_estimators=estimator, max_depth=depth, random_state=0)
                ml_rf.fit(x_train, y_train)
                predicted = ml_rf.predict(x_test)
                wmaes.append(wmae(x_test, y_test, predicted))
            print('WMAE:', np.mean(wmaes))
            resultado.append({'Max_Depth': depth, 'Estimators': estimator, 'WMAE': np.mean(wmaes)})
    return pd.DataFrame(result)
#-- creating fuction to calculate error
def wmae(db, y, y_h):
    weights = db.IsHoliday_train.apply(lambda x: 5 if x else 1)
    return np.round(np.sum(weights*abs(y-y_h))/(np.sum(weights)), 2)
#-- applying ml
n_estimators = [56, 58, 60]
max_depth = [25, 27, 30]

random_forest(n_estimators, max_depth)
#-- creating columns based on date, I'll test if those columns will improve models evaluation
db_train_store_features['week'] = db_train_store_features['Date'].dt.week
db_train_store_features['month'] = db_train_store_features['Date'].dt.month
db_train_store_features['day_year'] = db_train_store_features['Date'].dt.dayofyear
#-- selecting features
X_train = db_train_store_features[['Store','Dept','IsHoliday_train','Size', 'Type', 'week', 'month', 'day_year']]
Y_train = db_train_store_features['Weekly_Sales']
#-- running second model
n_estimators = [56, 58, 60]
max_depth = [25, 27, 30]

random_forest(n_estimators, max_depth)
#-- defining model to use in test db
ml_rf = RandomForestRegressor(n_estimators=60, max_depth=27, max_features=6, min_samples_split=3, min_samples_leaf=1, random_state=0)
ml_rf.fit(X_train, Y_train)
#-- printing head
db_test.head()
#-- checking data
db_test.info()
#-- converting date field
db_test.Date = db_test.Date.apply(pd.to_datetime)
#-- checking convertion
db_test.info()
#-- merging db_train + db_store
db_train_store = pd.merge(left=db_train, right=db_stores, on='Store', how='left')
db_train_store.head()
#-- merging dbs
db_test_store = pd.merge(left=db_test, right=db_stores, on='Store', how='left' )
db_test_store.head()
#-- merging dbs
db_test_store_features = pd.merge(left=db_test_store, right=db_features, on=['Store', 'Date'], how='left', suffixes=('_train', '_features'))
db_test_store_features.head()
#-- converting
db_test_store_features.Type = db_test_store_features.Type.map({'A':1, 'B':2, 'C':3})
#-- creating same features
db_test_store_features['week'] = db_test_store_features['Date'].dt.week
db_test_store_features['month'] = db_test_store_features['Date'].dt.month
db_test_store_features['day_year'] = db_test_store_features['Date'].dt.dayofyear
#-- selecting features
X_test = db_test_store_features[['Store','Dept','IsHoliday_train','Size', 'Type', 'week', 'month', 'day_year']]
#-- running prediction
predict = RF.predict(X_test)
#-- print prediction
print(predict)
#-- concat store + dept + date
db_test_store_features['Id'] = db_test_store_features['Store'].astype(str) + '_' + db_test_store_features['Dept'].astype(str) + '_' + db_test_store_features['Date'].astype(str)
#-- concat ID + Prediction
db_Submission = pd.concat([db_test_store_features['Id'], pd.DataFrame(predict)], axis=1)
db_Submission.columns = ['Id', 'Weekly_Sales']
#-- saving results into .csv
db_Submission.to_csv('walmart_thiago_mauricio.csv', index=False)