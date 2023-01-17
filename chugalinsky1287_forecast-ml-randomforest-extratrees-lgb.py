%%time
import pandas as pd
import numpy as np
from numba import jit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import lightgbm as lgb
#from catboost import CatBoost
#import pyodbc
import warnings
warnings
warnings.filterwarnings('ignore')
#import matplotlib.pyplot as plt
#matplotlib inline
import gc
#import category_encoders as ce
from sklearn.model_selection import train_test_split
import pyodbc
from sklearn import metrics
def mape(y_true, y_pred):
    ABS = np.abs(y_true - y_pred)
    return (ABS.sum()/y_true.sum()) * 100
seed = 123
sales = pd.read_csv('../input/sales data-set.csv')
stores = pd.read_csv('../input/stores data-set.csv')
features = pd.read_csv('../input/Features data set.csv')

df = pd.merge(sales,stores, on='Store',how='left')
df1 = pd.merge(df,features, on=['Store','Date'],how='left')
df1.head()
df1.Date = pd.to_datetime(df1.Date,format='%d/%m/%Y')
df1['week'] = df1.Date.dt.week
df1['year'] = df1.Date.dt.year
df1['year_week'] =  df1['year'].astype(str)+ df1['week'].astype(str)

df1.head()
df1['Weekly_Sales']=df1['Weekly_Sales'].astype(float)
# добавляем ноль перед 1,2,3,4 и т.д. неделей для правильной сортировки
df1['year_week'] = df1['year'].astype(str) + df1['week'].astype(str)
df1.loc[df1['week']<10, 'year_week'] = df1.loc[df1['week']<10, 'year'].astype(str) + '0' + df1.loc[df1['week']<10, 'week'].astype(str)
df1['year_week'] =  df1['year_week'].astype(int)

#убираем продажи меньше 0
df1 = df1[df1['Weekly_Sales']>=0]
    
#логарифмируем продажи
df1['Weekly_Sales_log'] = np.log1p(df1.Weekly_Sales)
df1.drop('Weekly_Sales', axis=1, inplace=True)

df1= pd.get_dummies(data=df1, columns=['year','Store','Dept','Type'])

df1['dec1'] = np.where((df1.week==52),1,0)
df1['dec2'] = np.where((df1.week==51),1,0)
#все категориальные числа переводим в числа
df1_numeric = df1.select_dtypes(exclude=['object','bool'])
df1_obj = df1.select_dtypes(include=['object','bool']).copy()
for c in df1_obj:
    df1_obj[c] = pd.factorize(df1_obj[c])[0]
df1 = pd.concat([df1_numeric, df1_obj], axis=1)
df1.head()
df1 = df1.fillna(0)
df1.head()
df1.drop('Date', axis=1, inplace=True)
df1.info()
df1.year_week.unique()
test = df1[df1.year_week.isin([201240, 201241, 201242, 201243])]
df2 = df1[~df1.year_week.isin([201240, 201241, 201242, 201243])]
X_test = test.drop('Weekly_Sales_log', axis=1)
y_test = test['Weekly_Sales_log']
X_train = df2.drop('Weekly_Sales_log', axis=1)
y_train = df2['Weekly_Sales_log']

model = ExtraTreesRegressor(n_estimators=20, criterion='mse', bootstrap=True, n_jobs=-1, random_state=seed)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
100-mape(np.expm1(y_test), np.expm1(y_pred))
model = RandomForestRegressor(n_estimators=20, criterion='mse', bootstrap=True, n_jobs=-1, random_state=seed)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
100-mape(np.expm1(y_test), np.expm1(y_pred))
df1.columns