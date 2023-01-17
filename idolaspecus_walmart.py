import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv')

test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv')

store = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')

ft = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv')



train['year'] = train['Date'].astype('datetime64').dt.year

train['month'] = train['Date'].astype('datetime64').dt.month

train['week'] = train['Date'].astype('datetime64').dt.week

train['day'] = train['Date'].astype('datetime64').dt.day



test['year'] = test['Date'].astype('datetime64').dt.year

test['month'] = test['Date'].astype('datetime64').dt.month

test['week'] = test['Date'].astype('datetime64').dt.week

test['day'] = test['Date'].astype('datetime64').dt.day
ft['Temperature'] = ft['Temperature'].astype('float32')

ft['Fuel_Price'] = ft['Fuel_Price'].astype('float32')

ft['Unemployment'] = ft['Unemployment'].astype('float32')

ft['CPI'] = ft['CPI'].astype('float32')
store['type2'] = 0

store['type2'].loc[store['Type']=='A'] = 1

store['type2'].loc[store['Type']=='B'] = 2

store['type2'].loc[store['Type']=='C'] = 3
train['size'] = pd.merge(train, store)['Size']

test['size'] = pd.merge(test, store)['Size']



train['type'] = pd.merge(train, store)['type2']

test['type'] = pd.merge(test, store)['type2']
train = pd.merge(train, ft)

test = pd.merge(test, ft)



print(train.describe())

print(train.corr())
print(train.columns)
Y = train['Weekly_Sales']



X = train.drop(['Date','Weekly_Sales','Temperature', 'type', 'Temperature', 'MarkDown1'

               , 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Fuel_Price'

               , 'CPI', 'Unemployment'], axis=1)

test = test.drop(['Date', 'Temperature', 'type', 'Temperature', 'MarkDown1'

                 , 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Fuel_Price'

                 ,'CPI', 'Unemployment'], axis=1)
print(X.head(1))

print(test.head(1))
import matplotlib.pyplot as plt

import seaborn as sns



#train3 = train[train['Weekly_Sales']<50000]



# print(train.groupby('type')['Weekly_Sales'].mean(), train.groupby('type')['Weekly_Sales'].median(),

#       train.groupby('type')['Weekly_Sales'].max(), train.groupby('type')['Weekly_Sales'].min())



#a, b = plt.subplots(1,1,figsize=(20, 10))

#sns.boxplot(train['year'][train['Weekly_Sales']<50000], train['Weekly_Sales'][train['Weekly_Sales']<50000])

#sns.boxplot(train3['temp'], train3['Weekly_Sales'])
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)

rf.fit(X, np.log(Y+5000))

pred = np.exp(rf.predict(test))-5000



submission = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv")

submission["Weekly_Sales"] = pred

submission.to_csv("/kaggle/working/submission.csv", index=False)
importance_df = pd.DataFrame(rf.feature_importances_)

importance_df['columns'] = X.columns

importance_df = importance_df.sort_values(0, ascending = False)

print(importance_df)