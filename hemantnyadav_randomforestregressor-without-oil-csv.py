# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}

data = {

    'tra': pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=['date']),

    'tes': pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date']),

    'ite': pd.read_csv('../input/items.csv'),

    'sto': pd.read_csv('../input/stores.csv'),

    'trn': pd.read_csv('../input/transactions.csv', parse_dates=['date']),

    'hol': pd.read_csv('../input/holidays_events.csv', dtype={'transferred':str}, parse_dates=['date']),

    'oil': pd.read_csv('../input/oil.csv', parse_dates=['date']),

    }

train = data['tra'][(data['tra']['date'].dt.month == 8) & (data['tra']['date'].dt.day > 15)]
train = train_all[(train_all['date'].dt.month == 8) & (train_all['date'].dt.day > 15)]

train["onpromotion"].fillna(0, inplace=True)

train['id'] = train['id'].astype(np.uint32)

train['store_nbr'] = train['store_nbr'].astype(np.uint8)

train['item_nbr'] = train['item_nbr'].astype(np.uint32)

train['unit_sales'] = train['unit_sales'].astype(np.uint32)

train["onpromotion"]=train["onpromotion"].astype(np.int8)
test = pd.read_csv("../input/test.csv",parse_dates=["date"])
test["onpromotion"].fillna(0, inplace=True)

test['id'] = test['id'].astype(np.uint32)

test['store_nbr'] = test['store_nbr'].astype(np.uint8)

test['item_nbr'] = test['item_nbr'].astype(np.uint32)

test["onpromotion"]=test["onpromotion"].astype(np.int8)
items = pd.read_csv('../input/items.csv')

transaction = pd.read_csv('../input/transactions.csv')

stores = pd.read_csv('../input/stores.csv')
print("Total Obsevations before sampling",len(train))

strain = train#.sample(frac=1.0,replace=True)

print("Total Obsevations after sampling",len(strain))
df = strain.merge(right = items, on='item_nbr', right_index  = True)

df = df.merge(right=stores,on='store_nbr', right_index  = True)
df_test = test.merge(right = items, on='item_nbr', right_index  = True)

df_test = df_test.merge(right=stores,on='store_nbr', right_index  = True)
from sklearn.model_selection import GridSearchCV

reduced_df = df.drop(['city','state','id'], axis=1)

reduced_df_test = df_test.drop(['city','state','id'], axis=1)

ohe_df = pd.get_dummies(df, columns=['family','type','city','state']) 

ohe_df_test = pd.get_dummies(df_test, columns=['family','type','city','state']) 

ohe_reduced_df = pd.get_dummies(reduced_df, columns=['family','type'])

ohe_reduced_df_test = pd.get_dummies(reduced_df_test, columns=['family','type'])
from sklearn.model_selection   import train_test_split

unitSales,reduced_unitSales = ohe_df['unit_sales'],ohe_reduced_df['unit_sales']

features,reduced_features = ohe_df.drop('unit_sales', axis = 1),ohe_reduced_df.drop('unit_sales', axis = 1)



X_train, X_test, y_train, y_test = train_test_split(features, unitSales, test_size=0.2, random_state=42)

#X_train, X_test, y_train, y_test = train_test_split(features, unitSales, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import cross_validation

from sklearn.metrics import classification_report,confusion_matrix

rf = RandomForestRegressor()

rf.fit(X_train, y_train)
prediction = rf.predict(X_test)
prediction = pd.DataFrame(prediction, index=X_test.index,columns=['PedictedSale'])

#Use below code to when predicting with test dataset

#prediction = pd.DataFrame(prediction, index=test.index,columns=['PedictedSale'])

sub = pd.concat([prediction, y_test.to_frame()], axis=1)

sub = pd.concat([sub, X_test['perishable'].to_frame()], axis=1)
sub['newPS'] = sub.apply(lambda row: 0 if(row['PedictedSale']<0) else row['PedictedSale'] , axis=1)

sub['newUS'] = sub.apply(lambda row: 0 if(row['unit_sales']<0) else row['unit_sales'] , axis=1)
sub['newPS'] = np.log(sub.newPS + 1 )

sub['newUS'] = np.log(sub.newPS + 1 )

sub['yhatminusy'] = (sub['newPS']-sub['newUS'])**2

sub['perishable'] = sub.perishable>0

sub['perishableW'] = sub.apply(lambda row: 1.5 if(row['perishable']) else 1, axis=1)

sub['comp1'] = sub.yhatminusy*sub.perishableW
ax = sub['PedictedSale'].resample('m').mean().plot(figsize = (15, 6), color='red')

fig = sub['unit_sales'].resample('m').mean().plot(ax=ax).get_figure()

plt.legend(['Predicted Sale', 'actual'], loc='upper right')

plt.show()
(sub.comp1.sum()/sub.perishableW.sum())**0.5