# import necessary modules

import numpy as np

import pandas as pd

import random

import seaborn as sns

import matplotlib.pyplot as plt

import gc



import seaborn as sns

sns.set(style = 'whitegrid', color_codes = True)

%matplotlib inline



#For statistical tests

import scipy.stats as st



#For formula notation (similar to R)

import statsmodels.formula.api as smf



from sklearn.ensemble import RandomForestRegressor

from sklearn.cross_validation import train_test_split

import xgboost as xgb

import operator
df_train = pd.read_csv("../input/corporitasampled-train-data/train_rd.csv")
df_train.head()
print("we have taken ",len(df_train), "rows")
Input_Path = '../input/favorita-grocery-sales-forecasting/'
test  = pd.read_csv("../input/favorita-grocery-sales-forecasting/test.csv")

testg  = pd.read_csv("../input/favorita-grocery-sales-forecasting/test.csv")

store = pd.read_csv("../input/favorita-grocery-sales-forecasting/stores.csv")

holiday = pd.read_csv("../input/favorita-grocery-sales-forecasting/holidays_events.csv")

item = pd.read_csv("../input/favorita-grocery-sales-forecasting/items.csv")

oil = pd.read_csv("../input/favorita-grocery-sales-forecasting/oil.csv")

trans = pd.read_csv("../input/favorita-grocery-sales-forecasting/transactions.csv")
df_train.head()
item.head()
print("There are",len(item['family'].unique()),"families of products or items")
oil.head()
trans.head()
holiday.head()
store.head()
print("There are",len(store['type'].unique()),"type of stores")
print("Stores are in ",len(store['city'].unique()),"cities in ", len(store['state'].unique()),"states")
test.head()
train = pd.merge(df_train, store, on= "store_nbr")

train = pd.merge(train, item, on= "item_nbr")

train = pd.merge(train, holiday, on="date")

train = pd.merge(train, oil, on ="date")
train.head()
train_items = pd.merge(df_train, item, how='inner')

train_items1 = pd.merge(df_train, item, how='inner')

train_items2 = pd.merge(df_train, item, how='inner')
train_items.head()
oil_nan = (oil.isnull().sum() / oil.shape[0]) * 100

oil_nan
store_nan = (store.isnull().sum() / store.shape[0]) * 100

store_nan
item_nan = (item.isnull().sum() / item.shape[0]) * 100

item_nan
df_train_nan = (df_train.isnull().sum() / df_train.shape[0]) * 100

df_train_nan
train['onpromotion'] = train['onpromotion'].fillna(2)

train['onpromotion'] = train['onpromotion'].replace(True,1)

train['onpromotion'] = train['onpromotion'].replace(False,0)
(train['onpromotion'].unique())
train['dcoilwtico'] = train['dcoilwtico'].fillna(0)
train['Year']  = train['date'].apply(lambda x: int(str(x)[:4]))

train['Month'] = train['date'].apply(lambda x: int(str(x)[5:7]))

train['date']  = train['date'].apply(lambda x: (str(x)[8:]))





test['Year']  = test['date'].apply(lambda x: int(str(x)[:4]))

test['Month'] = test['date'].apply(lambda x: int(str(x)[5:7]))

test['date']  = test['date'].apply(lambda x: (str(x)[8:]))



train.head()

train_items1['date'] = pd.to_datetime(train_items1['date'], format='%Y-%m-%d')

train_items1['day_item_purchased'] = train_items1['date'].dt.day

train_items1['month_item_purchased'] =train_items1['date'].dt.month

train_items1['quarter_item_purchased'] = train_items1['date'].dt.quarter

train_items1['year_item_purchased'] = train_items1['date'].dt.year

train_items1.drop('date', axis=1, inplace=True)



train_items2['date'] = pd.to_datetime(train_items2['date'], format='%Y-%m-%d')

train_items2['day_item_purchased'] = train_items2['date'].dt.day

train_items2['month_item_purchased'] =train_items2['date'].dt.month

train_items2['quarter_item_purchased'] = train_items2['date'].dt.quarter

train_items2['year_item_purchased'] = train_items2['date'].dt.year

train_items2.drop('date', axis=1, inplace=True)
#train_items['Year']  = train_items['date'].apply(lambda x: int(str(x)[:4]))

#train_items['Month'] = train_items['date'].apply(lambda x: int(str(x)[5:7]))

#train_items['date']  = train_items['date'].apply(lambda x: (str(x)[8:]))
train_items1.loc[(train_items1.unit_sales<0),'unit_sales'] = 1 

train_items1['unit_sales'] =  train_items1['unit_sales'].apply(pd.np.log1p) 



train_items1['family'] = train_items1['family'].astype('category')

train_items1['onpromotion'] = train_items1['onpromotion'].astype('category')

train_items1['perishable'] = train_items1['perishable'].astype('category')

cat_columns = train_items1.select_dtypes(['category']).columns

train_items1[cat_columns] = train_items1[cat_columns].apply(lambda x: x.cat.codes)



train_items2.loc[(train_items2.unit_sales<0),'unit_sales'] = 1 

train_items2['unit_sales'] =  train_items2['unit_sales'].apply(pd.np.log1p) 



train_items2['family'] = train_items2['family'].astype('category')

train_items2['onpromotion'] = train_items2['onpromotion'].astype('category')

train_items2['perishable'] = train_items2['perishable'].astype('category')

cat_columns = train_items2.select_dtypes(['category']).columns

train_items2[cat_columns] = train_items2[cat_columns].apply(lambda x: x.cat.codes)
train_items1.head()
strain = train.sample(frac=0.01,replace=True)
fig, (axis1) = plt.subplots(1,1,figsize=(30,4))

sns.barplot(x='onpromotion', y='unit_sales', data=strain, ax=axis1)
fig, (axis1) = plt.subplots(1,1,figsize=(30,4))

sns.barplot(x='family', y='unit_sales', data=strain, ax=axis1)
fig, (axis1) = plt.subplots(1,1,figsize=(30,4))

sns.countplot(x=strain['family'], data=strain, ax=axis1)
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))

sns.barplot(x='type_x', y='unit_sales', data=strain, ax=axis1)
fig, (axis1) = plt.subplots(1,1,figsize=(30,4))

sns.countplot(x=store['city'], data=store, ax=axis1)
fig, (axis1) = plt.subplots(1,1,figsize=(30,4))

sns.countplot(x=store['state'], data=store, ax=axis1)
fig, (axis1) = plt.subplots(1,1,figsize=(30,4))

sns.countplot(x='cluster', data=store, ax=axis1)
g = sns.FacetGrid(train, col='cluster', hue='cluster', size=4)

g.map(sns.barplot, 'type_x', 'unit_sales');
fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(15,8))



ax1 = oil.plot(legend=True,ax=axis1,marker='o',title="Oil Price")

average_sales = train.groupby('date')["unit_sales"].mean()

average_promo = train.groupby('date')["onpromotion"].mean()



fig, (axis1, axis2) = plt.subplots(2,1,figsize=(15,4))



ax1 = average_sales.plot(legend=True,ax=axis1,marker='o',title="Average Sales")

ax2 = average_promo.plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Average Promo")
train.plot(kind='scatter',x='store_nbr',y='unit_sales',figsize=(15,4))
train.plot(kind='scatter',x='item_nbr',y='unit_sales',figsize=(15,4))
store_number = train.groupby('store_nbr')["unit_sales"].mean()

item_number = train.groupby('item_nbr')["unit_sales"].mean()



fig, (axis1, axis2) = plt.subplots(2,1,figsize=(30,4))



ax1 = store_number.plot(legend=True,ax=axis1,marker='o',title="Sales with store")

ax2 = item_number.plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Sales with item")
# Contingency table

ct = pd.crosstab(store['type'], store['cluster'])

ct
ct.plot.bar(figsize = (15, 6), stacked=True)

plt.legend(title='cluster vs Type')

plt.show()
st.chi2_contingency(ct)
# Contingency table

ct2 = pd.crosstab(store['city'], store['cluster'])

ct2
ct2.plot.bar(figsize = (15, 6), stacked=True)

plt.legend(title='cluster')

plt.show()
st.chi2_contingency(ct2)
promo_sales = train[train['onpromotion'] == 1.0]['unit_sales']

nopromo_sales = train[train['onpromotion'] == 0.0]['unit_sales']

st.ttest_ind(promo_sales, nopromo_sales, equal_var = False)
lm0 = smf.ols(formula = 'unit_sales ~ dcoilwtico', data = train).fit()
#print the Result 

print(lm0.summary())
X_train = train.drop(['unit_sales', 'description', 'locale_name','locale','city','state','family','type_x','type_y','cluster','class','perishable','transferred', 'dcoilwtico'], axis = 1)

y_train = train.unit_sales
rf = RandomForestRegressor(n_jobs = -1, n_estimators = 15)

y = rf.fit(X_train, y_train)

print('model fit')
X_test = test

y_test = rf.predict(X_test)
result = pd.DataFrame({'id':test.id, 'unit_sales': y_test}).set_index('id')

result = result.sort_index()

result[result.unit_sales < 0] = 0

result.to_csv('submissionR.csv', index=False)

print('submission created')

train_items1 = train_items1.drop(['unit_sales','family','class','perishable'], axis = 1)
train_items1.head()
train_items2 = train_items2.drop(['id','store_nbr','item_nbr','onpromotion', 'day_item_purchased','month_item_purchased','quarter_item_purchased','year_item_purchased','family','class','perishable'], axis = 1)
train_items2.head()
Xg_train, Xg_valid = train_test_split(train_items1, test_size=0.012, random_state=10)

Yg_train, Yg_valid = train_test_split(train_items2, test_size=0.012, random_state=10)

features = list(train_items1.columns.values)

features2 = list(train_items2.columns.values)
features 
features2
#dtrain = xgb.DMatrix(Xg_train[features], Xg_train.unit_sales)

#dvalid = xgb.DMatrix(Xg_valid[features], Xg_valid.unit_sales)

#Xg_train.dtypes
dtrain = xgb.DMatrix(Xg_train[features], Yg_train[features2])

dvalid = xgb.DMatrix(Xg_valid[features], Yg_valid[features2])
def rmspe(y, yhat):

    return np.sqrt(np.mean((yhat/y-1) ** 2))
def rmspe_xg(yhat, y):

    y = np.expm1(y.get_label())

    yhat = np.expm1(yhat)

    return "rmspe", rmspe(y,yhat)
params = {"objective": "reg:linear",

          "booster" : "gbtree",

          "eta": 0.3,

          "max_depth": 10,

          "subsample": 0.9,

          "colsample_bytree": 0.7,

          "silent": 1,

          "seed": 1301

          }

num_boost_round = 30

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \

  early_stopping_rounds=20, feval=rmspe_xg, verbose_eval=True)

print("Validating")

yhat = gbm.predict(xgb.DMatrix(Xg_valid[features]))

error = rmspe(Yg_valid.unit_sales.values, np.expm1(yhat))

print('RMSPE: {:.6f}'.format(error))
testg.head()
testg['date'] = pd.to_datetime(testg['date'], format='%Y-%m-%d')

testg['day_item_purchased'] = testg['date'].dt.day

testg['month_item_purchased'] =testg['date'].dt.month

testg['quarter_item_purchased'] = testg['date'].dt.quarter

testg['year_item_purchased'] = testg['date'].dt.year

testg.drop('date', axis=1, inplace=True)
testg.head()
features
testg.loc[(train_items.unit_sales<0),'unit_sales'] = 1 

#testg['unit_sales'] =  train_items['unit_sales'].apply(pd.np.log1p) 

testg['onpromotion'] = testg['onpromotion'].astype('category')

cat_columns = testg.select_dtypes(['category']).columns

testg[cat_columns] = testg[cat_columns].apply(lambda x: x.cat.codes)
dtest = xgb.DMatrix(testg[features])
test_probs = gbm.predict(dtest)

print("Make predictions on the test set")
result = pd.DataFrame({"id": test["id"], 'unit_sales': np.expm1(test_probs)})

result.to_csv("submissionX2.csv", index=False)

print("Submission created")