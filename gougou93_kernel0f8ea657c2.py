import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn

from scipy import stats

from scipy.stats import norm, skew #for some statistics

url = 'https://raw.githubusercontent.com/goupilAnthony/house_pricing_prediction/master/train.csv'

url2 = 'https://raw.githubusercontent.com/goupilAnthony/house_pricing_prediction/master/test.csv'

data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
id_train = data_train['Id']

id_test = data_test['Id']
all_data = pd.concat([data_train,data_test])

data = all_data
data_train.shape
data_test.shape
all_data.index
#train

data.dtypes.unique()
#train

def recap_nan(data):



  feature_understanding = pd.DataFrame(columns=['feature_name','value_type','unique_value_number','unique_value_list','nan_percentage','nan_nbr']) 



  cols = list(data.columns)

  for col in cols:

    nb_nan = data[col].isna().sum()

    total = len(data)

    nan_perc = nb_nan / total *100

    type_val = []

    un_val_list = list(data[col].unique()) 

    for val in un_val_list:

      if type(val) not in type_val:

        type_val.append(type(val))

    un_val_nb = len(un_val_list)

    line = { 'feature_name' : col, 'value_type' : type_val, 'unique_value_number' : un_val_nb, 'unique_value_list' : un_val_list, 'nan_percentage' : nan_perc,'nan_nbr':nb_nan}

    feature_understanding = feature_understanding.append(line,ignore_index=True)

  return feature_understanding[feature_understanding['nan_percentage'] > 0]
#train

recap_nan(data)
data.loc[pd.isna(data['GarageArea']),:]
#train / test

def replace_nan(data):

  data.loc[pd.isna(data['LotFrontage']),'LotFrontage'] = 0

  data.loc[pd.isna(data['Alley']),'Alley'] = 'NO'

  data.loc[pd.isna(data['MasVnrType']),'MasVnrType'] = 'None'

  data.loc[pd.isna(data['MasVnrArea']),'MasVnrArea'] = 0

  data.loc[pd.isna(data['BsmtQual']),'BsmtQual'] = 'NO'

  data.loc[pd.isna(data['BsmtCond']),'BsmtCond'] = 'NO'

  data.loc[pd.isna(data['BsmtExposure']),'BsmtExposure'] = 'NO'

  data.loc[pd.isna(data['BsmtFinType1']),'BsmtFinType1'] = 'NO'

  data.loc[pd.isna(data['BsmtFinType2']),'BsmtFinType2'] = 'NO'

  data.loc[pd.isna(data['FireplaceQu']),'FireplaceQu'] = 'NO'

  data.loc[pd.isna(data['GarageType']),'GarageType'] = 'NO'

  data.loc[pd.isna(data['GarageYrBlt']),'GarageYrBlt'] = 'NO'

  data.loc[pd.isna(data['GarageFinish']),'GarageFinish'] = 'NO'

  data.loc[pd.isna(data['GarageCond']),'GarageCond'] = 'NO'

  data.loc[pd.isna(data['GarageQual']),'GarageQual'] = 'NO'

  data.loc[pd.isna(data['KitchenQual']),'KitchenQual'] = 'TA'

  data.loc[pd.isna(data['Fence']),'Fence'] = 'NO'	

  data.loc[pd.isna(data['MSZoning']),'MSZoning'] = 'C (all)'

  data.loc[pd.isna(data['Utilities']),'Utilities'] = 'AllPub' 

  data.loc[pd.isna(data['Functional']),'Functional'] = 'Mod' 

  data.loc[pd.isna(data['MiscFeature']),'MiscFeature'] = 'Othr' 

  data.loc[pd.isna(data['Exterior2nd']),'Exterior2nd'] = 'NA'

  data.loc[pd.isna(data['BsmtFinSF1']),'BsmtFinSF1'] = 0

  data.loc[pd.isna(data['BsmtFinSF2']),'BsmtFinSF2'] = 0

  data.loc[pd.isna(data['BsmtUnfSF']),'BsmtUnfSF'] = 0

  data.loc[pd.isna(data['TotalBsmtSF']),'TotalBsmtSF'] = 0 

  data.loc[pd.isna(data['BsmtFullBath']),'BsmtFullBath'] = 0

  data.loc[pd.isna(data['BsmtHalfBath']),'BsmtHalfBath'] = 0

  data.loc[pd.isna(data['Electrical']),'Electrical'] = 'Mix'

  data.loc[pd.isna(data['Exterior1st']),'Exterior1st'] = 'NA'

  data.loc[pd.isna(data['GarageCars']),'GarageCars'] = 0

  data.loc[pd.isna(data['GarageArea']),'GarageArea'] = 0

  data.loc[pd.isna(data['PoolQC']),'PoolQC'] = 'NO'

  data.loc[pd.isna(data['SaleType']),'SaleType'] = 'NA'

  return data

data = replace_nan(data)

cols = list(data.columns)

cols.pop(-1)

y = data['SalePrice']

#train

data.shape
cols = list(data.columns)

for col in cols:

  nb_null = data[col].isnull().sum()

  total = len(data)

  null_perc = nb_null / total *100

  index = feature_understanding[feature_understanding['feature_name'] == col ].index[0]

  feature_understanding.loc[index,'null_percentage'] = null_perc

  

feature_understanding[feature_understanding['null_percentage'] > 0]
#train

non_cat = ['LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',

          'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',

          'MiscVal','YrSold','SalePrice']

data = data.set_index('Id')
#train

categoricals= list(set(list(data.columns)).difference(set(non_cat)))
#train

categoricals
recap_cat = pd.DataFrame(columns=['name','unique'])

for col in categoricals:

  un = len(data[col].unique())

  recap_cat = recap_cat.append({'name':col,'unique':un},ignore_index=True)

  

recap_cat
data.shape
#train

for col in categoricals:

  dum = pd.get_dummies(data.loc[:,col])

  dum = dum.drop(dum.columns[0],axis=1)

  data = data.drop(col,axis=1)

  data = pd.concat([data,dum],axis=1)

  

data.shape #after dummies
cols = list(data.columns)

x = data.loc[id_train , :]

x = x.drop(columns=['SalePrice'])

test_set = data.loc[id_test,:]

test_set = test_set.drop(columns=['SalePrice'])

y = data.loc[id_train,'SalePrice']
x = test_set  ############  POUR PROCESING AVANT DE PREDICT 
x.shape
y.shape
test_set.head()
len(test_set.columns)
len(cols)
x.shape
#train

fig, ax = plt.subplots()

ax.scatter(x = x['GrLivArea'], y = y)

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
ind = x[(x['GrLivArea']>4000) & (y<300000)].index

ind
#train

x = x.drop(ind)

y = y.drop(ind)
#train

fig, ax = plt.subplots()

ax.scatter(x = x['GrLivArea'], y = y)

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#train

x.info()
#train

sns.distplot(y)
#train

y = np.log1p(y)
#train

sns.distplot(y)
#train

data.columns
#train

recap_skew = pd.DataFrame(columns=['col_name','skew_before_log','skew_after_log'])

numeric_feats = x.dtypes[x.dtypes != "object"].index

for col in x[numeric_feats]:

  skew_bef = skew(x[col])

  x[col] = np.log1p(x[col])

  skew_aft = skew(x[col])

  line = {'col_name':col,'skew_before_log':skew_bef,'skew_after_log':skew_aft}

  recap_skew = recap_skew.append(line,ignore_index=True)
#train

recap_skew
#train

sns.distplot(x['LotArea'])
#train

x.shape
test_set.shape
#train

x.info()
#train

x.loc[:,x.dtypes == object].head()
#train

x['GarageYrBlt'].unique()
#train

x.loc[x['GarageYrBlt'] == 'NO','GarageYrBlt'] = 0
#train

np.dtype(x['GarageYrBlt'])
#train

x['GarageYrBlt'] = x['GarageYrBlt'].astype(str).astype(float)
#train

x['GarageYrBlt'] = np.log1p(x['GarageYrBlt'])
np.dtype(x['GarageYrBlt'])
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

x = df_pca

y = data['SalePrice']

scores = pd.DataFrame(columns=['score'])

for i in range(101):

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33) 

  lin_reg = LinearRegression()

  lin_reg.fit(X_train,y_train)

  score = lin_reg.score(X_test,y_test)

  line = { 'score':score }

  scores = scores.append(line,ignore_index=True)

  

print(scores['score'].mean())
x.shape
y.shape
#train

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

scores = pd.DataFrame(columns=['score'])

for i in range(101):

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20) 

  lin_reg = LinearRegression()

  lin_reg.fit(X_train,y_train)

  score = lin_reg.score(X_test,y_test)

  line = { 'score':score }

  scores = scores.append(line,ignore_index=True)

  

print(scores['score'].mean())
from sklearn.model_selection import train_test_split
#train

from sklearn import model_selection

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn import metrics

kfold = model_selection.KFold(n_splits=5, random_state=42)

model = LinearRegression()

results = cross_val_score(model, x, y, cv=10)
results.mean()
#train

preds = cross_val_predict(model, x, y, cv=6)
scores = pd.DataFrame(columns=['scores'])

for i in range(101):

  from sklearn.linear_model import LassoCV

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

  lasso = LassoCV(cv=5)

  lasso.fit(X_train,y_train)

  score = lasso.score(X_test,y_test)

  line = {'scores':score}

  scores = scores.append(line,ignore_index=True)

scores.mean()
x.shape
scores = pd.DataFrame(columns=['scores'])

for i in range(101):

  from sklearn.linear_model import ElasticNetCV

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

  regr = ElasticNetCV(cv=5)

  regr.fit(X_train,y_train)

  score = regr.score(X_test,y_test)

  scores = scores.append(line,ignore_index=True)

scores.mean()
#train

x.shape
x.shape
pred = regr.predict(x)
pred = list(np.expm1(pred))
list(x.index)
sub = pd.DataFrame(columns=['Id','SalePrice'])

sub['Id'] = list(x.index)

sub['SalePrice'] = pred
sub.head()
sub.to_csv('submit.csv',index=False)