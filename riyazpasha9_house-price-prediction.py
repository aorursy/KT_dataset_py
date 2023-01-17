# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import chart_studio.plotly as py

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

cf.go_offline()

from scipy import stats

from scipy.stats import norm
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train.head()
print(train.shape)

print(test.shape)
test.head(2)
train.columns
plt.figure(figsize=(10,5))

sns.distplot(train['SalePrice'])

plt.show()
plt.figure(figsize=(12,5))

corrmat = train.corr()

sns.heatmap(corrmat,cmap='Blues')

plt.show()
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = train.corr().nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

plt.figure(figsize=(10,5))

hm = sns.heatmap(cm, annot=True,yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

sns.pairplot(data=train,vars=cols)

plt.show()
miss_per=train.isna().sum()/train.isna().count()*100

miss_per = miss_per.sort_values(ascending=False).head(20)

df_miss_per=pd.DataFrame(miss_per.values,index=miss_per.index,columns=['Total_percentage'])
df_miss_per
plt.figure(figsize=(12,5))

sns.barplot(x=df_miss_per['Total_percentage'],y=df_miss_per.index)

plt.title('Missing values percentage Plot')

plt.show()
train.drop(columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],inplace= True)
train['SalePrice'].describe()
sns.distplot(train['SalePrice'],fit=norm)

plt.figure()

stats.probplot(train['SalePrice'],plot=plt)
#dropping these extreme values

train=train.drop(axis=0,index=train.index[691])

train=train.drop(axis=0,index = train.index[1182])

train=train.drop(axis=0,index=train.index[495])

train=train.drop(axis=0,index = train.index[916])
sns.distplot(np.log(train['SalePrice']),fit=norm)

plt.figure()

stats.probplot(np.log(train['SalePrice']),plot=plt)
train['SalePrice'] = np.log(train['SalePrice'])

sns.scatterplot(train['GrLivArea'],train['SalePrice'])

plt.show()
train.drop(train[train['GrLivArea']>4000].index,axis=0,inplace=True)
sns.scatterplot(train['GrLivArea'],train['SalePrice'])

plt.show()
print(corrmat['SalePrice'].sort_values(ascending=False)[:15])

corrmat['SalePrice'].sort_values(ascending=False)[-5:]
features=list(corrmat['SalePrice'].sort_values(ascending=False)[:15].index)\

+list(corrmat['SalePrice'].sort_values(ascending=False)[-5:].index)

features.remove('SalePrice')
for col in features:

    sns.scatterplot(train[col],train['SalePrice'])

    plt.show()
train=train.drop(train[train['MasVnrArea']>1200].index,axis=0)

train=train.drop(train[train['BsmtFinSF1']>2000].index,axis=0)

train=train.drop(train[train['LowQualFinSF']>530].index,axis=0)

train=train.drop(train[(train['MSSubClass']<25) &(train['SalePrice']<11)].index,axis=0)
train[features].head()
cat_features=train.select_dtypes(include=['object']).columns

cat_features
for col in cat_features:

    sns.boxplot(x=train[col],y=train['SalePrice'])

    plt.show()
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].median())

train['GarageCond']=train['GarageCond'].fillna(train['GarageCond'].mode()[0])

train['GarageType']=train['GarageType'].fillna(train['GarageType'].mode()[0])

train['GarageYrBlt']=train['GarageYrBlt'].fillna(train['GarageYrBlt'].mode()[0])

train['GarageFinish']=train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])

train['GarageQual']=train['GarageQual'].fillna(train['GarageQual'].mode()[0])

train['BsmtExposure']=train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])

train['BsmtFinType2']=train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])

train['BsmtFinType1']=train['BsmtFinType1'].fillna(train['BsmtFinType1'].mode()[0])

train['BsmtCond']=train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])

train['BsmtQual']=train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])

train['MasVnrArea']=train['MasVnrArea'].fillna(train['MasVnrArea'].median())

train['MasVnrType']=train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])

train['Electrical']=train['Electrical'].fillna(train['Electrical'].mode()[0])

train['Utilities']=train['Utilities'].fillna(train['Utilities'].mode()[0])
train_1 = train.copy()
for col in cat_features:

    

    fea_du=train[col].value_counts().head(5)

    k=fea_du.index[:5]

    for cat in k:

        name= col+'_'+cat

        train_1[name] =(train_1[col]==cat).astype(int)

    del train_1[col]

    print(col)

    
train_1.head()

cat_features_train_1=train_1.columns     
corr=train_1.corr()

feature_select=list(corr['SalePrice'].sort_values(ascending=False)[:30].index)\

+list(corr['SalePrice'].sort_values(ascending=True)[:20].index)

feature_select.remove('SalePrice')
X = train_1[feature_select]

y = train_1['SalePrice']
from sklearn.preprocessing import StandardScaler,RobustScaler,normalize,MinMaxScaler

sc_x= StandardScaler()

X_1=sc_x.fit_transform(X)

df_scaled=pd.DataFrame(X_1,columns=X.columns)
np.mean(X_1),np.std(X_1)
from sklearn.decomposition import PCA

pca = PCA(n_components=0.98)

principalComponents = pca.fit_transform(X_1)

principalDf = pd.DataFrame(data = principalComponents)
principalDf.shape
pca.explained_variance_ratio_.sum()
plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Price Prediction Dataset Explained Variance')

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,SGDRegressor,Lasso

from sklearn.linear_model import Ridge,Lasso

from sklearn.metrics import mean_absolute_error,mean_squared_error

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

import statsmodels.api as sm
X = principalDf

y = train_1['SalePrice']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

model = LinearRegression()

model.fit(X_train,y_train)

print(model.score(X_train,y_train))

print(model.score(X_test,y_test))
y_pred = model.predict(X_test)

mean_absolute_error(y_test,y_pred)

np.sqrt(mean_squared_error(y_test,y_pred))
model1 = SGDRegressor(alpha=0.01)

model1.fit(X_train,y_train)

print(model1.score(X_train,y_train))

print(model1.score(X_test,y_test))
y_pred1 = model1.predict(X_test)

mean_absolute_error(y_test,y_pred1)

np.sqrt(mean_squared_error(y_test,y_pred1))
model2 = Ridge(alpha=0.01,fit_intercept=True)

model2.fit(X_train,y_train)

print(model2.score(X_train,y_train))

print(model2.score(X_test,y_test))
y_pred2 = model2.predict(X_test)

mean_absolute_error(y_test,y_pred2)

np.sqrt(mean_squared_error(y_test,y_pred2))
dt = DecisionTreeRegressor(max_depth=21,min_samples_split=92,criterion='mse')

dt.fit(X_train,y_train)

print(dt.score(X_train,y_train))

print(dt.score(X_test,y_test))
#rf = RandomForestRegressor(n_estimators=41,max_depth=32,warm_start=True)

rf = RandomForestRegressor(n_estimators=100,min_samples_split=15,min_samples_leaf=2,random_state=42,warm_start=True)

rf.fit(X_train,y_train)

print(rf.score(X_train,y_train))

print(rf.score(X_test,y_test))
bg = BaggingRegressor(base_estimator=rf,n_estimators=41)

bg.fit(X_train,y_train)

print(bg.score(X_train,y_train))

print(bg.score(X_test,y_test))
la = Lasso(alpha=0.001)

la.fit(X_train,y_train)

print(la.score(X_train,y_train))

print(la.score(X_test,y_test))
xgb = XGBRegressor(reg_alpha=0.001,reg_lambda=0.001)

xgb.fit(X_train,y_train)

print(xgb.score(X_train,y_train))

print(xgb.score(X_test,y_test))
from sklearn.model_selection import cross_val_score

for ml in [model,model1,model2,rf,la,bg]:

    

    accuracy = cross_val_score(estimator=ml,X=X_train,y=y_train,cv=10)

    print(ml,'Mean accuracy',accuracy.mean())

    print('Accuracy std',accuracy.std())
test.isna().sum().sort_values(ascending=False)[:40]
test.drop(columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],inplace= True)
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].median())

test['GarageCond']=test['GarageCond'].fillna(test['GarageCond'].mode()[0])

test['GarageType']=test['GarageType'].fillna(test['GarageType'].mode()[0])

test['GarageYrBlt']=test['GarageYrBlt'].fillna(test['GarageYrBlt'].mode()[0])

test['GarageFinish']=test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])

test['GarageQual']=test['GarageQual'].fillna(test['GarageQual'].mode()[0])

test['BsmtExposure']=test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])

test['BsmtFinType2']=test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])

test['BsmtFinType1']=test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])

test['BsmtCond']=test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])

test['BsmtQual']=test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])

test['MasVnrArea']=test['MasVnrArea'].fillna(test['MasVnrArea'].median())

test['MasVnrType']=test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])

test['Electrical']=test['Electrical'].fillna(test['Electrical'].mode()[0])

test['Utilities']=test['Utilities'].fillna(test['Utilities'].mode()[0])

test['GarageCars']=test['GarageCars'].fillna(test['GarageCars'].mode()[0])

test['GarageArea']=test['GarageArea'].fillna(test['GarageArea'].median())

test['KitchenQual']=test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])

test['Exterior1st']=test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])

test['SaleType']=test['SaleType'].fillna(test['SaleType'].mode()[0])

test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median())

test['BsmtUnfSF']=test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].median())

test['Exterior2nd']=test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])

test['BsmtFinSF1']=test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].median())

test['BsmtFinSF2'] =test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].median())

test['BsmtFullBath']=test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])

test['BsmtHalfBath'] =test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])

test['Functional']=test['Functional'].fillna(test['Functional'].mode()[0])

test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test.isna().sum().sort_values(ascending=False)[:40].head()
test_1 = test.copy()
for col in cat_features:

    

    fea_du_test=test[col].value_counts().head(5)

    k=fea_du_test.index[:5]

    for cat in k:

        name= col+'_'+cat

        test_1[name] =(test_1[col]==cat).astype(int)

    del test_1[col]

    print(col)
test_1.shape
test_1=test_1[feature_select]
#sc_test= StandardScaler()

df_test=sc_x.transform(test_1)

df_scaled_test=pd.DataFrame(df_test,columns=test_1.columns)
df_scaled_test.head(1)
#pc = PCA(n_components=30)

pc = pca.transform(df_test)

principalDf_test = pd.DataFrame(data = pc)
principalDf_test.head()
y_predictions=model.predict(principalDf_test)

predictions=np.expm1(y_predictions)
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": predictions

    })

submission.to_csv('submission.csv', index=False)