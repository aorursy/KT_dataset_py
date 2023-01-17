import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
from sklearn import ensemble, metrics
from sklearn import linear_model, preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
#from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.shape, test.shape)
train.head()
sns.distplot(train['SalePrice'], fit=norm)
mu, sigma = norm.fit(train['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
print('mu={:2f}, sigma={:.2f}'.format(mu,sigma))
print('skew={}'.format(skew(np.log1p(train['SalePrice']))))
train.describe(include=['number']).loc[["mean","min","max"]]
# ref. https://github.com/shan4224/Kaggle_House_Prices
n = train.select_dtypes(include=object)
for c in n.columns:
    print(c, ':  ', train[c].unique())
fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(4 * 4, 3 * 4), sharey=True)
axes = np.ravel(axes)
cols = ['OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual','BsmtCond','GarageQual','GarageCond',
        'MSSubClass','MSZoning','Neighborhood','BldgType','HouseStyle','Heating','Electrical','SaleType']
for i, c in zip(np.arange(len(axes)), cols):
    sns.boxplot(x=c, y='SalePrice', data=train, ax=axes[i])
all_data = train.append(test, sort=False).reset_index(drop=True)
all_data.shape
# to categorical feature
cols = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for c in cols:
    all_data[c] = all_data[c].astype(str)

# encode quality
# Ex(Excellent), Gd（Good）, TA（Typical/Average）, Fa（Fair）, Po（Poor）
cols = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
for c in cols:
    all_data[c].fillna(0, inplace=True)
    all_data[c].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
def pair_features_to_dummies(df, col1, col2, prefix):
    d_1 = pd.get_dummies(df[col1].astype(str), prefix=prefix)
    d_2 = pd.get_dummies(df[col2].astype(str), prefix=prefix)
    for c in list(set(list(d_1.columns) + list(d_2.columns))):
        if not c in d_1.columns: d_1[c] = 0
        if not c in d_2.columns: d_2[c] = 0
    return (d_1 + d_2).clip(0, 1)

cond = pair_features_to_dummies(all_data,'Condition1','Condition2','Condition')
exterior = pair_features_to_dummies(all_data,'Exterior1st','Exterior2nd','Exterior')
bsmtftype = pair_features_to_dummies(all_data,'BsmtFinType1','BsmtFinType2','BsmtFinType') 

all_data = pd.concat([all_data, cond, exterior, bsmtftype], axis=1)
all_data.drop(['Condition1','Condition2', 'Exterior1st','Exterior2nd','BsmtFinType1','BsmtFinType2'], axis=1, inplace=True)
all_data.head()
n = all_data.drop('SalePrice', axis=1).loc[:,all_data.isnull().any()].isnull().sum()
print(n.sort_values(ascending=False))
# fillna
for c in ['MiscFeature', 'Alley', 'Fence']:
    all_data[c].fillna('None', inplace=True)
    
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

all_data.loc[all_data.GarageYrBlt.isnull(),'GarageYrBlt'] = all_data.loc[all_data.GarageYrBlt.isnull(),'YearBuilt']
all_data['GarageType'].fillna('None', inplace=True)
all_data['GarageFinish'].fillna(0, inplace=True)

for c in ['GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    all_data[c].fillna(0, inplace=True)
for i, t in all_data.loc[:, all_data.columns != 'SalePrice'].dtypes.iteritems():
    if t == object:
        all_data[i].fillna(all_data[i].mode()[0], inplace=True)
        all_data[i] = LabelEncoder().fit_transform(all_data[i].astype(str))
    else:
        all_data[i].fillna(all_data[i].median(), inplace=True)
all_data['OverallQualCond'] = all_data['OverallQual'] * all_data['OverallCond']
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['Interaction'] = all_data['TotalSF'] * all_data['OverallQual']
train = all_data[all_data['SalePrice'].notnull()]
test = all_data[all_data['SalePrice'].isnull()].drop('SalePrice', axis=1)
corr = train.corr()
corr_price_abs = pd.DataFrame(corr.SalePrice.abs().sort_values(ascending=False))
for i, c in zip(np.arange(len(corr_price_abs)), corr_price_abs.index): print(i, ':', c)
#train[corr_price_abs.index].describe().loc[['min','max']]
fig, axes = plt.subplots(ncols=4, nrows=9, figsize=(20, 30))
axes = np.ravel(axes)
col_name = corr_price_abs[1:].index
for i in range(36):
    train.plot.scatter(ax=axes[i], x=col_name[i], y='SalePrice', c='OverallQual', sharey=True, colorbar=False, cmap='GnBu')
train = train[train['TotalSF'] < 6000]
train = train[train['TotalBsmtSF'] < 4000]
train = train[train['SalePrice'] < 700000]
_ = '''
from scipy.stats import skew

#log transform skewed numeric features:
numeric_feats = all_data[test.columns].dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

#all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
from scipy.special import boxcox1p
lam = 0.15
for feat in skewed_feats:
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
all_data = pd.get_dummies(all_data)
'''
X_train = train.drop(['SalePrice','Id'], axis=1)
Y_train = train['SalePrice']
X_test  = test.drop(['Id'], axis=1)

print(X_train.shape, Y_train.shape, X_test.shape)
def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse.mean()
reg = xgb.XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.05, subsample=0.6, colsample_bytree=0.6)

score = rmse_cv(reg, X_train, np.log1p(Y_train))
print(score.mean())

#col = X_test.columns
#feature_imp = pd.DataFrame(reg.feature_importances_, index=col, columns=["importance"])
#print(feature_imp.sort_values("importance", ascending=False).head(30))
Y_train_s = np.log1p(Y_train)

reg_1 = xgb.XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.05, subsample=0.6, colsample_bytree=0.6)
reg_1.fit(X_train, Y_train_s)
pred_1 = np.expm1(reg_1.predict(X_test))

scaler = preprocessing.RobustScaler(); #StandardScaler
scaler.fit(X_train)

reg_2 = linear_model.ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=100000)
reg_2.fit(scaler.transform(X_train), Y_train_s)
pred_2 = np.expm1(reg_2.predict(scaler.transform(X_test)))

reg_3 = linear_model.Ridge(alpha=60)
reg_3.fit(scaler.transform(X_train), Y_train_s)
pred_3 = np.expm1(reg_3.predict(scaler.transform(X_test)))

reg_4 = linear_model.BayesianRidge()
reg_4.fit(scaler.transform(X_train), Y_train_s)
pred_4 = np.expm1(reg_4.predict(scaler.transform(X_test)))

result = pred_1 * 0.5 + pred_2 * 0.2 + pred_3 * 0.2 + pred_4 * 0.1
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": result
})
submission.to_csv("submission.csv", index=False)
submission.head(10)
