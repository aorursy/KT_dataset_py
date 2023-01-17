#數據處理及可視化

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



#演算法

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from xgboost.sklearn import XGBRegressor

from sklearn.linear_model import RidgeCV 



from sklearn.model_selection import cross_val_score
!ls ../input/house-prices-advanced-regression-techniques/
#Obtain the data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

corr_matrix['SalePrice']
#資料探索

corr_matrix = train.corr()

corr_matrix['SalePrice'].sort_values(ascending=False)
f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_matrix, vmax=.8, square=True)
# neg_attr=['BsmtFinSF2','BsmtHalfBath','MiscVal','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr']

# train.drop(neg_attr,inplace=True)

# test.drop(neg_attr,inplace=True)
# train.info()
# SalePrice distribution

sns.distplot(train['SalePrice'], color="r", kde=False)

plt.title("Distribution of Sale Price")

plt.ylabel("Number of Occurences")

plt.xlabel("Sale Price");
#右偏分布，調整回常態分佈

sns.distplot(np.log1p(train['SalePrice']))

y = train['SalePrice']

y = np.log1p(y)
###NA's
#拼接數據並將缺失值數量可視化

#DataFrame中各列空值的占比

train['train_or_test']='train'

test['train_or_test']='test'

all_data = pd.concat((train.drop(['SalePrice'],axis=1), test))

nulls = all_data.isnull().sum()

nullcols = nulls[(nulls != 0)]

dtypes = all_data.dtypes

dtypes2 = dtypes.loc[(nulls != 0)]

info = pd.concat([nullcols,dtypes2],axis=1).sort_values(by=0,ascending=False)

print(info)

print("There are", len(nullcols), "columns with missing values")

#針對類別資料不能填None的，填充值

#家庭功用，填充典型功用 #電力系統，標準型 #房子外牆質地和付款方式，填最多人使用

all_data['Functional'] = all_data['Functional'].fillna('Typ')

all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr")

all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA")

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
#特別的案例

#if PoolArea>0,it exists PoolQC also

#使各類分佈一致，填fair一般的

all_data[all_data['PoolArea'] > 0 & all_data['PoolQC'].isnull()][['PoolArea','PoolQC']]

all_data.loc[960, 'PoolQC'] = 'Fa'

all_data.loc[1043, 'PoolQC'] = 'Fa'
#有倉庫型態但沒有建築時間

all_data[(all_data['GarageType'].notnull()) & (all_data['GarageYrBlt'].isnull())][["Neighborhood", "YearBuilt", "YearRemodAdd", "GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond"]]



# all_data['GarageFinish'].value_counts()

#數值型態填中位數，類別型態填眾數

all_data.loc[666,'GarageYrBlt'] = all_data['GarageYrBlt'].median()

all_data.loc[1116,'GarageYrBlt'] = all_data['GarageYrBlt'].median()



all_data.loc[666, 'GarageFinish'] = all_data['GarageFinish'].mode()[0]

all_data.loc[1116, 'GarageFinish'] = all_data['GarageFinish'].mode()[0]



all_data.loc[1116, 'GarageCars'] = all_data['GarageCars'].median()



all_data.loc[666, 'GarageArea'] = all_data['GarageArea'].median()

all_data.loc[1116, 'GarageArea'] = all_data['GarageArea'].median()



all_data.loc[666, 'GarageQual'] = all_data['GarageQual'].mode()[0]

all_data.loc[1116, 'GarageQual'] = all_data['GarageQual'].mode()[0]



all_data.loc[666, 'GarageCond'] = all_data['GarageCond'].mode()[0]

all_data.loc[1116, 'GarageCond'] = all_data['GarageCond'].mode()[0]

#地下室系列

basement_columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

                   'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',

                   'TotalBsmtSF']

tempdf = all_data[basement_columns]

tempdfnulls = tempdf[tempdf.isnull().any(axis=1)]
tempdfnulls[(tempdfnulls.isnull()).sum(axis=1) < 5]
#if BsmtFinType=Unf ,BsmtFinSF(finished square feet)!=0

#前面BsmtCond=TA(有點潮濕的)和 BsmtExposure(瀑光)=No 有關，所以照著填

all_data.loc[948,'BsmtExposure'] = 'No'

all_data.loc[27,'BsmtExposure'] = 'No'

all_data.loc[888,'BsmtExposure'] = 'No'



all_data.loc[725,'BsmtCond'] = 'TA'

all_data.loc[580,'BsmtCond'] = 'TA' #BsmtExposure='Mn',BsmtCond='TA'比較多

all_data.loc[580,'BsmtCond'] = 'TA' #BsmtExposure=Av the most of BsmtCond is TA







#impute the MSZoning by groupby MSSubClass's mode()[0]

# all_data[all_data['MSZoning'].isnull()]

subclass_group = all_data.groupby('MSSubClass')

Zoning_modes = subclass_group['MSZoning'].apply(lambda x:x.mode()[0])

Zoning_modes
all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x:x.fillna(x.mode()[0]))
#impute the NA of category = None

objects = []

for i in all_data.columns:

    if all_data[i].dtype == object:

        objects.append(i)

all_data.update(all_data[objects].fillna('None'))



nulls = np.sum(all_data.isnull())

nullcols = nulls[nulls != 0]

dtypes = all_data.dtypes

dtypes2 = dtypes[nulls != 0]

info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0,ascending=False)

print(info)

print("There are", len(nullcols), "columns with missing values")
neighborhood_group = all_data.groupby('Neighborhood')

lot_medians = neighborhood_group['LotFrontage'].median()

display(lot_medians)

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
#Garage 正常

all_data[(all_data['GarageYrBlt'].isnull()) & all_data['GarageArea']>0]

#MasVnrArea 正常

all_data[(all_data['MasVnrArea'].isnull()) & (all_data['MasVnrType']!='None')][['MasVnrArea','MasVnrType']]
#Fill the numerical col

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in all_data.columns:

    if all_data[i].dtype in numeric_dtypes:

        numerics.append(i)

all_data.update(all_data[numerics].fillna(0))

nulls = np.sum(all_data.isnull())

nullcols = nulls[nulls !=0 ]

dtypes = all_data.dtypes

dtypes2 = dtypes[nulls !=0]

info = pd.concat([nullcols, dtypes2],axis=1).sort_values(by=0,ascending=False)

print(info)

print("There are", len(nullcols), "columns with missing values")
#incorrect values

all_data.describe()

all_data[all_data['GarageYrBlt'] == 2207]

all_data.loc[1132,'GarageYrBlt'] = 2007
#There are features that are read in as numericals but are actually objects

factors = ['MSSubClass']

 

for i in factors:

    all_data.update(all_data[i].astype('str'))
#分割資料集

train = all_data[all_data['train_or_test']=='train']

train.drop('train_or_test', axis=1,inplace=True)

test = all_data[all_data['train_or_test']=='test']

test.drop('train_or_test', axis=1,inplace=True)

train
#針對train創造新的特徵

#簡化存在的特徵

# train["SimplOverallQual"] = train.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad

#                                                        4 : 2, 5 : 2, 6 : 2, # average

#                                                        7 : 3, 8 : 3, 9 : 3, 10 : 3 # good

#                                                       })

#合併特徵

# Overall quality of the house

# train["OverallGrade"] = train["OverallQual"] * train["OverallCond"]
#與價格關連性最大的3個特徵做多項式

# 3* Polynomials on the top 10 existing features

# train["OverallQual-s2"] = train["OverallQual"] ** 2

# train["OverallQual-s3"] = train["OverallQual"] ** 3

# train["OverallQual-Sq"] = np.sqrt(train["OverallQual"])



# train["GrLivArea-2"] = train["GrLivArea"] ** 2

# train["GrLivArea-3"] = train["GrLivArea"] ** 3

# train["GrLivArea-Sq"] = np.sqrt(train["GrLivArea"])

# train["SimplOverallQual-s2"] = train["SimplOverallQual"] ** 2

# train["SimplOverallQual-s3"] = train["SimplOverallQual"] ** 3
#探索離群值並刪除

train_dummies = pd.get_dummies(pd.concat((train.drop(["Id"], axis=1), test.drop(["Id"], axis=1)), axis=0)).iloc[: train.shape[0]]

test_dummies = pd.get_dummies(pd.concat((train.drop(["Id"], axis=1), test.drop(["Id"], axis=1)), axis=0)).iloc[train.shape[0]:]
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

ridge.fit(train_dummies, y)

alpha = ridge.alpha_

alpha
rr = Ridge(alpha=10)

rr.fit(train_dummies, y)

np.sqrt(-cross_val_score(rr, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
y_pred = rr.predict(train_dummies)

resid = y - y_pred

mean_resid = resid.mean()

std_resid = resid.std()

z = (resid - mean_resid) / std_resid

z = np.array(z)

outliers1 = np.where(abs(z) > abs(z).std() * 3)[0]

outliers1
plt.figure(figsize=(6, 6))

plt.scatter(y, y_pred)

plt.scatter(y.iloc[outliers1], y_pred[outliers1])

plt.plot(range(10, 15), range(10, 15), color="red")

er = ElasticNet()

er.fit(train_dummies, y)

np.sqrt(-cross_val_score(rr, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
y_pred = er.predict(train_dummies)

resid = y - y_pred

mean_resid = resid.mean()

std_resid = resid.std()

z = (resid - mean_resid) / std_resid

z = np.array(z)

outliers2 = np.where(abs(z) > abs(z).std() * 3)[0]

outliers2

plt.figure(figsize=(6, 6))

plt.scatter(y, y_pred)

plt.scatter(y.iloc[outliers2], y_pred[outliers2])

plt.plot(range(10, 15), range(10, 15), color="red")

outliers = []

for i in outliers1:

    for j in outliers2:

        if i == j:

            outliers.append(i)

outliers
#刪除離群值

train = train.drop(outliers)

y = y.drop(outliers)
train_dummies = pd.get_dummies(pd.concat((train.drop(["Id"], axis=1), test.drop(["Id"], axis=1)), axis=0)).iloc[: train.shape[0]]

test_dummies = pd.get_dummies(pd.concat((train.drop(["Id"], axis=1), test.drop(["Id"], axis=1)), axis=0)).iloc[train.shape[0]:]
xgbr = XGBRegressor(max_depth=5, n_estimators=400)

xgbr.fit(train_dummies, y)

np.sqrt(-cross_val_score(xgbr, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
lsr = Lasso(alpha=0.00047)

lsr.fit(train_dummies, y)

np.sqrt(-cross_val_score(lsr, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
rr = Ridge(alpha=10)

rr.fit(train_dummies, y)

np.sqrt(-cross_val_score(rr, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
train_predict =  0.4 * xgbr.predict(train_dummies)+ 0.3 * lsr.predict(train_dummies) + 0.3 * rr.predict(train_dummies)

train_predict
sample_submission.shape

test_dummies
test_predict =  0.4 * xgbr.predict(test_dummies)+ 0.3 * lsr.predict(test_dummies) + 0.3 * rr.predict(test_dummies)

test_predict = np.array(test_predict)

sample_submission["SalePrice"] = np.exp(test_predict)-1

sample_submission.to_csv('submission.csv',index=False)