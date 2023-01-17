# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #seaborn for data visualisation
import matplotlib as mpl #matplotlib for data visualisation
import matplotlib.pyplot as plt
#import missingno as msno
import scipy.stats as st

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data['PoolQC'].value_counts()
print("*" * 50)
print("Train shape is : ", train_data.shape)
print("Test shape is :", test_data.shape)
print("*" * 50)
train_data.describe(include='all')
test_data.describe(include='all')
def dataHasNaN(df):
    HasNaN = False
    if df.count().min()!=df.shape[0]:
        HasNaN = True
    return HasNaN    

def is_categorical(array_like):
    return array_like.dtype.name == 'object'
#Target variable distribution
sns.distplot(train_data['SalePrice'])
print("Train data['SalePrice']")
print("*" * 50)
print("Skew     : ", train_data['SalePrice'].skew())
print("Kurtosis : ", train_data['SalePrice'].kurtosis())
print("*" * 50)
train_data.columns
#GrLivArea: Above grade (ground) living area square feet
train_data.plot.scatter(x='GrLivArea', y='SalePrice')
#TotalBsmtSF: Total square feet of basement area
train_data.plot.scatter(x='TotalBsmtSF', y='SalePrice')
#OverallQual: Rates the overall material and finish of the house
f, ax = plt.subplots(figsize=(8, 6))
ax = sns.boxplot(x='OverallQual', y='SalePrice', data=train_data)
train_data.plot.scatter(x='YearBuilt', y='SalePrice')
corr = abs(train_data.corr())
f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, vmax=0.8, square=True)
cols = corr['SalePrice'].nlargest(10).index.tolist()
corr = abs(train_data[cols].corr())
sns.heatmap(corr, annot=True)
collinear_indices = ['TotRmsAbvGrd', '1stFlrSF', 'GarageCars']
cols = [index for index in cols if index not in collinear_indices]
sns.pairplot(train_data[cols])
plt.figure(3); plt.title('Log Normal')
sns.distplot(train_data['SalePrice'], kde=False, fit=st.lognorm)
res = st.probplot(train_data['SalePrice'], plot=plt)
#Normalising the target variable by taking its log.
#train_data['SalePrice']
target = np.log(train_data['SalePrice'])
#We can see in the plot that the target variable distribution now approaches a normal distribution.
sns.distplot(target, fit=st.norm)
fig = plt.figure()
res = st.probplot(target, plot=plt)
print("Log SalePrice skew and kurtosis :")
print("*"*50)
print("Skew     : ", target.skew())
print("Kurtosis : ", target.kurtosis())
print("*"*50)
sns.distplot(train_data['GrLivArea'], fit=st.norm)
fig = plt.figure()
res = st.probplot(train_data['GrLivArea'], plot=plt)
sns.distplot(train_data['TotalBsmtSF'], fit=st.norm)
fig = plt.figure()
res = st.probplot(train_data['TotalBsmtSF'], plot=plt)
#Whether data has NaN
dataHasNaN(train_data), dataHasNaN(test_data)
#Keeping the ids for final output

train_data_id = train_data['Id']
test_data_id = test_data['Id']

train_data.drop(['Id'], axis=1, inplace=True)
test_data.drop(['Id'], axis=1, inplace=True)
#Concatenate the frames for data preprocessing

frames = [train_data.drop(['SalePrice'], axis =1), test_data]
df = pd.concat(frames)
df.reset_index(drop=True, inplace=True)
df['MSSubClass'] = df['MSSubClass'].astype(dtype='object')
#We would want to avoid multicollinearity. Hence, if two variables strongly correlate to each other 
#we would want to remove one of them.

df.drop(collinear_indices, axis=1, inplace=True)
dataHasNaN(df)
na_df = (df.isnull().sum()/len(df)) * 100
na_df = na_df[na_df!=0].sort_values(ascending=False)
na_df = pd.DataFrame({'Na ratio': na_df})
na_df.head()
na_df.index
df['PoolQC'] = df['PoolQC'].fillna("None")
df['MiscFeature'] = df['MiscFeature'].fillna("None")
df['Alley'] = df['Alley'].fillna("None")
df['Fence'] = df['Fence'].fillna("None")
df['FireplaceQu'] = df['FireplaceQu'].fillna("None")
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
                        lambda LotFrontage_Grp : LotFrontage_Grp.fillna(
                            LotFrontage_Grp.median()
                        )
                    )
#df.groupby('Neighborhood')['LotFrontage'].get_group('Blmngtn').median()
for attr in ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType']:
    df[attr] = df[attr].fillna('None')
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
for attr in ['BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']:
    df[attr] = df[attr].fillna("None")
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
df['BsmtFullBath'] = df['BsmtHalfBath'] + 0.5 * df['BsmtFullBath']
df.drop('BsmtHalfBath', axis=1, inplace=True)
df['Functional'] = df['Functional'].fillna('Typ')
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)
df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['Utilities'].value_counts()
df.drop('Utilities', axis=1, inplace=True)
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['GarageArea'] = df['GarageArea'].fillna(0)
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
dataHasNaN(df)
df['GarageQual'].value_counts()
cat_col = [ elem for elem in df.columns.tolist() if is_categorical(df[elem])==True ]
num_col = [ elem for elem in df.columns.tolist() if elem not in cat_col ]
len(cat_col)+len(num_col)==len(df.columns)
skew_feats_series = abs(df[num_col].skew()).sort_values(ascending=False)
skewness = pd.DataFrame({'skew' : skew_feats_series}).dropna()

skewness.head(10)
dataHasNaN(skewness)
skewness = skewness[skewness['skew'] > 0.75]
skewness
len(skewness)
skewness.index
from scipy.stats import boxcox_normplot
lm,ppcc = boxcox_normplot(df['GrLivArea'], la=-5, lb=5)
plt.plot(lm, ppcc)
from scipy.stats import boxcox
skewed_features = skewness.index
tot_lm = 0
for feat in skewed_features:
    op, lm = boxcox(df[feat])
    tot_lm += lm
#df[skewness.index] = np.log1p(df[skewness.index])
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    df[feat] = boxcox1p(df[feat], lam)
df.head(10)
df.describe(include='all')
scale_1 = {'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
df['BsmtQual'] = df['BsmtQual'].map(scale_1)
df['BsmtCond'] = df['BsmtCond'].map(scale_1)
df['BsmtExposure'] = df['BsmtExposure'].map({'None':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4})
df['HeatingQC'] = df['HeatingQC'].map(scale_1)
df['KitchenQual'] = df['KitchenQual'].map(scale_1)
df['FireplaceQu'] = df['FireplaceQu'].map(scale_1)
df['GarageQual'] = df['GarageQual'].map(scale_1)
df['GarageCond'] = df['GarageCond'].map(scale_1)
df['PoolQC'] = df['PoolQC'].map(scale_1)

scale_2 = {'None':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}
df['BsmtFinType1'] = df['BsmtFinType1'].map(scale_2)
df['BsmtFinType2'] = df['BsmtFinType2'].map(scale_2)

'''
df['LotShape'] = df['LotShape'].map({'Reg':1, 'IR1':2, 'IR2':3, 'IR3':4})
df['Utilities'] = df['Utilities'].map({'AllPub':1, 'NoSeWa':2, 'No value':3})
df['LotConfig'] = df['LotConfig'].map({'Inside':1, 'Corner':2, 'CulDSac':3, 'FR2':4, 'FR3':5})
df['BldgType'] = df['BldgType'].map({'1Fam':1, '2fmCon':2, 'Duplex':3, 'TwnhsE':4, 'Twnhs':5})
df['HouseStyle'] = df['HouseStyle'].map({'1Story':1, '1.5Unf':2, '1.5Fin':3, '2Story':4, '2.5Unf':5, '2.5Fin':6})
df['CentralAir'] = df['CentralAir'].map({'Y':0, 'N':1})
df['Functional'] = df['Functional'].map({'No value':0, 'Typ':1, 'Min1':2, 'Min2':3, 'Mod':4, 'Maj1':5, 'Maj2':6, 'Sev':7, 'Sal':8})
df['GarageType'] = df['GarageType'].map({'No value':0, 'Detchd':1, 'CarPort':2, 'BuiltIn':3, 'Basment':4, 'Attchd':5, '2Types':6})
df['GarageFinish'] = df['GarageFinish'].map({'No value':0, 'Unf':1, 'RFn':2, 'Fin':3})
df['PavedDrive'] = df['PavedDr].map({'N':0, 'P':1, 'Y':2})
df['Fence'] = df['Fence'].map({'No value':0, 'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4})
'''
for col in df.columns:
    if df[col].count().min()!=df.shape[0]:
        print(col)
dataHasNaN(df)
df = pd.get_dummies(df, drop_first=True)
X_train = df.iloc[:train_data.shape[0]]
y_train = pd.DataFrame(target, columns=['SalePrice'])
X_test = df.iloc[train_data.shape[0]:]
plt.scatter(train_data['GrLivArea'], train_data['SalePrice'])
plt.scatter(X_train['GrLivArea'], y_train['SalePrice'])
plt.scatter(train_data['LotArea'], train_data['SalePrice'])
plt.scatter(X_train['LotArea'], y_train['SalePrice'])
plt.scatter(train_data['TotalBsmtSF'], train_data['SalePrice'])
plt.scatter(X_train['TotalBsmtSF'], y_train['SalePrice'])
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train.loc[:,num_col] = sc_X.fit_transform(X_train.loc[:, num_col])
X_test.loc[:, num_col] = sc_X.transform(X_test.loc[:, num_col])
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score

def rmse_cv(model):
    kf = KFold(n_splits=10,shuffle=True, random_state=333 )
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train['SalePrice'].ravel(), scoring="neg_mean_squared_error", cv=kf))
    return(rmse)
L1_model = Lasso(alpha = 0.0005, random_state=1)
L1_pipe = make_pipeline(RobustScaler(), L1_model)
L1_pipe.fit(X_train, y_train)
L1_pipe.score(X_train, y_train)
rmse_cv(L1_pipe).mean(), rmse_cv(L1_pipe).std()
coef_L1 = pd.Series(L1_model.coef_, index = df.columns)
print("Total coefficient removed : ", sum(coef_L1==0))

imp_coef_L1 = pd.concat([coef_L1.sort_values().head(10), 
                  coef_L1.sort_values().tail(10)])

#plt.figure(figsize=(20,10))
imp_coef_L1.plot(kind='barh')
y_pred_L1 = np.exp(L1_pipe.predict(X_test)).ravel()
L2_model = Ridge(alpha = 0.0005, random_state=1)
L2_pipe = make_pipeline(RobustScaler(), L2_model )
L2_pipe.fit(X_train, y_train)
L2_pipe.score(X_train, y_train)
rmse_cv(L2_pipe).mean(), rmse_cv(L2_pipe).std()
L2_model.coef_ = L2_model.coef_.reshape(L2_model.coef_.shape[1])
coef_L2 = pd.Series(L2_model.coef_, index = df.columns)
print("Total coefficient removed : ", sum(coef_L2==0))

imp_coef_L2 = pd.concat([coef_L2.sort_values().head(10), 
                  coef_L2.sort_values().tail(10)])

#plt.figure(figsize=(20,10))
imp_coef_L2.plot(kind='barh')
y_pred_L2 = np.exp(L2_pipe.predict(X_test)).ravel()
ENet_model = ElasticNet(alpha = 0.0005, l1_ratio=0.1, random_state = 102)
ENet_pipe = make_pipeline(RobustScaler(), ENet_model )
ENet_pipe.fit(X_train, y_train)
ENet_pipe.score(X_train, y_train)
rmse_cv(ENet_pipe).mean(), rmse_cv(ENet_pipe).std()
coef_ENet = pd.Series(ENet_model.coef_, index = df.columns)
print("Total coefficient removed : ", sum(coef_ENet==0))

imp_coef_ENet = pd.concat([coef_ENet.sort_values().head(10), 
                  coef_ENet.sort_values().tail(10)])

imp_coef_ENet.plot(kind='barh')
y_pred_ENet = np.exp(ENet_pipe.predict(X_test)).ravel()
y_pred_L1
y_pred_L2
y_pred_ENet
y_pred = y_pred_ENet
y_pred = pd.DataFrame({'Id': test_data_id, 'SalePrice': y_pred})
y_pred.to_csv("submission.csv", index=False)