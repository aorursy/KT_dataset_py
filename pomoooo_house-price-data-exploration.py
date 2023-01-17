import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
combine = [df_train, df_test]
print(df_train.shape)
print(df_test.shape)
fig = plt.figure(figsize=(12,9))
sns.distplot(df_train['SalePrice'], fit=norm)
df_train['SalePrice'].describe()
(mu, sigma) = norm.fit(df_train['SalePrice'])
# mu1 = df_train['SalePrice'].mean()
# std1 = df_train['SalePrice'].std()
print(mu, sigma)
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
fig = plt.figure(figsize=(12,9))
sns.distplot(df_train['SalePrice'], fit=norm)
df_train.head()
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
var = 'LotFrontage'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', xlim=(0,200), ylim=(0,800000))
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
plt.figure(figsize=(10,7))
sns.boxplot(x=var, y="SalePrice", data=data)
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
plt.figure(figsize=(25,7))
sns.boxplot(x=var, y="SalePrice", data=data)
plt.xticks(rotation=90);
# colormap = plt.cm.RdBu
corrmat = df_train.corr()
plt.figure(figsize=(30,30))
# sns.heatmap(df_train.corr(),linewidths=0.1,vmax=1.0, 
#             square=True, cmap=colormap, linecolor='white', annot=True)
sns.heatmap(corrmat, linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
# cm1 = np.corrcoef(df_train[cols].values.T)
plt.figure(figsize=(10,7))
cm = np.corrcoef(df_train[cols].T)  # 用df_train[cols].corr()直接计算是一样的
sns.heatmap(cm, annot=True, yticklabels=cols.values, xticklabels=cols.values)
# sns.heatmap(df_train[cols].corr(), annot=True)
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
# 处理前的缺失值比例
for df in combine:
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/len(df)).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data[missing_data['Total']>0])
for df in combine:
    df['PoolQC'] = df['PoolQC'].fillna("None")
    df['MiscFeature'] = df['MiscFeature'].fillna("None")
    df['Alley'] = df['Alley'].fillna("None")
    df['Fence'] = df['Fence'].fillna("None")
    df['FireplaceQu'] = df['FireplaceQu'].fillna("None")
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        df[col] = df[col].fillna("None")
    for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
        df[col] = df[col].fillna(0)

    for col in ['BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']:
        df[col] = df[col].fillna("None")
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        df[col] = df[col].fillna(0)

    df['MasVnrType'] = df['MasVnrType'].fillna("None")
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
    df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])
    # 'Utilities'字段中取值几乎一样，丢弃
    df.drop(['Utilities'], axis=1, inplace=True)
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
    
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
# 处理后缺失值比例
for df in combine:
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/len(df)).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data[missing_data['Total']>0])
print(df_train.shape)
print(df_test.shape)
# from sklearn.preprocessing import LabelEncoder
# cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
#         'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
#         'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
#         'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass')
# # process columns, apply LabelEncoder to categorical features
# for df in combine:
#     for c in cols:
#         lbl = LabelEncoder()
#         lbl.fit(list(df[c].values)) 
#         #print(lbl.classes_)
#         df[c] = lbl.transform(list(df[c].values))

LotShape_maping = {"Reg":0, "IR1":1, "IR2":2, "IR3":3}
for df in combine:
    df['LotShape'] = df['LotShape'].map(LotShape_maping)
    
LandContour_maping = {"Lvl":0, "Bnk":1, "HLS":2, "Low":3}
for df in combine:
    df['LandContour'] = df['LandContour'].map(LandContour_maping)

LandSlope_maping = {"Gtl":0, "Mod":1, "Sev":2}
for df in combine:
    df['LandSlope'] = df['LandSlope'].map(LandSlope_maping)

maping1 = {"Ex":0, "Gd":1, "TA":2, "Fa":3, "Po":4, "None":5}
for df in combine:
    df['ExterQual'] = df['ExterQual'].map(maping1)
    df['ExterCond'] = df['ExterCond'].map(maping1)
    df['BsmtQual'] = df['BsmtQual'].map(maping1)
    df['BsmtCond'] = df['BsmtCond'].map(maping1)
    df['HeatingQC'] = df['HeatingQC'].map(maping1)
    df['FireplaceQu'] = df['FireplaceQu'].map(maping1)
    df['GarageQual'] = df['GarageQual'].map(maping1)
    df['GarageCond'] = df['GarageCond'].map(maping1)
    df['KitchenQual'] = df['KitchenQual'].map(maping1)

BsmtExposure_maping = {"Gd":0, "Av":1, "Mn":2, "No":3, "None":4}
for df in combine:
    df['BsmtExposure'] = df['BsmtExposure'].map(BsmtExposure_maping)
    
BsmtFinType1_maping = {"GLQ":0, "ALQ":1, "BLQ":2, "Rec":3, "LwQ":4, "Unf":5, "None":6}
for df in combine:
    df['BsmtFinType1'] = df['BsmtFinType1'].map(BsmtFinType1_maping)
    df['BsmtFinType2'] = df['BsmtFinType2'].map(BsmtFinType1_maping)
    
CentralAir_maping = {"N":0, "Y":1}
for df in combine:
    df['CentralAir'] = df['CentralAir'].map(CentralAir_maping)
    
GarageFinish_maping = {"Fin":0, "RFn":1, "Unf":1, "None":1}
for df in combine:
    df['GarageFinish'] = df['GarageFinish'].map(GarageFinish_maping)
    
PavedDrive_maping = {"Y":0, "P":1, "N":1}
for df in combine:
    df['PavedDrive'] = df['PavedDrive'].map(PavedDrive_maping)

PoolQC_maping = {"Ex":0, "Gd":1, "TA":2, "Fa":3, "None":4}
for df in combine:
    df['PoolQC'] = df['PoolQC'].map(PoolQC_maping)

Fence_maping = {"GdPrv":0, "MnPrv":1, "GdWo":2, "MnWw":3, "None":4}
for df in combine:
    df['Fence'] = df['Fence'].map(Fence_maping)

Functional_maping = {"Typ":0, "Min1":1, "Min2":2, "Mod":3, "Maj1":4, "Maj2":5, "Sev":6, "Sal":7}
for df in combine:
    df['Functional'] = df['Functional'].map(Functional_maping)
for df in combine:
    df['MSSubClass'] = df['MSSubClass'].apply(str)
for df in combine:
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
y_train = df_train['SalePrice']
df_train.drop(['SalePrice'], axis=1, inplace=True)
all_data = pd.concat([df_train, df_test], axis=0,ignore_index=True)
print(all_data.shape)
# for df in combine:
#     df = pd.get_dummies(df)
#     print(df.shape)
#     print(df.columns)
all_data = pd.get_dummies(all_data)
print(all_data.shape)
x_train = all_data[all_data['Id']<1461]
x_train.drop(['Id'] ,axis=1, inplace=True)
x_test = all_data[all_data['Id']>=1461]
x_test_Id = x_test.Id
x_test.drop(['Id'],axis=1, inplace=True)
y_train.head()
# import xgboost as xgb
# from sklearn.model_selection import KFold, cross_val_score,cross_val_predict
# model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
#                              learning_rate=0.05, max_depth=3, 
#                              min_child_weight=1.7817, n_estimators=2200,
#                              reg_alpha=0.4640, reg_lambda=0.8571,
#                              subsample=0.5213, silent=1,
#                              random_state =7, nthread = -1)
# n_folds = 5
# kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
# rmse= np.sqrt(-cross_val_score(model_xgb, x_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
# print(rmse.mean())

# model_xgb.fit(x_train, y_train)
# y_pred_xbg = np.expm1(model_xgb.predict(x_test))
# # rmse = model_xgb.score()
# print(prediction)
# from sklearn import linear_model

# clf = linear_model.Lasso(alpha=0.1, max_iter=600000)
# n_folds = 5
# kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
# rmse= np.sqrt(-cross_val_score(clf, x_train, y_train, scoring="neg_mean_squared_error", cv = kf))
# print(rmse.mean())
# # print(rmse.mean())

# clf.fit(x_train, y_train)
# y_pred_lasso = np.expm1(clf.predict(x_test))

# print(y_pred)
# my_submission = pd.DataFrame({'Id': x_test_Id, 'SalePrice': y_pred})
# my_submission.to_csv('submission.csv', index=False)
# print(y_pred)
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

from sklearn.cross_validation import KFold

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, params=None):
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
ntrain = x_train.shape[0]
ntest = x_test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED) 

# 获取通过第一层模型产生新特征
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    # 将训练集分成5份，4份用作训练，1份用作预测，共进行5次
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr) # 4/5训练集用作训练

        oof_train[test_index] = clf.predict(x_te)  # 1/5训练集用作预测
        oof_test_skf[i, :] = clf.predict(x_test)   # 对测试集进行预测

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

la_params = {'alpha': 0.1, 'max_iter': 600000}
en_params = {'alpha': 0.0005, 'l1_ratio': .9, 'random_state': 3}
kr_params = {'alpha': 0.6, 'kernel': 'polynomial', 'degree': 2, 'coef0': 2.5}
gbr_params = {'n_estimators': 3000, 'learning_rate': 0.05,
               'max_depth': 4, 'max_features': 'sqrt',
               'min_samples_leaf': 15, 'min_samples_split': 10, 
               'loss': 'huber', 'random_state': 5}
xbgr_params = {'colsample_bytree': 0.4603, 'gamma': 0.0468, 
                 'learning_rate': 0.05, 'max_depth': 3, 
                 'min_child_weight': 1.7817, 'n_estimators': 2200,
                 'reg_alpha': 0.4640, 'reg_lambda': 0.8571,
                 'subsample': 0.5213, 'silent': 1,
                 'random_state': 7, 'nthread': -1}

la = SklearnHelper(clf=Lasso, params=la_params)
en = SklearnHelper(clf=ElasticNet, params=en_params)
kr = SklearnHelper(clf=KernelRidge, params=kr_params)
gbr = SklearnHelper(clf=GradientBoostingRegressor, params=gbr_params)
xbgr = SklearnHelper(clf=xgb.XGBRegressor, params=xbgr_params)


# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = y_train.ravel()
x_train = x_train.values # Creates an array of the train data
x_test = x_test.values # Creats an array of the test data

# Create our OOF train and test predictions. These base results will be used as new features
la_oof_train, la_oof_test = get_oof(la, x_train, y_train, x_test) # Extra Trees
en_oof_train, en_oof_test = get_oof(en,x_train, y_train, x_test) # Random Forest
kr_oof_train, kr_oof_test = get_oof(kr, x_train, y_train, x_test) # AdaBoost 
gbr_oof_train, gbr_oof_test = get_oof(gbr,x_train, y_train, x_test) # Gradient Boost
xbgr_oof_train, xbgr_oof_test = get_oof(xbgr,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")
import xgboost as xgb

x_train2 = np.concatenate((la_oof_train, en_oof_train, kr_oof_train, gbr_oof_train, xbgr_oof_train), axis=1)
x_test2 = np.concatenate((la_oof_test, en_oof_test, kr_oof_test, gbr_oof_test, xbgr_oof_test), axis=1)

gbm = xgb.XGBRegressor(
    colsample_bytree=0.4603, gamma=0.0468, 
     learning_rate=0.05, max_depth=3, 
     min_child_weight=1.7817, n_estimators=2200,
     reg_alpha=0.4640, reg_lambda=0.8571,
     subsample=0.5213, silent=1,
     random_state =7, nthread = -1).fit(x_train2, y_train)

predictions = np.expm1(gbm.predict(x_test2))
# prediction = 0.6 * y_pred_xbg + 0.4 * y_pred_lasso
# prediction = y_pred_xbg
my_submission = pd.DataFrame({'Id':x_test_Id, 'SalePrice':predictions})
my_submission.to_csv('submission.csv', index=False)
