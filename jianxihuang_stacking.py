import numpy as np

import pandas as pd



from scipy import stats

from scipy.stats import norm, skew #统计常用函数



# 导入可视化工具

import matplotlib.pyplot as plt

import matplotlib as mpl

import matplotlib.pylab as pylab

import seaborn as sns

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
data_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

data_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

data_test_ID = data_test['Id']

# 观察前5行

data_train.head()
print("训练集的大小为：{}\n测试集的大小为：{}".format(data_train.shape, data_test.shape))
data_train.info()
# DataFrame自带索引，这里去掉'Id'一列

data_train = data_train.drop(['Id'], axis=1)

data_test = data_test.drop(['Id'], axis=1)

data_train.head()
target = 'SalePrice'

data_cleaner = [data_train, data_test]
plt.figure(figsize=(8,6), dpi=80)

plt.scatter(x = data_train['GrLivArea'], y = data_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
# 如图示，右下方出现两个大面积低房价的样本，这里进行清除

data_train = data_train.drop(data_train[(data_train['GrLivArea']>4000) & (data_train['SalePrice']<300000)].index)



# 再次检查

plt.figure(figsize=(8,6), dpi=80)

plt.scatter(data_train['GrLivArea'], data_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
# 查看训练集大小，如期少了2条记录

data_train.shape
# 获取概率分布参数μ（mu）和σ（sigma）

(mu, sigma) = norm.fit(data_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



# 画出概率分布图

sns.distplot(data_train[target] , fit=norm)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



# 画出正态概率图QQ-plot

fig = plt.figure()

stats.probplot(data_train['SalePrice'], plot=plt)

plt.show()
# skewness & Kurtosis

print("Skewness: %f" % data_train[target].skew())

print("Kurtosis: %f" % data_train[target].kurt())
# 对数据进行box cox转换，用numpy log1p进行log(1+x)的变换，产生新的column [SalePrice_trans]

data_train['SalePrice_trans'] = np.log1p(data_train[target])

target_trans = 'SalePrice_trans'

# 画图观察

sns.distplot(data_train[target_trans])

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

# QQ-plot

fig = plt.figure()

stats.probplot(data_train[target_trans], plot=plt)

plt.show()

# skewness & Kurtosis

print("Skewness: %f" % data_train[target_trans].skew())

print("Kurtosis: %f" % data_train[target_trans].kurt())
# 定义目标值，第3部分进行训练时调用

data_target = data_train['SalePrice_trans']
missing_train = pd.DataFrame({'Missing Number': data_train.isnull().sum().sort_values(ascending=False)})

missing_train = missing_train.drop(missing_train[missing_train['Missing Number']==0].index)

missing_test = pd.DataFrame({'Missing Number':data_test.isnull().sum().sort_values(ascending=False)})

missing_test = missing_test.drop(missing_test[missing_test['Missing Number']==0].index)
missing_train
columns1 = ['PoolQC', 'FireplaceQu','GarageQual','GarageCond','FireplaceQu','GarageFinish','Fence','GarageType', 'GarageCars', 'GarageArea',

            'BsmtFinType2','BsmtQual','BsmtCond','BsmtFinType1','MasVnrType','MiscFeature','Alley','Fence','BsmtExposure']

columns2 = ['LotFrontage','MasVnrArea']

columns3 = ['Electrical','MasVnrType', 'MSZoning','KitchenQual','Exterior1st','Exterior2nd','SaleType']
for column1 in columns1:

    data_train[column1] = data_train[column1].fillna("None")

    data_test[column1] = data_test[column1].fillna("None")
# LotFrontage是街道离房子的距离，连续型，其结果与其邻居有关。通过分类后填充同类民居的平均值

data_train['LotFrontage'] = data_train['LotFrontage'].groupby(by=data_train['Neighborhood']).apply(lambda x: x.fillna(x.mean()))

data_train['MasVnrArea'] = data_train['MasVnrArea'].fillna(data_train['MasVnrArea'].median())

data_test['LotFrontage'] = data_test['LotFrontage'].groupby(by=data_test['Neighborhood']).apply(lambda x: x.fillna(x.mean()))

data_test['MasVnrArea'] = data_test['MasVnrArea'].fillna(data_test['MasVnrArea'].median())
# Electrical, MasVnrType是类别型，缺失通过mode填补

for column3 in columns3:

    data_train[column3] = data_train[column3].fillna(data_train[column3].mode()[0])

    data_test[column3] = data_test[column3].fillna(data_train[column3].mode()[0])
# 缺失GarageYrBlt，是因为不存在车库，这里由于年份是数值型，这里用较老年份1920代替

data_train['GarageYrBlt'] = data_train['GarageYrBlt'].fillna(1920)

data_test['GarageYrBlt'] = data_test['GarageYrBlt'].fillna(1920)
# Functional : 提供的数据描述中，表示NA是typical

data_train["Functional"] = data_train["Functional"].fillna("Typ")

data_test["Functional"] = data_test["Functional"].fillna("Typ")
# Utilities : 具体的特征值主要是"AllPub", 除了一个"NoSeWa" 和两个NA。没有太大的预测价值，这里将该特征移除

data_train = data_train.drop(['Utilities'], axis=1)

data_test = data_test.drop(['Utilities'], axis=1)
# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : 缺失的数据是由于没有basement，这里用0填充

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    data_train[col] = data_train[col].fillna(0)

    data_test[col] = data_test[col].fillna(0)
missing_train = pd.DataFrame({'Missing Number': data_train.isnull().sum().sort_values(ascending=False)})

missing_train = missing_train.drop(missing_train[missing_train['Missing Number']==0].index)

missing_test = pd.DataFrame({'Missing Number':data_test.isnull().sum().sort_values(ascending=False)})

missing_test = missing_test.drop(missing_test[missing_test['Missing Number']==0].index)
missing_train
missing_test
data_train['TotalSF'] = data_train['TotalBsmtSF'] + data_train['1stFlrSF'] + data_train['2ndFlrSF']

data_test['TotalSF'] = data_test['TotalBsmtSF'] + data_test['1stFlrSF'] + data_test['2ndFlrSF']
for dataset in data_cleaner:

#MSSubClass

    dataset['MSSubClass'] = dataset['MSSubClass'].astype(str)



#OverallCond

    dataset['OverallCond'] = dataset['OverallCond'].astype(str)



#Year and month sold

    dataset['YrSold'] = dataset['YrSold'].astype(str)

    dataset['MoSold'] = dataset['MoSold'].astype(str)
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

from sklearn.preprocessing import LabelEncoder

for c in cols:

    label = LabelEncoder()

    data_train[c] = label.fit_transform(data_train[c])

    data_test[c] = label.fit_transform(data_test[c])
#查看数据仍是字符串的特征有哪些

data_train.dtypes[data_train.dtypes == 'object'].index
feats_numeric = data_train.dtypes[data_train.dtypes != "object"].index



# 检查train训练集中需要偏差的特征值有哪些

data_skewness = pd.DataFrame({'Skew' :data_train[feats_numeric].apply(lambda x: skew(x)).sort_values(ascending=False)})

data_skewness = data_skewness[abs(data_skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(data_skewness.shape[0]))

data_skewness.head(5)
# 暂时不用最大似然估计求最佳lam值，因为特征中出现很多特征值为0的情况

# def lam_best(y):

#     lam_range = np.linspace(-2,5,100)  # default nums=50

#     for i,lam in enumerate(lam_range):

#         llf = np.zeros(lam_range.shape, dtype=float)

#         llf[i] = stats.boxcox_llf(lam, y) # y>0

#         lam_best = lam_range[llf.argmax()]

#     return(lam_best)



from scipy.special import boxcox1p

feats_skewed = data_skewness.index

lam = 0.15

for feat in feats_skewed:

    #all_data[feat] += 1

    data_train[feat] = boxcox1p(data_train[feat], lam)
# 对test测试集的偏差特征值进行变换

feats_numeric_test = data_test.dtypes[data_test.dtypes != "object"].index

data_skewness = pd.DataFrame({'Skew' :data_test[feats_numeric_test].apply(lambda x: skew(x)).sort_values(ascending=False)})

data_skewness = data_skewness[abs(data_skewness) > 0.75]

from scipy.special import boxcox1p

feats_skewed = data_skewness.index

lam = 0.15

for feat in feats_skewed:

    #all_data[feat] += 1

    data_test[feat] = boxcox1p(data_test[feat], lam)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost

import lightgbm
data_feats = data_train.drop(['SalePrice', 'SalePrice_trans'], axis=1)
data_all = pd.concat((data_feats, data_test))

num = data_feats.shape[0]

data_all = pd.get_dummies(data_all)

data_x = data_all[:num]

data_test = data_all[num:]
# 检查训练集特征值与目标值是否能对上

# 训练集特征值

data_x.shape
# 训练集目标值

data_target.shape
# 测试集

data_test.shape
n_folds = 5

def RMSE(alg):

    kf = KFold(n_folds, shuffle=True, random_state=20).get_n_splits(data_x)

    rmse= np.sqrt(-cross_val_score(alg, data_x, data_target, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

score = RMSE(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

score = RMSE(ENet)

print("\nENet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

score = RMSE(KRR)

print("\nKRR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

score = RMSE(GBoost)

print("\nGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb = xgboost.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

score = RMSE(model_xgb)

print("\nXGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_lgb = lightgbm.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score = RMSE(model_lgb)

print("\nLGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)  
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))



score = RMSE(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X.iloc[train_index], y.iloc[train_index])

                y_pred = instance.predict(X.iloc[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

                                                 meta_model = lasso)



score = RMSE(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
# StackingAveragedModels得分最佳
model_fit = stacked_averaged_models.fit(data_x, data_target)
result = np.expm1(stacked_averaged_models.predict(data_test)) # log(1+x)逆运算
# 保存预测结果

submission = pd.DataFrame()

submission['Id'] = data_test_ID

submission['SalePrice'] = result

submission.to_csv('/kaggle/working/submission',index=False)
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")