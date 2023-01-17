import numpy as np  # 科学计算工具包

import pandas as pd  # 数据分析工具包

import matplotlib.pyplot as plt # 图表绘制工具包

import seaborn as sns # 基于 matplot, 导入 seaborn 会修改默认的 matplotlib 配色方案和绘图样式，这会提高图表的可读性和美观性

import os

from scipy import stats

from scipy.stats import norm, skew 



plt.rc("font",family="SimHei",size="15")  #解决中文乱码问题



# 在 jupyter notebook 里面显示图表

%matplotlib inline 
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



df_train.head()
# 所有字段列表



df_train.columns
# 描述性统计数据



df_train['SalePrice'].describe()
# 房价直方图



plt.figure(figsize=(16, 5))

#去掉拟合的密度估计曲线，kde参数设为False

sns.distplot(df_train['SalePrice'],kde=False)  

# 'kde' 是控制密度估计曲线的参数，默认为 True，不设置会默认显示，如果我们将其设为 False，则不显示密度曲线。
print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#  Grlivarea 与 SalePrice 



plt.figure(figsize=(16, 8))

plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])

plt.ylim=(0,800000)  # y坐标轴范围

plt.xlabel('Grlivarea ')  # x轴名称

plt.ylabel('SalePrice ')  # y轴名称

plt.title(' Grlivarea and SalePrice') #标题
#  TotalBsmtSF 与 SalePrice 



plt.figure(figsize=(16, 8))

plt.scatter(x=df_train['TotalBsmtSF'], y=df_train['SalePrice'])

plt.ylim=(0,800000)  # y坐标轴范围

plt.xlabel('TotalBsmtSF ')  # x轴名称

plt.ylabel('SalePrice ')  # y轴名称

plt.title('TotalBsmtSF and SalePrice') #标题
#  OverallQual 与 SalePrice 



plt.figure(figsize=(16, 8))

sns.boxplot(x=df_train['OverallQual'], y=df_train['SalePrice'])

plt.ylim=(0,800000)  # y坐标轴范围

plt.xlabel('OverallQual')  # x轴名称

plt.ylabel('SalePrice ')  # y轴名称

plt.title('OverallQual and SalePrice') #标题
#  YearBuilt 与 SalePrice 



plt.figure(figsize=(16, 8))

sns.boxplot(x=df_train['YearBuilt'], y=df_train['SalePrice'])

plt.ylim=(0,800000)  # y坐标轴范围

plt.xticks(rotation=90) # x轴标签旋转90度

plt.xlabel('Yearbuilt')  # x轴名称

plt.ylabel('SalePrice')  # y轴名称

plt.title(' YearBuilt and SalePrice') #标题
#将生成的相关系数矩阵以热力图的形式展现出来

#这个corr是计算DataFrame中列之间的相关系数，如果存在非数值类型，则会自动排除



corrmat = df_train.corr()#相关系数，比如有n列，则相关系数矩阵大小为n*n，其中的(i,j)表示的是第i列和第j列之间的相关系数，最后生成的是DataFrame

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);#vmax=.8表示设置最大值为0.8；square表示将Axes横纵比设置为相等，这样每个单元格都是正方形
#分析与目标特征房价 SalePrice 相关度最高的十个变量



k = 10

#corrmat.nlargest表示的是在SalePrice列中选出最大的前k个对应的行，也就是说这样做之后的效果是生成了10*38的DataFrame

cols_10 = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

corrs_10 = df_train[cols_10].corr()

plt.figure(figsize=(12, 9))

sns.heatmap(corrs_10, annot=True) # annot: 如果为true，数据写到每个单元上
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], height = 2.5)

plt.show()
# 把“Id”这一列从数据集中单独挑出，便于操作



#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(df_train.shape))

print("The test data size before dropping Id feature is : {} ".format(df_test.shape))



#Save the 'Id' column

train_ID = df_train['Id']

test_ID = df_test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

df_train.drop("Id", axis = 1, inplace = True)

df_test.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(df_train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(df_test.shape))
#  Grlivarea 与 SalePrice 



plt.figure(figsize=(16, 8))

plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])

plt.ylim=(0,800000)  # y坐标轴范围

plt.xlabel('Grlivarea')  # x轴名称

plt.ylabel('SalePrice')  # y轴名称

plt.title('Grlivarea and SalePrice') #标题
# 删除这两个离群值

df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)



# 重绘图以查看是否剔除异常值

plt.figure(figsize=(16, 8))

plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])

plt.ylim=(0,800000)  # y坐标轴范围

plt.xlabel('Grlivarea')  # x轴名称

plt.ylabel('SalePrice')  # y轴名称

plt.title(' Grlivarea and SalePrice ') #标题
# 绘制目标特征的概率分布图

plt.figure(figsize=(16, 8))

sns.distplot(df_train['SalePrice'], fit=norm) # 拟合标准正态分布

plt.legend(['Normal dist'], loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



# 获取目标特征的正态分布参数

(mu, sigma) = norm.fit(df_train['SalePrice']) # fit方法：对一组随机取样进行拟合，最大似然估计方法找出最适合取样数据的概率密度函数系数。

print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# 绘制QQ-plot图



plt.figure(figsize=(16, 8))

res = stats.probplot(df_train['SalePrice'], plot=plt)
# 正态化Y

df_train['SalePrice'] = np.log1p(df_train['SalePrice'])



# 绘制分布图检查新的分布

sns.distplot(df_train['SalePrice'], fit=norm);

plt.legend(['Normal dist'], loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



# 获取新的正态分布参数

(mu, sigma) = norm.fit(df_train['SalePrice'])

print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



# 绘制QQ-plot图

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
ntrain = df_train.shape[0] # 训练集数目

ntest = df_test.shape[0] # 测试集数目

y_train = df_train.SalePrice.values # 训练集的Y

all_data = pd.concat((df_train, df_test)).reset_index(drop=True) # 删除目标特征 SalePrice

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
# 统计缺失值数量和占比



def missing_info(data,num):

    # func:统计缺失值数量和占比函数

    # data: dataframe类型

    # num: 数字类型，显示前几行数据

    # return: 缺失值统计\占比



    null_data = data.isnull().sum().sort_values(ascending=False)

    percent_1 = data.isnull().sum()/data.isnull().count()

    missing_data = pd.concat([null_data,percent_1.apply(lambda x: format(x, '.2%'))],axis=1,keys=['total missing','missing percentage'])

    return missing_data.head(num)



missing_data = missing_info(all_data,40) 

missing_data
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
print(df_train['Utilities'].unique())

print(df_test['Utilities'].unique())
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
# 再次检查缺失值

missing_data = missing_info(all_data,5) 

missing_data
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',

        'YrSold', 'MoSold')



for c in cols:

    lbl = LabelEncoder() # 创建 labelencoder 实例

    lbl.fit(list(all_data[c].values))

    all_data[c] = lbl.transform(list(all_data[c].values)) 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data.info()
# 筛选出所有数值型的特征

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# 检查所有数值型特征的偏态

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    all_data[feat] = boxcox1p(all_data[feat], lam)
all_data = pd.get_dummies(all_data)

print(all_data.shape)
df_train = all_data[:ntrain]

df_test = all_data[ntrain:]
# 导入库



from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC #线性回归模型

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor #集成模型

from sklearn.kernel_ridge import KernelRidge #核岭回归

from sklearn.pipeline import make_pipeline #pipeline

from sklearn.preprocessing import RobustScaler #标准化

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone #自定义类的API

from sklearn.model_selection import KFold, cross_val_score, train_test_split #交叉验证

from sklearn.metrics import mean_squared_error #均方误差

import xgboost as xgb #XGBoost

# import lightgbm as lgb #lightGBM
# 5折交叉验证

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df_train.values)

    rmse= np.sqrt(-cross_val_score(model, df_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
models_name = ['Lasso', 'ElasticNet', 'Ridge']

models = [lasso, ENet, KRR]

for i, model in enumerate(models):

    score = rmsle_cv(model)

    print('{} score: {}({})'.format(models_name[i], score.mean(), score.std()))
KRR.fit(df_train.values, y_train)

KRR_pred= np.expm1(KRR.predict(df_test.values))



sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = KRR_pred

sub.to_csv('submission.csv',index=False)