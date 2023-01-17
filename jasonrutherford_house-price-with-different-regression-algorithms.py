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
import warnings

warnings.filterwarnings('ignore')

import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



Id = test_data['Id']

train_data.drop(['Id'] , axis = 1 , inplace = True)

test_data.drop(['Id'] , axis = 1 , inplace = True)
train_data.head()
train_data.info() , test_data.info()
missing_train_data = train_data.isnull().sum()

missing_train_data[missing_train_data > 0].sort_values(ascending = False)
missing_test_data = test_data.isnull().sum()

missing_test_data[missing_test_data > 0].sort_values(ascending = False)
train_data.drop(['Alley' , 'Fence'] , axis = 1 , inplace = True)

test_data.drop(['Alley' , 'Fence'] , axis = 1 , inplace = True)
train_data['PoolQC'].fillna('None' , inplace = True)

train_data['MiscFeature'].fillna('None' , inplace = True)

train_data['FireplaceQu'].fillna('None' , inplace = True)



test_data['PoolQC'].fillna('None' , inplace = True)

test_data['MiscFeature'].fillna('None' , inplace = True)

test_data['FireplaceQu'].fillna('None' , inplace = True)
a = train_data.isnull().sum()

a[a > 0].sort_values(ascending = False)
train_data["LotAreaCut"] = pd.qcut(train_data.LotArea,10)

train_data.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count'])
train_data['LotFrontage'] = train_data.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# Since some combinations of LotArea and Neighborhood are not available, so we just LotAreaCut alone.

train_data['LotFrontage'] = train_data.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
cols = ['MasVnrArea' , 'GarageCars' , 'GarageArea']

for col in cols:

    train_data[col].fillna(0, inplace = True)

    

cols1 = ['MasVnrType' , 'GarageQual' , 'GarageCond' , 'GarageFinish' , 'GarageYrBlt' , 

         'GarageType' , 'BsmtExposure' , 'BsmtCond' , 'BsmtQual' , 'BsmtFinType2' , 'BsmtFinType1']

for col in cols1:

    train_data[col].fillna('None' , inplace = True)

    

cols2 = ['Electrical']

for col in cols2:

    train_data[col].fillna(train_data[col].mode()[0] , inplace = True)
b = test_data.isnull().sum()

b[b > 0].sort_values(ascending = False)
test_data["LotAreaCut"] = pd.qcut(test_data.LotArea,10)

test_data.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count'])
test_data['LotFrontage'] = test_data.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# Since some combinations of LotArea and Neighborhood are not available, so we just LotAreaCut alone.

test_data['LotFrontage'] = test_data.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
cols=['MasVnrArea' , 'GarageCars' , 'GarageArea' , 

      'BsmtUnfSF' , 'BsmtFinSF2' , 'BsmtFinSF1' , 

      'TotalBsmtSF']

for col in cols:

    test_data[col].fillna(0, inplace = True)

    

cols1 = ['MasVnrType' , 'GarageQual' , 'GarageCond' , 'GarageFinish' , 'GarageYrBlt' , 'GarageType' , 

         'BsmtExposure' , 'BsmtCond' , 'BsmtQual' , 'BsmtFinType2' , 'BsmtFinType1']

for col in cols1:

    test_data[col].fillna("None", inplace=True)

    

# fill in with mode

cols2 = ['MSZoning' , 

         'BsmtFullBath' , 'BsmtHalfBath' , 

         'Utilities' , 'Functional' , 'KitchenQual' , 'SaleType' , 

         'Exterior1st' , 'Exterior2nd']

for col in cols2:

    test_data[col].fillna(test_data[col].mode()[0], inplace=True)
train_data.isnull().sum() , test_data.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(15,8))

sns.boxplot(train_data.YearBuilt, train_data.SalePrice)
plt.figure(figsize=(12,6))

plt.scatter(x = train_data.GrLivArea, y = train_data.SalePrice)

plt.xlabel("GrLivArea", fontsize=13)

plt.ylabel("SalePrice", fontsize=13)

plt.ylim(0,800000)
train_data.drop(train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 300000)].index , inplace = True)
corrmat = train_data.corr()

f, ax = plt.subplots(figsize=(20, 12))

sns.heatmap(corrmat, vmax=1.0, square=True)
corr = train_data.corr()

features = abs(corr['SalePrice']).sort_values(ascending = False)

features
category_list = ['MSZoning' , 'Street' , 'LotShape' , 'LandContour' , 'Utilities' , 'LotConfig' , 'LandSlope' , 'Neighborhood' ,

                 'Condition1' , 'Condition2' , 'BldgType' , 'HouseStyle' , 'RoofStyle' , 'RoofMatl' , 'Exterior1st' , 'Exterior2nd' ,    

                 'MasVnrType' , 'ExterQual' , 'ExterCond' , 'Foundation' , 'BsmtQual' , 'BsmtCond' , 'BsmtExposure' , 'BsmtFinType1' ,     

                 'BsmtFinType2' , 'Heating' , 'HeatingQC' , 'CentralAir' , 'Electrical' , 'KitchenQual' , 'Functional' ,

                 'GarageType' , 'GarageFinish' , 'GarageQual' , 'GarageCond' , 'PavedDrive' , 'SaleType' , 'SaleCondition' ]
train_data['Street'].value_counts()
cate_drop_list = ['Street' , 'LandContour' , 'Utilities' , 'LandSlope' ,

                  'Condition1' , 'Condition2' , 'BldgType' , 'RoofMatl' ,   

                  'ExterCond' , 'BsmtCond' , 'BsmtExposure' ,     

                  'BsmtFinType2' , 'Heating' , 'CentralAir' , 'Electrical' , 'Functional' , 

                  'GarageQual' , 'GarageCond' , 'PavedDrive' , 'SaleType' , 'SaleCondition']
train_data.drop(cate_drop_list , axis = 1 , inplace = True)

test_data.drop(cate_drop_list , axis = 1 , inplace = True)
Member_list1 = ['MSZoning','LotShape','LotConfig','RoofStyle','MasVnrType','ExterQual','BsmtQual',

                'BsmtFinType1','HeatingQC','KitchenQual','GarageFinish','Foundation','GarageType',

                'PoolQC','MiscFeature','FireplaceQu']
for i in Member_list1 :

    train_data[i] = pd.factorize(train_data[i])[0].astype(np.int64)

    test_data[i] = pd.factorize(test_data[i])[0].astype(np.int64)
train_data['Neighborhood'].replace({'MeadowV' : 1 , 'IDOTRR' : 2 , 'BrDale' : 2 ,

                                    'OldTown' : 3 , 'Edwards' : 3 , 'BrkSide' : 3 ,

                                    'Sawyer' : 4 , 'Blueste' : 4 , 

                                    'SWISU' : 4 , 'NAmes' : 4 ,

                                    'NPkVill' : 5 , 'Mitchel' : 5 ,

                                    'SawyerW' : 6 , 'Gilbert' : 6 , 'NWAmes' : 6 ,

                                    'Blmngtn' : 7 , 'CollgCr' : 7 , 

                                    'ClearCr' : 7 , 'Crawfor' : 7 ,

                                    'Veenker' : 8 , 'Somerst' : 8 , 'Timber' : 8 ,

                                    'StoneBr' : 9 ,

                                    'NoRidge' : 10 , 'NridgHt' : 10} , inplace = True)



train_data['HouseStyle'].replace({'1.5Unf' : 1 , 

                                  '1.5Fin' : 2 , '2.5Unf' : 2 , 'SFoyer' : 2 , 

                                  '1Story' : 3 , 'SLvl' : 3 ,

                                  '2Story' : 4 , '2.5Fin' : 4} , inplace = True)



train_data['Exterior1st'].replace({'BrkComm' : 1 ,

                                   'AsphShn' : 2 , 'CBlock' : 2 , 'AsbShng' : 2 ,

                                   'WdShing' : 3 , 'Wd Sdng' : 3 , 'MetalSd' : 3 , 

                                   'Stucco' : 3 , 'HdBoard' : 3 ,

                                   'BrkFace' : 4 , 'Plywood' : 4 ,                                                       

                                   'VinylSd' : 5 ,

                                   'CemntBd' : 6 ,

                                   'Stone' : 7 , 'ImStucc' : 7} , inplace = True)



train_data['Exterior2nd'].replace({'Other':1,

                                   'AsphShn' : 2 , 'CBlock' : 2 , 'AsbShng' : 2 ,

                                   'Wd Shng' : 3 , 'Wd Sdng' : 3 , 'MetalSd' : 3 , 

                                   'Stucco' : 3 , 'HdBoard' : 3 ,

                                   'BrkFace' : 4 , 'Plywood' : 4 , 'Brk Cmn' : 4 ,

                                   'VinylSd' : 5 ,

                                   'CmentBd' : 6 ,

                                   'Stone' : 7 , 'ImStucc' : 7} , inplace = True)
test_data['Neighborhood'].replace({'MeadowV' : 1 , 'IDOTRR' : 2 , 'BrDale' : 2 ,

                                   'OldTown' : 3 , 'Edwards' : 3 , 'BrkSide' : 3 ,

                                   'Sawyer' : 4 , 'Blueste' : 4 , 

                                   'SWISU' : 4 , 'NAmes' : 4 ,

                                   'NPkVill' : 5 , 'Mitchel' : 5 ,

                                   'SawyerW' : 6 , 'Gilbert' : 6 , 'NWAmes' : 6 ,

                                   'Blmngtn' : 7 , 'CollgCr' : 7 , 

                                   'ClearCr' : 7 , 'Crawfor' : 7 ,

                                   'Veenker' : 8 , 'Somerst' : 8 , 'Timber' : 8 ,

                                   'StoneBr' : 9 ,

                                   'NoRidge' : 10 , 'NridgHt' : 10} , inplace = True)



test_data['HouseStyle'].replace({'1.5Unf' : 1 , 

                                 '1.5Fin' : 2 , '2.5Unf' : 2 , 'SFoyer' : 2 , 

                                 '1Story' : 3 , 'SLvl' : 3 ,

                                 '2Story' : 4 , '2.5Fin' : 4} , inplace = True)



test_data['Exterior1st'].replace({'BrkComm' : 1 ,

                                  'AsphShn' : 2 , 'CBlock' : 2 , 'AsbShng' : 2 ,

                                  'WdShing' : 3 , 'Wd Sdng' : 3 , 'MetalSd' : 3 , 

                                  'Stucco' : 3 , 'HdBoard' : 3 ,

                                  'BrkFace' : 4 , 'Plywood' : 4 ,

                                  'VinylSd' : 5 ,

                                  'CemntBd' : 6 ,

                                  'Stone' : 7 , 'ImStucc' : 7} , inplace = True)



test_data['Exterior2nd'].replace({'Other':1,

                                  'AsphShn' : 2 , 'CBlock' : 2 , 'AsbShng' : 2 ,

                                  'Wd Shng' : 3 , 'Wd Sdng' : 3 , 'MetalSd' : 3 , 

                                  'Stucco' : 3 , 'HdBoard' : 3 ,

                                  'BrkFace' : 4 , 'Plywood' : 4 , 'Brk Cmn' : 4 ,

                                  'VinylSd' : 5 ,

                                  'CmentBd' : 6 ,

                                  'Stone' : 7 , 'ImStucc' : 7} , inplace = True)
train_data.drop(['LotAreaCut'] , axis = 1 , inplace = True)

test_data.drop(['LotAreaCut'] , axis = 1 , inplace = True)
test_data[['HouseStyle' , 'Exterior1st']]
test_data['HouseStyle'] = test_data['HouseStyle'].astype(np.int64)

test_data['Exterior1st'] = test_data['Exterior1st'].astype(np.int64)
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline, make_pipeline

from scipy.stats import skew

from sklearn.decomposition import PCA, KernelPCA

from sklearn.preprocessing import Imputer
class labelenc(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self,X,y=None):

        return self

    

    def transform(self,X):

        lab=LabelEncoder()

        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])

        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])

        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])

        return X



class skew_dummies(BaseEstimator, TransformerMixin):

    def __init__(self,skew=0.5):

        self.skew = skew

    

    def fit(self,X,y=None):

        return self

    

    def transform(self,X):

        X_numeric=X.select_dtypes(exclude=["object"])

        skewness = X_numeric.apply(lambda x: skew(x))

        skewness_features = skewness[abs(skewness) >= self.skew].index

        X[skewness_features] = np.log1p(X[skewness_features])

        X = pd.get_dummies(X)

        return X
NumStr = ['YearBuilt' , 'YearRemodAdd' , 'GarageYrBlt']

for col in NumStr:

    train_data[col] = train_data[col].astype(str)

    test_data[col] = test_data[col].astype(str)
# build pipeline

pipe = Pipeline([('labenc', labelenc()),('skew_dummies', skew_dummies(skew=1))])



train_data = pipe.fit_transform(train_data)

test_data = pipe.fit_transform(test_data)
train_data = train_data.join(pd.get_dummies(train_data['YrSold'] , prefix = 'YrSold'))

test_data = test_data.join(pd.get_dummies(test_data['YrSold'] , prefix = 'YrSold'))

train_data.drop(['YrSold'] , axis = 1 , inplace = True)

test_data.drop(['YrSold'] , axis = 1 , inplace = True)
train_data.head()
test_data.head()
train_data.shape , test_data.shape
SalePrice_train_data = train_data.SalePrice

train_data.drop(['SalePrice'] , axis = 1 , inplace = True)

train_data.insert(0 , 'SalePrice' , SalePrice_train_data)
corr = train_data.corr()

features_importance = abs(corr['SalePrice']).sort_values(ascending = False)

features_importance
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score , train_test_split

import numpy as np

X , y = train_data.iloc[: , 1:] , train_data.iloc[: , 0]



std = StandardScaler()

X_std = std.fit_transform(X)



mms = MinMaxScaler()

X_mms = mms.fit_transform(X)



norm = Normalizer()

X_norm = norm.fit_transform(X)
lab_enc = LabelEncoder()

y = lab_enc.fit_transform(y)
neighbors = np.arange(1, 30)

kfold = 10

val_accuracy = { 'std' : [] , 'mms' : [] , 'norm' : [] }



bestKnr = None

bestAcc = 0.0

bestScaling = None



# 在不同的K值条件下与不同的标准化后的数值处理后的模型精确度之间的关系

for i, k in enumerate(neighbors):

    

    knr = KNeighborsRegressor(n_neighbors = k)

    

    # 交叉验证集的accuracy

    

    s1 = np.mean(cross_val_score(knr, X_std, y, cv=kfold))

    val_accuracy['std'].append(s1)

    

    s2 = np.mean(cross_val_score(knr, X_mms, y, cv=kfold))

    val_accuracy['mms'].append(s2)

    

    s3 = np.mean(cross_val_score(knr, X_norm, y, cv=kfold))

    val_accuracy['norm'].append(s3)

    

    if s1 > bestAcc:

        bestAcc = s1

        bestKnr = knr

        bestScaling = 'std'

        

    elif s2 > bestAcc:

        bestAcc = s2

        bestKnr = knr

        bestScaling = 'mms'

        

    elif s3 > bestAcc:

        bestAcc = s3

        bestKnr = knr

        bestScaling = 'norm'



# Plotting

plt.figure(figsize=[13,8])



plt.plot(neighbors, val_accuracy['std'], label = 'CV Accuracy with std')

plt.plot(neighbors, val_accuracy['mms'], label = 'CV Accuracy with mms')

plt.plot(neighbors, val_accuracy['norm'], label = 'CV Accuracy with norm')



plt.legend()

plt.title('k value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neighbors)

plt.show()



print('Best Accuracy with feature scaling:', bestAcc)

print('Best kNN classifier:', bestKnr)

print('Best scaling:', bestScaling)
from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.001)

lasso.fit(X , y)



FI_lasso = pd.DataFrame({'Feature Importance' : lasso.coef_}, index = train_data.columns[1:61])

FI_lasso.sort_values('Feature Importance' , ascending = False)



FI_lasso[FI_lasso['Feature Importance'] != 0].sort_values('Feature Importance').plot(kind = 'barh' , figsize = (15 , 25))

plt.xticks(rotation = 90)

plt.show()
from sklearn import linear_model, svm, gaussian_process

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



X_train , X_test , y_train , y_test = train_test_split(X_std , y , test_size = 0.2 , random_state = 42)



clfs = {

        'svm':svm.SVR(), 

        'RandomForestRegressor':RandomForestRegressor(n_estimators=400),

        'BayesianRidge':linear_model.BayesianRidge()

       }

for clf in clfs:

    try:

        clfs[clf].fit(X_train, y_train)

        y_pred = clfs[clf].predict(X_test)

        print(clf + " cost:" + str(np.sum(y_pred-y_test)/len(y_pred)) )

    except Exception as e:

        print(clf + " Error:")

        print(str(e))
x = train_data.iloc[: , 1:].values

y = train_data.iloc[: , 0].values

X_train , X_test , y_train , y_test = train_test_split(x , y , test_size = 0.33 , random_state = 42)



clf = RandomForestRegressor(n_estimators = 400)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



# 保存clf，共下面计算测试集数据使用

# rfr = clf
from sklearn.model_selection import learning_curve



def plot_learning_curve(estimator, title, X, y, ylim = None, cv = None, n_jobs = 1, 

                        train_sizes = np.linspace(.05, 1., 20), verbose = 0, plot = True):

    """

    画出data在某模型上的learning curve.

    参数解释

    ----------

    estimator : 你用的分类器。

    title : 表格的标题。

    X : 输入的feature，numpy类型

    y : 输入的target vector

    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点

    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)

    n_jobs : 并行的的任务数(默认1)

    """

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    

    if plot:

        plt.figure()

        plt.title(title)

        if ylim is not None:

            plt.ylim(*ylim)

        plt.xlabel(u"number")

        plt.ylabel(u"score")

        plt.gca().invert_yaxis()

        plt.grid()

    

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 

                         alpha=0.1, color="b")

        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 

                         alpha=0.1, color="r")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"train—score")

        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"incross—score")

    

        plt.legend(loc="best")

        

        plt.draw()

        plt.gca().invert_yaxis()

        plt.show()

    

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2

    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff



plot_learning_curve(clf, "learning curve", X_train, y_train)
features_drop_list = ['MasVnrArea' , 'YrSold_2007' , 'TotalBsmtSF' , 'GarageYrBlt' , '1stFlrSF' ,

                      '2ndFlrSF' , 'GarageArea' , 'BsmtFinSF1' , 'YrSold_2010' , 'BsmtUnfSF']
train_data.drop(features_drop_list , axis = 1 , inplace = True)

test_data.drop(features_drop_list , axis = 1 , inplace = True)
x = train_data.iloc[: , 1:].values

y = train_data.iloc[: , 0].values

X_train , X_test , y_train , y_test = train_test_split(x , y , test_size = 0.33 , random_state = 42)



clf = RandomForestRegressor(n_estimators = 400)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



# 保存clf，共下面计算测试集数据使用

# rfr = clf
plot_learning_curve(clf, "learning curve", X_train, y_train)
add_train_features = pd.DataFrame()

 

add_train_features['GrLivArea_OverallQual']    = train_data['GrLivArea']    * train_data['OverallQual']

add_train_features['Neighborhood_OverallQual'] = train_data['Neighborhood'] + train_data['OverallQual']

add_train_features['LotArea_OverallQual']      = train_data['LotArea']      * train_data['OverallQual']

add_train_features['MSZoning_OverallQual']     = train_data['MSZoning']     + train_data['OverallQual']

add_train_features['MSZoning_YearBuilt']       = train_data['MSZoning']     + train_data['YearBuilt']

add_train_features['Neighborhood_YearBuilt']   = train_data['Neighborhood'] + train_data['YearBuilt']

add_train_features['Rooms']                    = train_data['FullBath']     + train_data['TotRmsAbvGrd']

add_train_features['PorchArea']                = train_data['OpenPorchSF']  + train_data['EnclosedPorch'] + train_data['3SsnPorch'] + train_data['ScreenPorch']
add_test_features = pd.DataFrame()

 

add_test_features['GrLivArea_OverallQual']    = test_data['GrLivArea']    * test_data['OverallQual']

add_test_features['Neighborhood_OverallQual'] = test_data['Neighborhood'] + test_data['OverallQual']

add_test_features['LotArea_OverallQual']      = test_data['LotArea']      * test_data['OverallQual']

add_test_features['MSZoning_OverallQual']     = test_data['MSZoning']     + test_data['OverallQual']

add_test_features['MSZoning_YearBuilt']       = test_data['MSZoning']     + test_data['YearBuilt']

add_test_features['Neighborhood_YearBuilt']   = test_data['Neighborhood'] + test_data['YearBuilt']

add_test_features['Rooms']                    = test_data['FullBath']     + test_data['TotRmsAbvGrd']

add_test_features['PorchArea']                = test_data['OpenPorchSF']  + test_data['EnclosedPorch'] + train_data['3SsnPorch'] + train_data['ScreenPorch']
train_data = pd.concat([train_data , add_train_features] , axis = 1)

test_data = pd.concat([test_data , add_test_features] , axis = 1)
corr = train_data.corr()

features = abs(corr['SalePrice']).sort_values(ascending = False)

features
# 各特征之间的相关性



corrmat = train_data.corr()

f, ax = plt.subplots(figsize = (30, 24))

sns.heatmap(corrmat , linewidth = 0.01 , vmax = 1.0 , vmin = 0.0 , fmt = '.2f' , square = True)
x = train_data.iloc[: , 1:].values

y = train_data.iloc[: , 0].values



# 标准化

std = StandardScaler()

x = std.fit_transform(x)
# define cross validation strategy

def rmse_cv(model,x,y):

    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=5))

    return rmse
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.svm import SVR , LinearSVR

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge

from sklearn.kernel_ridge import KernelRidge

from xgboost import XGBRegressor



# 导入GridSearchCV模块为下面的调参做准备，当然xgboost可以自动地调参，更加方便快捷。

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import seaborn as sns



models = [LinearRegression(),

          Ridge(),

          Lasso(alpha=0.01,max_iter=10000),

          RandomForestRegressor(),

          GradientBoostingRegressor(),

          SVR(),

          LinearSVR(),

          ElasticNet(alpha=0.001,max_iter=10000),

          SGDRegressor(max_iter=1000,tol=1e-3),

          BayesianRidge(),

          KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),

          ExtraTreesRegressor(),

          XGBRegressor()]



names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb"]

for name, model in zip(names, models):

    score = rmse_cv(model, x , y)

    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
from sklearn.model_selection import GridSearchCV

class grid():

    def __init__(self,model):

        self.model = model

    

    def grid_get(self,x,y,param_grid):

        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")

        grid_search.fit(x,y)

        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))

        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])

        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
# LinearRegression

param_grid = {'normalize' : [True , False]}

grid(LinearRegression()).grid_get(x ,y , param_grid)
# Ridge

param_grid = {'alpha' : [35 , 40 , 45 , 50 , 55 , 60 , 65 , 70 , 80 , 90]}

grid(Ridge()).grid_get(x ,y , param_grid)
# ElasticNet

param_grid = {'alpha' : [0.001 , 0.004 , 0.005 , 0.008] , 

              'l1_ratio' : [0.1 , 0.3 , 0.5] ,

              'max_iter' : [2000 , 4000 , 6000]}

grid(ElasticNet()).grid_get(x ,y , param_grid)
# SGDRegressor

param_grid = {'alpha' : [0.005 , 0.01 , 0.02] , 

              'l1_ratio' : [0.1 , 0.3 , 0.5 , 0.8] ,

              'max_iter' : [3000 , 5000 , 8000 , 9000]}

grid(SGDRegressor()).grid_get(x ,y , param_grid)
# BayesianRidge

param_grid = {'alpha_1' : [6 , 7 , 8 , 9 , 10] , 

              'lambda_1' : [6 , 7 , 8 , 9 , 10]}

grid(BayesianRidge()).grid_get(x ,y , param_grid)
# Kernel Ridge

param_grid = {'alpha' : [2.0 , 3.0 , 4.0] , 

              'kernel' : ['polynomial'] , 

              'degree' : [2 , 3 , 4] , 

              'coef0' : [2.0 , 3.0 , 4.0]}

grid(KernelRidge()).grid_get(x ,y , param_grid)
from sklearn.base import RegressorMixin , clone



class AverageWeight(BaseEstimator, RegressorMixin):

    def __init__(self,mod,weight):

        self.mod = mod

        self.weight = weight

        

    def fit(self,x,y):

        self.models_ = [clone(x) for x in self.mod]

        for model in self.models_:

            model.fit(x,y)

        return self

    

    def predict(self,x):

        w = list()

        pred = np.array([model.predict(x) for model in self.models_])

        # for every data point, single model prediction times weight, then add them together

        for data in range(pred.shape[1]):

            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]

            w.append(np.sum(single))

        return w
# LinearRegression



lr = LinearRegression(normalize = False)

ridge = Ridge(alpha = 35)

ela = ElasticNet(alpha = 0.005 , l1_ratio = 0.3 , max_iter = 2000)

sgdr = SGDRegressor(alpha = 0.02 , l1_ratio = 0.3 , max_iter = 8000)

bayr = BayesianRidge(alpha_1 = 10 , lambda_1 = 6)

ker = KernelRidge(alpha = 4.0 , kernel = 'polynomial' , degree = 2 , coef0 = 4.0)



# assign weights based on their gridsearch score

w1 = 0.164

w2 = 0.165

w3 = 0.168

w4 = 0.162

w5 = 0.168

w6 = 0.173



weight_avg = AverageWeight(mod = [lr , ridge , ela , sgdr , bayr , ker] , weight = [w1 , w2 , w3 , w4 , w5 , w6])



score = rmse_cv(weight_avg , x , y)

print(score.mean())
weight_avg = AverageWeight(mod = [ela , ker] , weight = [0.3 , 0.7])

score = rmse_cv(weight_avg , x , y)

print(score.mean())
from sklearn.model_selection import KFold



class stacking(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self,mod,meta_model , n_folds = 5):

        self.mod = mod

        self.meta_model = meta_model

        self.n_folds = n_folds

        

    def fit(self,X,y):

        self.saved_model = [list() for i in self.mod]

        oof_train = np.zeros((X.shape[0], len(self.mod)))

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        for i,model in enumerate(self.mod):

            for train_index, val_index in kfold.split(X, y):

                renew_model = clone(model)

                renew_model.fit(X[train_index], y[train_index])

                self.saved_model[i].append(renew_model)

                oof_train[val_index,i] = renew_model.predict(X[val_index])

        

        self.meta_model.fit(oof_train,y)

        return self

    

    def predict(self,X):

        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) 

                                      for single_model in self.saved_model]) 

        return self.meta_model.predict(whole_test)

    

    def get_oof(self,X,y,test_X):

        oof = np.zeros((X.shape[0],len(self.mod)))

        test_single = np.zeros((test_X.shape[0],5))

        test_mean = np.zeros((test_X.shape[0],len(self.mod)))

        for i,model in enumerate(self.mod):

            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):

                clone_model = clone(model)

                clone_model.fit(X[train_index],y[train_index])

                oof[val_index,i] = clone_model.predict(X[val_index])

                test_single[:,j] = clone_model.predict(test_X)

            test_mean[:,i] = test_single.mean(axis=1)

        return oof, test_mean
stack_model = stacking(mod = [ela , ker] , meta_model = ker)

score = rmse_cv(stack_model,x,y)

print(score.mean())
stack_model.fit(x,y)
test_data['PorchArea'].fillna(test_data['PorchArea'].mean() , inplace = True)
test_x = std.fit_transform(test_data)
pred = stack_model.predict(test_x)
prediction = pd.DataFrame(pred , columns = ['SalePrice'])

result = pd.concat([Id , prediction] , axis = 1)

result.to_csv('submission10.csv',index=False)