#导入本次实验所需要用到的python库

import pandas as pd

import numpy as np

import seaborn as sns

from scipy import stats

from scipy.stats import skew

from scipy.stats import norm

import matplotlib

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.ensemble import GradientBoostingRegressor

import lightgbm as lgb

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold,GridSearchCV

from pactools.grid_search import GridSearchCVProgressBar
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all" 
train_data = pd.read_csv('./data/train.csv')

test_data = pd.read_csv('./data/test.csv')
train_data.shape

test_data.shape #test没有SalePrice
#删掉比较明显的异常值

#SalePrice-OverallQual没有太明显的异常值，暂时留着

#train_data.drop(train_data[(train_data['OverallQual']<5) & (train_data['SalePrice']>200000)].index,inplace=True)

train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<200000)].index,inplace=True)

train_data.drop(train_data[(train_data['YearBuilt']<1900) & (train_data['SalePrice']>400000)].index,inplace=True)

train_data.drop(train_data[(train_data['TotalBsmtSF']>6000) & (train_data['SalePrice']<200000)].index,inplace=True)

train_data.reset_index(drop=True, inplace=True)
train_data.shape #上一步删掉了3个样本

test_data.shape
#合并数据集，便于统一进行数据清洗和特征工程

#id没啥用,注意train中只取特征，不取label(SalePrice)

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

all_features.shape #(1457+1459)*79(第一列为索引)
del_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu']  #缺失率高(80%以上,可酌情选择)，可以考虑删掉

#'PoolQC', 'MiscFeature', 'Alley', 'Fence' > 80%

#'PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu' > 50%

#'PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu','LotFrontage' > 15% (最多删到这儿)
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

numeric_features

len(numeric_features) #36

category_features = all_features.dtypes[all_features.dtypes == 'object'].index

category_features

len(category_features) #43
#一些特征其被表示成数值特征缺乏意义，例如年份还有类别(有些类别使用数字表示，会被误认为是数值变量)，这里将其转换为字符串，即类别型变量

#但要注意只适用于取值个数不多的特征，而像YearBuilt这种虽然也是年份，但取值空间很大1872-2010,如果转化成类别特征会非常稀疏，可能就不太合适

all_features['YrSold'] = all_features['YrSold'].astype(str) #取值不多

all_features['MoSold'] = all_features['MoSold'].astype(str)

all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)

all_features['OverallQual'] = all_features['OverallQual'].astype(str)

all_features['OverallCond'] = all_features['OverallCond'].astype(str)



numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

category_features = all_features.dtypes[all_features.dtypes == 'object'].index
#统计特殊情况：一部分特征值数据的缺失是由于房屋确实不存在此种类型的特征

#pands默认会将NA当做缺失值，但对本问题，NA不一定是缺失值（详见data_description.txt）

#NA	None:NA代表缺失值

#NA	No *: 表示改房屋没有相关属性

#因此缺失值一共有两种情况：(1)NA	None:NA代表缺失值 (2)未用NA标注的缺失值

#下面是12个特征：NA	No *: 表示该房屋没有相关属性 

special_features = [

    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

    'PoolQC', 'Fence'

]

len(special_features)
#显示数值变量的分布

def show_distribution(df,columns):

    f = pd.melt(df, value_vars=columns)

    g = sns.FacetGrid(f, col="variable",  col_wrap=5, sharex=False, sharey=False)

    g = g.map(sns.distplot, "value")

    plt.show()



#计算各数值变量的偏度（skewness）

def show_skew(df, numeric_features):

    skewed_feats = df[numeric_features].apply(

        lambda x: skew(x.dropna())).sort_values(ascending=False)

    skewness = pd.DataFrame({'Skew': skewed_feats})

    return skewness

#数值log变换之前的分布和偏度

#show_distribution(all_features, numeric_features)

skewness = show_skew(all_features, numeric_features)

#skewness.head(20)
#数值特征log变换：详见 偏态数据转换为正态分布https://drivingc.com/p/5bc43e20d249877c5c5e4c76

#len(skewness[skewness>0].index) #31 注意这种写法才对

#len(skewness[skewness['Skew']>0].index) #27，wrong, but why?

def log_transform(df, skewness, threshold=0.15, alpha=1.01):

    #alpha是为了防止log(0)的错误,Kaggle竞赛-房价预测（House Prices）小结中采用1.01，但一般似乎都是用一个比较小的量？

    to_log = skewness[abs(skewness) > threshold].index.tolist()

    df[to_log] = df[to_log].apply(lambda x: np.log(x + alpha))

    return df
#数值log变换之后的分布和偏度，可见log变换之后数据的偏度明显降低

all_features = log_transform(all_features,skewness,threshold=0.15,alpha=1.01)

#show_distribution(all_features, numeric_features)

skewness = show_skew(all_features, numeric_features)

#skewness.head(20)
#填充缺失值,对不同类型的缺失值采取了不同的填充方式，便于修改

def fill_missings(df,

                  numeric_features,

                  category_features,

                  special_features,

                  del_features,

                  standardization=True,

                  delete=False):

    #deal with special_features

    for feature in special_features:

        if feature in numeric_features:

            df[feature].fillna(0)  #数值型的NA	No * ，填0

        elif feature in category_features:

            df[feature].fillna('missing')  #类别型的NA	No * ，填充新类别missing



    #数值特征标准化

    if standardization:

        df[numeric_features] = df[numeric_features].apply(

        lambda x: (x - x.mean()) / (x.std()))

        # 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值

        df[numeric_features] = df[numeric_features].apply(

            lambda x: x.fillna(0))  #数值特征用均值初始化

    else:

        df[numeric_features] = df[numeric_features].apply(

            lambda x: x.fillna(x.mean()))  #数值特征用均值初始化



    #离散特征独热化,暂不区分有顺序信息的类别特征和普通的类别特征

    df[category_features] = df[category_features].apply(

        lambda x: x.fillna(x.mode()[0]))  #离散特征用众数初始化

    #填充完之后可以统一决定是否删除

    if delete:

        df.drop(columns=del_features, inplace=True) #目前选定的4个del_features都是类别特征，如果要删需要在get_dummies之前

    # dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征

    df = pd.get_dummies(df, dummy_na=False)  #暂忽略缺失值



    return df
all_features = fill_missings(all_features,

                             numeric_features,

                             category_features,

                             special_features,

                             del_features,

                             standardization=False,

                             delete=True)



all_features.shape #(2916, 334)
n_train = train_data.shape[0]

train_x=all_features[:n_train]

train_x.columns[0]

len(train_x.columns)

test_x=all_features[n_train:]

train_x.shape #(1457, 334)

X=np.array(train_x)

X.shape #(1457, 334)

# X=np.delete(X,0,1) 删掉第1列，前面已经闪过了，so，跳过

# X.shape #(1457, 333)

y=np.log(1+np.array(train_data['SalePrice'])) #+1的效果比不+1略好

# y=np.log(np.array(train_data['SalePrice'])) #因为房价都是比较大的整数，也许可以不用+1
#??GridSearchCVProgressBar 可打印网格搜索的进度 详见https://github.com/pactools/pactools/blob/master/pactools/grid_search.py
def grid_search(model, parameters, train_x, train_y, progress_bar=False, cv=5):

    #sklearn的0.22版本默认采用5-fold cv，当前版本默认3折

    models = GridSearchCVProgressBar(

        model, parameters, cv=cv, verbose=1,

        n_jobs=6) if progress_bar else GridSearchCV(

            model, parameters, cv=cv, n_jobs=6)

    models.fit(train_x, train_y)

    print(models.best_params_)

    print(models.best_score_)

    #print(models.best_estimator_)
#1.ridge

params1 = {

    'alpha':

    [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

}

grid_search(Ridge(), params1, X, y, progress_bar=True, cv=5)



# {'alpha': 6.0}

# 0.9145055927678927
#2. lasso

params2 = {

    'alpha': [

        0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.001, 0.01, 0.1,

        0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0

    ]

}

grid_search(Lasso(), params2, X, y, progress_bar=True)



# {'alpha': 0.0004}

# 0.9180910018284797
#3. GBR

params3 = {

    'learning_rate': [0.01, 0.02, 0.05, 0.1],

    'n_estimators': [3000, 5000, 10000],

    'max_depth': [3, 4, 5],

    'min_samples_leaf': [1, 5, 10],

    'min_samples_split': [2, 10, 20, 40],

}

grid_search(GradientBoostingRegressor(), params3, X, y, progress_bar=True)
#4. ENet

params4 = {

    'alpha': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 5.0],

    'l1_ratio': [0, 0.005, 0.01, 0.1, 0.5, 0.9, 0.99, 0.995, 1],

}

grid_search(ElasticNet(), params4, X, y, progress_bar=True)



# {'alpha': 0.01, 'l1_ratio': 0.005}

# 0.9132364219434982
#5. lightgbm.LGBMRegressor

params5 = {

    'num_leaves':

    [3,7,15,31,63],

    'learning_rate': [0.01,0.05,0.1,0.2],

    'n_estimators':[100,300,500,600,800],

    'max_bin':[50,255], #max number of bins that feature values will be bucketed in. default = 255

    'bagging_fraction':[0.1,0.5,0.6,1.0], #default = 1.0

    'bagging_freq':[0,5,10], #default = 0, frequency for bagging

    'feature_fraction':[0.1,0.5,1.0], #randomly select part of features on each iteration (tree)

    'min_data_in_leaf':[6,10,20,30], #default = 20,minimal number of data in one leaf. Can be used to deal with over-fitting

    'min_sum_hessian_in_leaf':[1e-3,1,10,11], #default = 1e-3,minimal sum hessian in one leaf. Like min_data_in_leaf, it can be used to deal with over-fitting

    

}

grid_search(lgb.LGBMRegressor(),params5,X,y,progress_bar=True)



# {'feature_fraction': 0.1, 'learning_rate': 0.05, 'min_data_in_leaf': 10, 'n_estimators': 500, 'num_leaves': 7}

# 0.9110044883960132