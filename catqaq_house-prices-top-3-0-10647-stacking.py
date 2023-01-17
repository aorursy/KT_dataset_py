from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all" 
import numpy as np  

import pandas as pd 

from datetime import datetime

from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, Ridge 

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import make_scorer

from sklearn.neighbors import LocalOutlierFactor

from sklearn.linear_model import LinearRegression

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import seaborn as sns

import os
print(os.listdir("../input"))



train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print("Train set size:", train.shape)

print("Test set size:", test.shape)

print('start data processing', datetime.now(), )
# know your target

train['SalePrice'].describe()
sns.distplot(train['SalePrice']);
#skewness and kurtosis: 可以看到SalePrice的偏度较大，log变换可以缓解这个问题，而且比赛的损失函数也正好是log-rmse，所以随后会对SalePrice作log-transformation

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])



#much better

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show()
def detect_outliers(x, y, top=5, plot=True):

    lof = LocalOutlierFactor(n_neighbors=40, contamination=0.1)

    x_ =np.array(x).reshape(-1,1)

    preds = lof.fit_predict(x_)

    lof_scr = lof.negative_outlier_factor_

    out_idx = pd.Series(lof_scr).sort_values()[:top].index

    if plot:

        f, ax = plt.subplots(figsize=(9, 6))

        plt.scatter(x=x, y=y, c=np.exp(lof_scr), cmap='RdBu')

    return out_idx
outs = detect_outliers(train['GrLivArea'], train['SalePrice'],top=5) #got 1298,523

outs
outs = detect_outliers(train['LowQualFinSF'], train['SalePrice'],top=5)#got 88

outs
#很多public kernel中都用这些点，88,523,1298很容易找到，对于其他的outliers后面会补充说明

#改进点８：more or less outliers

outliers = [30, 88, 462, 523, 632, 1298, 1324]
#all_outliers只包含30,88,523,1298，其他的outliers是怎么得到的？

#可能的原因：

#1.detect_outliers函数中的参数设置问题

#2.这里仅从特征与train['SalePrice']的关系来寻找outliers,或许也可以从特征与特征之间的关系来寻找outliers

from collections import Counter

all_outliers=[]

numeric_features = train.dtypes[train.dtypes != 'object'].index

for feature in numeric_features:

    try:

        outs = detect_outliers(train[feature], train['SalePrice'],top=5, plot=False)

    except:

        continue

    all_outliers.extend(outs)



print(Counter(all_outliers).most_common())

for i in outliers:

    if i in all_outliers:

        print(i)
#delete outliers

train = train.drop(train.index[outliers])

train.shape
#合并train,test的特征，便于统一进行特征工程

y = train.SalePrice.reset_index(drop=True)

train_features = train.drop(['SalePrice'], axis=1)

test_features = test

features = pd.concat([train_features, test_features]).reset_index(drop=True)

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.

features.drop(['Id'], axis=1, inplace=True)

print(features.shape)
#2.1 一些特征其被表示成数值特征缺乏意义，例如年份还有类别(有些类别使用数字表示，会被误认为是数值变量)，这里将其转换为字符串，即类别型变量

features['MSSubClass'] = features['MSSubClass'].apply(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)

# 改进点1：OverallQual，OverallCond也是由数字表示的类别变量，但内含顺序信息

# features['OverallQual'] = features['OverallQual'].astype(str)

# features['OverallCond'] = features['OverallCond'].astype(str)
#2.2 numeric_features and 

numeric_features = features.dtypes[features.dtypes != 'object'].index

numeric_features

len(numeric_features) #33

category_features = features.dtypes[features.dtypes == 'object'].index

category_features

len(category_features) #46
#2.3 special features with NA---> NO such feature（NA不是真正的缺失值，而是该样本没有这个特征)

special_features = [

    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

    'PoolQC', 'Fence'

]

len(special_features)
#1.类别型特征，但明说了具有典型值的：fillna with Typical values

features['Functional'] = features['Functional'].fillna('Typ') #Typ	Typical Functionality

features['Electrical'] = features['Electrical'].fillna("SBrkr") #SBrkr	Standard Circuit Breakers & Romex

features['KitchenQual'] = features['KitchenQual'].fillna("TA") #TA	Typical/Average



#2.分组填充

#groupby：Group DataFrame or Series using a mapper or by a Series of columns.

#transform是与groupby（pandas中最有用的操作之一）组合使用的,恢复维度

#对MSZoning按MSSubClass分组填充众数

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

#对LotFrontage按Neighborhood分组填充中位数(房子到街道的距离先按照地理位置分组再填充各自的中位数)

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



#3. fillna with new type: ‘None’(或者其他不会和已有类名重复的str）

features["PoolQC"] = features["PoolQC"].fillna("None") #note "None" is a str, (NA	No Pool)

#车库相关的类别变量，使用新类别字符串'None'填充空值。

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')

#地下室相关的类别变量，使用字符串'None'填充空值。

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')



#4. fillna with 0: 数值型的特殊变量

#车库相关的数值型变量，使用0填充空值。

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)



#5.填充众数

#对于列名为'Exterior1st'、'Exterior2nd'、'SaleType'的特征列，使用列中的众数填充空值。

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])



#6. 统一填充剩余的数值特征和类别特征

features[numeric_features] = features[numeric_features].apply(

            lambda x: x.fillna(0)) #改进点2：没做标准化，这里把0换成均值更好吧？

features[category_features] = features[category_features].apply(

            lambda x: x.fillna('None')) #改进点3：可以考虑将新类别'None'换成众数
#2.5 data transformation

#数字型数据列偏度校正

#使用skew()方法，计算所有整型和浮点型数据列中，数据分布的偏度（skewness）。

#偏度是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。亦称偏态、偏态系数。 

skew_features = features[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)



#改进点5：调整阈值，原文以0.5作为基准，统计偏度超过此数值的高偏度分布数据列，获取这些数据列的index

high_skew = skew_features[skew_features > 0.15]

skew_index = high_skew.index



#对高偏度数据进行处理，将其转化为正态分布

#Box和Cox提出的变换可以使线性回归模型满足线性性、独立性、方差齐次以及正态性的同时，又不丢失信息

#也可以使用简单的log变换

for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
#2.6 特征删除和融合创建新特征

#features['Utilities'].describe()

#Utilities: all values are the same(AllPub 2914/2915)

#Street: Pave 2905/2917

#PoolQC: too many missing values, del_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu'] missing>50%

#改进点4：删除更多特征del_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu']

features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1) 

#features = features.drop(['Utilities', 'Street', 'PoolQC','MiscFeature', 'Alley', 'Fence'], axis=1) #FireplaceQu建议保留



#融合多个特征，生成新特征

#改进点6：可以尝试组合出更多的特征

features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])



#简化特征。对于某些分布单调（比如100个数据中有99个的数值是0.9，另1个是0.1）的数字型数据列，进行01取值处理。

#PoolArea: unique      13, top          0, freq      2905/2917

#2ndFlrSF: unique      633, top          0, freq      1668/2917

#2ndFlrSF: unique      5, top          0, freq      1420/2917

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

#2.7 get_dummies

print("before get_dummies:",features.shape)

final_features = pd.get_dummies(features).reset_index(drop=True)

print("after get_dummies:",final_features.shape)



X = final_features.iloc[:len(y), :]	

X_sub = final_features.iloc[len(y):, :]

print("after get_dummies, the dataset size:",'X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)
#2.8 #删除取值过于单一（比如某个值出现了99%以上）的特征

overfit = []

for i in X.columns:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 99.94: #改进点7：99.94是可以调整的，80,90,95，99...

        overfit.append(i)



overfit = list(overfit)

overfit.append('MSZoning_C (all)')



X = np.array(X.drop(overfit, axis=1).copy())

y = np.array(y)

X_sub = np.array(X_sub.drop(overfit, axis=1).copy())



print('X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)



print('feature engineering finished!', datetime.now())
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)





#定义均方根对数误差（Root Mean Squared Logarithmic Error ，RMSLE）

def rmsle(y, y_pred):

    return np.sqrt(mse(y, y_pred))



#创建模型评分函数

def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
#3.1 parameters(for grid search)

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
#3.2 single model

#改进点9:more models

#改进点10: 对svr，GradientBoostingRegressor，LGBMRegressor，XGBRegressor等做GridSearchCV

#ridge

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))



#lasso

lasso = make_pipeline(

    RobustScaler(),

    LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))



#elastic net

elasticnet = make_pipeline(

    RobustScaler(),

    ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))



#svm

svr = make_pipeline(RobustScaler(), SVR(

    C=20,

    epsilon=0.008,

    gamma=0.0003,

))



#GradientBoosting（展开到一阶导数）

gbr = GradientBoostingRegressor(n_estimators=3000,

                                learning_rate=0.05,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)



#lightgbm

lightgbm = LGBMRegressor(

    objective='regression',

    num_leaves=4,

    learning_rate=0.01,

    n_estimators=5000,

    max_bin=200,

    bagging_fraction=0.75,

    bagging_freq=5,

    bagging_seed=7,

    feature_fraction=0.2,

    feature_fraction_seed=7,

    verbose=-1,

    #min_data_in_leaf=2,

    #min_sum_hessian_in_leaf=11

)



#xgboost（展开到二阶导数）

xgboost = XGBRegressor(learning_rate=0.01,

                       n_estimators=3460,

                       max_depth=3,

                       min_child_weight=0,

                       gamma=0,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:linear',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       reg_alpha=0.00006)
#3.3 stacking

#StackingCVRegressor：A 'Stacking Cross-Validation' regressor for scikit-learn estimators.

#regressors=(...)中并没有纳入前面的svr模型,似乎纳入svr之后性能反而变差(why?)：stacking模型的性能0.11748--->0.11873

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
#3.4 观察单模型的效果

print('TEST score on CV')



score = cv_rmse(ridge) #cross_val_score(RidgeCV(alphas),X, y) 外层k-fold交叉验证, 每次调用modelCV.fit时内部也会进行k-fold交叉验证

print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), ) #0.1024



score = cv_rmse(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), ) #0.1031



score = cv_rmse(elasticnet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )#0.1031 



score = cv_rmse(svr)

print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), ) #0.1023



score = cv_rmse(lightgbm)

print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )#0.1061



score = cv_rmse(gbr)

print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )#0.1072



score = cv_rmse(xgboost)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), ) #0.1064
#3.5 train the stacking model

#stacking 3步走(可不用管细节，fit会完成stacking的整个流程)：

#1.1 learn first-level model

#1.2 construct a training set for second-level model

#2. train the second-level model:学习第2层的模型，也就是学习如何融合第1层的模型

#3. re-learn first-level model on the entire train set

print('START Fit')

print(datetime.now(), 'StackingCVRegressor')

stack_gen_model = stack_gen.fit(X, y) #Fit ensemble regressors and the meta-regressor.
#4.1 submit stacking result

print('Predict submission', datetime.now(),)

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(stack_gen_model.predict(X_sub)))

submission.head()

submission.to_csv("submission_stacking.csv", index=False) #0.11674
#4.1 在整个训练集上重新训练第1层的单模型和svr，后面blending用(如果直接拿stacking第1层的模型，会报not fit的错误)

print(datetime.now(), 'ridge')

ridge_model_full_data = ridge.fit(X, y)

print(datetime.now(), 'lasso')

lasso_model_full_data = lasso.fit(X, y)

print(datetime.now(), 'elasticnet')

elastic_model_full_data = elasticnet.fit(X, y)

print(datetime.now(), 'GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)

print(datetime.now(), 'xgboost')

xgb_model_full_data = xgboost.fit(X, y)

print(datetime.now(), 'lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)

print(datetime.now(), 'svr')

svr_model_full_data = svr.fit(X, y)
#待混合的models

models = [

    ridge_model_full_data, lasso_model_full_data, elastic_model_full_data,

    gbr_model_full_data, xgb_model_full_data, lgb_model_full_data,

    svr_model_full_data, stack_gen_model

]

len(models)
#linear blending coefficients: public coefs

#order: ridge, lasso, elasticnet, gbr, xgboost, lightgbm, svr, stack

public_coefs = [0.1, 0.1, 0.1, 0.1, 0.15, 0.1, 0.1, 0.25]

bias = 0
def linear_blend_models_predict(data_x,models,coefs, bias):

    tmp=[model.predict(data_x) for model in models]

    tmp = [c*d for c,d in zip(coefs,tmp)]

    pres=np.array(tmp).swapaxes(0,1) #numpy中的reshape不能用于交换维度，一开始的种种问题，皆由此来

    pres=np.sum(pres,axis=1)

    return pres
#4.2 submit blend_models_with_public_coefs 



print('blending models RMSLE score on train data:')

print(rmsle(y, linear_blend_models_predict(X,models,public_coefs, bias)))



#before Blend with Top Kernals submissions

print('Predict submission', datetime.now(),)

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(linear_blend_models_predict(X_sub,models,public_coefs, bias))) #expm1: exp(x) - 1; 注意还要取整

# submission.iloc[:,1] = np.expm1(blend_models_predict(X_sub)) 

submission.head()

submission.to_csv("submission_blend_models_with_public_coefs.csv", index=False) #0.11413
#linear blending coefficients: coefs got by linear regression 

#注意：这里很容易过拟合，所以alphas3的不宜过小

#alphas3 = np.linspace(0,1e3,1001) #如果从０开始，RidgeCV会选择０，train_rmse:0.027818，但test_rmse会很大，即过拟合！

#改进点11：如何得到更合适的系数

alphas3 = [70] #可以继续优化，使train_rmse与public_coefs的结果接近



def blend_models(train_x, train_y, models):

    tmp = [model.predict(train_x) for model in models]

    pres = np.array(tmp).swapaxes(0,1) #一开始用的reshape，注意这与pytorch中不同，不能用于多维的维度间的交换！！！

    print(pres.shape)  #(1457,8)

    #注意要设置fit_intercept=False，否则bias会很大，占主导地位，而系数coef_都很小

    #fit_intercept=False时不求截距，但要求数据提前中心化，并且此时会忽略normalize参数

    #linear = LinearRegression(fit_intercept=False)

    linear = RidgeCV(alphas=alphas3,

                     cv=kfolds,

                     fit_intercept=False,

                     scoring=make_scorer(rmsle, greater_is_better=False)

                    )

    linear = linear.fit(pres, train_y)

    print('linear coefficient:')

    print(linear.coef_)

    print('linear bias:')

    print(linear.intercept_)

    print('best alpha: %f'%(linear.alpha_))

    print('best score: %f'%(rmsle(linear.predict(pres), train_y)))

    return linear.coef_, linear.intercept_
#可以对coefs归一化

coefs, bias = blend_models(X, y, models)

sum(coefs)

# coefs=[i/sum(coefs) for i in coefs.tolist()]

# coefs

# from scipy.special import softmax

# coefs=softmax(coefs)

# coefs
#4.3 submit blend_models_with_regression_coefs 

print('blending models RMSLE score on train data:')

print(rmsle(y, linear_blend_models_predict(X,models,coefs,bias))) #0.059305



#before Blend with Top Kernals submissions

print('Predict submission', datetime.now(),)

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(linear_blend_models_predict(X_sub,models,coefs,bias))) #expm1: exp(x) - 1; 注意还要取整

# submission.iloc[:,1] = np.expm1(blend_models_predict(X_sub)) #expm1: exp(x) - 1; 注意还要取整

submission.head()

submission.to_csv("submission_blend_models_with_regression_coefs.csv", index=False) #0.11492 可以得到相当的效果
#4.4 mixing with the top kernels



print('Blend with Top Kernals submissions', datetime.now(),)

sub_1 = pd.read_csv('../input/top-10-0-10943-stacking-mice-and-brutal-force/House_Prices_submit.csv')

sub_2 = pd.read_csv('../input/hybrid-svm-benchmark-approach-0-11180-lb-top-2/hybrid_solution.csv')

sub_3 = pd.read_csv('../input/lasso-model-for-regression-problem/lasso_sol.csv')

submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(linear_blend_models_predict(X_sub,models,public_coefs, bias)))) + 

                                (0.25 * sub_1.iloc[:,1]) + 

                                (0.25 * sub_2.iloc[:,1]) + 

                                (0.25 * sub_3.iloc[:,1]))  

submission.to_csv("submission_blend_top.csv", index=False) #0.11115

print('Save submission', datetime.now())
#4.5 Brutal approach to deal with predictions close to outer range 

#第超低的房价更低，让超高的房价更高(通常来说，会将小者放大，大者缩小，但房价有其特殊性：有些偏远地区的房子比预测更低，

#有些房子比预测高得多)，这里让这两种极端情况更极端一些，这样更符合房价的特性

#注意缩放的分位数0.005,0.995以及缩放系数0.77,1.1可以适当调整，相关public kernel中并未提到这块儿的参数如何选择，猜测：唯结果论

q1 = submission['SalePrice'].quantile(0.0045) 

q2 = submission['SalePrice'].quantile(0.998)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)



submission.to_csv("submission_blend_top_Scale extremes.csv", index=False) #0.10647(best result)

print('Save submission', datetime.now())
submission = pd.read_csv('../input/house-price-best/submission_best.csv')

submission.to_csv("submission_best.csv", index=False) #0.10647(best result)

print('Save submission best', datetime.now())
#如果不重新训练每个单模型(跳过4.1)，直接拿stack_gen_model.regressors来用，会发现：

#只有第一个模型能直接predict，其他模型会报not fit错误

stack_gen_model.regressors[0].predict(X)

#stack_gen_model.regressors[1].predict(X) #error

#stack_gen_model.regressors[2].predict(X) #error
#how to optimize hyperparameters?

# from pactools.grid_search import GridSearchCVProgressBar #if u need progress bar

# def grid_search(model, parameters, train_x, train_y, progress_bar=False, cv=5):

#     #sklearn的0.22版本默认采用5-fold cv，当前版本默认3折

#     models = GridSearchCVProgressBar(

#         model, parameters, cv=cv, verbose=1,

#         n_jobs=6) if progress_bar else GridSearchCV(

#             model, parameters, cv=cv, n_jobs=6)

#     models.fit(train_x, train_y)

#     print(models.best_params_)

#     print(models.best_score_)

#     #print(models.best_estimator_)



# params1 = {

#     'alpha':

#     [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# }

# grid_search(Ridge(), params1, X, y, progress_bar=True, cv=5)
#一般情况下：小者放大，大者缩小，让极端值尽可能变得正常,但不适用于本问题

q1 = submission['SalePrice'].quantile(0.005)

q2 = submission['SalePrice'].quantile(0.995)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*1.1)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*0.77)



submission.to_csv("submission_base_blend_top_Scale extremes2.csv", index=False) #0.11602 反效果

print('Save submission', datetime.now())