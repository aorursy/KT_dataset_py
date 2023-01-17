# -*- coding: utf-8 -*-

import numpy as np 

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt 

import seaborn as sns

color = sns.color_palette()

sns.set_style('whitegrid')

import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows = 8000





from scipy import stats

from scipy.stats import norm, skew 

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))# 所有float保留3位有效数字



from subprocess import check_output

#查看当前可用的数据目录

print(check_output(["ls", "../input"]).decode("utf8")) 

#导入数据

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print("success")
train.head(3)

print(train.shape)
test.head(3)

print(train.shape)
#数据本来就有列索引，因此去掉Id列

train_ID = train['Id']

test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)

print("去掉Id后的train: %s  去掉Id后的test: %s" %(train.shape,test.shape))
#数据可视化有利于异常点的检测

fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#Deleting outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#要对整体数据进行处理，因此要组合数据。

ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))

all_data.head()
#缺失值检测

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data
plt.figure(figsize=(15,5))

sns.heatmap(all_data.isnull(),cbar= False, yticklabels=False, cmap = "cividis")

plt.title('Missing data by feature', fontsize=15)

f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=10)

plt.ylabel('Percent of missing values', fontsize=10)

plt.title('Percent missing data by feature', fontsize=15)
#丢弃  Utilities的所有值几乎没有变，因此丢弃。

all_data = all_data.drop(['PoolQC','MiscFeature','Alley','Fence','Utilities'], axis=1)

#填充：根据属性值的情况进行填充

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))# 均值填充

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])#众数填充

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

#检查一下现在的确实率

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
#将一下数据转为string类型

#MSSubClass：销售的住房类型

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



#OverallCond：房子整体状况评价

all_data['OverallCond'] = all_data['OverallCond'].astype(str)



#YrSold，MoSold：销售年份和月份

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
# LabelEncoder编码

from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))

       

print('Shape all_data: {}'.format(all_data.shape))

all_data.head()

#Dummy编码

all_data = pd.get_dummies(all_data)

print(all_data.shape)

all_data.head()
# 增加“TotalSF”（总住房面积）属性

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# 特征变量的数据分布



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric = [] # 数值型属性

for i in all_data.columns:

    if all_data[i].dtype in numeric_dtypes:

        numeric.append(i)

print('数值型属性共：',len(numeric),'种')   



sns.set_style("white")

f, ax = plt.subplots(figsize=(9, 10))

ax.set_xscale("log")

ax = sns.boxplot(data=all_data[numeric] , orient="h", palette="Set1")

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
# 找到那些 倾斜度>0.5 的特征

skew_features = all_data[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))

skewness = pd.DataFrame({'Skew' :high_skew})

skew_features.head(10)
# boxcox1p函数进行数据变换使其服从正态分布

for i in skew_index:

    all_data[i] = boxcox1p(all_data[i], boxcox_normmax(all_data[i] + 1))
#观察变换后的盒图

sns.set_style("white")

f, ax = plt.subplots(figsize=(9,10))

ax.set_xscale("log")

ax = sns.boxplot(data=all_data[skew_index] , orient="h", palette="Set1")

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
# 目标变量的数据分布



sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

sns.distplot(y_train, color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")

sns.despine(trim=True, left=True)

plt.show()



#QQ-plot

fig = plt.figure()

res = stats.probplot(y_train, plot=plt)

plt.show()
# 做log(1+x) 变换

y_train = np.log1p(y_train)



sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

sns.distplot(y_train , fit=norm, color="b");



#获取y_train的均值和方差

(mu, sigma) = norm.fit(y_train)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))





plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")

sns.despine(trim=True, left=True)



plt.show()



#QQ-plot

fig = plt.figure()

res = stats.probplot(y_train, plot=plt)

plt.show()
#查看现在的数据情况

all_data.head(3)

print(all_data.shape)
y_train
from sklearn.model_selection import train_test_split

import lightgbm as lgb

import xgboost as xgb

from sklearn.metrics import mean_absolute_error



def lgb_mae(X,y):

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

    # Light Gradient Boosting Regressor

    model = lgb.LGBMRegressor(objective='regression', 

                           num_leaves=6,

                           learning_rate=0.01, 

                           n_estimators=7000,

                           max_bin=200, 

                           bagging_fraction=0.8,

                           bagging_freq=4, 

                           bagging_seed=8,

                           feature_fraction=0.2,

                           feature_fraction_seed=8,

                           min_sum_hessian_in_leaf = 11,

                           verbose=-1,

                           random_state=42)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)

def xgb_mae(X,y):

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

    # XGBoost Regressor

    model = xgb.XGBRegressor(learning_rate=0.01,

                           n_estimators=6000,

                           max_depth=4,

                           min_child_weight=0,

                           gamma=0.6,

                           subsample=0.7,

                           colsample_bytree=0.7,

                           objective='reg:linear',

                           nthread=-1,

                           scale_pos_weight=1,

                           seed=27,

                           reg_alpha=0.00006,

                           random_state=42)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)

#maelist有[lgb][xgb]两列用于存储模型的mae值

maelist=[]
#降维前

proper_lgb=lgb_mae(all_data[:1458],y_train)

proper_xgb=xgb_mae(all_data[:1458],y_train)

print("lgb:%s xgb:%s" %(proper_lgb,proper_xgb))

maelist.append([proper_lgb,proper_xgb])
from sklearn.decomposition import PCA

pca = PCA(0.99)

pca_dataset_X = pca.fit_transform(all_data) 

print("从212维降至：",pca.n_components_)

pd.DataFrame(pca_dataset_X).plot(title='PCA_dataset')
pca_lgb=lgb_mae(pca_dataset_X[:1458],y_train)

pca_xgb=xgb_mae(pca_dataset_X[:1458],y_train)

print("lgb:%s xgp:%s" %(pca_lgb,pca_xgb))

maelist.append([pca_lgb,pca_xgb])
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#lda = LinearDiscriminantAnalysis(0.99)

#lda_dataset_X = lda.fit(x_train, y_train).transform(x_train)

#pd.DataFrame(lda_dataset_X).plot(title='LDA_dataset')
from sklearn.decomposition import FactorAnalysis

fa=FactorAnalysis(n_components=5)

fa_dataset_X = fa.fit_transform(all_data)

pd.DataFrame(fa_dataset_X).plot(title='ICA_dataset')
fa_lgb=lgb_mae(fa_dataset_X[:1458],y_train)

fa_xgb=xgb_mae(fa_dataset_X[:1458],y_train)

print("lgb:%s xgp:%s" %(fa_lgb,fa_xgb))

maelist.append([fa_lgb,fa_xgb])
from sklearn.decomposition import FastICA

ica = FastICA(n_components=5)

ica_dataset_X = ica.fit_transform(all_data)

pd.DataFrame(ica_dataset_X).plot(title='ICA_dataset')
ica_lgb=lgb_mae(ica_dataset_X[:1458],y_train)

ica_xgb=xgb_mae(ica_dataset_X[:1458],y_train)

print("lgb:%s xgp:%s" %(ica_lgb,ica_xgb))

maelist.append([ica_lgb,ica_xgb])
#方差选择法，返回值为特征选择后的数据

from sklearn.feature_selection import VarianceThreshold

#参数threshold为方差的阈值

new_data=VarianceThreshold(threshold=3).fit_transform(all_data)

new_data.shape
f1_lgb=lgb_mae(new_data[:1458],y_train)

f1_xgb=xgb_mae(new_data[:1458],y_train)

print("lgb:%s xgp:%s" %(f1_lgb,f1_xgb))

maelist.append([f1_lgb,f1_xgb])
#SelectKBest函数

from sklearn.feature_selection import SelectKBest  

from array import array

from sklearn.feature_selection import f_regression

skb_model=SelectKBest(f_regression)

skb_model.fit_transform(all_data[:1458],y_train)

new_data.shape
f2_lgb=lgb_mae(new_data[:1458],y_train)

f2_xgb=xgb_mae(new_data[:1458],y_train)

print("lgb:%s xgp:%s" %(f2_lgb,f2_xgb))

maelist.append([f2_lgb,f2_xgb])
# 递归特征消除法

from sklearn.feature_selection import RFE 



#递归特征消除法，返回特征选择后的数据 

#参数estimator为基模型 

#参数n_features_to_select为选择的特征个数 



train_X, val_X, train_y, val_y = train_test_split(all_data[:1458],y_train, random_state = 0)

model1=RFE(estimator=lgb.LGBMRegressor(), n_features_to_select=20)

model1.fit(train_X, train_y)

preds_val1 = model1.predict(val_X)

mae1 = mean_absolute_error(val_y, preds_val1)



model2=RFE(estimator= xgb.XGBRegressor(), n_features_to_select=20)

model2.fit(train_X, train_y)

preds_val2 = model2.predict(val_X)

mae2 = mean_absolute_error(val_y, preds_val2)
print("lgb:%s xgp:%s" %(mae1,mae2))

maelist.append([mae1,mae2])
#基于惩罚项的特征选择法

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression



#带L1惩罚项的线性回归作为基模型的特征选择

one_model = lgb.LGBMRegressor(penalty="l1",

                              objective='regression', 

                           num_leaves=6,

                           learning_rate=0.01, 

                           n_estimators=7000,

                           max_bin=200, 

                           bagging_fraction=0.8,

                           bagging_freq=4, 

                           bagging_seed=8,

                           feature_fraction=0.2,

                           feature_fraction_seed=8,

                           min_sum_hessian_in_leaf = 11,

                           verbose=-1,

                           random_state=42)

one_model.fit(all_data[:1458], y_train)

fe_model = SelectFromModel(one_model,prefit=True)

X_new1 = fe_model.transform(all_data[:1458])

print("X_new1 共有 %s 个特征"%X_new1.shape[1])



two_model = xgb.XGBRegressor( penalty="l1",

                         learning_rate=0.01,

                           n_estimators=6000,

                           max_depth=4,

                           min_child_weight=0,

                           gamma=0.6,

                           subsample=0.7,

                           colsample_bytree=0.7,

                           objective='reg:linear',

                           nthread=-1,

                           scale_pos_weight=1,

                           seed=27,

                           reg_alpha=0.00006,

                           random_state=42)

two_model.fit(all_data[:1458],y_train)

fe_model = SelectFromModel(two_model,prefit=True)

X_new2 = fe_model.transform(all_data[:1458])

print("X_new2 共有 %s 个特征"%X_new2.shape[1])
train_X, val_X, train_y, val_y = train_test_split(X_new1,y_train, random_state = 0)

model1 = lgb.LGBMRegressor(objective='regression', 

                           num_leaves=6,

                           learning_rate=0.01, 

                           n_estimators=7000,

                           max_bin=200, 

                           bagging_fraction=0.8,

                           bagging_freq=4, 

                           bagging_seed=8,

                           feature_fraction=0.2,

                           feature_fraction_seed=8,

                           min_sum_hessian_in_leaf = 11,

                           verbose=-1,

                           random_state=42)

model1.fit(train_X, train_y)

preds_val1 = model1.predict(val_X)

mae1 = mean_absolute_error(val_y, preds_val1)



train_X, val_X, train_y, val_y = train_test_split(X_new2,y_train, random_state = 0)

model2= xgb.XGBRegressor(learning_rate=0.01,

                           n_estimators=6000,

                           max_depth=4,

                           min_child_weight=0,

                           gamma=0.6,

                           subsample=0.7,

                           colsample_bytree=0.7,

                           objective='reg:linear',

                           nthread=-1,

                           scale_pos_weight=1,

                           seed=27,

                           reg_alpha=0.00006,

                           random_state=42)

model2.fit(train_X, train_y)

preds_val2 = model2.predict(val_X)

mae2 = mean_absolute_error(val_y, preds_val2)

print("lgb:%s xgp:%s" %(mae1,mae2))

maelist.append([mae1,mae2])
#基于树模型的特征选择法

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor

train_X, val_X, train_y, val_y = train_test_split(all_data[:1458],y_train, random_state = 0)



model=RandomForestRegressor()

model.fit(train_X, train_y)

preds_val = model.predict(val_X)

mae = mean_absolute_error(val_y, preds_val)

print('随机森林：',mae)
plt.figure(figsize=(17, 7)) 

title = ('all above features selection methods mae results')

plt.title(title,fontsize=20)

plt.ylabel('mae_score')

plt.xticks([0,1,2,3,4,5,6,7,8,9],['proper','jw_PCA', 'jw_FA', 'jw_ICA', 'Filter_VT', 'Filter_F','Wrapper_RFE','Embedded_L1','Embedded_RF'])

plt.grid(axis='y',color="grey",linestyle='--',lw=0.5,alpha=0.5)

plt.tick_params(axis='both',labelsize=14)



import seaborn as sns

sns.despine(left=True,bottom=True)

x=range(8)

data1=[x[0] for x in maelist]

data2=[x[1] for x in maelist]



l1=plt.plot(x,data1)

l2=plt.plot(x,data2)

#随机森林的mae点单独画

l3=plt.scatter(8,mae)

plt.annotate("%s" %round(mae,3),xy=(8,mae) , xytext=(-20, 10), textcoords='offset points')

for x,y1,y2 in zip(x,data1,data2):

    plt.annotate("%s" %round(y1,3),xy=(x,y1) , xytext=(-20, 10), textcoords='offset points')

    plt.annotate("%s" %round(y2,3),xy=(x,y2), xytext=(-20, 10), textcoords='offset points')

plt.text(2,0.128,"xgb",fontdict={'size': 15, 'color':  'red'})

plt.text(2,0.142,"lgb",fontdict={'size': 15, 'color':  'red'})

plt.text(7,0.1,"RandomForest",fontdict={'size': 15, 'color':  'red'})



#提交模型

model = lgb.LGBMRegressor(objective='regression', 

                           num_leaves=6,

                           learning_rate=0.01, 

                           n_estimators=7000,

                           max_bin=200, 

                           bagging_fraction=0.8,

                           bagging_freq=4, 

                           bagging_seed=8,

                           feature_fraction=0.2,

                           feature_fraction_seed=8,

                           min_sum_hessian_in_leaf = 11,

                           verbose=-1,

                           random_state=42)

model.fit(all_data[:1458],y_train)

preds_val = model.predict(all_data[1458:2917])

output = pd.DataFrame({'Id': test_ID,

                       'SalePrice': preds_val})

output.to_csv('submission.csv', index=False)

print("Your first model has finished!")