import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

#中文

plt.rcParams['font.sans-serif']=['SimHei']

plt.rcParams['axes.unicode_minus']=False
#读取训练集

data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

data
data.info()
# 缺失值处理

# 很多特征还不及数据个数的三分之一，所以选择舍去

data.drop(columns = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis = 1,inplace = True)
data.describe()
#填补数值型缺失值

data['LotFrontage'].fillna(data['LotFrontage'].median(),inplace = True)

data['MasVnrArea'].fillna(data['MasVnrArea'].mean(),inplace = True)
#字符型变量缺失值

miss_index_list = data.isnull().any()[data.isnull().any().values == True].index.tolist()

print(miss_index_list)
miss_list = []

for i in miss_index_list:

    miss_list.append(data[i].values.reshape(-1,1))

print(miss_list)
from sklearn.impute import SimpleImputer

#用众数填补数值型变量

for i in range(len(miss_list)):

    imp_most = SimpleImputer(strategy='most_frequent')

    imp_most = imp_most.fit_transform(miss_list[i])

    data.loc[:,miss_index_list[i]] = imp_most
data.info()
#哑变量/独热编码

from sklearn.preprocessing import OneHotEncoder

data_ = data.copy()

data_.drop('Id',axis = 1,inplace = True)

#选出字符型的特征

ob_features = data_.select_dtypes(include=['object']).columns.tolist()

ob_features
#哑变量转化

OneHot = OneHotEncoder(categories='auto')

result = OneHot.fit_transform(data_.loc[:,ob_features]).toarray()

print(result,"\n")

print(result.shape)
#矩阵转DataFrame

OneHotnames = OneHot.get_feature_names().tolist()#获取特征名

OneHotDf = pd.DataFrame(result,columns=OneHotnames)

OneHotDf
#删去原始字符型无序变量

data_.drop(columns=ob_features,inplace=True)

data_.head()
#合并哑变量和数值型变量成一个新DataFrame

data_ = pd.concat([OneHotDf,data_],axis=1)

data_
from sklearn.feature_selection import VarianceThreshold

#过滤掉方差小于0.1的特征

transfer = VarianceThreshold(threshold=0.1)

new_data1 = transfer.fit_transform(data_)

#get_support得到的需要留下的下标索引

var_index = transfer.get_support(True).tolist()

print(len(data_.columns.tolist())-len(var_index))
#过滤方差后剩下的特征

data1 = data_.iloc[:,var_index]

data1.head()
#皮尔逊相关系数

from scipy.stats import pearsonr

pear_num = [] #存系数 

pear_name = [] #存特征名称

feature_names = data1.columns.tolist()

#得到每个特征与SalePrice间的相关系数

for i in range(0,len(feature_names)-1):

    print('%s和%s之间的皮尔逊相关系数为%f'%(feature_names[i],feature_names[-1],pearsonr(data1[feature_names[i]],data1[feature_names[-1]])[0]))

    if (abs(pearsonr(data1[feature_names[i]],data1[feature_names[-1]])[0])>0.5):

        pear_num.append(pearsonr(data1[feature_names[i]],data1[feature_names[-1]])[0])

        pear_name.append(feature_names[i])
print(pear_num)

print(pear_name)

for i,v in enumerate(ob_features):

    if i in [17,29,32]:

        print(v)
#相关系数较高的特征合并成一个新的dataframe

pear_dict = {"features":pear_name,"pearsonr":pear_num}

highpear_df = pd.DataFrame(pear_dict)

highpear_df = highpear_df.sort_values(by = ['pearsonr'],ascending = False)

highpear_df = highpear_df.reset_index(drop = True)

highpear_df
#关系矩阵图

cols = highpear_df['features'].tolist()

cols.append('SalePrice')

corr = np.corrcoef(data1[cols].values.T)

plt.figure(figsize=(12,12))

sns.set(font_scale=1.25)

sns.heatmap(corr,cbar=True,annot=True,square=True,fmt='.2f',yticklabels=cols, xticklabels=cols)

plt.show()
#TotalBsmtSF（房间总数）和GrLivArea（地面以上居住面积）相关系数0.82

#GarageCars（车库可装车辆个数）和GarageArea（车库面积）相关系数0.84 ，比较冲突

#TotalBsmtSF（地下室总面积）和1stFlrSF（第一层面积）相关系数0.85 ， 比较冲突

#个人觉得可以将TotalBsmtSF和GrLivArea都保留，剩下取GarageArea和otalBsmtSF两个特征
highpear_df.drop([2,5],inplace=True)

#重置索引

highpear_df = highpear_df.reset_index(drop = True)

highpear_df
#data2高相关特征和一个类标签

x = highpear_df.loc[:,'features'].values.tolist()

x.append('SalePrice')

data2 = data1.loc[:,x]

data2
sns.pairplot(data2[x], size = 2.5)

plt.show()
plt.figure(figsize=(10,12),dpi = 80)

plt.subplot(311)

sns.boxplot(x = 'OverallQual',y = 'SalePrice',data = data2)

plt.subplot(312)

sns.boxplot(x = 'FullBath',y = 'SalePrice',data = data2)

plt.subplot(313)

sns.boxplot(x = 'TotRmsAbvGrd',y = 'SalePrice',data = data2)

plt.show()
plt.figure(figsize=(10,12),dpi = 80)

plt.subplot(311)

sns.boxplot(x = 'x32_Unf',y = 'SalePrice',data = data2)

plt.subplot(312)

sns.boxplot(x = 'x29_TA',y = 'SalePrice',data = data2)

plt.subplot(313)

sns.boxplot(x = 'x17_TA',y = 'SalePrice',data = data2)

plt.show()
plt.figure(figsize=(10, 20),dpi = 80)

#绘制连续变量的散点图

plt.subplot(511)

sns.scatterplot(x = 'GrLivArea',y = 'SalePrice',data = data2)

plt.subplot(512)

sns.scatterplot(x = 'GarageArea',y = 'SalePrice',data = data2)

plt.subplot(513)

sns.scatterplot(x = 'TotalBsmtSF',y = 'SalePrice',data = data2)

plt.subplot(514)

sns.scatterplot(x = 'YearBuilt',y = 'SalePrice',data = data2)

plt.subplot(515)

sns.scatterplot(x = 'YearRemodAdd',y = 'SalePrice',data = data2)

plt.show()
plt.figure(figsize=(16, 12),dpi = 80)

sns.boxplot(x = 'MSSubClass',y = 'SalePrice',data = data1)

plt.show()
plt.figure(figsize=(16, 12),dpi = 80)

sns.boxplot(x = 'Neighborhood',y = 'SalePrice',data = data)

plt.show()
#随机森林API

from sklearn.ensemble import RandomForestRegressor

#线性回归API

from sklearn.linear_model import LinearRegression

##网格搜索交叉验证

from sklearn.model_selection import GridSearchCV,cross_val_score

#划分数据集

from sklearn.model_selection import train_test_split

#标准化

from sklearn.preprocessing import StandardScaler

#误差

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')
#目标变量

class_label = data2.iloc[:,-1]

class_label
#data3删去类标签，开始建模，在训练集上挑选较优算法

data3 = data2.copy()

data3.drop("SalePrice",axis = 1,inplace = True)

cols = data3.columns.tolist()

x = data3[cols].values

y = class_label.values

#取三成数据当成测试数据

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)

#实例化

sta = StandardScaler()

#标准化数据

x_train_sta = sta.fit_transform(x_train)

x_test_sta = sta.fit_transform(x_test)

y_train_sta = sta.fit_transform(y_train.reshape(-1,1))

y_test_sta = sta.fit_transform(y_test.reshape(-1,1))
#计算模型的均方误差

clfs = {'rfr':RandomForestRegressor(),

      'LR':LinearRegression()}

for clf in clfs:

        clfs[clf].fit(x_train_sta,y_train_sta)

        prediction = clfs[clf].predict(x_test_sta)

        print(clf + " RMSE:" + str(np.sqrt(metrics.mean_squared_error(y_test_sta,prediction))))
#计算调参之前模型得分

score = cross_val_score(RandomForestRegressor(),x_train,y_train,cv=5).mean()

print(score)
#随机森林进行调参

param_grid = {

    #'n_estimators':np.arange(100,400,50),

    #'max_depth':np.arange(10,25,2),

    #'max_features':np.arange(1,10,1)

    #'min_samples_leaf':np.arange(1,5,1)

    #'min_samples_split':np.arange(2,10,1)

}

rf = RandomForestRegressor(n_estimators=300,max_depth =20,max_features =5,

                            min_samples_leaf =2,min_samples_split=2)



grid = GridSearchCV(rf,param_grid=param_grid,cv = 5)

grid.fit(x_train,y_train)
#grid.best_params_

grid.best_score_
#调参后提高的分数

grid.best_score_-score
#最优模型

rf_reg = grid.best_estimator_
#特征重要度

rf_reg.feature_importances_
index_ = data2.columns.tolist()

index_.remove('SalePrice')

#将特征重要度重构一个dataframe

feature_df = pd.DataFrame(rf_reg.feature_importances_,columns=['imp'])

feature_df.index = index_

#从大到小排序

feature_df = feature_df.sort_values(by = 'imp',ascending=False)

feature_df
value = feature_df["imp"].values.tolist()

plt.figure(figsize=(10,10))

plt.pie(value,autopct="%0.1f%%",labels=feature_df.index)

plt.show()
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

test_data
test_data.info()
#对测试集做与训练集一样的处理

test_data.drop(columns = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis = 1,inplace = True)

test_data['LotFrontage'].fillna(test_data['LotFrontage'].median(),inplace = True)

test_data['MasVnrArea'].fillna(test_data['MasVnrArea'].mean(),inplace = True)

miss_index_list = test_data.isnull().any()[test_data.isnull().any().values == True].index.tolist()

print(miss_index_list)

miss_list = []

for i in miss_index_list:

    miss_list.append(test_data[i].values.reshape(-1,1))

print(miss_list)

#用众数填补数值型变量

for i in range(len(miss_list)):

    imp_most = SimpleImputer(strategy='most_frequent')

    imp_most = imp_most.fit_transform(miss_list[i])

    test_data.loc[:,miss_index_list[i]] = imp_most
ob_features = test_data.select_dtypes(include=['object']).columns.tolist()

OneHot1 = OneHotEncoder(categories='auto')

result = OneHot1.fit_transform(test_data.loc[:,ob_features]).toarray()

result
OneHotnames = OneHot1.get_feature_names().tolist()#获取特证名

OneHotDf = pd.DataFrame(result,columns=OneHotnames)

test_data = pd.concat([OneHotDf,test_data],axis=1)

test_data
test_value_x = test_data[index_].values

#test_value_x = sta.fit_transform(test_value_x)

test_value_x
#用建好模型预测SalePrice

test_value_y = rf_reg.predict(test_value_x)

print(test_value_y)

print(test_value_y.shape)
submission_Id = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv",usecols=['Id'])

SalePrice = pd.DataFrame(test_value_y,columns=['SalePrice'])

Submission = pd.concat([submission_Id,SalePrice],axis=1)

Submission.to_csv("./submission.csv",index = False)