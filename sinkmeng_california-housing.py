import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
data_path='/kaggle/input/california-housing-prices/housing.csv'
data_init = pd.read_csv(data_path)

data = data_init.copy()
data.head()
#可以看出名为’total_bedrooms‘的属性部分值有缺失，后续将考虑填充。此外除'ocean_proximity'为对象类型外，其余属性均为浮点型数值类型。

data.info()
data.describe()
feature=data.drop('median_house_value',axis=1)

label=data['median_house_value']
# 房价中位数在500001.0处的统计最多，共计3842个唯一值,占到总数据长度的18.6%

label.value_counts()
len(label.unique())/len(label)
label.describe()
label.hist(bins=50,figsize=(8,6),color='b',alpha=.7)



plt.title('label')

plt.xlabel('house_value')

plt.ylabel('counts')

plt.grid(False)      #不显示网格

plt.show()
feature.hist(bins=50,figsize=(20,15),color='b',alpha=.7)

plt.show()
#筛选出数据类型为'object'的特征

category_list=[column for column in feature.columns if feature[column].dtype=='object']

category_list

feature[category_list]
#统计该类别特征的值的分布，因为dataframe格式无法使用value_counts()方法，所以将该特征用series格式呈现

feature.loc[:,category_list[0]].value_counts()
corr = data.corr()

corr
#只考虑相关性大小，不考虑正负，并按照绝对值大小排序

corr['median_house_value'].abs().sort_values(ascending=False)
#结果显示，bedrooms_per_population的相关性要比total_bedrooms和population各自与标签的相关性都要高

#而bedrooms_per_house的相关性就没有total_bedrooms和households各自与标签的相关性都要高

data_1=data.copy()

data_1['bedrooms_per_population']=data_1['total_bedrooms']/data_1['population']

data_1['bedrooms_per_house']=data_1['total_bedrooms']/data_1['households']

data_1.corr()['median_house_value'].abs().sort_values(ascending=False)
data.plot(kind='scatter',x='longitude',y='latitude'

          ,s=data['population']/100,label='population'#以人口密度值区别散点的大小

          ,alpha=.4                                 #设置小的透明度，会突出颜色更深的点

          ,figsize=(16,12)                           #设置画布大小

          ,c='median_house_value'                   #颜色深度以房价高低衡量

          ,cmap=plt.get_cmap('jet')                 #选择colormap

          ,colorbar=True

         ) 

plt.legend()

plt.show()
import pandas_profiling #数据探索分析库，简单高效生成交互式数据报告

data_profile=data.profile_report(style={'full_width':True})

data_profile
def split_train_test(data,test_ratio):

    np.random.seed(42)

    shuffled_index = np.random.permutation(len(data))#随机生成指定长度范围内不重复随机数序列

    test_set_size = int(len(data)*test_ratio)

    test_index = shuffled_index[:test_set_size]

    train_index = shuffled_index[test_set_size:]

    return data.iloc[train_index],data.iloc[test_index]
test_ratio=0.2

train_set,test_set=split_train_test(data,test_ratio)

print(len(train_set),'train_set',len(test_set),'test_set')
from sklearn.model_selection import train_test_split

train_f,test_f = train_test_split(data,test_size=0.2,random_state=42)#添加随机种子保证每次运行划分的都是一样的结果
train_x,test_x,train_y,test_y= train_test_split(feature,label,test_size=0.2,random_state=42)
corr_matrix = data.corr()

corr_matrix['median_house_value'].abs().sort_values(ascending=False)
data['median_income'].describe()
plt.figure()

plt.hist(data['median_income'],bins=50)

# plt.grid()

plt.show()
data['income_cat'] = np.ceil(data['median_income']/1.5)

data['income_cat'].where(data['income_cat']<5,5.0,inplace=True)# series.where(),小于5则保持原样，大于5赋值为5.0

data['income_cat'].value_counts()
plt.figure()

plt.hist(data['income_cat'])

# plt.grid()

plt.show()
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1       #进行一次划分

                               ,test_size=0.2    #测试集占比0.2

                               ,random_state=42  #设置随机种子保证每次运行划分的数据集不会发生变化

                              )

for train_index,test_index in split.split(data

                                          ,data['income_cat']#选择分层抽样依据的属性（或特征）

                                         ):

    strat_train_set=data.loc[train_index]

    strat_test_set=data.loc[test_index]
x_strat=strat_train_set['income_cat'].value_counts(normalize=True)

x_strat
x_full=data['income_cat'].value_counts(normalize=True)

x_full
train_rand,test_rand=train_test_split(data,test_size=0.2,random_state=42)

x_rand=train_rand['income_cat'].value_counts(normalize=True)

x_rand
strat_bia=(x_strat-x_full)/x_full*100

rand_bia=(x_rand-x_full)/x_full*100

compare=pd.DataFrame([x_full,x_strat,x_rand,strat_bia,rand_bia],index=['x_full','x_strat','x_rand','strat_bia_%','rand_bia_%']).T

compare
for set in (strat_train_set,strat_test_set):

    set.drop(['income_cat'],axis=1)

    
data1=data.drop('income_cat',axis=1)
data1.isnull().sum()
# hdata.drop(['total_bedrooms'],axis=1,inplace=True)

#data.dropna(subset=['total_bedrooms'])

data_median_fill=data1.fillna(data1.median())

data_median_fill.info()
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy='median')#以中位数填充缺失值

data_num=data.drop('ocean_proximity',axis=1)

columns=data_num.columns

data_num=imputer.fit_transform(data_num)

data_num=pd.DataFrame(data_num,columns=columns)

data_median_fill['bedrooms_per_population']=data_median_fill['total_bedrooms']/data_median_fill['population']
#分层抽样,这里我们将先前分层抽样的几步抽象成一个函数，传入要分层抽样的数据集和分层抽样的特征，返回训练集和测试集

def stratifiedshufflesplit(data_,feature='median_income'):

    data_['cat'] = np.ceil(data_[feature]/1.5)           #此处我们默认使用收入中位数作为分层抽样的依据

    data_['cat'].where(data_['cat']<5,5.0,inplace=True)



    from sklearn.model_selection import StratifiedShuffleSplit

    split_ = StratifiedShuffleSplit(n_splits=1   #进行一次划分

                               ,test_size=0.2    #测试集占比0.2

                               ,random_state=42  #设置随机种子保证每次运行划分的数据集不会发生变化

                              )

    for train_index,test_index in split.split(data_

                                          ,data_['cat']#选择分层抽样依据的属性（或特征）

                                         ):

        strat_train=data_.loc[train_index]

        strat_test=data_.loc[test_index]

        strat_train=strat_train.drop('cat',axis=1)

        strat_test=strat_test.drop('cat',axis=1)

    data_.drop('cat',axis=1,inplace=True)

    return strat_train,strat_test



strat_train,strat_test=stratifiedshufflesplit(data_median_fill,feature='median_income')



train_x,train_y=strat_train.drop(['median_house_value','ocean_proximity'],axis=1),strat_train['median_house_value']

test_x,test_y=strat_test.drop(['median_house_value','ocean_proximity'],axis=1),strat_test['median_house_value']
from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor(max_depth=10)

dtr=dtr.fit(train_x,train_y)



from sklearn.metrics import mean_squared_error

predict=dtr.predict(test_x)

mse=mean_squared_error(test_y,predict)

rmse1=np.sqrt(mse)

rmse1
from sklearn.ensemble import RandomForestRegressor

rfg=RandomForestRegressor(n_estimators=10,max_depth=10,random_state=0)

rfg=rfg.fit(train_x,train_y)



predicts=rfg.predict(test_x)

mse_=mean_squared_error(test_y,predicts)

rmse_2=np.sqrt(mse_)

rmse_2
data2=data_median_fill.copy()

data2_num=data2.drop(['ocean_proximity','median_house_value'],axis=1)

data2_cat=data2['ocean_proximity']

data2_label=data2['median_house_value']
from sklearn.preprocessing import OneHotEncoder

data2_cat=pd.DataFrame(data2_cat)



encoder=OneHotEncoder(categories='auto')

data2_cat_onehot=encoder.fit_transform(data2_cat).toarray()    #输出是稀疏矩阵的一种存储方式，需转换成数组



#独热编码实例的学习参数，显示多个文本类型数据在编码前的文本值数组，每一个文本特征存储一个数组

encoder.categories_
data2_cat_onehot=pd.DataFrame(data2_cat_onehot,columns=encoder.categories_)

data2_cat_onehot.head()
columns_list=data2_num.columns.tolist()
columns_list.extend(encoder.categories_[0].tolist())

columns_list
data2_num = data2_num.sub(data2_num.min())/(data2_num.max()-data2_num.min())
data2_num.describe()
data2_num_onehot=pd.DataFrame(np.c_[data2_num,data2_cat_onehot],columns=columns_list)#np.c_[]数组横向拼接成数组,np.r_[]纵向拼接

data2_num_onehot
data2=pd.concat([data2_num_onehot,data2_label],axis=1)

data2.info()
strat_train,strat_test=stratifiedshufflesplit(data2,feature='median_income')



train_x,train_y=strat_train.drop(['median_house_value'],axis=1),strat_train['median_house_value']

test_x,test_y=strat_test.drop(['median_house_value'],axis=1),strat_test['median_house_value']
from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor(max_depth=10)

dtr=dtr.fit(train_x,train_y)



from sklearn.metrics import mean_squared_error

predict=dtr.predict(test_x)

mse=mean_squared_error(test_y,predict)

rmse3=np.sqrt(mse)

print('独热编码前后决策树表现分别为：\n前：{}，\n后：{}'.format(rmse1,rmse3))
dtr.feature_importances_
feature_sort_dtr=list(zip(dtr.feature_importances_,train_x.columns))#重要性在前，排序按照元祖对第一个元素排序

sorted(feature_sort_dtr,reverse=True)
corr['median_house_value'].abs().sort_values(ascending=False)
train_x.columns[np.argmax(dtr.feature_importances_)]
from sklearn.ensemble import RandomForestRegressor

rfg=RandomForestRegressor(n_estimators=10,max_depth=10)

rfg=rfg.fit(train_x,train_y)



predicts=rfg.predict(test_x)

mse_=mean_squared_error(test_y,predicts)

rmse_4=np.sqrt(mse_)

print('独热编码前后随机森林表现分别为：\n前：{}，\n后：{}'.format(rmse_2,rmse_4))
feature_sort_rfg=list(zip(rfg.feature_importances_,train_x.columns))

sorted(feature_sort_rfg,reverse=True)
train_x.columns[np.argmax(rfg.feature_importances_)]
# pip install joblib 直接安装joblib,无需从sklearn.externals模块导入
# from sklearn.externals import joblib

import joblib

joblib.dump(dtr,'dtr.pkl')

joblib.dump(rfg,'rfg.pkl')
dtr_load=joblib.load('dtr.pkl')

dtr_load
dtr
rfg_load=joblib.load('rfg.pkl')

rfg_load
from sklearn.model_selection import GridSearchCV

param_grid=dict(n_estimators=[10,30],max_depth=[4,6,8,10],min_samples_split=[2,3,4],min_samples_leaf=[2,3,4])

rfg_gs=RandomForestRegressor()

grid_search=GridSearchCV(rfg_gs,param_grid,cv=5,scoring='neg_mean_squared_error')



grid_search.fit(train_x,train_y)
grid_search.best_params_
grid_search.best_estimator_
grid_search.cv_results_
best_model=grid_search.best_estimator_

joblib.dump(best_model,'best_model_rfg.pkl')
test_predictions=best_model.predict(test_x)

test_mse=mean_squared_error(test_y,test_predictions)

test_rmse=np.sqrt(test_mse)
test_rmse
best_model.feature_importances_
best_model.get_params()
best_model.set_params()
param_grid=dict(n_estimators=list(range(20,41)),max_depth=[10],min_samples_split=[3],min_samples_leaf=[4])

rfg_gs_estimator=RandomForestRegressor()

grid_search=GridSearchCV(rfg_gs_estimator,param_grid,cv=5,scoring='neg_mean_squared_error')



grid_search.fit(train_x,train_y)
test_predictions=grid_search.predict(test_x)

test_mse=mean_squared_error(test_y,test_predictions)

test_rmse=np.sqrt(test_mse)

test_rmse
grid_search.best_params_
best_model_1=grid_search.best_estimator_

joblib.dump(best_model_1,'best_model_1.pkl')