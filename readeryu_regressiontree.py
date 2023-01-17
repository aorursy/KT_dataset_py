import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn import ensemble

from copy import copy

from sklearn.metrics import mean_absolute_error

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

## 基础的数据探索，观察缺失值

print('The train data have {} columns and {} rows.'.format(train.shape[1],train.shape[0]))

print('There are {} missing values in total.'.format(train.isnull().sum().sum()))

null_series = train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)

null_data=pd.DataFrame(null_series,columns=['NullValues'])

null_data['Percentage']= null_series.apply(lambda x:'{:.2%}'.format(x/1460))

null_data['Type']=None

for idx in null_data.index:

    null_data.loc[idx,'Type'] = str(train[idx].dtype)

bar_plot = null_data['NullValues']/1460

missing_data_fig=plt.figure(figsize=((5,5))) 

bar_plot.plot(kind='bar')

missing_data_fig.savefig('missing_data_fig.png')

null_data.T
## 找出所有数值变量，除了Id和Saleprice

coltype = {}

for col in train.columns:

    if train[col].dtype!='object':

        coltype[col] = train[col].dtype

num_vars = list(set(coltype.keys())-{'Id','SalePrice'})



## 缺失值占比太大的变量,通过后续的箱型图进一步决定是否放弃，
## 观察训练数据集和测试集合数值变量是否同分布

train.hist(bins=50,figsize=(30,20),column=num_vars)

plt.tight_layout(pad=0.4)

plt.savefig('train_distribution.png')
test.hist(bins=50,figsize=(30,20),column=num_vars)

plt.tight_layout(pad=0.4)

plt.savefig('test_distribution.png')
## 观察训练数据集和测试集合字符变量是否同分布

train.describe(include=np.object)
test.describe(include=np.object)
corr_fig = plt.figure(figsize=(30,15))

tmp = num_vars

tmp.append('SalePrice')

corrdata = train[tmp].corr()

del tmp

sns.heatmap(corrdata,vmax=1,cmap="Blues",annot=True)

corr_fig.savefig('corr.png')
# 按照相关系数选择十个数值变量参与建模

label_col='SalePrice'

corrdata[label_col].sort_values(ascending=False)[1:11]
char_var = list(train.select_dtypes(include=np.object).columns)

base_color = sns.color_palette()[0]

i=1

char_distribution = plt.figure(figsize=(100,20))

for col_name in char_var:

    plt.subplot(9,5,i)

    sns.boxplot(data = train, x = col_name, y = 'SalePrice',color=base_color)

    i+=1

char_distribution.tight_layout(pad=0.4)

char_distribution.show()

char_distribution.savefig('char_distribution.png')
num_feature = list(corrdata[label_col].sort_values(ascending=False)[1:11].index)

print(num_feature)

char_feature = ['MSZoning','Alley','Neighborhood','Condition2','RoofMatl','Exterior2nd',

                'BsmtQual','KitchenQual','PoolQC','MiscFeature','SaleType','SaleCondition']

print(char_feature)
print('需要处理缺失值的字符特征:')

print(set(list(null_data.index))&set(char_feature))

print('需要处理缺失值的数值特征:')

print(set(list(null_data.index))&set(num_feature))
train[['Alley','PoolQC','BsmtQual','MiscFeature']].describe()
## 根据上表，从备选特征中剔除PoolQC,MiscFeature和Alley。对BsmtQual进行众数('TA')填充

train['BsmtQual'] = train['BsmtQual'].fillna('TA')



features = copy(char_feature)

features.extend(num_feature)

print(len(features))

features.remove('PoolQC')

features.remove('MiscFeature')

features.remove('Alley')

print((len(features)))
## 以下数据参与建模

X = pd.get_dummies(train[features])

Y = train['SalePrice']
from sklearn.metrics import mean_squared_error

params = {'n_estimators': 3000, 'max_depth':3, 'min_samples_split': 10,'min_samples_leaf':15,

           'subsample':0.3,'learning_rate': 0.05, 'loss': 'huber'}



# 寻找合适参数

for n in [100,500,1000,2000]:

    params['n_estimators']=n

    clf = ensemble.GradientBoostingRegressor(**params)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.33)

    

    clf.fit(X_train,y_train)

    mse = mean_squared_error(y_test, clf.predict(X_test))

    print('n_estimators={:d}时：'.format(n))

    print("测试集上MSE: {:.4f}\t测试集上MAE: {:.4f}" .format(mse,mean_absolute_error(y_test, clf.predict(X_test))))

    

    print("训练集上MSE: {:.4f}\t训练集上MAE: {:.4f}" .format(mean_squared_error(y_train, clf.predict(X_train)),

                                                  mean_absolute_error(y_train, clf.predict(X_train))))  

#确定最终模型

final_params = {'n_estimators': 1000, 'max_depth':3, 'min_samples_split': 10,'min_samples_leaf':15,

           'subsample':0.3,'learning_rate': 0.05, 'loss': 'huber'}

model = ensemble.GradientBoostingRegressor(**final_params)

model.fit(X,Y)
print('score:{:.5f}'.format(mean_squared_error(np.log(y_test),np.log(model.predict(X_test)))))
x_pred = test[features]

missing_in_pred = list(x_pred.isnull().sum()[x_pred.isnull().sum()>0].sort_values(ascending=False).index)

print(missing_in_pred)

x_pred.describe(include=np.object)
## 字符变量众数填充，数值变量平均数填充

values = {'BsmtQual':'TA', 'MSZoning':'RL','SaleType':'WD','KitchenQual':'TA','Exterior2nd':'VinylSd',

         'TotalBsmtSF': 1046.1179698216736, 'GarageArea': 472.76886145404666, 'GarageCars': 1.7661179698216736}

x_for_pred = x_pred.copy()

x_for_pred[missing_in_pred] = x_pred.fillna(value=values)[missing_in_pred]

x_for_pred.describe()
pred_input = pd.get_dummies(x_for_pred)

for item in ['Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'Exterior2nd_Other', 'RoofMatl_ClyTile',

           'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll']:

    pred_input[item]=0
result = pd.DataFrame(model.predict(pred_input),index = test.Id,columns=['SalePrice'])

result.to_csv('my_submition.csv')
import zipfile

zf = zipfile.ZipFile('zipfile_write.zip',mode='w')

try:

    print('adding missing_data_fig.png')

    zf.write('missing_data_fig.png')

    print('adding train_distribution.png')

    zf.write('train_distribution.png')

    print('adding char_distribution.png')

    zf.write('char_distribution.png')

    print('adding corr.png')

    zf.write('corr.png')

    print('adding corr.png')

    zf.write('test_distribution.png')

finally:

    print('closing')

    zf.close()