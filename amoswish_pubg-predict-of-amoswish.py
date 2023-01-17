# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns
from scipy import stats
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
%matplotlib inline
sns.set_style('darkgrid')
sns.set_palette('bone')
f1=open('../input/train_V2.csv')
f2=open('../input/test_V2.csv')
df_train=pd.read_csv(f1)
df_test=pd.read_csv(f2)
#删除空值
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)
# all_data = train.append(test, sort=False).reset_index(drop=True)
# del df_tatrain, test
# gc.collect()
all_data = df_train
all_data1 = df_test
all_data1['winPlacePerc'] = 0
match = all_data.groupby('matchId')
all_data['killPlacePerc'] = match['kills'].rank(pct=True).values
all_data['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values
match1 = all_data1.groupby('matchId')
all_data1['killPlacePerc'] = match1['kills'].rank(pct=True).values
all_data1['walkDistancePerc'] = match1['walkDistance'].rank(pct=True).values
all_data['_totalDistance'] = all_data['rideDistance'] + all_data['walkDistance'] + all_data['swimDistance']
all_data1['_totalDistance'] = all_data1['rideDistance'] + all_data1['walkDistance'] + all_data1['swimDistance']
all_data['zombi'] = ((all_data['_totalDistance'] == 0) | (all_data['kills'] == 0)
                     | (all_data['weaponsAcquired'] == 0) 
                     | (all_data['matchType'].str.contains('solo'))).astype(int)
all_data['cheater'] = ((all_data['kills'] / all_data['_totalDistance'] >= 1)
                       | (all_data['kills'] > 30) | (all_data['roadKills'] > 10)).astype(int)
pd.concat([all_data['zombi'].value_counts(), all_data['cheater'].value_counts()], axis=1).T
all_data1['zombi'] = ((all_data1['_totalDistance'] == 0) | (all_data1['kills'] == 0)
                     | (all_data1['weaponsAcquired'] == 0) 
                     | (all_data1['matchType'].str.contains('solo'))).astype(int)
all_data1['cheater'] = ((all_data1['kills'] / all_data1['_totalDistance'] >= 1)
                       | (all_data1['kills'] > 30) | (all_data1['roadKills'] > 10)).astype(int)
pd.concat([all_data1['zombi'].value_counts(), all_data1['cheater'].value_counts()], axis=1).T
def fillInf(df, val):#删除inf值
    numcols = df.select_dtypes(include='number').columns
    cols = numcols[numcols != 'winPlacePerc']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    for c in cols: df[c].fillna(val, inplace=True)
all_data['_healthAndBoosts'] = all_data['heals'] + all_data['boosts']
all_data['_killDamage'] = all_data['kills'] * 100 + all_data['damageDealt']
#all_data['_headshotKillRate'] = all_data['headshotKills'] / all_data['kills']
all_data['_killPlaceOverMaxPlace'] = all_data['killPlace'] / all_data['maxPlace']
all_data['_killsOverWalkDistance'] = all_data['kills'] / all_data['walkDistance']
#all_data['_killsOverDistance'] = all_data['kills'] / all_data['_totalDistance']
all_data['_walkDistancePerSec'] = all_data['walkDistance'] / all_data['matchDuration']

# suicide: solo and teamKills > 0
#all_data['_suicide'] = ((all_data['players'] == 1) & (all_data['teamKills'] > 0)).astype(int)

fillInf(all_data, 0)
all_data1['_healthAndBoosts'] = all_data1['heals'] + all_data1['boosts']
all_data1['_killDamage'] = all_data1['kills'] * 100 + all_data1['damageDealt']
#all_data['_headshotKillRate'] = all_data['headshotKills'] / all_data['kills']
all_data1['_killPlaceOverMaxPlace'] = all_data1['killPlace'] / all_data1['maxPlace']
all_data1['_killsOverWalkDistance'] = all_data1['kills'] / all_data1['walkDistance']
#all_data['_killsOverDistance'] = all_data['kills'] / all_data['_totalDistance']
all_data1['_walkDistancePerSec'] = all_data1['walkDistance'] / all_data1['matchDuration']

# suicide: solo and teamKills > 0
#all_data['_suicide'] = ((all_data['players'] == 1) & (all_data['teamKills'] > 0)).astype(int)

fillInf(all_data1, 0)
all_data.drop(['headshotKills','teamKills','roadKills','vehicleDestroys'], axis=1, inplace=True)
all_data.drop(['rideDistance','swimDistance','matchDuration'], axis=1, inplace=True)
all_data.drop(['rankPoints','killPoints','winPoints'], axis=1, inplace=True)
all_data1.drop(['headshotKills','teamKills','roadKills','vehicleDestroys'], axis=1, inplace=True)
all_data1.drop(['rideDistance','swimDistance','matchDuration'], axis=1, inplace=True)
all_data1.drop(['rankPoints','killPoints','winPoints'], axis=1, inplace=True)
all_data.drop(['Id','groupId','matchId'], axis=1, inplace=True)
all_data1.drop(['Id','groupId','matchId'], axis=1, inplace=True)
mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
#mapper = lambda x: 'solo' if ('solo' in x) else 'team'
all_data['matchType'] = all_data['matchType'].map(mapper)
all_data1['matchType'] = all_data['matchType'].map(mapper)
# 设置哑变量
a = pd.get_dummies(all_data['matchType'],prefix='matchType')
all_data = pd.concat([all_data,a],axis=1)
all_data1 = pd.concat([all_data1,a],axis=1)
all_data.drop(['matchType'], axis=1, inplace=True)
all_data1.drop(['matchType'], axis=1, inplace=True)
train_y = all_data['winPlacePerc']#吃鸡概率
all_data.drop(['winPlacePerc'], axis=1, inplace=True)
train_x = all_data
predict = all_data1
from sklearn import preprocessing
for item in list(train_x.columns):
#     all_data1[item].apply(preprocessing.scale,axis = 0)
#     all_data1[item] = preprocessing.scale(all_data1[item])
    train_x[item] = np.log1p(train_x[item])
    predict[item] = np.log1p(predict[item])
predict.drop(['winPlacePerc'], axis=1, inplace=True)
#删除空值
predict.dropna(inplace=True)
train_x.dropna(inplace=True)
import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, random_state=1)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print (linreg.intercept_)
print (linreg.coef_)
y_pred = linreg.predict(X_test)
from sklearn import metrics
# 用scikit-learn计算MSE
print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#十折交叉验证
y = train_y#吃鸡概率
x = train_x
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg, x, y, cv=10)
# 用scikit-learn计算MSE
print ("MSE:",metrics.mean_squared_error(y, predicted))
# 用scikit-learn计算RMSE
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
result = linreg.predict(predict)
f3=open('../input/sample_submission_V2.csv')
submit=pd.read_csv(f3)
sample_result = pd.DataFrame(result,columns = ['winPlacePerc'])
submit['winPlacePerc'] = sample_result
submit.to_csv(r'sample_submission_lineregression.csv', index=False)
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取
# ========Lasso回归========
# model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
# model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
alpha = np.logspace(-3,2,10)
model = LassoCV(alphas=alpha,cv=5)  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
model.fit(X_train, y_train)   # 线性回归建模
print('系数矩阵:\n',model.coef_)
print('线性回归模型:\n',model)
print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效

# 使用模型预测
predicted_lasso = model.predict(predict)
# 存储文件
f4=open('../input/sample_submission_V2.csv')
submit_lasso=pd.read_csv(f4)
sample_result_lasso = pd.DataFrame(predicted_lasso,columns = ['winPlacePerc'])
submit_lasso['winPlacePerc'] = sample_result_lasso
submit_lasso.to_csv(r'sample_submission_lasso.csv', index=False)
from sklearn.linear_model import RidgeCV,LassoCV#用这个自带交叉验证参数
from sklearn.model_selection import GridSearchCV#如果使用RidgeCV就不用GridSearchCV这个API了
#使用RidgeCV来建立参数
alpha = np.logspace(-3,2,10)#生成超参数，10的-3次方到10的2次方的等差数列
ridge = RidgeCV(alpha,cv=5)
ridge.fit(X_train,y_train)
ridge.alpha_#输出超参数的值
#使用Ridge配合GridSearchCV来做
from sklearn.linear_model import Ridge,Lasso
ridge_model = GridSearchCV(Ridge(),param_grid={'alpha':alpha},cv=5)
ridge_model.fit(X_train,y_train)
ridge_model.best_params_#验证模型效果
# 使用模型预测
predicted_ridge = ridge_model.predict(predict)
# 存储文件
f4=open('../input/sample_submission_V2.csv')
submit_ridge=pd.read_csv(f4)
sample_result_ridge = pd.DataFrame(predicted_ridge,columns = ['winPlacePerc'])
submit_ridge['winPlacePerc'] = sample_result_ridge
submit_ridge.to_csv(r'sample_submission_ridge.csv', index=False)
from sklearn.linear_model import ElasticNetCV#用这个自带交叉验证参数
from sklearn.model_selection import GridSearchCV#如果使用RidgeCV就不用GridSearchCV这个API了
#ElasticNetCV
alpha = np.logspace(-3,2,10)#生成超参数，10的-3次方到10的2次方的等差数列
elasticNet = ElasticNetCV(alpha,cv=10)
elasticNet.fit(X_train,y_train)
elasticNet.alpha_#输出超参数的值
#使用Ridge配合GridSearchCV来做
from sklearn.linear_model import ElasticNet
elasticNet_model = GridSearchCV(ElasticNet(),param_grid={'alpha':alpha},cv=10)
elasticNet_model.fit(X_train,y_train)
elasticNet_model.best_params_#验证模型效果
# 使用模型预测
predicted_elasticNet = elasticNet_model.predict(predict)
# 存储文件
f4=open('../input/sample_submission_V2.csv')
submit_elasticNet=pd.read_csv(f4)
sample_result_elasticNet = pd.DataFrame(predicted_elasticNet,columns = ['winPlacePerc'])
submit_elasticNet['winPlacePerc'] = sample_result_elasticNet
submit_elasticNet.to_csv(r'sample_submission_elasticNet.csv', index=False)
