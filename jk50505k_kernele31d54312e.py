# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np 

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from datetime import datetime

from sklearn import preprocessing

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import os

print(os.listdir('../input/longhu87v2'))

from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['KaiTi']

# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.model_selection import train_test_split
df=pd.read_excel('../input/longhu87v2/longhu-8.7v3.xlsx',encoding='gbk')
df.head()
df['人均消费'].fillna(df['人均消费'].mean(),inplace=True)
df['中学学校距离'].describe()
df.isnull().describe()
df=df[df['价格']>5000]

# df['价格'] = np.log1p(df['价格'])

df=df[df['价格']<15000]
# df[['小学学校能级','中学学校能级','三甲医院个数','特殊医院个数','景区个数','中型商业个数','三甲医院门诊量','其他因素个数','交通类个数','不利景观个数','公交线路个数','大型商业个数']]=\

# df[['小学学校能级','中学学校能级','三甲医院个数','特殊医院个数','景区个数','中型商业个数','三甲医院门诊量','其他因素个数','交通类个数','不利景观个数','公交线路个数','大型商业个数']]\

# .fillna(0)
df['中学学校距离'].fillna(10000,inplace=True)

df['小学学校距离'].fillna(10000,inplace=True)

df.fillna(4000,inplace=True)
tmp=df.columns

df.columns=['X' + str(i) for i in range(1,56)]

corrmat = df.corr()

f, ax = plt.subplots(figsize=(24, 24))

sns.heatmap(corrmat, vmax= .8, square=True); 

df.columns=tmp
#saleprice correlation matrix

k = 30   #number of variables for heatmap，热力图变量数量 



# nlargest - 根据SalePrice列排序，返回前10个跟SalePrice相关性最高的行

cols = corrmat.nlargest(k,'X4')['X4'].index 



# cm = corrmat.loc[cols,cols] 同以下cm赋值相同

# 训练集中取出目标列的样本，转置，计算10个特征之间的相关性

cm = np.corrcoef(df[cols].values.T)

f, ax = plt.subplots(figsize=(24, 24))

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
X=df.loc[:,['大型商业个数','轨交线路数','体育馆个数','中型商业个数','公园个数','人均消费','生鲜超市个数','幼儿园平均距离','景观湖距离','小学学校距离',\

            '文化宫科技馆博物馆最短距离','三甲医院开车距离','大型商业开车距离','健身房最短距离','小学学校能级']]

y=df['价格']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
train_data=np.asarray(x_train)

test_data=np.asarray(x_test)

train_target=np.asarray(y_train)

test_target=np.asarray(y_test)
mean=train_data.mean(axis=0)

train_data-=mean

std=train_data.std(axis=0)

train_data/=std



test_data-=mean

test_data/=std
from keras import models 

from keras import layers



def build_model():

    model=models.Sequential()

    model.add(layers.Dense(60,activation='relu',input_shape=(train_data.shape[1],)))

#     model.add(layers.Dense(32,activation='relu'))

    model.add(layers.Dense(120,activation='relu'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(60,activation='relu'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

    return model



#采用k折验证方法

k=4 

num_val_samples=len(train_data)//k

num_epochs=500

all_scores=[]

all_mae_histories=[]



for i in range(k):

    print('processing fold #',i)

    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]

    val_targets=train_target[i*num_val_samples:(i+1)*num_val_samples]

    

    partial_train_data=np.concatenate(

    [train_data[:i*num_val_samples],

    train_data[(i+1)*num_val_samples:]],axis=0)

    

    partial_train_target=np.concatenate(

    [train_target[:i*num_val_samples],

    train_target[(i+1)*num_val_samples:]],axis=0)

#     print(val_data,val_targets)

    model=build_model()

    history=model.fit(partial_train_data,partial_train_target,epochs=num_epochs,batch_size=64,verbose=0,validation_data=(val_data,val_targets))

    val_mse,val_mae=model.evaluate(val_data,val_targets,verbose=0)

    mae_history=history.history['val_mean_absolute_error']

    all_mae_histories.append(mae_history)

    all_scores.append(val_mae)
np.mean(all_scores)
average_mae_history=[np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt



plt.plot(range(1,len(average_mae_history)+1),average_mae_history)

def smooth_curve(points,factor=0.9):

    smoothed_points=[]

    for point in points:

        if smoothed_points:

            previous=smoothed_points[-1]

            smoothed_points.append(previous*factor+point*(1-factor))

        else:

            smoothed_points.append(point)

    return smoothed_points



smooth_mae_history=smooth_curve(average_mae_history[100:])

plt.figure(figsize=(15,8))

plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
test_mse_score,test_mae_socre=model.evaluate(test_data,test_target)

test_mae_socre
dnn=model
#设置k折交叉验证的参数。

kfolds = KFold(n_splits=10, shuffle=True, random_state=10)





#定义均方根对数误差（Root Mean Squared Logarithmic Error ，RMSLE）

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))





#创建模型评分函数，根据不同模型的表现打分

#cv表示Cross-validation,交叉验证的意思。

def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
alphas_alt = [13.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



#定义ridge岭回归模型（使用二范数作为正则化项。不论是使用一范数还是二范数，正则化项的引入均是为了降低过拟合风险。）

#注：正则化项如果使用二范数，那么对于任何需要寻优的参数值，在寻优终止时，它都无法将某些参数值变为严格的0，尽管某些参数估计值变得非常小以至于可以忽略。即使用二范数会保留变量的所有信息，不会进行类似PCA的变量凸显。

#注：正则化项如果使用一范数，它比L2范数更易于获得“稀疏(sparse)”解，即它的求解结果会有更多的零分量。

ridge = make_pipeline(RobustScaler(), RidgeCV( cv=kfolds))



#定义LASSO收缩模型（使用L1范数作为正则化项）（由于对目标函数的求解结果中将得到很多的零分量，它也被称为收缩模型。）

#注：正则化项如果使用二范数，那么对于任何需要寻优的参数值，在寻优终止时，它都无法将某些参数值变为严格的0，尽管某些参数估计值变得非常小以至于可以忽略。即使用二范数会保留变量的所有信息，不会进行类似PCA的变量凸显。

#注：正则化项如果使用一范数，它比L2范数更易于获得“稀疏(sparse)”解，即它的求解结果会有更多的零分量。										

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7,cv=kfolds))



#定义elastic net弹性网络模型（弹性网络实际上是结合了岭回归和lasso的特点，同时使用了L1和L2作为正则化项。）									

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7,))



#定义SVM支持向量机模型                                     

svr = make_pipeline(RobustScaler(), SVR())



#定义GB梯度提升模型（展开到一阶导数）									

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)                             



#定义lightgbm模型									

lightgbm = LGBMRegressor(objective='regression', 

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



#定义xgboost模型（展开到二阶导数）                                      

xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,

                                     max_depth=2, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
print('进行交叉验证，计算不同模型的得分TEST score on CV')



#打印二范数rideg岭回归模型的得分

score = cv_rmse(ridge)

print("二范数rideg岭回归模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



#打印一范数LASSO收缩模型的得分

score = cv_rmse(lasso)

print("一范数LASSO收缩模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



#打印elastic net弹性网络模型的得分

score = cv_rmse(elasticnet)

print("elastic net弹性网络模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



#打印SVR支持向量机模型的得分

score = cv_rmse(svr)

print("SVR支持向量机模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



#打印lightgbm轻梯度提升模型的得分

score = cv_rmse(lightgbm)

print("lightgbm轻梯度提升模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



#打印gbr梯度提升回归模型的得分

score = cv_rmse(gbr)

print("gbr梯度提升回归模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



#打印xgboost模型的得分

score = cv_rmse(xgboost)

print("xgboost模型的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), ) 



stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
#########使用训练数据特征矩阵作为输入，训练数据对数处理后的预测房价作为输出，进行各个模型的训练-【开始】#########

#开始集合所有模型，使用stacking方法

print('进行模型参数训练 START Fit')



print(datetime.now(), '对stack_gen集成器模型进行参数训练')

stack_gen_model = stack_gen.fit(np.array(x_train), np.array(y_train))



print(datetime.now(), '对elasticnet弹性网络模型进行参数训练')

elastic_model_full_data = elasticnet.fit(x_train, y_train)



print(datetime.now(), '对一范数lasso收缩模型进行参数训练')

lasso_model_full_data = lasso.fit(x_train, y_train)



print(datetime.now(), '对二范数ridge岭回归模型进行参数训练')

ridge_model_full_data = ridge.fit(x_train, y_train)



print(datetime.now(), '对svr支持向量机模型进行参数训练')

svr_model_full_data = svr.fit(x_train, y_train)



print(datetime.now(), '对GradientBoosting梯度提升模型进行参数训练')

gbr_model_full_data = gbr.fit(x_train, y_train)



print(datetime.now(), '对xgboost二阶梯度提升模型进行参数训练')

xgb_model_full_data = xgboost.fit(x_train, y_train)



print(datetime.now(), '对lightgbm轻梯度提升模型进行参数训练')

lgb_model_full_data = lightgbm.fit(x_train, y_train)

#########使用训练数据特征矩阵作为输入，训练数据对数处理后的预测房价作为输出，进行各个模型的训练-【结束】#########
from sklearn import preprocessing



def blend_models_predict(X):

    return ((0.03 * elastic_model_full_data.predict(X)) + \

            (0.03 * lasso_model_full_data.predict(X)) + \

            (0.05 * ridge_model_full_data.predict(X)) + \

            (0.02 * svr_model_full_data.predict(X)) + \

            (0.1 * gbr_model_full_data.predict(X)) + \

            (0.15 * xgb_model_full_data.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.2*np.array([i[0] for i in dnn.predict(preprocessing.scale(X))]))+\

            (0.3 * stack_gen_model.predict(np.array(X))))

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(blend_models_predict(x_train),y_train))

print(mean_absolute_error(blend_models_predict(x_test),y_test))
blend_models_predict(x_test)
predict_df = pd.DataFrame({'price': y_test, 'pre_test': blend_models_predict(x_test)})



predict_df.to_csv('Submission.csv')
model=XGBRegressor()

moldel=model.fit(X,y)

weights={}

# print(model.feature_importances_)

for i in range(len(X.columns)):

    weights[X.columns[i]]=model.feature_importances_[i]

print(weights)
model=LGBMRegressor()

moldel=model.fit(X,y)

weights={}

# print(model.feature_importances_)

for i in range(len(X.columns)):

    weights[X.columns[i]]=model.feature_importances_[i]

print(weights)
lasso= LassoCV(max_iter=1e7)

model=lasso.fit(train_data,train_target)

weights={}

for i in range(len(X.columns)):

    weights[X.columns[i]]=model.coef_[i]

    

print(weights)


from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor(criterion='mae')

model=clf.fit(X,y)

for i in range(len(X.columns)):

    weights[X.columns[i]]=model.feature_importances_[i]

print(weights)