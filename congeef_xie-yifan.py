import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
train = pd.read_csv(r"D:\housing_price_prediction\train.csv")
test= pd.read_csv(r"D:\housing_price_prediction\test.csv")
train.head(5)
plt.hist(train['median_house_value'],color='blue')
plt.show()
train=train[train['median_house_value']<500000]
plt.hist(train['median_house_value'],color='blue')
plt.show()
train.shape
train.shape,test.shape
y_train=train.pop('median_house_value')  #删除并返回数据集中median_house_value标签列
all_df=pd.concat((train,test),axis=0)
all_df.drop(['ocean_proximity'], axis = 1,inplace = True)
all_df.head(5)
all_df.info()
all_df.drop(['id'], axis = 1,inplace = True)#处理掉ID列
all_df.info()
total=all_df.isnull().sum().sort_values(ascending=False)  #每列缺失数量
percent=(all_df.isnull().sum()/len(all_df)).sort_values(ascending=False) #每列缺失率
miss_data=pd.concat([total,percent],axis=1,keys=['total','percent'])
miss_data #显示每个列及其对应的缺失率
#部分少量的缺失值，不是很重要，可以用one-hotd转变离散值，然后均值补齐
all_dummies_df=pd.get_dummies(all_df)
mean_col=all_dummies_df.mean()
all_df.fillna(mean_col,inplace=True)
N=len(test)
#处理后的训练集(不含median_house_value)
df_train1=all_df.iloc[:int(len(all_df)-N),:]    

df_train_train=df_train1.iloc[0:int(0.8*len(df_train1)),:]  #train中的训练集(不含median_house_value)
df_train_test=df_train1.iloc[int(0.8*len(df_train1)):,:]    #train中的测试集(不含median_house_value)

df_train_train_y=y_train.iloc[0:int(0.8*len(y_train))]     #train中训练集的target
df_train_test_y=y_train.iloc[int(0.8*len(df_train1)):]     #train中测试集的target

#处理后的测试集
df_test1=all_df.iloc[int(len(all_df)-N):,:] 
#加载相关库
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
#调参,对随机森林的最大特征选择进行调试  ，也需要用到交叉验证
from sklearn.ensemble import RandomForestRegressor
max_features=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
test_score=[]
for max_feature in max_features:
    clf=RandomForestRegressor(max_features=max_feature,n_estimators=100)
    score=np.sqrt(cross_val_score(clf,df_train_train,df_train_train_y,cv=5))
    test_score.append(1-np.mean(score))

plt.plot(max_features,test_score) #得出误差得分图
rf=RandomForestRegressor(max_features=0.9,n_estimators=100)
rf.fit(df_train_train,df_train_train_y)
#用均方误差来判断模型好坏，结果越小越好
(((df_train_test_y-rf.predict(df_train_test))**2).sum())/len(df_train_test_y)
#计算一下R方
from sklearn.metrics import r2_score
r2_score(df_train_test_y, rf.predict(df_train_test))
pred=rf.predict(df_test1)
pred.shape
result=pd.DataFrame({'id':test.id, 'predicted':pred})
result.to_csv("D:\housing_price_prediction\pred.csv",index=False)
result
