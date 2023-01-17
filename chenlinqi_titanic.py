#参考https://zhuanlan.zhihu.com/p/28802636

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train_data = pd.read_csv('/kaggle/input/titanic/train.csv',index_col = 'PassengerId')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv',index_col = 'PassengerId')

data_all = pd.concat([train_data,test_data],axis=0)
#1~891训练集，891~1309测试集

data_all.info()
data_all.tail()
#查看Embarked缺失的两列情况

data_all.loc[data_all.loc[:,'Embarked'].isnull(),:]
#62号乘客和830号乘客的票价Fare都是80， Pclass都是1，那么先假设票价、客舱和等级相同的乘客是在同一个登船港口登船。

data_all.groupby(['Embarked','Pclass'])['Fare'].median()
#Embarked为C且Pclass为1的乘客的Fare中位数为80。因此可以将缺失的Embarked值设置为“C”

data_all['Embarked'].fillna('C',inplace = True)
#Fare值的缺失位置和相关信息

data_all.loc[data_all['Fare'].isnull(),:]
data_all.groupby(['Embarked','Pclass'])['Fare'].median()
#S登船港口上船且Pclass为3的乘客费用Fare的中位数为8.05，因此Fare的空缺值补充为8.05。

data_all['Fare'].fillna(8.05,inplace = True)
#age原始数据的分布情况

data_all['Age'].hist(bins = 10,grid = False,density = True,figsize = (10,6))
#age先考虑用随机森林填充

#选择的预测变量Sex,Embarked,Fare,SibSp,Parch,Pclass

from sklearn.preprocessing import OneHotEncoder

data_ = data_all.copy()

X_co = data_[['Sex','Embarked']]#需要编码的变量Sex,Embarked

enc = OneHotEncoder().fit(X_co)

result = OneHotEncoder().fit_transform(X_co).toarray()

enc.get_feature_names()
#拼接填补缺失值用的特征矩阵

X = pd.concat([data_all.loc[:,['Fare','SibSp','Parch','Pclass']],pd.DataFrame(result,index = np.arange(1,1310),columns = ['female','male','C','Q','S'])],axis=1)
X.head()
#标签

Y = data_all['Age']
#划分测试集，训练集

Y_tr = Y[Y.notnull()]

Y_te = Y[Y.isnull()]

X_tr = X.loc[Y_tr.index,:]

X_te = X.loc[Y_te.index,:]
#随机森林填补缺失值

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=1000,random_state=0)

rfr.fit(X_tr,Y_tr)

Y_te = rfr.predict(X_te)
data_.loc[data_['Age'].isnull(),'Age'] = Y_te
plt.subplot(121)

data_all['Age'].hist(bins = 10,grid = False,density = True,figsize = (10,6))

plt.title('before fillna')

plt.subplot(122)

data_['Age'].hist(bins=10,grid = False,density = True,figsize = (10,6))

plt.title('after fillna')
data_all['Age'] = data_['Age']
#数据分析
#Pclass对生存率的影响

print(data_all.groupby(['Pclass','Survived'])['Survived'].count())

print(data_all.groupby(['Pclass'])['Survived'].mean())
Pclass_count = data_all.groupby(['Pclass','Survived'])['Survived'].count()

Pclass_mean = data_all.groupby(['Pclass'])['Survived'].mean()

Pclass_count.unstack().plot(kind = 'bar',figsize = (10,6),fontsize = 15,rot = 0)

Pclass_mean.plot(kind = 'bar',figsize = (10,6),fontsize = 15,rot =0)
Pclass_mean.plot(kind = 'bar',figsize = (15,6),fontsize = 15,rot =0)
#Name映射

title_list = []

for i in data_all['Name']:

    title_list.append(i.split(',')[1].split('.')[0].strip())
pd.Series(title_list).value_counts()
#map()函数作用与Series，apply函数作用于DataFrame的一行或一列；applymap函数作用与DataFrame的每一个元素

#坑爹玩意··········分出来的title前面有空格，匹配半天匹配不上！！！！！！！！

title_mapDict = {'Capt':'Officer'

                 ,"Col":"Officer"

                 ,"Major":"Officer"

                 ,"Jonkheer":"Royalty"

                 ,"Don":"Royalty"

                 ,'Dona':'Royalty'

                 ,"Sir":"Royalty"

                 ,"Dr":"Officer"

                 ,"Rev":"Officer"

                 ,"the Countess":"Royalty"

                 ,"Mme":"Mrs"

                 ,"Mlle":"Miss"

                 ,"Ms":"Mrs"

                 ,"Mr":"Mr"

                 ,"Mrs" :"Mrs"

                 ,"Miss" :"Miss"

                 ,"Master" :"Officer"

                 ,"Lady" :"Royalty"

                    }

title_ = pd.Series(title_list).str.strip().map(title_mapDict)
data_all['Name'] = title_
Name_count = data_all.groupby(['Name','Survived'])['Survived'].count()

print(Name_count)

Name_count.unstack().plot(kind = 'bar',figsize = (10,6),fontsize = 15,rot =0,stacked = True)
Name_mean = data_all.groupby(['Name'])['Survived'].mean()

print(Name_mean)

Name_mean.plot(kind = 'bar',figsize = (10,6),fontsize = 15,rot =0)
#Sex对生存率的影响

Sex_count = data_all.groupby(['Sex','Survived'])['Survived'].count()

print(Sex_count)

Sex_count.unstack().plot(kind = 'bar',figsize = (10,6),fontsize = 15,rot =0)
Sex_mean = data_all.groupby(['Sex'])['Survived'].mean()

print(Sex_mean)

Sex_mean.plot(kind = 'bar',figsize = (10,6),fontsize = 15,rot =0)
#不等舱位等级，不同性别的生还概率

print('不同舱位等级的生还人数：\n',data_all.groupby(['Pclass','Sex','Survived'])['Survived'].count())

print('--------------------------')

print('不同舱位等级的生还概率：\n',data_all.groupby(['Pclass','Sex'])['Survived'].mean())
data_all.groupby(['Pclass','Sex'])['Survived'].mean().unstack().plot(kind = 'bar',rot = 0,figsize = (10,6),fontsize = 15)

plt.xlabel('Pclass',fontdict = {'size' : 15})

plt.ylabel('Survived Rate',fontdict = {'size':15})

plt.legend(fontsize = 15)
#家庭人口对生存的影响

data_all['Family'] = data_all['SibSp'] + data_all['Parch']

Family_count = data_all.groupby(['Family','Survived'])['Survived'].count()

print(Family_count)

Family_count.unstack().plot(kind = 'bar',figsize = (10,6),fontsize = 15,rot =0)
Family_mean = data_all.groupby(['Family'])['Survived'].mean()

print(Family_mean)

Family_mean.plot(kind = 'bar',figsize = (10,6),fontsize = 15,rot =0)
#age对生存的影响

data_all.loc[data_all['Survived'] == 0,:]['Age'].hist(bins = 10,figsize = (10,6),label = 'Not Survived',grid = False)

data_all.loc[data_all['Survived'] == 1,:]['Age'].hist(bins = 10,figsize = (10,6),label = 'Survived',grid = False)

plt.legend()
#Age_new分箱

bins = [0,18,40,81]

Age_new = pd.cut(data_all['Age'],bins)
data_all['Age_new'] = Age_new
Age_count = data_all.groupby(['Age_new','Survived'])['Survived'].count()

print(Age_count)

Age_count.unstack().plot(kind = 'bar',figsize = (10,6),fontsize = 15,rot =0)
Age_sex_count = data_all.groupby(['Age_new','Sex','Survived'])['Survived'].count()

print(Age_sex_count)

Age_sex_count.unstack().plot(kind = 'bar',figsize = (20,6),fontsize = 15,rot =0)
#登船港口对生存的影响

Embarked_count = data_all.groupby(['Embarked','Survived'])['Survived'].count()

print(Embarked_count)

Embarked_count.unstack().plot(kind = 'bar',figsize = (10,6),fontsize = 15,rot =0)
Embarked_mean = data_all.groupby(['Embarked'])['Survived'].mean()

print(Embarked_mean)

Embarked_mean.plot(kind = 'bar',figsize = (10,6),fontsize = 15,rot = 0)
data_all.info()
#Age_new\Embarked\Fare\Pclass\Sex\Family

#其中Age_new/Embarked/Sex需要进行编码

from sklearn.preprocessing import OneHotEncoder

X_code = data_all.loc[:,['Age_new','Embarked','Sex']]

ohe = OneHotEncoder().fit(X_code)

result_X_code = ohe.fit_transform(X_code).toarray()

ohe.get_feature_names()
new_col = ['(0, 18]','(18, 40]','(40, 81]','C','Q','S','female','male']

X = pd.concat([data_all.loc[:,['Fare','Pclass','Family']],pd.DataFrame(result_X_code,columns = new_col,index = np.arange(1,1310))],axis=1)
Y_train = data_all.loc[1:891,'Survived']

Y_test = data_all.loc[892:1309,'Survived']

X_train = X.loc[1:891,:]

X_test = X.loc[892:1309,:]
X.tail()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.tree import export_graphviz

import graphviz
clf = DecisionTreeClassifier(random_state=16,criterion='entropy',max_depth=8)

cross_val_score(clf,X_train,Y_train,cv=10).mean()
clf.fit(X_train,Y_train)

clf.feature_importances_
feature_name = ['Fare','Pclass','Family','(0,18]','(18,40]','(40,81]','C','Q','S','female','male']

dot_data = export_graphviz(clf

                          ,feature_names = feature_name

                          ,filled = True

                          ,rounded = True

                          )

graph = graphviz.Source(dot_data)

graph
#max_depth学习曲线

score = []

for i in range(1,15):

    clf = DecisionTreeClassifier(random_state=0,criterion='entropy',max_depth=i)

    score.append(cross_val_score(clf,X_train,Y_train,cv=10).mean())



print(max(score),score.index(max(score)))

plt.plot(np.arange(1,15),score)
#min_sample_split学习曲线

score = []

for i in range(1,15):

    clf = DecisionTreeClassifier(random_state=15,criterion='entropy',max_depth=8,min_samples_split=i)

    score.append(cross_val_score(clf,X_train,Y_train,cv=10).mean())



print(max(score),score.index(max(score)))

plt.plot(np.arange(1,15),score)
#min_sample_leaf学习曲线

score = []

for i in range(1,15):

    clf = DecisionTreeClassifier(random_state=0,criterion='entropy',max_depth=8,min_samples_leaf=i)

    score.append(cross_val_score(clf,X_train,Y_train,cv=10).mean())



print(max(score),score.index(max(score)))

plt.plot(np.arange(1,15),score)
#random_state学习曲线

score = []

for i in range(1,300):

    clf = DecisionTreeClassifier(random_state=i,criterion='entropy',max_depth=8)

    score.append(cross_val_score(clf,X_train,Y_train,cv=10).mean())



print(max(score),score.index(max(score)))

plt.plot(np.arange(1,300),score)
#最小信息增益min_impurity_decrease

score = []

for i in np.linspace(0,0.5,20):

    clf = DecisionTreeClassifier(random_state=16,criterion='entropy',max_depth=8,min_impurity_decrease=i)

    score.append(cross_val_score(clf,X_train,Y_train,cv=10).mean())



print(max(score),score.index(max(score)))

plt.plot(np.linspace(0,0.5,20),score)
#预测结果

clf = DecisionTreeClassifier(random_state=16,criterion='entropy',max_depth=8)

clf.fit(X_train,Y_train)

Y_test = clf.predict(X_test)
result = pd.concat([pd.Series(np.arange(892,1310)),pd.Series(Y_test)],axis=1)

result.columns =  ['PassengerId','Survived']
result
result.to_csv(r'submission.csv',index = False)