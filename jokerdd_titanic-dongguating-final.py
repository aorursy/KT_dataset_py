# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd #数据分析

import numpy as np #科学计算

from pandas import Series,DataFrame



data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_train

data_train.info()
%matplotlib inline 

import matplotlib.pyplot as plt

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



# plt.subplot2grid((2,3),(0,0))             # 这段我注释掉了，因为这个同学的baseline的语法已经被淘汰了

data_train.Survived.value_counts().plot(kind='bar')# 图表呈现为柱状图 

plt.title(u"whether survived(0,1)") # 设置标题

plt.ylabel(u"population")  # y轴的轴标

plt.show()



 # 剩下的三个图原理一样，就是利用matplotlib库画图，大同小异。

# plt.subplot2grid((2,3),(0,1))

data_train.Pclass.value_counts().plot(kind="bar")

plt.ylabel(u"population")

plt.title(u"Pclass")

plt.show()





plt.subplot2grid((2,3),(0,2))

plt.scatter(data_train.Survived, data_train.Age)

plt.ylabel(u"Age")                         # 设定纵坐标名称

plt.grid(b=True, which='major', axis='y') 

plt.title(u"relationship between Age and survived(0,1)")

plt.show()





# plt.subplot2grid((2,3),(1,0), colspan=2)

data_train.Age[data_train.Pclass == 1].plot(kind='kde')   

data_train.Age[data_train.Pclass == 2].plot(kind='kde')

data_train.Age[data_train.Pclass == 3].plot(kind='kde')

plt.xlabel(u"Age")# plots an axis lable

plt.ylabel(u"density") 

plt.title(u"relationship between Age and Pclass")

plt.legend((u'1st', u'2nd',u'3rd'),loc='best') # sets our legend for our graph.

plt.show()





# plt.subplot2grid((2,3),(1,2))

data_train.Embarked.value_counts().plot(kind='bar')

plt.title(u"Embarked")

plt.ylabel(u"population")  

plt.show()


#### 查看Survived与Pclass的关系

Survived_Pclass=data_train['Pclass'].groupby(data_train['Survived'])

Survived_Pclass.value_counts().unstack()

 

Survived_Pclass.value_counts().unstack().plot(kind='bar',stacked = True)

plt.show()



#### 查看Survived与Pclass的关系，横纵坐标调换位置。

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()

df=pd.DataFrame({u'survived':Survived_1, u'unsurvived':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"relationship between survived and Pclass")

plt.xlabel(u"Pclass") 

plt.ylabel(u"population") 

plt.show()





#### 查看Survived与Sex的关系

 

Survived_Sex=data_train['Sex'].groupby(data_train['Survived'])

Survived_Sex.value_counts().unstack()

 

Survived_Sex.value_counts().unstack().plot(kind='bar',stacked=True)

plt.show()
#然后我们再来看看各种舱级别情况下各性别的获救情况

fig=plt.figure()

fig.set(alpha=0.65) # 设置图像透明度，无所谓

plt.title(u"Pclass and age")



ax1=fig.add_subplot(141)   # 依旧是进行画图。将pclass与survived读入。

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')

ax1.set_xticklabels([u"survived", u"unsurvived"], rotation=0)

ax1.legend([u"female/higher Pclass"], loc='best')



ax2=fig.add_subplot(142, sharey=ax1)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')

ax2.set_xticklabels([u"unsurvived", u"survived"], rotation=0)

plt.legend([u"female/lower Pclass"], loc='best')



ax3=fig.add_subplot(143, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')

ax3.set_xticklabels([u"unsurvived", u"survived"], rotation=0)

plt.legend([u"male/higher Pclass"], loc='best')



ax4=fig.add_subplot(144, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')

ax4.set_xticklabels([u"unsurvived", u"survived"], rotation=0)

plt.legend([u"male/lower Pclass"], loc='best')



plt.show()
### 利用交叉表crosstab

 

pd.crosstab(data_train.Age,data_train['Survived'])

 

# pd.crosstab(data_train.Age,data_train['Survived']).plot(kind='bar',stacked=True)  #也是柱状图，交叉表有一点点bug，我还不太熟练，但感觉效果特别直观！！

pd.crosstab(data_train.Age,data_train['Survived']).plot(kind='bar')

plt.show()

from sklearn.ensemble import RandomForestRegressor

 

### 使用 RandomForestClassifier 填补缺失的年龄属性

def set_missing_ages(df):

    

    # 把已有的数值型特征取出来丢进Random Forest Regressor中

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]



    # 乘客分成已知年龄和未知年龄两部分

    known_age = age_df[age_df.Age.notnull()].as_matrix()

    unknown_age = age_df[age_df.Age.isnull()].as_matrix()



    # y即目标年龄，known age是一个特征矩阵，第一行是age，其他行以用特征属性值，这里取了第一行

    y = known_age[:, 0]



    # X即特征属性值

    X = known_age[:, 1:]



    # fit到RandomForestRegressor之中

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

    rfr.fit(X, y)

    

    # 用得到的模型进行未知年龄结果预测

    predictedAges = rfr.predict(unknown_age[:, 1::])

    

    # 用得到的预测结果填补原缺失数据

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df, rfr



def set_Cabin_type(df):

    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"

    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"

    return df



data_train, rfr = set_missing_ages(data_train)

data_train = set_Cabin_type(data_train)



dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')      #代码解释见下方markdown



dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')



dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')



dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')



df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df

import sklearn.preprocessing as preprocessing  #解释见markdown

scaler = preprocessing.StandardScaler()

age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))

df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)

fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))

df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)

df.head()
from sklearn import linear_model #就是调用线性回归方程，机械化的操作，并不是很难。



# 用正则取出我们要的属性值

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

train_np = train_df.values



# y即第0列：Survival结果

y = train_np[:, 0]



# X即第1列及以后：特征属性值

X = train_np[:, 1:]



# fit到LogisticRegression之中

clf = linear_model.LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)

clf.fit(X, y)



clf
data_test = pd.read_csv("/kaggle/input/titanic/test.csv")

data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0



# 接着我们对test_data做和train_data中一致的特征变换

# 首先用同样的RandomForestRegressor模型填上丢失的年龄



tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

null_age = tmp_df[data_test.Age.isnull()].values



# 根据特征属性X预测年龄并补上

X = null_age[:, 1:]

predictedAges = rfr.predict(X)

data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges



data_test = set_Cabin_type(data_test)

dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')





df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)

df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)

df_test.head()
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions = clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})

result.to_csv("/kaggle/working/logical.csv", index=False)
pd.read_csv("/kaggle/working/logical.csv").head()
pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})
from sklearn.model_selection import cross_val_score, train_test_split



 #简单看看打分情况

clf = linear_model.LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)

all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

X = all_data.values[:,1:]

y = all_data.values[:,0]

# print(cross_validation.cross_val_score(clf, X, y, cv=5))

print(cross_val_score(clf, X, y, cv=5))
split_train, split_cv = train_test_split(df, test_size=0.3, random_state=42)



train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

# 生成模型

clf = linear_model.LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)

clf.fit(train_df.values[:,1:], train_df.values[:,0])



# 对cross validation数据进行预测



cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions = clf.predict(cv_df.values[:,1:])



origin_data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:,0]]['PassengerId'].values)]

bad_cases.head(10)
import numpy as np

import matplotlib.pyplot as plt

# from sklearn.learning_curve import learning_curve  修改以fix learning_curve DeprecationWarning

from sklearn.model_selection import learning_curve



# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 

                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):



#     estimator : 你用的分类器。

#     title : 表格的标题。

#     X : 输入的feature，numpy类型

#     y : 输入的target vector

#     ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点

#     cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)

#     n_jobs : 并行的的任务数(默认1)

#     

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    

    if plot:

        plt.figure()

        plt.title(title)

        if ylim is not None:

            plt.ylim(*ylim)

        plt.xlabel(u"train_numbers")

        plt.ylabel(u"score")

        plt.gca().invert_yaxis()

        plt.grid()

    

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 

                         alpha=0.1, color="b")

        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 

                         alpha=0.1, color="r")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"score of train")

        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"score of cross_val")

    

        plt.legend(loc="best")

        

        plt.draw()

        plt.gca().invert_yaxis()

        plt.show()

    

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2

    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff

data = pd.read_csv("/kaggle/input/titanic-wcg-xgboost-0-84688/WCG_XGBoost2.csv")

plot_learning_curve(clf, u"learning_curve", X, y)
from sklearn.ensemble import BaggingClassifier



train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')

train_np = train_df.values



# y即Survival结果

y = train_np[:, 0]



# X即特征属性值

X = train_np[:, 1:]



# fit到BaggingRegressor之中

clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)

bagging_clf = BaggingClassifier(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)

bagging_clf.fit(X, y)



test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')

predictions = bagging_clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})

result = data

result.to_csv("/kaggle/working/logical.csv", index=False)
pd.read_csv("/kaggle/working/logical.csv").head(10)