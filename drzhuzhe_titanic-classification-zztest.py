# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pandas import Series,DataFrame

data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_train
"""

PassengerId => 乘客ID

Pclass => 乘客等级(1/2/3等舱位)

Name => 乘客姓名

Sex => 性别

Age => 年龄

SibSp => 堂兄弟/妹个数

Parch => 父母与小孩个数

Ticket => 船票信息

Fare => 票价

Cabin => 客舱

Embarked => 登船港口

"""



data_train.info()
data_train.describe()
# 以上内容为初步处理

import matplotlib.pyplot as plt

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



#plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图

data_train.Survived.value_counts().plot(kind='bar')# 柱状图 

plt.title(u"survived rate (1 for survived)") # 标题

plt.ylabel(u"count")  



#plt.subplot2grid((2,3),(0,1))

plt.figure()

data_train.Pclass.value_counts().plot(kind="bar")

plt.ylabel(u"count")

plt.title(u"carbin level")



#"""

#plt.subplot2grid((2,3),(0,2))

plt.figure()

plt.scatter(data_train.Survived, data_train.Age)

plt.ylabel(u"age")                         # 设定纵坐标名称

plt.grid(b=True, which='major', axis='y') 

plt.title(u"survived per age (1 for survived)")





#plt.subplot2grid((2,3),(1,0), colspan=2)

plt.figure()

data_train.Age[data_train.Pclass == 1].plot(kind='kde')   

data_train.Age[data_train.Pclass == 2].plot(kind='kde')

data_train.Age[data_train.Pclass == 3].plot(kind='kde')

plt.xlabel(u"age")# plots an axis lable

plt.ylabel(u"probility density") 

plt.title(u"age per carbin")

plt.legend((u'top carbin', u'2 carbin',u'3 carbin'),loc='best') # sets our legend for our graph.





#plt.subplot2grid((2,3),(1,2))

plt.figure()

data_train.Embarked.value_counts().plot(kind='bar')

plt.title(u"onboard per port")

plt.ylabel(u"count")  

#"""





plt.show()

#看看各乘客等级的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()

df=pd.DataFrame({u'survied':Survived_1, u'descreased':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"survived per carbin")

plt.xlabel(u"carbin level") 

plt.ylabel(u"count") 

plt.show()

#看看各性别的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()

Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()

df=pd.DataFrame({u'male':Survived_m, u'female':Survived_f})

df.plot(kind='bar', stacked=True)

plt.title(u"survied per sex")

plt.xlabel(u"sex") 

plt.ylabel(u"count")

plt.show()


 #然后我们再来看看各种舱级别情况下各性别的获救情况

fig=plt.figure()

#fig.set(alpha=0.65) # 设置图像透明度，无所谓

plt.title(u"survived per age or carbin")



ax1=fig.add_subplot(141)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')

ax1.set_xticklabels([u"survived", u"decreased"], rotation=0)

ax1.legend([u"female/top"], loc='best')



ax2=fig.add_subplot(142, sharey=ax1)

#plt.figure()

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')

ax2.set_xticklabels([u"decreased", u"survived"], rotation=0)

plt.legend([u"female/low"], loc='best')



ax3=fig.add_subplot(143, sharey=ax1)

#plt.figure()

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')

ax3.set_xticklabels([u"decreased", u"survived"], rotation=0)

plt.legend([u"male/top"], loc='best')



ax4=fig.add_subplot(144, sharey=ax1)

#plt.figure()

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')

ax4.set_xticklabels([u"decreased", u"survived"], rotation=0)

plt.legend([u"male/low"], loc='best')



#plt.show()
fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()

df=pd.DataFrame({u'survied':Survived_1, u'decreased':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"survived per port")

plt.xlabel(u"port") 

plt.ylabel(u"count") 

g = data_train.groupby(['SibSp','Survived'])

df = pd.DataFrame(g.count()['PassengerId'])

print(df)



g = data_train.groupby(['SibSp','Survived'])

df = pd.DataFrame(g.count()['PassengerId'])

print(df)


#ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴把

#cabin只有204个乘客有值，我们先看看它的一个分布

data_train.Cabin.value_counts()

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()

Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()

df=pd.DataFrame({u'has':Survived_cabin, u'no':Survived_nocabin}).transpose()

df.plot(kind='bar', stacked=True)

plt.title(u"Cabin has or not")

plt.xlabel(u"Cabin has or not") 

plt.ylabel(u"count")

plt.show()
from sklearn.ensemble import RandomForestRegressor

 

### 使用 RandomForestClassifier 填补缺失的年龄属性

def set_missing_ages(df):

    

    # 把已有的数值型特征取出来丢进Random Forest Regressor中

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    

    

    # 乘客分成已知年龄和未知年龄两部分

    #known_age = age_df[age_df.Age.notnull()].as_matrix()

    #unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    known_age = age_df[age_df.Age.notnull()].values

    unknown_age = age_df[age_df.Age.isnull()].values

    

    # y即目标年龄 取第一行后面的

    y = known_age[:, 0]



    # X即特征属性值 取第一行后面的

    X = known_age[:, 1:]

    

    print(X.shape, y.shape, unknown_age.shape)

    

    # fit到RandomForestRegressor之中 

    # 注：RandomForest是一个用在原始数据中做不同采样，建立多颗DecisionTree，再进行average等等来降低过拟合现象，提高结果的机器学习算法，我们之后会介绍到

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

    rfr.fit(X, y)

        

    # 用得到的模型进行未知年龄结果预测

    predictedAges = rfr.predict(unknown_age[:, 1:])

    

    print(predictedAges)

    

    # 用得到的预测结果填补原缺失数据

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df, rfr



def set_Cabin_type(df):

    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"

    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"

    return df



#data_train.info

data_train, rfr = set_missing_ages(data_train)

data_train = set_Cabin_type(data_train)



def DataPreProccess(data_train):

    data_train["Title"] = data_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



    #pd.crosstab(data_train['Title'], data_train['Sex'])



    data_train['Title'] = data_train['Title'].replace('Mlle', 'Miss')

    data_train['Title'] = data_train['Title'].replace('Ms', 'Miss')

    data_train['Title'] = data_train['Title'].replace('Mme', 'Mrs')



    data_train['Title'] = data_train['Title'].replace(['Lady', 'Countess','Capt','Col','Don', 'Dr', 'Major','Rev', 'Sir', 'Jonkheer', 'Dona'], 'Not married')

    data_train['Title'] = data_train['Title'].replace(['Mr', 'Mrs'], 'Married')



    #pd.crosstab(data_train['Title'], data_train['Sex'])

    data_train["Surname"] = data_train['Name'].str.split(',').str.get(0)



    # 补充family 属性

    data_train['Family']=data_train['SibSp']+data_train['Parch']+1

    #data_train=data_train.drop(['SibSp','Parch'],axis=1)



    def FamilyGroup(family):

        a=''

        if family<=1:

            a='Solo'

        elif family<=4:

            a='Small'

        else:

            a='Large'

        return a

    data_train['FamilyGroup']=data_train['Family'].map(FamilyGroup)

    #data_train=data_train.drop(['Family'],axis=1)  



    # 年龄分层

    def AgeGroup(age):

        a=''

        if age<=15:

            a='Child'

        elif age<=30:

            a='Young'

        elif age<=50:

            a='Adult'

        else:

            a='Old'

        return a

    data_train['AgeGroup']=data_train['Age'].map(AgeGroup)

    #data_train=data_train.drop(['Age'],axis=1)



    # 后面会 drop 名字的

    #data_train



    df=pd.get_dummies(data_train,columns=['Sex','Embarked','FamilyGroup','AgeGroup','Cabin', 'Pclass', 'Title'])



    #df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

    df.drop(['Name', 'Ticket', 'Family', 'SibSp','Parch', 'Surname'], axis=1, inplace=True)

    return df

df = DataPreProccess(data_train)

df
data_train.info

data_train.describe
# 特征因子化 二元的应该不用吧

# https://blog.csdn.net/zs15321583801/article/details/79652045

"""

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')



dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')



dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')



dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

"""



df=pd.get_dummies(data_train,columns=['Sex','Embarked','FamilyGroup','AgeGroup','Cabin', 'Pclass', 'Title'])



#df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df.drop(['Name', 'Ticket', 'Family', 'SibSp','Parch', 'Surname'], axis=1, inplace=True)

df
# 对  age 和 fare 做归一化 

import sklearn.preprocessing as preprocessing



def stdAgeAndFare(df):

    scaler = preprocessing.StandardScaler()

    col_names = ['Age', 'Fare']

    #age_scale_param = scaler.fit(df['Age'].values)

    features = df[col_names]

    scaler = scaler.fit(features.values)

    features = scaler.transform(features.values)

    df[col_names] = features

    return df

df = stdAgeAndFare(df)

df
from sklearn import linear_model



# 用正则取出我们要的属性值

#train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

# 这个直接取列不就行了 ， 班门弄斧

__regTxt__ = 'Survived|Age|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|FamilyGroup_.*|AgeGroup_.*|Title_.*'

train_df = df.filter(regex=__regTxt__)

train_np = train_df.values



# y即Survival结果

y = train_np[:, 0]



# X即特征属性值

X = train_np[:, 1:]



# fit到RandomForestRegressor之中

clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)

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

"""

dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')





df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

"""

df_test = DataPreProccess(data_test)

df_test = stdAgeAndFare(df_test)



"""

scaler = preprocessing.StandardScaler()





col_names = ['Age', 'Fare']

#col_names2 = ['Age_scaled', 'Fare_scaled']

#age_scale_param = scaler.fit(df['Age'].values)

features = df_test[col_names]

scaler = scaler.fit(features.values)

features = scaler.transform(features.values)

df_test[col_names] = features

"""



#data_test, rfr_test = set_missing_ages(data_test)

#data_test = set_Cabin_type(data_test)

#df_test = DataPreProccess(data_test)

#df_test = stdAgeAndFare(df_test)

#df_test

data_test.info()
#test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

test = df_test.filter(regex=__regTxt__)

predictions = clf.predict(test)

#result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})

#result.to_csv("/logistic_regression_predictions.csv", index=False)

result.to_csv("logistic_regression_predictions.csv", index=False)
#!cat /logistic_regression_predictions.csv

#result.describe
# 线性回归的相关性

DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})
from sklearn.model_selection import cross_val_score, train_test_split



 #简单看看打分情况

clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)

#all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

all_data = df.filter(regex=__regTxt__)

X = all_data.values[:,1:]

y = all_data.values[:,0]

cross_val_score(clf, X, y, cv=5)
# 查badcase

# 分割数据，按照 训练数据:cv数据 = 7:3的比例

#reg_txt = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*'

#reg_txt = 'Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*'

split_train, split_cv = train_test_split(df, test_size=0.3, random_state=0)

train_df = split_train.filter(regex=__regTxt__)

# 生成模型

clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)

clf.fit(train_df.values[:,1:], train_df.values[:,0])



# 对cross validation数据进行预测



cv_df = split_cv.filter(regex=__regTxt__)

predictions = clf.predict(cv_df.values[:,1:])



origin_data_train = pd.read_csv("../input/titanic/train.csv")

bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:,0]]['PassengerId'].values)]

bad_cases
data_train[data_train['Name'].str.contains("Major")]
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve



# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 

                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):

    """

    画出data在某模型上的learning curve.

    参数解释

    ----------

    estimator : 你用的分类器。

    title : 表格的标题。

    X : 输入的feature，numpy类型

    y : 输入的target vector

    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点

    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)

    n_jobs : 并行的的任务数(默认1)

    """

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

        plt.xlabel(u"sample count")

        plt.ylabel(u"score")

        plt.gca().invert_yaxis()

        plt.grid()

    

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 

                         alpha=0.1, color="b")

        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 

                         alpha=0.1, color="r")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"trainning sorce")

        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"cross validate sorce")

    

        plt.legend(loc="best")

        

        plt.draw()

        plt.show()

        plt.gca().invert_yaxis()

    

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2

    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff



plot_learning_curve(clf, u"learning curve", X, y)
from sklearn.ensemble import BaggingRegressor



#train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')

#_reg_txt = 'Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title'

train_df = df.filter(regex=__regTxt__)

train_np = train_df.values



# y即Survival结果

y = train_np[:, 0]



# X即特征属性值

X = train_np[:, 1:]



# fit到BaggingRegressor之中

clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)

bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)

bagging_clf.fit(X, y)



test = df_test.filter(regex=__regTxt__)

predictions = bagging_clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})

result.to_csv("logistic_regression_bagging_predictions.csv", index=False)