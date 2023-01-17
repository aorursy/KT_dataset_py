# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pandas import Series, DataFrame

import pandas as pd

import numpy as np



data_train = pd.read_csv("../input/titanic/train.csv")

data_train.head()
data_train.info()
# 乘客各属性分布

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Droid Sans Fallback'] #用来正常显示中文标签

plt.rcParams['axes.unicode_minus']=False

%matplotlib inline



fig = plt.figure()

fig.set(alpha=0.2)



plt.subplot2grid((2,3), (0,0))

data_train.Survived.value_counts().plot(kind='bar')

plt.title('获救情况（1为获救）')

plt.ylabel('人数')



plt.subplot2grid((2,3), (0,1))

data_train.Pclass.value_counts().plot(kind='bar')

plt.ylabel('人数')

plt.title('乘客等级分布')



plt.subplot2grid((2,3), (0,2))

plt.scatter(data_train.Survived, data_train.Age)

plt.ylabel('年龄')

plt.grid(b=True, which='major', axis='y')

plt.title('按年龄看获救分布 (1为获救)')



plt.subplot2grid((2,3), (1, 0), colspan=2)

data_train.Age[data_train.Pclass==1].plot(kind="kde")

data_train.Age[data_train.Pclass==2].plot(kind="kde")

data_train.Age[data_train.Pclass==3].plot(kind="kde")

plt.xlabel("年龄")# plots an axis lable

plt.ylabel("密度") 

plt.title("各等级的乘客年龄分布")

plt.legend(('头等舱', '2等舱','3等舱'),loc='best') # sets our legend for our graph.



plt.subplot2grid((2,3),(1,2))

data_train.Embarked.value_counts().plot(kind='bar')

plt.title("各登船口岸上船人数")

plt.ylabel("人数")  

plt.show()
#看看各乘客等级的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()



df = pd.DataFrame({'获救': Survived_1, '未获救': Survived_0})

df.plot(kind = 'bar', stacked=True)

plt.xlabel("乘客等级") 

plt.ylabel("人数") 

plt.show()
fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()

Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()

df=pd.DataFrame({'男性':Survived_m, '女性':Survived_f})

df.plot(kind='bar', stacked=True)

plt.title("按性别看获救情况")

plt.xlabel("性别") 

plt.ylabel("人数")

plt.show()
#然后我们再来看看各种舱级别情况下各性别的获救情况

fig=plt.figure()

fig.set(alpha=0.65) # 设置图像透明度，无所谓

plt.title(u"根据舱等级和性别的获救情况")



ax1=fig.add_subplot(141)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')

ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)

ax1.legend([u"女性/高级舱"], loc='best')



ax2=fig.add_subplot(142, sharey=ax1)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')

ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)

plt.legend([u"女性/低级舱"], loc='best')



ax3=fig.add_subplot(143, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')

ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)

plt.legend([u"男性/高级舱"], loc='best')



ax4=fig.add_subplot(144, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')

ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)

plt.legend([u"男性/低级舱"], loc='best')

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()

df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"各登录港口乘客的获救情况")

plt.xlabel(u"登录港口") 

plt.ylabel(u"人数") 



plt.show()
from sklearn.ensemble import RandomForestRegressor

 

### 使用 RandomForestClassifier 填补缺失的年龄属性

def set_missing_ages(df):

    

    # 把已有的数值型特征取出来丢进Random Forest Regressor中

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]



    # 乘客分成已知年龄和未知年龄两部分

    known_age = age_df[age_df.Age.notnull()].as_matrix()

    unknown_age = age_df[age_df.Age.isnull()].as_matrix()



    # y即目标年龄

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
data_train.head()
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')



dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')



dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')



dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')



df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df.head()
import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()

age_scale_param = scaler.fit(df['Age'].values.reshape(1,-1))

age_scale_param
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(1,-1), age_scale_param)[0]

fare_scale_param = scaler.fit(df['Fare'].values.reshape(1,-1))

df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(1,-1), fare_scale_param)[0]

df.head()
from sklearn import linear_model



# 用正则取出我们要的属性值

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

train_np = train_df.values



# y即Survival结果

y = train_np[:, 0]



# X即特征属性值

X = train_np[:, 1:]



# fit到RandomForestRegressor之中

clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)

clf.fit(X, y)
data_test = pd.read_csv("../input/titanic/test.csv")

data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

null_age = tmp_df[data_test.Age.isnull()].values
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

df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(1,-1), age_scale_param)[0]

df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(1,-1), fare_scale_param)[0]

df_test.head()
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions = clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})



result.to_csv("./logistic_regression_predictions.csv", index=False)