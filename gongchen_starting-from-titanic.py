# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# Imports 

import numpy as np

import pandas as pd

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns 

sns.set_style('whitegrid')

%matplotlib inline 



import warnings

warnings.filterwarnings('ignore') # 禁止warning出现
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
titanic_train = pd.read_csv("../input/train.csv")

titanic_test = pd.read_csv("../input/test.csv")

titanic_train.head(5)
# 描述各类数据的具体情况

# 从中可以看出Age数据有所缺失

print(titanic_train.describe())

print(titanic_test.describe())
titanic_train.info()

print("————————————————")

titanic_test.info()
# 删除没有用的部分

titanic_train = titanic_train.drop(["PassengerId","Name","Ticket"], axis = 1)

titanic_test = titanic_test.drop(["Name","Ticket"], axis = 1)



# 分段分析数据 Embarked

# 使用fillna 完成缺失值的填充

titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")

# 检查是否填充

titanic_train.info()



# plot 绘图分析Embarked与Survived的关系 size，aspect两个参数用于控制图形大小比例

sns.factorplot("Embarked","Survived", data = titanic_train, size = 4, aspect = 3)

    

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

# sns.factorplot("Embarked", data = titanic_train, kind = "count", order = ["S","C","Q"],ax = axis1)

#sns.factorplot("Survived", hue = "Embarked",data = titanic_train, kind = "count", order = [1,0], ax = axis2)

# 每个码头上船人数

sns.countplot(x = "Embarked",data = titanic_train, ax = axis1)

# 被救人当中每个码头生死情况

sns.countplot(x = "Survived", hue = "Embarked", data = titanic_train, order = [1,0], ax = axis2)



# 每个码头获救比例

embark_perc = titanic_train[["Embarked","Survived"]].groupby(["Embarked"],as_index = False).mean()

sns.barplot(x="Embarked", y="Survived",data = embark_perc, order = ["S","C","Q"],ax = axis3)



embark_dummies_titanic = pd.get_dummies(titanic_train["Embarked"])

embark_dummies_titanic.drop(["S"], axis=1,inplace=True)



embark_dummies_test = pd.get_dummies(titanic_test["Embarked"])

embark_dummies_test.drop(["S"], axis=1,inplace=True)

titanic_train = titanic_train.join(embark_dummies_titanic)

titanic_test = titanic_test.join(embark_dummies_test)



titanic_train.drop(["Embarked"],axis=1, inplace=True)

titanic_test.drop(["Embarked"],axis=1, inplace=True)
# Fare(船票价格) 字段分析 缺失值处理

# 通过开始数据统计显示，测试集中缺失一数值，用中位数填充

titanic_test["Fare"].fillna(titanic_test["Fare"].median(),inplace=True)



# 数值转换

titanic_train["Fare"] = titanic_train["Fare"].astype(int)

titanic_test["Fare"] = titanic_test["Fare"].astype(int)



# 在买票人当中被救或没有被救的人数

fare_not_survived = titanic_train["Fare"][titanic_train["Survived"]==0]

fare_survived = titanic_train["Fare"][titanic_train["Survived"]==1]



# get avgerage_fare and std_fare for fare of survived/not survived passengers

avgerage_fare = DataFrame([fare_not_survived.mean(),fare_survived.mean()])

print("avgerage_fare")

print (avgerage_fare)

print("-----------------------")

std_fare = DataFrame([fare_not_survived.std(),fare_survived.std()])

print("std_fare")

print(std_fare)



# plot

titanic_train["Fare"].plot(kind = "hist", figsize = (10,5),bins=100, xlim=(0,50))



avgerage_fare.index.name = std_fare.index.name = ["Survived"]

avgerage_fare.plot(yerr=std_fare, kind="bar", legend=False)
# Age



fig, (axis1,axis2)=plt.subplots(1,2,figsize=(15,4))

axis1.set_title("Original Age values - Titanic")

axis2.set_title("New Age values - Titanic")



# 得到平均值，标准差 在训练集上

average_age_titanic = titanic_train["Age"].mean()

std_age_titanic = titanic_train["Age"].std()



# ??? 缺失值

count_nan_age_titanic = titanic_train["Age"].isnull().sum()

print(count_nan_age_titanic) #177



# 得到平均值，标准差 ,缺失值 在测试集

average_age_test = titanic_test["Age"].mean()

std_age_test = titanic_test["Age"].std()

count_nan_age_test = titanic_test["Age"].isnull().sum()



# 随机生成，平均值-标准差 是为了使得到的数据更加接近真实数据

rand_1 = np.random.randint(average_age_titanic - std_age_titanic,average_age_titanic + std_age_titanic,size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)



#可视化Age的分布情况

# Note ：drop all null value,and convert to int

titanic_train["Age"].dropna().astype(int).hist(bins=70, ax=axis1)



titanic_train["Age"][np.isnan(titanic_train["Age"])] = rand_1

titanic_test["Age"][np.isnan(titanic_test["Age"])] = rand_2



titanic_train["Age"] = titanic_train["Age"].astype(int)

titanic_test["Age"] = titanic_test["Age"].astype(int)



#plot new Age value



titanic_train["Age"].hist(bins=70,ax=axis2)
# ....continue with plot Age column



# 获得被救或死亡年龄的最大值

facet = sns.FacetGrid(titanic_train,hue="Survived",aspect=4)   

#使用map函数映射kde，以Age作为X轴

facet.map(sns.kdeplot,"Age",shade=True)

#取最大年龄

##oldest = titanic_train["Age"].max()

#设置x轴的取值范围为0到oldest

facet.set(xlim=(0,titanic_train["Age"].max()))

#添加图标，印记

facet.add_legend()



# average survived passengers by age

fig,axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = titanic_train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)
# 船舱标号

# 缺失值比较大，所以删除处理

titanic_train.drop("Cabin",axis = 1,inplace=True)

titanic_test.drop("Cabin",axis = 1,inplace=True)
# Family 



# Instead of having two columns Parch(父母和孩子) & SibSp（兄弟姐妹和配偶）,

# 乘客是否有家人在船上 我们只能通过一列来代表，

# 那么，那么有许多的家人是否能增加存活率呢？



titanic_train["Family"] = titanic_train["Parch"] + titanic_train["SibSp"]

titanic_train["Family"].loc[titanic_train["Family"] > 0] = 1

titanic_train["Family"].loc[titanic_train["Family"] == 0] = 0



titanic_test["Family"] = titanic_test["Parch"] + titanic_test["SibSp"]

titanic_test["Family"].loc[titanic_test["Family"] > 0] = 1

titanic_test["Family"].loc[titanic_test["Family"] == 0] = 0



# drop Parch & SibSp ？？？

titanic_train = titanic_train.drop(["Parch","SibSp"], axis = 1)

titanic_test = titanic_test.drop(["Parch","SibSp"], axis = 1)



# plot 

fig,(axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



sns.countplot(data=titanic_train,x = "Family",order=[1,0],ax = axis1)

family_perc = titanic_train[["Family","Survived"]].groupby(["Family"],as_index=False).mean()

sns.barplot(x = "Family",y = "Survived", data = family_perc,order=[1,0],ax=axis2)



axis1.set_xticklabels(["With Family","Alone"],rotation = 0)
# Sex



def get_person(passenger):

    age,sex = passenger

    return "child" if age < 16 else sex



titanic_train["Person"] = titanic_train[["Age","Sex"]].apply(get_person, axis = 1)

titanic_test["Person"] = titanic_test[["Age","Sex"]].apply(get_person, axis = 1)



# 因为我们创建了Person列，所以不需要Sex列 

# 

titanic_test.drop(["Sex"], axis=1,inplace=True)

titanic_train.drop(["Sex"], axis=1,inplace=True)



person_dummies_titanic = pd.get_dummies(titanic_train["Person"])

person_dummies_titanic.columns = ["Child","Female","Male"]

person_dummies_titanic.drop(["Male"],axis=1,inplace=True)



person_dummies_titanic_test = pd.get_dummies(titanic_train["Person"])

person_dummies_titanic_test.columns = ["Child","Female","Male"]

person_dummies_titanic_test.drop(["Male"],axis=1,inplace=True)



titanic_train = titanic_train.join(person_dummies_titanic)

titanic_test = titanic_test.join(person_dummies_titanic_test)



fig,(axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x="Person",data=titanic_train,ax=axis1)



# 每个人（男性，女性或者儿童）的平均生存情况

person_perc = titanic_train[["Person","Survived"]].groupby(["Person"],as_index = False).mean()

sns.barplot(x="Person",y="Survived",data=person_perc, ax=axis2,order=["male","female","child"])



titanic_train.drop(["Person"],axis=1,inplace=True)

titanic_test.drop(["Person"],axis=1,inplace=True)
# Pclass



sns.factorplot("Pclass","Survived",data = titanic_train,order=[1,2,3],size = 4)



pclass_dummies_titanic = pd.get_dummies(titanic_train["Pclass"])

pclass_dummies_titanic.columns = ["Class_1","Class_2","Class_3"]

pclass_dummies_titanic.drop(["Class_3"],axis=1,inplace=True)



pclass_dummies_titanic_test = pd.get_dummies(titanic_train["Pclass"])

pclass_dummies_titanic_test.columns = ["Class_1","Class_2","Class_3"]

pclass_dummies_titanic_test.drop(["Class_3"],axis=1,inplace=True)



titanic_train.drop(["Pclass"],axis=1,inplace=True)

titanic_test.drop(["Pclass"],axis=1,inplace=True)



titanic_train = titanic_train.join(pclass_dummies_titanic)

titanic_test = titanic_test.join(pclass_dummies_titanic_test)
# define training and testing sets



X_train = titanic_train.drop("Survived",axis=1)

Y_train = titanic_train["Survived"]

X_text = titanic_test.drop("PassengerId",axis=1).copy()
# Logistic Regression



logred = LogisticRegression()

logred.fit(X_train,Y_train)

Y_pred = logred.predict(X_text)

logred.score(X_train,Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train,Y_train)



Y_pred = random_forest.predict(X_text)



random_forest.score(X_train,Y_train)
coeff_df = DataFrame(titanic_train.columns.delete(0))

coeff_df.columns = ["Features"]

coeff_df["Coefficient Estimate"] = pd.Series(logred.coef_[0])



# preview(预览)

coeff_df
submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv("titanic.csv",index = False)