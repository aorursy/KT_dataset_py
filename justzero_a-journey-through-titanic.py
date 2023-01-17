# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
# 加载训练 & 预测数据为 DataFrame

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# 预览数据

titanic_df.head()
titanic_df.info()

print("-----------------------------------------------")

test_df.info()
# 删除不需要的列, 这些列对分析和预测起不到什么作用

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)

test_df    = test_df.drop(['Name','Ticket'], axis=1)
# Embarked（上船的港口）



# 用最常出现 "S" 填充 titanic_df.Embarker 中的缺失值

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")



# plot

sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)

# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)

sns.countplot(x='Embarked', data=titanic_df, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)



# group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)



# Either to consider Embarked column in predictions,

# and remove "S" dummy variable, 

# and leave "C" & "Q", since they seem to have a good rate for Survival.



# OR, don't create dummy variables for Embarked column, just drop it, 

# because logically, Embarked doesn't seem to be useful in prediction.



embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)



embark_dummies_test  = pd.get_dummies(test_df['Embarked'])

embark_dummies_test.drop(['S'], axis=1, inplace=True)



titanic_df = titanic_df.join(embark_dummies_titanic)

test_df    = test_df.join(embark_dummies_test)



titanic_df.drop(['Embarked'], axis=1,inplace=True)

test_df.drop(['Embarked'], axis=1,inplace=True)
# Fare（乘客费）



# 用中位数填充 test_df.Fare 中的缺失值

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)



# 将 Fare 数值类型转换为 int

titanic_df['Fare'] = titanic_df['Fare'].astype(int)

test_df['Fare']    = test_df['Fare'].astype(int)



# 根据票价查看存活 & 未存活乘客

fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]

fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]



# 计算存活/未存活乘客票价的平均值和标准差

avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])



# plot

titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))



avgerage_fare.index.names = std_fare.index.names = ["Survived"]

avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
# Age（年龄）



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values - Titanic') # 初始年龄值

axis2.set_title('New Age values - Titanic')      # 新年龄值



# axis3.set_title('Original Age values - Test')

# axis4.set_title('New Age values - Test')



# 计算 titanic_df.Age 的均值，标准差和空值个数

average_age_titanic   = titanic_df["Age"].mean()

std_age_titanic       = titanic_df["Age"].std()

count_nan_age_titanic = titanic_df["Age"].isnull().sum()



# 计算 test_df.Age 的均值，标准差和空值个数

average_age_test   = test_df["Age"].mean()

std_age_test       = test_df["Age"].std()

count_nan_age_test = test_df["Age"].isnull().sum()



# 在 (mean +/- std) 范围内，生成随机数

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)



# plot 初始年龄值

# NOTE: 删除所有空值，数值类型转换为 int

titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



# 用生成的随机数填充 Age 列的缺失值

titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1

test_df["Age"][np.isnan(test_df["Age"])] = rand_2



# 数值类型转换为 int

titanic_df['Age'] = titanic_df['Age'].astype(int)

test_df['Age']    = test_df['Age'].astype(int)

        

# plot 新年龄值

titanic_df['Age'].hist(bins=70, ax=axis2)

# test_df['Age'].hist(bins=70, ax=axis4)
# .... continue with plot Age column



# 面积表示存活/未存活乘客数

#facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)

#facet.map(sns.kdeplot,'Age',shade=True)

#facet.set(xlim=(0, titanic_df['Age'].max()))

#facet.add_legend()



# 每个年龄乘客的存活概率

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)
# Cabin（船舱号）



# 此列有大量的空值，所以它不会对预测产生明显的影响

titanic_df.drop("Cabin", axis=1, inplace=True)

test_df.drop("Cabin", axis=1, inplace=True)
# Family（家庭）



# 替换 Parch & SibSp 列（是否父母孩子在船上，是否配偶兄弟姐妹在船上）, 

# 我可以替换为一列为是否有“亲人”在船上

# 意味着：如果有亲人在船上，存活的几率会受到影响

titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]

titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1

titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0



test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]

test_df['Family'].loc[test_df['Family'] > 0] = 1

test_df['Family'].loc[test_df['Family'] == 0] = 0



# 删除 Parch & SibSp 列

titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)

test_df    = test_df.drop(['SibSp','Parch'], axis=1)



# plot

fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)

sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)



# 是否有家庭成员在船上，乘客存活的平均值

family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)



axis1.set_xticklabels(["With Family","Alone"], rotation=0)
# Sex（性别）





# 我们可以看出，小孩（age < ~16）似乎有着很高的生存机会。

# 所以我们可以把乘客分为男人、女人和孩子

def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)

test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)



# 引入 Person 列后，Sex 列不在需要

titanic_df.drop(['Sex'],axis=1,inplace=True)

test_df.drop(['Sex'],axis=1,inplace=True)



# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers

# 为 Person 列创建假设变量，并删除男性行（男性为生存几率最低的人群）

person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(test_df['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



titanic_df = titanic_df.join(person_dummies_titanic)

test_df    = test_df.join(person_dummies_test)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)

sns.countplot(x='Person', data=titanic_df, ax=axis1)



# average of survived for each Person(male, female, or child)

person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])



titanic_df.drop(['Person'],axis=1,inplace=True)

test_df.drop(['Person'],axis=1,inplace=True)
# Pclass（船票类别）



# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])

sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)



# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers

pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



titanic_df.drop(['Pclass'],axis=1,inplace=True)

test_df.drop(['Pclass'],axis=1,inplace=True)



titanic_df = titanic_df.join(pclass_dummies_titanic)

test_df    = test_df.join(pclass_dummies_test)
X_train, X_test, y_train, y_test = train_test_split(

    titanic_df.drop("Survived",axis=1), titanic_df["Survived"],

    train_size=0.25, random_state=33)



#X_train = np.array(X_train, dtype=np.int32).reshape( (len(X_train), 9) )
# Logistic Regression



lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_y_pred = lr.predict(X_test)

print(lr.score(X_test, y_test))

print(classification_report(lr_y_pred, y_test))
# Support Vector Machines



lsvc = LinearSVC()

lsvc.fit(X_train, y_train)

lsvc_y_pred = lsvc.predict(X_test)

print(lsvc.score(X_test, y_test))

print(classification_report(lsvc_y_pred, y_test))
# Random Forests



rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

rfc_y_pred = rfc.predict(X_test)

print(rfc.score(X_test, y_test))

print(classification_report(rfc_y_pred, y_test))
# # K-nearest-Neighbors



# knn = KNeighborsClassifier(n_neighbors = 3)

# knn.fit(X_train, y_train)

# knn_y_pred = knn.predict(X_test)

# print(knn.score(X_test, y_test))

# print(classification_report(knn_y_pred, y_test))
# XGBoost Classifier



xgbc = XGBClassifier()

xgbc.fit(X_train, y_train)

xgbc_y_pred = xgbc.predict(X_test)

print(xgbc.score(X_test, y_test))

print(classification_report(xgbc_y_pred, y_test))
# define training and testing sets



X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
# Logistic Regression



lr = LogisticRegression()

lr.fit(X_train, Y_train)

lr_y_pred = lr.predict(X_test)
# Support Vector Machines



lsvc = LinearSVC()

lsvc.fit(X_train, Y_train)

lsvc_y_pred = lsvc.predict(X_test)
# Random Forests



rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, Y_train)

rfc_y_pred = rfc.predict(X_test)
# K-nearest-Neighbors



# knn = KNeighborsClassifier(n_neighbors = 3)

# knn.fit(X_train, Y_train)

# knn_y_pred = knn.predict(X_test)
# XGBoost Classifier



xgbc = XGBClassifier()

xgbc.fit(X_train, Y_train)

xgbc_y_pred = xgbc.predict(X_test)
# 使用逻辑回归获得每个特征的相关系数

coeff_df = DataFrame(titanic_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(lr.coef_[0])



# preview

coeff_df
Y_pred = lr_y_pred



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('lr_submission_titanic.csv', index=False)
Y_pred = lsvc_y_pred



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('lsvc_submission_titanic.csv', index=False)
Y_pred = rfc_y_pred



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('rfc_submission_titanic.csv', index=False)
Y_pred = xgbc_y_pred



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('xgbc_submission_titanic.csv', index=False)