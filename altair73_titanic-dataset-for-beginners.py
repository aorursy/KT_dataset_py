# To analyse

import pandas as pd

import numpy as np



#To visualise

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

%matplotlib inline   

#This will display the plots below the code and store it in the notebook itself



#To ignore warnings

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.columns)

train.sample(5)

train.describe(include='all')
print(pd.isnull(train).sum())

print(pd.isnull(train).mean())
sns.barplot(x="Sex",y="Survived",data = train)

plt.show()
train[['Sex','Survived']].groupby('Sex').mean()*100
sns.barplot(x="Pclass",y="Survived",data = train)

plt.show()
train[['Pclass','Survived']].groupby('Pclass').mean()*100
sns.catplot(x='Sex', y='Survived',  kind='bar', data=train, hue='Pclass')

plt.show()
sns.barplot(x="SibSp",y="Survived",data = train)

plt.show()
train[['SibSp','Survived']].groupby("SibSp").mean()*100
train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)

value = [-1, 0, 5, 12, 18, 30, 65, 100]

names = ['Missing', 'Baby', 'Child', 'Teen', 'Youth', 'Adult', 'Elder']

train['AgeGroup'] = pd.cut(train["Age"], value, labels = names)

test['AgeGroup'] = pd.cut(test["Age"], value, labels = names)





sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()
train['Name'].head()
for item in [train,test]:

    item['Title'] = item['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()

    # Here we split the second word from the name and stripped the excess whitespaces.
pd.crosstab(train['Title'],train['Sex'])

train["Age"] = train["Age"].replace({-0.5:np.nan})

test["Age"] = test["Age"].replace({-0.5:np.nan})
Null_List = train[train['Age'].isna()].groupby('Title').count()['Survived']

Null_Title = Null_List.index.values

Null_List
for item in Null_Title:

        val = train[train.Title == item]['Age'].median()

        train_list = train[(train.Title==item)& (train.AgeGroup == 'Missing')].index

        for elem in train_list:

            train.iloc[elem,train.columns.get_loc('Age')]=val

    
Null_List = test[test['Age'].isna()].groupby('Title').count()['PassengerId']

Null_Title = Null_List.index.values

for item in Null_Title:

        val = train[train.Title == item]['Age'].median()

        test_list = test[(test.Title==item) & (test.AgeGroup == 'Missing')].index

        for elem in test_list:

            test.iloc[elem,test.columns.get_loc('Age')]=val

    
# First we can fill the missing value



test[pd.isnull(test)['Fare']]





val = int(test[pd.isnull(test)['Fare']].Pclass)

amount = round(test[test.Pclass ==val].Fare.mean())

id = test[pd.isnull(test)['Fare']].index

test.iloc[id,test.columns.get_loc('Fare')]=amount
train['Fare'] = pd.cut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['Fare'] = pd.cut(test['Fare'], 4, labels = [1, 2, 3, 4])
pd.isnull(train).sum()
train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)
print(train.groupby("Embarked").count()["Survived"])
train = train.fillna({"Embarked": "S"})
train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
sex_num = {"male":0,"female":1}

train["Sex"] = train["Sex"].map(sex_num)

test["Sex"] = test["Sex"].map(sex_num)
embarked_num = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_num)

test['Embarked'] = test['Embarked'].map(embarked_num)

print(pd.isnull(test).sum())

print(pd.isnull(train).sum())
train.sample(5)
group = list(map(str,train.AgeGroup.unique().sort_values()))

val = pd.Series(group)

print(val)
value = [0, 5, 12, 18, 30, 65, 100]

names = ['Baby', 'Child', 'Teen', 'Youth', 'Adult', 'Elder']

train['AgeGroup'] = pd.cut(train["Age"], value, labels = names)

test['AgeGroup'] = pd.cut(test["Age"], value, labels = names)

group = list(map(str,train.AgeGroup.unique().sort_values()))

val = pd.Series(group)

print(val)
item = val.to_dict()

item
item = {v: k for k, v in item.items()}

item
train['AgeGroup'] = train['AgeGroup'].map(item)

test['AgeGroup'] = test['AgeGroup'].map(item)

train['Title'] = train['Title'].replace('Ms', 'Miss').replace('Mme', 'Mrs').replace('Mlle', 'Miss').replace(['Countess', 'Lady', 'Sir'], 'Royal').replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Special')

test['Title'] = test['Title'].replace('Ms', 'Miss').replace('Mme', 'Mrs').replace('Mlle', 'Miss').replace(['Countess', 'Lady', 'Sir'], 'Royal').replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Special')



title_num = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Special": 6}

train['Title'] = train['Title'].map(title_num)

train['Title'] = train['Title'].fillna(0)

test['Title'] = test['Title'].map(title_num)

test['Title'] = test['Title'].fillna(0)
from sklearn.model_selection import train_test_split



predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, random_state = 0)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_predict = logreg.predict(x_val)

result1 = round(accuracy_score(y_predict, y_val) * 100, 2)

print(result1)
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



svc = SVC()

svc.fit(x_train, y_train)

y_predict = svc.predict(x_val)

result2 = round(accuracy_score(y_predict, y_val) * 100, 2)

print(result2)



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_predict = knn.predict(x_val)

result3 = round(accuracy_score(y_predict, y_val) * 100, 2)

print(result3)
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_predict = gbk.predict(x_val)

result4 = round(accuracy_score(y_predict, y_val) * 100, 2)

print(result4)
from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_predict = randomforest.predict(x_val)

result5 = round(accuracy_score(y_predict, y_val) * 100, 2)

print(result5)
model = ["Logistic Regression","Support Vector Machines (SVM)","K-Nearest Neighbours (KNN)","Gradient Boosting Classifier","Random Forest Classifier"]

value = [result1,result2,result3,result4,result5]

result = pd.DataFrame({"Model":model,"Value":value}).sort_values(by="Value", ascending = False)

result
index = test['PassengerId']

prediction = gbk.predict(test.drop('PassengerId', axis=1))



output = pd.DataFrame({ 'PassengerId' : index, 'Survived': prediction })

output.to_csv('submission.csv', index=False)