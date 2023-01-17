import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

train.info()
train.head()
train.describe()
train.isnull().sum()
test.info()
# For Train

median_1 = train['Age'].median()

train['Age'].fillna(median_1,inplace = True)



# For Test

median_2 = test['Age'].median()

test['Age'].fillna(median_2,inplace = True)
a = train['Embarked'].value_counts()

print("Embarked Count:\n", a)

train[train['Embarked'].isnull()]
train['Embarked'].fillna('S',inplace = True)
print("Count of Missing value for Fare Feature in Test Data: ",test['Fare'].isnull().sum())

test[test['Fare'].isnull()]
mean_fare = test[test['Pclass'] == 3]['Fare'].mean()

test['Fare'].fillna(mean_fare,inplace = True)
Id = test['PassengerId']

train = train.drop(['PassengerId','Cabin','Ticket','Name'],axis = 1)

test = test.drop(['PassengerId','Cabin','Ticket','Name'],axis = 1)
print("Check of Null Values for Train Dataset : \n",train.isnull().sum())
print("Check of Null Values for Test Dataset : \n",test.isnull().sum())
plt.figure(figsize=[14,10])



plt.subplot(221)

train['Survived'].value_counts().plot(kind = 'bar',color = 'c',legend = True)

plt.xlabel("Not-Survived vs Survived")

plt.ylabel("Count")

plt.title("Distribution by Survival")



plt.subplot(222)

train['Pclass'].value_counts().plot(kind = 'bar',color = 'c',legend = True)

plt.xlabel("Classes")

plt.ylabel("Count")

plt.title("Distribution of People by class")



plt.subplot(223)

train['Sex'].value_counts().plot(kind = 'bar',color = 'c',legend = True)

plt.xlabel("Sex")

plt.ylabel("Count")

plt.title("Distribution of People by Sex")



plt.subplot(224)

train['Embarked'].value_counts().plot(kind = 'bar',color = 'c',legend = True)

plt.xlabel("Port of Embarkation")

plt.ylabel("Count")

plt.title("Distribution of People by Embarked")
#Optional Same plotting using Pivot table

plt.figure(figsize=[14,10])



#plt.subplot(121)

Sex_pivot = train.pivot_table(index="Sex",values="Survived")

Sex_pivot.plot.bar()

plt.show()



#plt.subplot(122)

Class_pivot = train.pivot_table(index="Pclass",values="Survived")

Class_pivot.plot.bar()

plt.show()
grp_sex = train.groupby('Survived')['Sex'].value_counts()

grp_sex.unstack().plot(kind = 'bar')

plt.xlabel("Not-Survived vs Survived")

plt.ylabel("Count")

plt.title("Survival Distribution by Sex")
age_0_10 = train[train['Age'] < 10]['Age'].count()

age_10_20 = train[(train['Age'] >= 10) & (train['Age'] < 20) ]['Age'].count()

age_20_30 = train[(train['Age'] >= 20) & (train['Age'] < 30) ]['Age'].count()

age_30_40 = train[(train['Age'] >= 30) & (train['Age'] < 40) ]['Age'].count()

age_40_50 = train[(train['Age'] >= 40) & (train['Age'] < 50) ]['Age'].count()

age_50_60 = train[(train['Age'] >= 50) & (train['Age'] < 60) ]['Age'].count()

age_60_70 = train[(train['Age'] >= 60) & (train['Age'] < 70) ]['Age'].count()

age_70_80 = train[(train['Age'] >= 70) & (train['Age'] <= 80) ]['Age'].count()



Age_dist = [age_0_10,age_10_20,age_20_30,age_30_40,age_40_50,age_50_60,age_60_70,age_70_80]

Age_graph = pd.DataFrame({'Age Range':['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80'],'Count' : Age_dist})



Age_graph.set_index('Age Range',inplace = True)



Age_graph.plot(kind = 'bar',color = 'g')



#plt.pie(Age_graph['Count'],labels = Age_graph['Age Range'])

plt.xlabel("Age Range")

plt.ylabel("Count")

plt.title("Distribution by Age")
grp_class = train.groupby('Survived')['Pclass'].value_counts()

grp_class.unstack().plot(kind = 'bar')

plt.xlabel("Not-Survived vs Survived")

plt.ylabel("Count")

plt.title("Survival Distribution by Class")
grp_emb = train.groupby('Survived')['Embarked'].value_counts()

grp_emb.unstack().plot(kind = 'bar',stacked = True)

plt.xlabel("Not-Survived vs Survived")

plt.ylabel("Count")

plt.title("Survival Distribution by Embarked")
plt.figure(figsize=(12,9))

sns.countplot(x="SibSp", hue="Survived", data=train)

plt.title("Survival On Basis of Number of sibling")

plt.legend(loc='best')
plt.figure(figsize=(12,9))

sns.countplot(x="Parch", hue="Survived", data=train)

plt.title("Survival On Basis of Number of Parent")

plt.legend()
age_0_10 = train[train['Age'] < 10]

age_10_20 = train[(train['Age'] >= 10) & (train['Age'] < 20) ]

age_20_30 = train[(train['Age'] >= 20) & (train['Age'] < 30) ]

age_30_40 = train[(train['Age'] >= 30) & (train['Age'] < 40) ]

age_40_50 = train[(train['Age'] >= 40) & (train['Age'] < 50) ]

age_50_60 = train[(train['Age'] >= 50) & (train['Age'] < 60) ]

age_60_70 = train[(train['Age'] >= 60) & (train['Age'] < 70) ]

age_70_80 = train[(train['Age'] >= 70) & (train['Age'] <= 80) ]



x1 = age_0_10.groupby('Age')['Survived'].value_counts().rename('Count').reset_index()

a1 = x1[x1['Survived'] == 0]['Count'].sum()

b1 = x1[x1['Survived'] == 1]['Count'].sum()



x2 = age_10_20.groupby('Age')['Survived'].value_counts().rename('Count').reset_index()

a2 = x2[x2['Survived'] == 0]['Count'].sum()

b2 = x2[x2['Survived'] == 1]['Count'].sum()



x3 = age_20_30.groupby('Age')['Survived'].value_counts().rename('Count').reset_index()

a3 = x3[x3['Survived'] == 0]['Count'].sum()

b3 = x3[x3['Survived'] == 1]['Count'].sum()



x4 = age_30_40.groupby('Age')['Survived'].value_counts().rename('Count').reset_index()

a4 = x4[x4['Survived'] == 0]['Count'].sum()

b4 = x4[x4['Survived'] == 1]['Count'].sum()



x5 = age_40_50.groupby('Age')['Survived'].value_counts().rename('Count').reset_index()

a5 = x5[x5['Survived'] == 0]['Count'].sum()

b5 = x5[x5['Survived'] == 1]['Count'].sum()



x6 = age_50_60.groupby('Age')['Survived'].value_counts().rename('Count').reset_index()

a6 = x6[x6['Survived'] == 0]['Count'].sum()

b6 = x6[x6['Survived'] == 1]['Count'].sum()



x7 = age_60_70.groupby('Age')['Survived'].value_counts().rename('Count').reset_index()

a7 = x7[x7['Survived'] == 0]['Count'].sum()

b7 = x7[x7['Survived'] == 1]['Count'].sum()



x8 = age_70_80.groupby('Age')['Survived'].value_counts().rename('Count').reset_index()

a8 = x8[x8['Survived'] == 0]['Count'].sum()

b8 = x8[x8['Survived'] == 1]['Count'].sum()



Age_distribution = pd.DataFrame({'Age Range':['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80'],

                         'Not-Survived' : [a1,a2,a3,a4,a5,a6,a7,a8],'Survived' : [b1,b2,b3,b4,b5,b6,b7,b8]})

Age_distribution.set_index('Age Range',inplace = True)

Age_distribution.plot(kind = 'bar')

plt.ylabel("Count")

plt.title("Survival Distribution by Age Ranges")
#Mapping

sex_map = {'male':0,'female': 1}

train['Sex'] = train['Sex'].map(sex_map)

test['Sex'] = test['Sex'].map(sex_map)
embarked_map = {'S':1,'C':2,'Q':3}

train['Embarked'] = train['Embarked'].map(embarked_map)

test['Embarked'] = test['Embarked'].map(embarked_map)
#Check if all values are numerical or not

train.head()

test.head()
X = train.drop(['Survived'],axis = 1)

y = train['Survived']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
LR = LogisticRegression(random_state=0)

LR.fit(X_train,y_train)

y_pred = LR.predict(X_test)

LR_Score = accuracy_score(y_pred,y_test)

print("Accuracy Using Logistic Regression : ", LR_Score)
gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_test)

GB_Score = accuracy_score(y_pred,y_test)

print("Accuracy Using Gaussian Naive Bayes  : ", GB_Score)
DT = DecisionTreeClassifier(random_state=10)

DT.fit(X_train, y_train)

y_pred = DT.predict(X_test)

DT_Score = accuracy_score(y_pred,y_test)

print("Accuracy Using Decision Tree : ",DT_Score)
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=100,max_features='auto',

                            n_jobs = -1,random_state=0,

                            oob_score=True,max_leaf_nodes=30

                             )

RF.fit(X_train,y_train)

y_pred = RF.predict(X_test)

RF_Score = accuracy_score(y_pred,y_test)

print("Accuracy Using RF  : ", RF_Score)
Results = pd.DataFrame({'Model': ['Logistic Regression','Gaussian Naive Bayes','Decision Tree','Random Forest'],

                        'Accuracy Score' : [LR_Score,GB_Score,DT_Score,RF_Score]})
Final_Results = Results.sort_values(by = 'Accuracy Score', ascending=False)

Final_Results = Final_Results.set_index('Model')

print(Final_Results)
Predictions = RF.predict(test)

#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : Id, 'Survived': Predictions })

output.to_csv('submission.csv', index=False)