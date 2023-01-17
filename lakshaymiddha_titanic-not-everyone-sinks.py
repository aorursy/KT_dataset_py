import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import os

print(os.listdir("../input/"))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
train_data.info()
test_data.info()
survived = train_data[train_data['Survived']==1]

not_survived = train_data[train_data['Survived']==0]

print ("Survival ratio: %.1f"%float(len(survived)/(len(survived)+len(not_survived))))
#variables info:



#pclass:

#    1 = 1st

#    2 = 2nd

#    3 = 3rd

#sibsp:

#     number of siblings/spouse aboard the Titanic

#parch:

#    number of parents/children aboard the Titanic

#embarked:

#    C = Cherbourgh

#    Q = Queenstown

#    S = Southampton

data = pd.concat([train_data,test_data],sort=False)
data.tail()
data['Age'] = data['Age'].fillna(value=data['Age'].median())

data['Fare'] = data['Fare'].fillna(value=data['Fare'].median())

data['Embarked'] = data['Embarked'].fillna(value='S')
df = data[:891]

df.head()
pd.crosstab(df.Embarked,df.Survived).plot(kind='bar',figsize=(3,3))

plt.ylabel('No. of Passengers')

plt.show()

# passengers who embarked from Southampton have higher ration of deaths as compared to other cities
pd.crosstab(df.Sex,df.Survived).plot(kind='bar',figsize=(3,3))

plt.ylabel('No. of Passengers')

plt.show()

# ratio of deaths of male is higher than that of female

gender_survival_average = df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean()

print (gender_survival_average)

# females have higher survival ratio than males
pd.crosstab(df.Cabin.str[0],df.Survived).plot(kind='bar',figsize=(3,3))

# type of cabin can be of much use in predicting survival

# so it can be splitted but it contains some missing values 

# so another type= missing can be added to it
# now the changes has to be made on both training and test set

data['Cabin'] = data['Cabin'].fillna('Missing')

data['Cabin'] = data['Cabin'].str[0]

data['Cabin'].value_counts()
pd.crosstab(df.Cabin.str[0],df.Survived).plot(kind='bar',figsize=(3,3))

plt.show()

# it can be  observed that the ones whose cabins are missing have much lower survival raion as compared to others
pd.crosstab(df.Parch,df.Survived).plot(kind='bar',figsize=(3,3))

plt.show()

# as the values are low we need to print actual values to get more information

print(df.groupby('Parch').Survived.value_counts())

#  those with less no companions have higher lower survival ratio.  
pd.crosstab(df.Pclass,df.Survived).plot(kind='bar',figsize=(3,3))

# pclass=3(lower class) have very less survival ratio as compared to other pclass

plt.show()

pclass_survival_average = df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean()

print (pclass_survival_average)

# class 1 (higher class) have highest survival ratio
pd.crosstab(df.SibSp,df.Survived).plot(kind='bar',figsize=(3,3))

plt.show()

# as values are low , we can print to get more information

print(df.groupby('SibSp').Survived.value_counts())



#for SibSp = 0 i.e. travellig alone class have lower survival ratio .
x = np.array(df['Age'])

y = np.array(df['Survived'])

sns.scatterplot(x,y)

plt.xlabel('Age')

plt.ylabel('Survival')

plt.show()

# no much prediction  can be made on the basis of this scatter plot 

# combining age with other variables to get more info

fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(131)

ax2 = fig.add_subplot(132)

ax3 = fig.add_subplot(133)

sns.boxplot(x="Embarked", y="Age",hue="Survived",data=df,ax=ax1)

sns.boxplot(x="Pclass", y="Age",hue="Survived",data=df,ax=ax2)

sns.boxplot(x="Sex", y="Age",hue="Survived",data=df,ax=ax3)

# deductions:

# in first plot those who boarded from C and having age > 30 have 
sns.heatmap(df.drop('PassengerId',axis=1).corr(),vmax=0.6,square=True,annot=True)

#this heatmap shows the correlation between features, more the absolute value more is 

# the dependance so we can neglect features with lower correlation coefficient
# cabin and sex can be extracted from data

# coefficients of fare and age are high so they need to be categorized

# into categories or bands for better visualization.

# one way is to one hot encode variables but by doing so the number of features will

# increase which will not be reliable for feature selection 

# other useful and appropriate way is to label the categories as integers.
data.info()
data.head()
train_test_data = [data]
#'embarked' has three categories S,C,Q

# we can label it as 0,1 and 2 respectively.

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# labelling sex as 1 and 0 

for item in train_test_data:

    item['Sex'] = item['Sex'].map({'male':1,'female':0})
# labelling cabin

data.Cabin.unique()
for item in train_test_data:

    item['Cabin'] = item['Cabin'].map({'M':0, 'C':1, 'E':2, 'G':3, 'D':4, 'A':5, 'B':6, 'F':7, 'T':8})
# merging sibsp and parch features to generate a new feature like companions

for item in train_test_data:

    item['Companions'] = item['SibSp'] + item['Parch']

print (data[['Companions','Survived']].groupby(['Companions'],as_index=False).mean())
sns.barplot(x = 'Companions',y = 'Survived',ci=None,data=data)

# those who travel alone have less surival ratio.

# those with 4,5,6 companions also have less survival ratio.

# so it can be converted into travelling alone (is_alone) feature
# lets check if is_alone feature will be reliable or not

for item in train_test_data:

    item['Is_alone'] = 0

    item.loc[item['Companions'] == 0,'Is_alone'] = 1

print (data[['Is_alone', 'Survived']].groupby(['Is_alone'], as_index=False).mean())

# those who are travlling alone have only 30% chances of survival

# so this feature can be used in place of companions i.e. Sibsp and parch
#categorising fare

cat = [0,7,10,21, 41,512]

lab = ['0','1','2','3','4']

data['FareBand'] = pd.cut(data['Age'], bins = cat, labels = lab)
#categorising age

cat = [0,16,32,48, 64,80]

lab = ['0','1','2','3','4']

data['AgeBand'] = pd.cut(data['Age'], bins = cat, labels = lab)

data.head()
# feature selection for training the classifier

train = data[0:891]

train.head()

test = data[891:]

test.head()

training_data = train.drop(['PassengerId','Name','SibSp','Age','Parch','Ticket','Fare'],axis=1)

testing_data = test.drop(['Name','SibSp','Parch','Ticket','Fare','Survived','Age'],axis=1)
y = training_data['Survived']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(training_data.drop(['Survived'],axis=1),y,random_state=0,test_size=0.2)
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
clf1 = LogisticRegression(solver='liblinear').fit(x_train,y_train)

clf2 = SVC().fit(x_train,y_train)

clf3 = DecisionTreeClassifier().fit(x_train,y_train)

clf4 = RandomForestClassifier(n_estimators=1000).fit(x_train,y_train)
pred1 = clf1.predict(x_test)

pred2 = clf2.predict(x_test)

pred3 = clf3.predict(x_test)

pred4 = clf4.predict(x_test)
from sklearn.metrics import accuracy_score

print ('accuracy score of Logistic regression classifier: ', accuracy_score(pred1,y_test))

print ('accuracy score of support vector classifier: ', accuracy_score(pred2,y_test))

print ('accuracy score of decision tree: ', accuracy_score(pred3,y_test))

print ('accuracy_score: ', accuracy_score(pred4,y_test))
from sklearn.metrics import classification_report

print(classification_report(y_test,pred1))

print(classification_report(y_test,pred2))

print(classification_report(y_test,pred3))

print(classification_report(y_test,pred4))

testing_data.tail()
Testing = testing_data.drop(['PassengerId'], axis = 1)
T_pred = clf2.predict(Testing).astype(int)

PassengerId = testing_data['PassengerId']

result = pd.DataFrame({'PassengerId': PassengerId, 'Survived':T_pred })

result.head()
result.to_csv("Titanic_sinking_Submission2.csv", index = False)