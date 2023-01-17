import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')

sns.set(font_scale=2.5) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.

import missingno as msno



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
data.isnull().sum()
f, ax =plt.subplots(1,2, figsize=(18,8))

data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')



sns.countplot('Survived', data=data, ax=ax[1])

ax[1].set_title('Survived')

plt.show()
data.groupby(['Sex', 'Survived'])['Survived'].count()
data.groupby(['Sex', 'Survived'])['Survived'].count()
f,ax = plt.subplots(1,2,figsize=(18,8))

data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')



sns.countplot('Sex', hue='Survived', data=data, ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')



plt.show()
pd.crosstab(data.Pclass, data.Survived, margins=True).style.background_gradient(cmap='summer_r')
f, ax=plt.subplots(1,2, figsize=(18,8))



data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number Of Passengers By Pclass')

ax[0].set_ylabel('Count')



sns.countplot('Pclass', hue='Survived', data=data, ax=ax[1])

ax[1].set_title('Pclass:Survived vs Dead')



plt.show()
pd.crosstab([data.Sex, data.Survived], data.Pclass, margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass', 'Survived', hue='Sex', data=data)

plt.show()
print('Oldest Passenger was of:', data['Age'].max(), 'Years')

print('Youngest Passenger was of:', data['Age'].min(), 'Years')

print('Average Age on the ship:', data['Age'].mean(), 'Years')
f,ax=plt.subplots(1,2, figsize=(18,8))



sns.violinplot("Pclass", "Age", hue = "Survived", data=data, split=True, ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))



sns.violinplot("Sex", "Age", hue="Survived", data=data, split=True, ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))



plt.show()
data['Initial']=0

for i in data:

    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')

    
pd.crosstab(data.Initial, data.Sex).T.style.background_gradient(cmap='summer_r')
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

data.groupby('Initial')['Age'].mean()
data.loc[(data.Age.isnull())&(data.Initial=='Master'), 'Age']=5

data.loc[(data.Age.isnull())&(data.Initial=='Miss'), 'Age']=22

data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33

data.loc[(data.Age.isnull())&(data.Initial=='Mrs'), 'Age']=36

data.loc[(data.Age.isnull())&(data.Initial=='Other'), 'Age']=46

    
data.Age.isnull().any()
f,ax=plt.subplots(1,2,figsize=(30,10))



data[data['Survived']==0].Age.plot.hist(ax=ax[0], bins=20, edgecolor='black', color='red')

ax[0].set_title('Survived=0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)



data[data['Survived']==1].Age.plot.hist(ax=ax[1], color='green', bins=20, edgecolor='black')

ax[1].set_title('Survived=1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)



plt.show()
sns.factorplot('Pclass', 'Survived', col='Initial', data=data)

plt.show()
pd.crosstab([data.Embarked, data.Pclass], [data.Sex, data.Survived], margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Embarked', 'Survived', data=data)

fig=plt.gcf()

fig.set_size_inches(5,3)

plt.show()
f,ax=plt.subplots(2,2,figsize=(20,15))



sns.countplot('Embarked',data=data,ax=ax[0,0])

ax[0,0].set_title('No. Of Passengers Boarded')



sns.countplot('Embarked', hue='Sex', data=data, ax=ax[0,1])

ax[0,1].set_title('Male-Female Split for Embarked')



sns.countplot('Embarked', hue='Survived', data=data, ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')



sns.countplot('Embarked', hue='Pclass', data=data, ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
data['Embarked'].fillna('S', inplace=True)
data.Embarked.isnull().any()
pd.crosstab([data.SibSp], data.Survived).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(20,8))

sns.barplot('SibSp','Survived',data=data,ax=ax[0])

ax[0].set_title('SibSp vs Survived')

sns.factorplot('SibSp','Survived',data=data,ax=ax[1])

ax[1].set_title('SibSp vs Survived')

plt.close(2)

plt.show()
pd.crosstab(data.SibSp, data.Pclass).style.background_gradient(cmap='summer_r')
pd.crosstab(data.Parch, data.Pclass).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(20,8))



sns.barplot('Parch','Survived',data=data,ax=ax[0])

ax[0].set_title('Parch vs Survived')



sns.factorplot('Parch','Survived',data=data,ax=ax[1])

ax[1].set_title('Parch vs Survived')



plt.close(2)

plt.show()
print('Highest Fare was:', data['Fare'].max())

print('Lowest Fare was:', data['Fare'].min())

print('Average Fare was:', data['Fare'].mean())
f, ax = plt.subplots(1,3, figsize=(20,8))



sns.distplot(data[data['Pclass']==1].Fare, ax=ax[0])

ax[0].set_title('Fares in Pclass 1')



sns.distplot(data[data['Pclass']==2].Fare, ax=ax[1])

ax[1].set_title('Fares in Pclass 2')



sns.distplot(data[data['Pclass']==3].Fare, ax=ax[2])

ax[2].set_title('Fares in Pclass 3')



plt.show()

sns.heatmap(data.corr(), annot=True, cmap='RdYlGn',  linewidths=0.2)

fig= plt.gcf()

fig.set_size_inches(10,8)

plt.show()
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn',  linewidths=0.2)
data['Age_band']=0



data.loc[data['Age']<=16, 'Age_band']=0

data.loc[(data['Age']>16)&(data['Age']<=32), 'Age_band']=1

data.loc[(data['Age']>32)&(data['Age']<=48), 'Age_band']=2

data.loc[(data['Age']>48)&(data['Age']<=64), 'Age_band']=3

data.loc[(data['Age']>64), 'Age_band']=4



data.head(2)

data['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')

sns.factorplot('Age_band', 'Survived', data=data, col='Pclass')

plt.show
data['Family_Size']=0

data['Family_Size']=data['Parch']+data['SibSp']

data['Alone']=0

data.loc[data.Family_Size==0, 'Alone']=1



f, ax=plt.subplots(1,2, figsize=(18,6))

sns.factorplot('Family_Size', 'Survived', data=data, ax=ax[0])

ax[0].set_title('Family_Size vs Survived')

sns.factorplot('Alone', 'Survived', data=data, ax=ax[1])

ax[1].set_title('Alone vs Survived')

plt.close(2)

plt.close(3)

plt.show()

sns.factorplot('Alone', 'Survived', data=data, hue='Sex', col='Pclass')

plt.show()
data['Fare_Range']=pd.qcut(data['Fare'], 4)

data.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
data['Fare_Range']=pd.qcut(data['Fare'], 4)

data.groupby(['Fare_Range'])['Survived'].mean().to_frame()
data['Fare_cat']=0

data.loc[data['Fare']<=7.91, 'Fare_cat']=0

data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454), 'Fare_cat']=1

data.loc[(data['Fare']>14.454)&(data['Fare']<=31), 'Fare_cat']=2

data.loc[(data['Fare']>31)&(data['Fare']<=513), 'Fare_cata']=3

sns.factorplot('Fare_cat', 'Survived', data=data, hue='Sex')

plt.show()
data['Sex'].replace(['male', 'female'],[0,1], inplace=True)

data['Embarked'].replace(['S', 'C', 'Q'], [0,1,2], inplace=True)

data['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'], [0,1,2,3,4], inplace=True)
data.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin', 'Fare_Range', 'PassengerId'], axis=1, inplace=True)

sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})

fig=plt.gcf()

fig.set_size_inches(18,15)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])

train_X=train[train.columns[1:]]

train_Y=train[train.columns[:1]]

test_X=test[test.columns[1:]]

test_Y=test[test.columns[:1]]

X=data[data.columns[1:]]

Y=data['Survived']
model=svm.SVC(kernel='rbf',C=1,gamma=0.1)

model.fit(train_X,train_Y)

prediction1=model.predict(test_X)

print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,test_Y))
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)

model.fit(train_X,train_Y)

prediction2=model.predict(test_X)

print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,test_Y))
model = LogisticRegression()

model.fit(train_X,train_Y)

prediction3=model.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
a_index = list(range(1,11))

a = pd.Series()

x = [0,1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model = KNeighborsClassifier(n_neighbors=i)

    model.fit(train_X, train_Y)

    prediction=model.predict(test_X)

    a=a.append(pd.Series(metrics.accuracy_score(prediction, test_Y)))

plt.plot(a_index, a)

plt.xtricks(x)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()

print('Accuracies for different values of n are: ', a.values, 'with the max value as', a.values.max())
from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

xyz=[]

accuracy=[]

std=[]

classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']

models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]

for i in models:

    model = i

    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")

    cv_result=cv_result

    xyz.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)

new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       

new_models_dataframe2