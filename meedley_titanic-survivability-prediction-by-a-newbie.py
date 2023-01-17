import matplotlib.pyplot as plt

from pandas import read_csv

import seaborn as sns

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import re

import os



train_df = read_csv('/kaggle/input/titanic/train.csv')

print(train_df.columns.tolist())#column names

print(train_df.shape) #Gives num_rows, num_cols

n = train_df.shape[0] #save number of observations
train_df.head(6)
print(train_df.isna().sum())
#unique values or range for feature set

print('Genders:', train_df['Sex'].unique())

print('Embarked:', train_df['Embarked'].unique())

print('Pclass:', train_df['Pclass'].unique())

print('Survived:', train_df['Survived'].unique())

print('SibSp Range:', train_df['SibSp'].min(),'-',train_df['SibSp'].max())

print('Parch Range:', train_df['Parch'].min(),'-',train_df['Parch'].max())

print('Family size range', (train_df['Parch']+train_df['SibSp']).min(),'-',(train_df['Parch']+train_df['SibSp']).max())

print('Fare Range:', train_df['Fare'].min(),'-',train_df['Fare'].max())

print('Total percent survived: %.2f' % (100*(train_df['Survived']==1).sum()/n))
#indexes of passengers that survived the disaster

didSurvive = train_df['Survived'] == 1
train_df.groupby("Sex")["Age"].describe()
train_df.groupby("Sex")["Age"].median()
#Null ages are filled by gender-based median of age.

train_df['Age'] = train_df["Age"].fillna(train_df.groupby("Sex")["Age"].transform('median'))
sns.distplot(train_df['Age'], bins = 9,kde=False);

sns.distplot(train_df[didSurvive]['Age'], bins = 9,kde=False);

plt.ylabel('Count'); plt.legend(['Total Travellers','Survived Travellers']);
###Gender analysis

#check male and female percentage

print('Gender Data: Percent travellers');

genderPercent = 100*train_df.groupby('Sex')['Sex'].count()/n;

print(genderPercent);

print('\nGender Data: Percent survived travellers')

#Check percent survivabiliy based on sex 

genderX = 100*train_df.groupby('Sex')['Survived'].sum()/n

print(genderX);
fig, ax =plt.subplots(1,2)

fig.tight_layout()

#Gender survivability

sns.countplot(x="Sex", data=train_df, ax=ax[0],order=['male','female']);

sns.countplot(x="Sex", data=train_df[didSurvive],ax=ax[1],order=['male','female']);

plt.ylim(0,600)

ax[0].set_title('Total Travellers');

ax[1].set_title('Survived Travellers');
isMale = train_df['Sex'] == 'male'

isFemale = train_df['Sex'] == 'female'



fig, ax =plt.subplots(1,2)

fig.tight_layout()



sns.distplot(train_df[isMale]['Age'],kde=False,bins=9,ax=ax[0])

sns.distplot(train_df[isFemale]['Age'],kde=False,bins=9,ax=ax[0])



ax[0].set_ylabel('Count'); 

ax[0].set_title('Total travellers')



sns.distplot(train_df[isMale&didSurvive]['Age'],kde=False,bins=9,ax=ax[1])

sns.distplot(train_df[isFemale&didSurvive]['Age'],kde=False,bins=9,ax=ax[1])

ax[1].set_ylabel('Count');

ax[1].set_title('Survived travellers')



plt.legend(['Male','Female']);
train_df.groupby('Pclass')['Fare'].describe()
#Fare Analysis

sns.distplot(train_df['Fare'],kde=False);

sns.distplot(train_df[didSurvive]['Fare'],kde=False);

plt.ylabel('Count'); plt.legend(['Total Travellers','Survived Travellers']);
famSize = train_df['SibSp']+train_df['Parch'];

sns.scatterplot(x=famSize,y=train_df['Fare']);
sns.scatterplot(x=train_df['Pclass'],y=train_df['Fare']);
#Travel Class Analysis

#Class data

print('Travel class Data: Percent travellers')

pclassPercent = 100*train_df.groupby('Pclass')['Pclass'].count()/n

print(pclassPercent)

print('\nTravel class Data: survival percentage')

pclassX = 100*train_df.groupby('Pclass')['Survived'].sum()/n

print(pclassX)

fig, ax =plt.subplots(1,2)

fig.tight_layout()

sns.countplot(x="Pclass", data=train_df,ax=ax[0]);

sns.countplot(x="Pclass", data=train_df[didSurvive],ax=ax[1]);

ax[0].set_title('Total Travellers')

ax[1].set_title('Survived Travellers')

plt.ylim(0,500);
train_df['Cabin'].unique().size
train_df['Deck']= train_df['Cabin'].astype(str).str[0]

print(train_df['Deck'].unique())

sns.countplot(x='Deck', hue='Pclass',data=train_df);
#Travel Companion Analysis

hasParch = train_df['Parch'] > 0 #is traveling with parent

hasSibsp = train_df['SibSp'] > 0 #is traveling with sibling or spouse

hasFamily = train_df['Parch'] + train_df['SibSp'] > 0#is traveling with family

isAlone = train_df['Parch'] + train_df['SibSp'] == 0#is traveling alone



print("Has parent or child aboard: %.2f" % (100*sum(hasParch)/n))

print("Has sibling or spouse aboard: %.2f " % (100*sum(hasSibsp)/n))

print('Is traveling alone:  %.2f '% (100*sum(isAlone)/n))

print('Has family survived:  %.2f '% ( 100*sum(hasFamily&didSurvive)/n))

print('Alone survived:  %.2f '% (100*sum(isAlone&didSurvive)/n))



fig, ax =plt.subplots(1,2)

fig.tight_layout()

familySize = train_df['SibSp'] + train_df['Parch']

sns.countplot(x=familySize,ax=ax[0])

ax[0].set_xlabel('Family Size')

sns.countplot(x=familySize[didSurvive],ax=ax[1])

ax[0].set_title('Total Travellers')

ax[1].set_title('Survived Travellers')

ax[1].set_xlabel('Family Size')

plt.ylim(0,600);
train_df['Titles'] = train_df['Name'].apply(lambda x: re.search(' [A-z][a-z]+\.',x).group(0))

titleOrder = train_df['Titles'].unique()

print(titleOrder)
sns.countplot(x="Titles", data=train_df,order=titleOrder);

plt.xticks(rotation=45)

plt.title('Total Travellers')

plt.figure()

plt.xticks(rotation=45)

sns.countplot(x="Titles", data=train_df[didSurvive],order=titleOrder);

plt.title('Survived Travellers')

plt.ylim(0,600);
#Embarkment Analysis

print('Travel embarkment Data: Percent travellers')

embarkedPercent = 100*train_df.groupby('Embarked')['Embarked'].count()/n

print(embarkedPercent)

print('\nTravel embarkment Data: survival percentage')

embarkedX = 100*train_df.groupby('Embarked')['Survived'].sum()/n

print(embarkedX);
#Fill null values by 'S', which has the highest frequency

train_df['Embarked'] = train_df["Embarked"].fillna('S')
fig, ax =plt.subplots(1,2);fig.tight_layout();

sns.countplot(x="Embarked", data=train_df,ax=ax[0],order=['S','C','Q']);

sns.countplot(x="Embarked", data=train_df[didSurvive],ax=ax[1],order=['S','C','Q']);

ax[0].set_title('Total Travellers');

ax[1].set_title('Survived Travellers');

plt.ylim(0,500);
train_df["Sex"] = train_df["Sex"].astype('category')

train_df["SexCode"] = train_df["Sex"].cat.codes

print(train_df[['Sex','SexCode']].head(5))
#categorize 3 embarked stations to

train_df["Embarked"] = train_df["Embarked"].astype('category')

train_df["EmbarkedCode"] = train_df["Embarked"].cat.codes

print(train_df[['Embarked','EmbarkedCode']].head(5))
train_df["FamSize"] = (train_df['Parch'] + train_df['SibSp'])#is traveling with family

print(train_df[['Parch', 'SibSp', 'FamSize']].head(5))
train_df["Titles"] = train_df["Titles"].astype('category')

train_df["TitleCode"] = train_df["Titles"].cat.codes

print(train_df[['Name', 'Titles','TitleCode']].head(10))
#Make similar changes to the Prediction DATA

predict_df = read_csv('/kaggle/input/titanic/test.csv')

predict_df['Age'] = predict_df["Age"].fillna(predict_df.groupby("Sex")["Age"].transform("median"))



predict_df["Sex"] = predict_df["Sex"].astype('category')

predict_df["SexCode"] = predict_df["Sex"].cat.codes



predict_df["Embarked"] = predict_df["Embarked"].astype('category')

predict_df["EmbarkedCode"] = predict_df["Embarked"].cat.codes



predict_df["FamSize"] = predict_df['Parch'] + predict_df['SibSp']



predict_df['Titles'] = predict_df['Name'].apply(lambda x: re.search(' [A-z][a-z]+\.',x).group(0))

predict_df["Titles"] = predict_df["Titles"].astype('category')

predict_df["TitleCode"] = predict_df["Titles"].cat.codes

#Replace missing fare value by the median of Fare in similar class

predict_df['Fare'] = predict_df["Fare"].fillna(predict_df.groupby("Pclass")["Fare"].transform("median"))
features= ['Pclass', 'SexCode','Age','EmbarkedCode','FamSize','TitleCode','Fare']

x = train_df[features];

y = train_df['Survived'];
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 20,shuffle=True)
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, train_test_split

#using svm for classification

svm = SVC(kernel='linear')

svm.fit(x_train,y_train)

scores = cross_val_score(svm,x,y)#get cross validation score

print("SVM Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.linear_model import LogisticRegression

#Using logistic regression

LR = LogisticRegression(max_iter=200).fit(x_train, y_train)

scores = cross_val_score(LR,x,y)

print("LR Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.ensemble import RandomForestClassifier

#Using random forest

RF = RandomForestClassifier(max_depth=100, n_estimators=200)

RF.fit(x_train, y_train)

scores = cross_val_score(RF,x,y)

#print(scores)

print("RF Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = RF.predict(x_test) #using random forest classifier

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True, cmap="Oranges");
from sklearn.neural_network import MLPClassifier

#using neural network classifier

NN = MLPClassifier(random_state = 4, max_iter = 5000)

NN.fit(x_train, y_train)

scores = cross_val_score(NN,x,y)

print("NN Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn import tree

#I experiemented with split criteria and method, and ended up using the default values.

clf = tree.DecisionTreeClassifier(criterion='gini',splitter='random')

clf = clf.fit(x_train,y_train)

scores = cross_val_score(clf,x,y)

print("DT Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

#y_predict= clf.predict(x_predict) #using logistic regression classifier
x_predict= predict_df[features]

y_predict= RF.predict(x_predict) #using random forest classifier

predict_titanic = {'PassengerID': predict_df['PassengerId'],'Survived': y_predict}

df = pd.DataFrame(predict_titanic, columns= ['PassengerID', 'Survived'])

df.to_csv('/kaggle/working/y_predict.csv',sep=',',index=False)