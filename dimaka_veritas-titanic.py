import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
display(train_df.head())

print('Shape of Data : ',train_df.shape)
train_df.describe()
train_df.isnull().sum()
train_df['Initial']=0

for i in train_df:

    train_df['Initial']=train_df.Name.str.extract('([A-Za-z]+)\.')

    

pd.crosstab(train_df.Initial,train_df.Sex).T.style.background_gradient(cmap='gist_rainbow')    
train_df['Initial'].replace(['Dr','Mlle','Mme','Ms','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],

                            ['Other','Miss','Miss','Miss','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],

                            inplace=True)

pd.crosstab(train_df.Initial,train_df.Sex).T.style.background_gradient(cmap='gist_rainbow')    
train_df.groupby('Initial')['Age'].mean()

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mr'),'Age']=32.5

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mrs'),'Age']=36

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Master'),'Age']=4.5

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Miss'),'Age']=22

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Other'),'Age']=44.5
f,ax=plt.subplots()

train_df['Survived'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',shadow=True)

ax.set_title('Survived')

ax.set_ylabel('')

plt.show()
sns.barplot(x= 'Sex', y= 'Survived' , data = train_df);
sns.violinplot(x= 'Sex',y='Age',data = train_df, hue = 'Survived', split = True);
sns.stripplot(x= 'Initial',y='Age',data = train_df, jitter = True , hue = 'Survived');
sns.factorplot(x= 'Embarked',y='Age',data = train_df, kind = 'bar', hue = 'Survived');
f,ax=plt.subplots(1,1,figsize=(6,5))

train_df['Embarked'].value_counts().plot.pie(explode=[0,0,0],autopct='%1.1f%%',ax=ax)

plt.show()
train_df['Embarked'].fillna('S',inplace=True)
train_df['FamilySize'] = train_df['Parch'] + train_df['SibSp']

sns.barplot(x= 'FamilySize', y= 'Survived' , data = train_df);
sns.pairplot(train_df, hue = 'Sex',palette='coolwarm');
train_df['Age_cat']=0

train_df.loc[train_df['Age']<=12,'Age_cat']=0

train_df.loc[(train_df['Age']>12)&(train_df['Age']<=18),'Age_cat']=1

train_df.loc[(train_df['Age']>18)&(train_df['Age']<=40),'Age_cat']=2

train_df.loc[(train_df['Age']>40)&(train_df['Age']<=50),'Age_cat']=3

train_df.loc[(train_df['Age']>50)&(train_df['Age']<=70),'Age_cat']=4

train_df.loc[train_df['Age']>70,'Age_cat']=5
train_df['Sex'].replace(['male','female'],[0,1],inplace=True)

train_df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

train_df['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
train_df['Fare_cat']=0

train_df.loc[train_df['Fare']<=7.775,'Fare_cat']=0

train_df.loc[(train_df['Fare']>7.775)&(train_df['Fare']<=8.662),'Fare_cat']=1

train_df.loc[(train_df['Fare']>8.662)&(train_df['Fare']<=14.454),'Fare_cat']=2

train_df.loc[(train_df['Fare']>14.454)&(train_df['Fare']<=26.0),'Fare_cat']=3

train_df.loc[(train_df['Fare']>26.0)&(train_df['Fare']<=52.369),'Fare_cat']=4

train_df.loc[train_df['Fare']>52.369,'Fare_cat']=5
train_df.drop(['Name','Age','Ticket','Cabin','SibSp','Parch','Fare','PassengerId'],axis=1,inplace=True)
train_df.head(15)
sns.heatmap(train_df.corr(),annot=True,cmap='BuPu',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()


test_df['Initial']=0

for i in test_df:

    test_df['Initial']=test_df.Name.str.extract('([A-Za-z]+)\.')



test_df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],

                           ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Other'],inplace=True)



test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Mr'),'Age']=32.5

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Mrs'),'Age']=36

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Master'),'Age']=4.5

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Miss'),'Age']=22

test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Other'),'Age']=44.5





test_df['Embarked'].fillna('S',inplace=True)





test_df['FamilySize'] = test_df['Parch'] + test_df['SibSp']



test_df['Age_cat']=0

test_df.loc[test_df['Age']<=12,'Age_cat']=0

test_df.loc[(test_df['Age']>12)&(test_df['Age']<=18),'Age_cat']=1

test_df.loc[(test_df['Age']>18)&(test_df['Age']<=40),'Age_cat']=2

test_df.loc[(test_df['Age']>40)&(test_df['Age']<=50),'Age_cat']=3

test_df.loc[(test_df['Age']>50)&(test_df['Age']<=70),'Age_cat']=4

test_df.loc[test_df['Age']>70,'Age_cat']=5







test_df['Sex'].replace(['male','female'],[0,1],inplace=True)

test_df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

test_df['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)





test_df['Fare_cat']=0

test_df.loc[test_df['Fare']<=7.775,'Fare_cat']=0

test_df.loc[(test_df['Fare']>7.775)&(test_df['Fare']<=8.662),'Fare_cat']=1

test_df.loc[(test_df['Fare']>8.662)&(test_df['Fare']<=14.454),'Fare_cat']=2

test_df.loc[(test_df['Fare']>14.454)&(test_df['Fare']<=26.0),'Fare_cat']=3

test_df.loc[(test_df['Fare']>26.0)&(test_df['Fare']<=52.369),'Fare_cat']=4

test_df.loc[test_df['Fare']>52.369,'Fare_cat']=5





test_df.drop(['Name','Age','Ticket','Cabin','SibSp','Parch','Fare'],axis=1,inplace=True)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn import metrics







from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.metrics import confusion_matrix #for confusion matrix
X_train, X_test, y_train, y_test = train_test_split(train_df.drop('Survived',axis=1), train_df['Survived'], test_size=0.30,random_state=1801)



#X_train=train_df[train_df.columns[1:]]

#y_train=train_df[train_df.columns[:1]]

#X_test=test_df[test_df.columns[1:]]

#y_test=test_df[test_df.columns[:1]]
train_df.info()
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print(classification_report(y_test,predictions))
print('The Logistic Regression model accuracy \t',metrics.accuracy_score(predictions,y_test))
# Naive Bayes

Naivemodel=GaussianNB()

Naivemodel.fit(X_train,y_train.values.ravel())

NB_prediction=Naivemodel.predict(X_test)

print('The accuracy of the NaiveBayes model is\t\t\t',metrics.accuracy_score(NB_prediction,y_test))



# Decision Tree

modelTree=DecisionTreeClassifier()

modelTree.fit(X_train,y_train)

DT_prediction=modelTree.predict(X_test)

print('The accuracy of the Decision Tree is \t\t\t',metrics.accuracy_score(DT_prediction,y_test))
logmodel_predictions = logmodel.predict(test_df[test_df.columns[1:]])

NaiveBayes_predictions = Naivemodel.predict(test_df[test_df.columns[1:]])

DecisionTree_predictions =modelTree.predict(test_df[test_df.columns[1:]])
logsubmission = pd.DataFrame({"PassengerId": test_df["PassengerId"],"Survived": logmodel_predictions})

logsubmission.to_csv('titanic logistic.csv', index=False)



NaiveBayessubmission = pd.DataFrame({"PassengerId": test_df["PassengerId"],"Survived": NaiveBayes_predictions})

NaiveBayessubmission.to_csv('titanic NaiveBayes.csv', index=False)





DecisionTree_submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],"Survived": DecisionTree_predictions})

DecisionTree_submission.to_csv('titanic DecisionTree.csv', index=False)
