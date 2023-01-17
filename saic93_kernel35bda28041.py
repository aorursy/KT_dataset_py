import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
gender_submission.head()
sns.heatmap(train.isnull(),yticklabels=False,cmap='magma')
train.drop(['Cabin','Ticket'],axis=1,inplace=True)
train.info()
# Lets check the correlation

train.corr()
# lets do some visualization

sns.countplot(train['Survived'])
sns.countplot(train['Sex'])
sns.countplot(train['Survived'],hue=train['Sex'])
sns.countplot(train['Pclass'])
sns.countplot(train['Survived'],hue=train['Pclass'])
sns.countplot(train['Parch'])
sns.countplot(train['Survived'],hue=train['Parch'])
sns.countplot(train['SibSp'])
sns.countplot(train['Survived'],hue=train['SibSp'])

plt.legend(loc='upper right')
sns.distplot(train['Fare'],bins=50)
sns.pairplot(train)
def impute_age(cols):

    age=cols[0]

    pclass=cols[1]

    sibsp=cols[2]

    parch=cols[3]

    

    age_t=train[(train['Pclass']==pclass)&(train['SibSp']==sibsp)&(train['Parch']==parch)]['Age'].median()

    age_s=train[(train['Pclass']==pclass)]['Age'].median()

    if pd.isnull(age):

        if pd.isnull(age_t):

            age=age_s

        else:

            age=age_t

    else:

        age=age

    return age
train['Age']=train[['Age','Pclass','SibSp','Parch']].apply(impute_age,axis=1)
train.info()
train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
train.info()
sns.boxplot(train['Age'])
sns.boxplot(train['Fare'])
from collections import Counter
Counter([1,1,1,3,3,2,3,2,5,2,3,2,2,3])
def detect_outliers(df,n,features):

    outlier_indices=[]

    

    for col in features:

        Q1=np.percentile(df[col],25)

        Q3=np.percentile(df[col],75)

        IQR=Q3-Q1

        outlier_step=IQR*1.5

        outlier_list_col=df[(df[col]<Q1-outlier_step)|(df[col]>Q3+outlier_step)].index

        outlier_indices.extend(outlier_list_col)

    outlier_indices=Counter(outlier_indices)

    multiple_outliers=[k for k,v in outlier_indices.items() if v>n]

    return multiple_outliers
outliers_to_drop=detect_outliers(train,1,['Age','SibSp','Parch','Fare'])
len(outliers_to_drop)
train.loc[outliers_to_drop]
train.drop(outliers_to_drop,inplace=True)

train.reset_index(drop=True,inplace=True)

train.info()
sns.distplot(train['Fare'],bins=40)
train['Fare']=train['Fare'].apply(lambda x:np.log(x) if x>0 else 0)
sns.distplot(train['Fare'],bins=40)
train['title']=train['Name'].apply(lambda x: x.split(',')[1].strip().split('.')[0])
train['title'].unique()
plt.figure(figsize=(20,10))

sns.countplot('title',hue='Survived',data=train)
train['title']=train['title'].replace(['Don','Rev','Dr','Mme','Ms','Major','Lady','Sir','Mlle','Col','Capt','the Countess','Jonkheer'],'Rare')
train['title']=train['title'].map({'Mr':0,'Mrs':1,'Miss':1,'Master':2,'Rare':3})
train['Embarked']=pd.get_dummies(train['Embarked'],drop_first=True)
train['Sex']=pd.get_dummies(train['Sex'],drop_first=True)
train.head()
train.drop('Name',axis=1,inplace=True)
train.head()
x=train.drop(['PassengerId','Survived'],axis=1)

y=train['Survived']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(x_train,y_train)
predictions=lg.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))

print(accuracy_score(y_test,predictions))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score
mns=[]

stds=[]

clfs=[KNeighborsClassifier(),SVC(),DecisionTreeClassifier(),RandomForestClassifier(),ExtraTreesClassifier(),AdaBoostClassifier(),MultinomialNB()]

for i in clfs:

    cvs=cross_val_score(i,x,y,scoring='accuracy',cv=5,n_jobs=-1,verbose=1)

    mns.append(cvs.mean())

    stds.append(cvs.std())
for i in range(7):

    print(clfs[i].__class__.__name__,':',mns[i]*100)
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train) #training the model

features=x.columns

importances = rfc.feature_importances_ #taking the feature importance values into the a variable

indices = np.argsort(importances)

plt.figure(figsize=(10,10)) #plotting these in to a horizontal barplot

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), features[indices])

plt.xlabel('Relative Importance')
predictions=rfc.predict(x_test)
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
train['Survived'].value_counts()
from xgboost import XGBClassifier
xgbc=XGBClassifier()
xgbc.fit(x_train,y_train)
predictions=xgbc.predict(x_test)
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
test['title']=test['Name'].apply(lambda x: x.split(',')[1].strip().split('.')[0])
test['title']=test['title'].replace(['Don','Rev','Dr','Mme','Ms','Major','Lady','Sir','Mlle','Col','Capt','the Countess','Jonkheer'],'Rare')
test['title']=test['title'].map({'Mr':0,'Mrs':1,'Miss':1,'Master':2,'Rare':3})
test['Embarked']=pd.get_dummies(test['Embarked'],drop_first=True)
test['Sex']=pd.get_dummies(test['Sex'],drop_first=True)
test.drop(['Name','Cabin','PassengerId','Ticket'],axis=1,inplace=True)
test.head()
test_predictions=xgbc.predict(test)
df= pd.read_csv("../input/titanic/test.csv")
Submission=df[['PassengerId','Fare']]
Submission['Survived']=test_predictions
Submission.drop('Fare',axis=1,inplace=True)

Submission.head()
Submission.to_csv('Submission_titanic.csv',index=False)