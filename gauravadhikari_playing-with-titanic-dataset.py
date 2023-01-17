# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

import numpy as np 

import pandas as pd

import seaborn as sns

sns.set_style('darkgrid')

import matplotlib.pyplot as plt

%matplotlib inline

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import VotingClassifier
testDf=pd.read_csv("../input/test.csv")

trainDf=pd.read_csv("../input/train.csv")

genderDf=pd.read_csv('../input/gender_submission.csv')
testDf.info()
passengerID=testDf['PassengerId']
titanicDf=pd.concat([testDf,trainDf],keys=['Test','Train'],names=['Dataset','Dataset ID'])
titanicDf.head()
titanicDf.tail()
# titanicDf.xs('Train').head()  # Another method for doing so

titanicDf.loc['Train'].head()
titanicDf.loc['Test'].head()
titanicDf.info()
titanicDf.xs('Train').info()
titanicDf.xs('Test').info()
titanicDf.xs('Train').describe()
titanicDf.xs("Test").describe()
titanicDf.xs('Train').hist(bins=20,figsize=(15,10))
# As we can see there are a couple of null values that we have to resolve

titanicDf.xs('Train').isnull().sum()
# To find the most repetative data in the Embarked column

embarked_modeSeries=titanicDf.xs('Train')['Embarked'].dropna().mode()
embarked_mode=embarked_modeSeries[0]

embarked_mode
titanicDf['Embarked'].fillna(embarked_mode,inplace=True)
titanicDf['Embarked'][titanicDf['Embarked'].isnull()==True]
titanicDf.isnull().sum()
FareMode=titanicDf['Fare'].mode()

FareMode
titanicDf['Fare'].fillna(FareMode[0],inplace=True)
titanicDf['Fare'].isnull().sum()
titanicDf.xs('Train').corr()['Age'].sort_values(ascending=False)
titanicDf.xs('Train')[['Age','Sex']].groupby('Sex').mean().sort_values(by='Age',ascending=False)
titanicDf['Pclass'].unique()
for valAge in ['male','female']:

    for x in range(0,3):

        titanicDfMedianAge=titanicDf.xs('Train')[(titanicDf.xs('Train')['Sex']==valAge) &

                                                 (titanicDf.xs('Train')['Pclass']==x+1)]['Age'].dropna().median()

        print('the median age is ',titanicDfMedianAge)

        

        titanicDf.loc[(titanicDf["Age"].isnull()) & (titanicDf["Sex"] == valAge) & (titanicDf["Pclass"] == x+1), "Age"] = titanicDfMedianAge

        
# Display specified ages for test 

# titanicDf.loc[(titanicDf["Sex"] == valAge) & (titanicDf["Pclass"] == x+1),"Age"]
titanicDf.loc['Train','Cabin'].unique()
titanicDf.loc['Train','Cabin'].isnull().sum()
titanicDf.fillna('None',inplace=True)
titanicDf.loc['Train','Cabin'].isnull().sum()
titanicDf.isnull().sum()
titanicDf['Title']=titanicDf['Name'].str.extract("([A-Za-z]+)\.",expand=False)
# Listing out the unique titles that we have created

set(titanicDf['Title'])
pd.crosstab(titanicDf['Title'],titanicDf['Sex'])
titanicDf['Title'].replace('Mme','Mrs',inplace=True)

titanicDf['Title'].replace('Ms','Miss',inplace=True)

titanicDf["Title"].replace(["Capt", "Col", "Countess", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir"], "Special", inplace=True)

titanicDf['Title'].replace('Mlle','Miss',inplace=True)
# titanicDf.xs('Train')[['Survived','Title']].groupby(['Title']).mean().sort_values(by='Survived',ascending=False)titanicDf
titanicDf.loc['Train'][['Title','Survived']].groupby('Title').sum().sort_values(by='Survived',ascending=False)
pd.cut(titanicDf.loc['Train','Age'],bins=5).dtype
titanicDf.loc[titanicDf['Age']<16,'Age']=0

titanicDf.loc[(titanicDf['Age']>=16) & (titanicDf['Age']<32),'Age']=1

titanicDf.loc[(titanicDf['Age']>=32) & (titanicDf['Age']<48),'Age']=2

titanicDf.loc[(titanicDf['Age']>=48) & (titanicDf['Age']<64),'Age']=3

titanicDf.loc[(titanicDf['Age']>=64),'Age']=4
titanicDf['Age'].sort_values().unique()
# Its better to have such values in int type 

titanicDf['Age']=titanicDf['Age'].astype(int)
titanicDf['Age'].value_counts()
titanicDf.loc['Train'][['Age','Survived']].groupby('Age').sum().sort_values(by='Survived',ascending=False)
# Including the passenger on board

titanicDf['Family']=titanicDf['SibSp']+titanicDf['SibSp']+1 
titanicDf.loc['Train'][['Family','Survived']].groupby('Family').sum()
titanicDf['IsAlone']=0

titanicDf.loc[titanicDf['Family']>1,"IsAlone"]=1
titanicDf.loc['Train'][['IsAlone','Survived']].groupby('Survived').mean().sort_values(by='IsAlone',ascending=False)
set(pd.qcut(titanicDf['Fare'],q=4))
titanicDf.loc[titanicDf['Fare']<=7.896,'Fare']=0

titanicDf.loc[(titanicDf['Fare']>7.896) & (titanicDf['Fare']<=14.454),'Fare']=1

titanicDf.loc[(titanicDf['Fare']>14.454) & (titanicDf['Fare']<=31.275),'Fare']=2

titanicDf.loc[(titanicDf['Fare']>31.275),'Fare']=3
titanicDf['Fare'].astype(int)
titanicDf['Fare'].unique()
titanicDf['Fare'].value_counts()
titanicDf.loc['Train'][['Fare','Survived']].groupby('Fare').sum().sort_values(by='Survived',ascending=False)
titanicDf['Cabin'].isnull().sum()
titanicDf['Cabin']=titanicDf['Cabin'].str.extract("([A-Za-z]+)",expand=False)
titanicDf.loc['Train'][['Cabin','Survived']].groupby(['Cabin']).sum().sort_values(by='Survived',ascending=False)
titanicDf.info()
titanicDf.drop(['Name','PassengerId','Ticket'],axis=1,inplace=True)
titanicDf.head()
titanicDf['Survived'].value_counts()
labelEncoder=LabelEncoder()

titanicDfEncodedTrain=titanicDf.loc['Train'].apply(labelEncoder.fit_transform)

titanicDfEncodedTest=titanicDf.loc['Test'].apply(labelEncoder.fit_transform)
titanicDfEncodedTrain.head()
titanicDfEncodedTest.head()
plt.figure(figsize=(20,10))

sns.heatmap(titanicDfEncodedTrain.corr(),annot=True,)
X_train=titanicDfEncodedTrain.drop('Survived',axis=1)

y_train=titanicDfEncodedTrain['Survived']
randomForestClassifier=RandomForestClassifier()

randomForestClassifier.fit(X_train,y_train)
randomForestClassifier.feature_importances_
# Zips the feature columns to the  feature importances 

feature_importances=zip(list(X_train.columns.values),randomForestClassifier.feature_importances_)



# sort acc to the feature importances 

feature_importances=sorted(feature_importances,key=lambda feature:feature[1],reverse=True)



# print the columns names and its importances in a good fashion

for name,score in feature_importances:

    print("{:10} | {}".format(name,score))
titanicDf.drop('Cabin',axis=1,inplace=True)
y_train=titanicDf.loc['Train']['Survived']
X_titanicdf=pd.get_dummies(titanicDf.drop('Survived',axis=1))

y_titanic=titanicDf['Survived']
X_train=X_titanicdf.loc['Train']

y_train=y_titanic.loc['Train'].astype(int)

X_test=X_titanicdf.loc['Test']
X_train.head()
y_train.head()
X_test.head()
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.fit_transform(X_test)
logisticClassifier=LogisticRegression()

cross_val_score(logisticClassifier,X_train,y_train,cv=10,scoring='accuracy').mean()
sgcClassifer=SGDClassifier()

cross_val_score(sgcClassifer,X_train,y_train,scoring='accuracy').mean()
svcClassifier=SVC()

cross_val_score(svcClassifier,X_train,y_train,scoring='accuracy').mean()
ldaClassifier=LinearDiscriminantAnalysis()

cross_val_score(ldaClassifier,X_train,y_train,scoring='accuracy').mean()
grid_params=[

    {

    "C":[4,5,6],

    "kernel":["rbf"],

    "tol":[0.00001,0.00003,0.00005,0.00008],

    "gamma":["auto","scale"],

    "class_weight": ["balanced", None],

    "shrinking":[True,False],

    "probability":[True]

    },

    {

        "kernel":["linear"],

        "degree":[1,3,5],

        "gamma":['auto',"scale"],

        "probability":[True]

    }

    ]
gridsearchCV=GridSearchCV(estimator=svcClassifier,param_grid=grid_params,verbose=2,scoring="accuracy")
gridsearchCV.fit(X_train,y_train)
gridsearchCV.best_params_
gridsearchCV.best_score_
svcClassifier=gridsearchCV.best_estimator_
cross_val_score(svcClassifier,X_train,y_train,scoring='accuracy').mean()
votingClassifierEstimators=[("svc",svcClassifier),

                           ("lda",ldaClassifier),

                            ("Logistic Classifier",logisticClassifier)

                           ]
votingClassifier=VotingClassifier(estimators=votingClassifierEstimators,voting="soft")
votingClassifier.fit(X_train,y_train)
cross_val_score(votingClassifier, X_train, y_train, cv=10, scoring="accuracy").mean()
predictions=votingClassifier.predict(X_test)
submissions=pd.DataFrame(

{

    'PassengerId':passengerID,

    'Survived':predictions

})
submissions.head(7)
# Writing the submissions to a csv file 

submissions.to_csv("submissions.csv",index=False)