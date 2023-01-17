# Imports



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

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import tree, linear_model
# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# preview the data

titanic_df.head()
titanic_df['Age'].hist()


for i in range(0,9):

    titanic_df.loc[titanic_df['Age'].map(lambda x:(x<(10*i+10)) & (x>=(10*i))),'Age']=i

titanic_df['Age'].hist()
####Split variable Nom####

NameVar=titanic_df['Name']

NameVar2=test_df['Name']

split1= lambda x: x.split(", ")

split2= lambda x: x.split(". ")

i=0

for l in NameVar:

    a=split1(l)

    b=split2(a[1])

    titanic_df.loc[i,'Name']=b[0]

    i=i+1

i=0

for l in NameVar2:

    a=split1(l)

    b=split2(a[1])

    test_df.loc[i,'Name']=b[0]

    i=i+1

titanic_df.groupby('Name') ['Name'].count()

Titles=['Master','Miss','Mr','Mrs'] 



titanic_df.loc[titanic_df['Name'].map(lambda x: x not in Titles),'Name']=4

titanic_df.loc[titanic_df['Name']=="Mr",'Name']=0

titanic_df.loc[titanic_df['Name']=="Mrs",'Name']=1

titanic_df.loc[titanic_df['Name']=="Miss",'Name']=2

titanic_df.loc[titanic_df['Name']=="Master",'Name']=3

test_df.loc[test_df['Name'].map(lambda x: x not in Titles),'Name']=4

test_df.loc[test_df['Name']=="Mr",'Name']=0

test_df.loc[test_df['Name']=="Mrs",'Name']=1

test_df.loc[test_df['Name']=="Miss",'Name']=2

test_df.loc[test_df['Name']=="Master",'Name']=3



####Code SEX####

titanic_df['Sex']=(titanic_df['Sex']=='male')*1

test_df['Sex']=(test_df['Sex']=='male')*1

###SibSP, Parch, Family####

titanic_df.loc[titanic_df['SibSp']>2,'SibSp']=3

titanic_df.loc[titanic_df['Parch']==2,'Parch']=1

titanic_df.loc[titanic_df['Parch']==3,'Parch']=1

titanic_df.loc[titanic_df['Parch']>3,'Parch']=2

titanic_df['Family']=titanic_df['SibSp']+titanic_df['Parch']

titanic_df.loc[titanic_df['Family']>0,'Family']=1



test_df.loc[test_df['SibSp']>2,'SibSp']=3

test_df.loc[test_df['Parch']==2,'Parch']=1

test_df.loc[test_df['Parch']==3,'Parch']=1

test_df.loc[test_df['Parch']>3,'Parch']=2

test_df['Family']=test_df['SibSp']+test_df['Parch']

test_df.loc[test_df['Family']>0,'Family']=1

####Split Cabine######

Cabine=titanic_df['Cabin']

i=0

for l in Cabine:

    if str(l)=='nan':

        titanic_df.loc[i,'Cabin']="No"

    else:

        if len(str(l))==1:

            titanic_df.loc[i,'Cabin']=l

        else:

            titanic_df.loc[i,'Cabin']=(str(l))[0]

    i=i+1

Cabine=test_df['Cabin']

i=0

for l in Cabine:

    if str(l)=='nan':

        test_df.loc[i,'Cabin']="No"

    else:

        if len(str(l))==1:

            test_df.loc[i,'Cabin']=l

        else:

            test_df.loc[i,'Cabin']=(str(l))[0]

    i=i+1



###Split Fare - Chercher meilleur split en 3 parties en maximisant la variance Inter groupes

s1=5

s2=20

titanic_df.loc[titanic_df['Fare']<s1,'CFare']=0     

titanic_df.loc[((titanic_df['Fare']<s2) & (titanic_df['Fare']>=s1)),'CFare']=1 

titanic_df.loc[titanic_df['Fare']>=s2,'CFare']=2

difExp=np.square(titanic_df.groupby('CFare')['Survived'].mean()-titanic_df['Survived'].mean())

vInter=sum(titanic_df.groupby('CFare')['Survived'].count()*difExp)/titanic_df['Survived'].count()

best=vInter

bestFare=[5,20]

for l1 in range(0,50):

    for l2 in range(l1,40):

        s1=5+l1*2

        s2=10+l2*2

        titanic_df.loc[titanic_df['Fare']<s1,'CFare']=0     

        titanic_df.loc[((titanic_df['Fare']<s2) & (titanic_df['Fare']>=s1)),'CFare']=1 

        titanic_df.loc[titanic_df['Fare']>=s2,'CFare']=2

        difExp=np.square(titanic_df.groupby('CFare')['Survived'].mean()-titanic_df['Survived'].mean())

        vInter=sum(titanic_df.groupby('CFare')['Survived'].count()*difExp)/titanic_df['Survived'].count()

        if vInter>best:

            best=vInter

            bestFare=[s1,s2]

s1=bestFare[0]

s2=bestFare[1]

titanic_df.loc[titanic_df['Fare']<s1,'CFare']=0     

titanic_df.loc[((titanic_df['Fare']<s2) & (titanic_df['Fare']>=s1)),'CFare']=1 

titanic_df.loc[titanic_df['Fare']>=s2,'CFare']=2

titanic_df['Fare']=titanic_df['CFare']

titanic_df.drop('CFare',axis=1,inplace=True)



test_df.loc[test_df['Fare']<s1,'Fare']=0     

test_df.loc[((test_df['Fare']<s2) & (test_df['Fare']>=s1)),'Fare']=1 

test_df.loc[test_df['Fare']>=s2,'Fare']=2

test_df['Fare'] = titanic_df['Fare'].fillna("0")

###Drop Ticket, PassenID

titanic_df.drop('Ticket',axis=1,inplace=True)

titanic_df.drop('PassengerId',axis=1,inplace=True)

test_df.drop('Ticket',axis=1,inplace=True)

test_df.drop('PassengerId',axis=1,inplace=True)



####Imputer Embarked avec la median

titanic_df['Embarked'] = titanic_df['Embarked'].fillna("S")

titanic_df.loc[titanic_df['Embarked']=="S",'Embarked']=0

titanic_df.loc[titanic_df['Embarked']=="C",'Embarked']=1

titanic_df.loc[titanic_df['Embarked']=="Q",'Embarked']=2

test_df.loc[test_df['Embarked']=="S",'Embarked']=0

test_df.loc[test_df['Embarked']=="C",'Embarked']=1

test_df.loc[test_df['Embarked']=="Q",'Embarked']=2



#####Split cabin

titanic_df.loc[titanic_df['Cabin']=="B",'Cabin']=0

titanic_df.loc[titanic_df['Cabin']=="C",'Cabin']=1

titanic_df.loc[titanic_df['Cabin']=="D",'Cabin']=2

titanic_df.loc[titanic_df['Cabin']=="E",'Cabin']=3

cab=["B","C","D","E"]

titanic_df.loc[titanic_df['Cabin'].map(lambda x: x not in cab),'Cabin']=4



test_df.loc[test_df['Cabin']=="B",'Cabin']=0

test_df.loc[test_df['Cabin']=="C",'Cabin']=1

test_df.loc[test_df['Cabin']=="D",'Cabin']=2

test_df.loc[test_df['Cabin']=="E",'Cabin']=3

test_df.loc[test_df['Cabin'].map(lambda x: x not in cab),'Cabin']=4
####Dummy names

#Dum1=pd.get_dummies(titanic_df['Name'])

#Dum2=pd.get_dummies(test_df['Name'])

#titanic_df.drop('Name',axis=1,inplace=True)

#test_df.drop('Name',axis=1,inplace=True)

#titanic_df=pd.concat([titanic_df,Dum1],axis=1)

#test_df=pd.concat([test_df,Dum2],axis=1)





###Dummy fare

#Dum1=pd.get_dummies(titanic_df['Fare'])

#Dum2=pd.get_dummies(test_df['Fare'])

#titanic_df.drop('Fare',axis=1,inplace=True)

#test_df.drop('Fare',axis=1,inplace=True)

#titanic_df=pd.concat([titanic_df,Dum1],axis=1)

#test_df=pd.concat([test_df,Dum2],axis=1)



###Dummy SibSp

#Dum1=pd.get_dummies(titanic_df['SibSp'])

#Dum2=pd.get_dummies(test_df['SibSp'])

#titanic_df.drop('SibSp',axis=1,inplace=True)

#test_df.drop('SibSp',axis=1,inplace=True)

#titanic_df=pd.concat([titanic_df,Dum1],axis=1)

#test_df=pd.concat([test_df,Dum2],axis=1)



###Dummy Parch

#Dum1=pd.get_dummies(titanic_df['Parch'])

#Dum2=pd.get_dummies(test_df['Parch'])

#titanic_df.drop('Parch',axis=1,inplace=True)

#test_df.drop('Parch',axis=1,inplace=True)

#titanic_df=pd.concat([titanic_df,Dum1],axis=1)

#test_df=pd.concat([test_df,Dum2],axis=1)



###Dummy Embarked

#Dum1=pd.get_dummies(titanic_df['Embarked'])

#Dum2=pd.get_dummies(test_df['Embarked'])

#titanic_df.drop('Embarked',axis=1,inplace=True)

#test_df.drop('Embarked',axis=1,inplace=True)

#titanic_df=pd.concat([titanic_df,Dum1],axis=1)

#test_df=pd.concat([test_df,Dum2],axis=1)
####Age Imputation

X_train=(titanic_df.drop(['Age','Survived'],axis=1)).loc[np.logical_not(titanic_df['Age'].map(np.isnan)),:]

Y_train=titanic_df.loc[np.logical_not(titanic_df['Age'].map(np.isnan)),'Age']

regr = linear_model.LinearRegression()

regr.fit(X_train,Y_train)

X_train2=(titanic_df.drop('Survived',axis=1))

X_test=test_df

X_train2.loc[titanic_df['Age'].map(np.isnan),'Age']=regr.predict((titanic_df.drop(['Age','Survived'],axis=1)).loc[titanic_df['Age'].map(np.isnan),:])

X_test.loc[test_df['Age'].map(np.isnan),'Age']=regr.predict((test_df.drop('Age',axis=1)).loc[test_df['Age'].map(np.isnan),:])

X_train=X_train2

Y_train=titanic_df['Survived']
X_train=(titanic_df.drop('Survived',axis=1)).loc[np.logical_not(titanic_df['Age'].map(np.isnan)),:]

Y_train=titanic_df.loc[np.logical_not(titanic_df['Age'].map(np.isnan)),'Survived']



# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



#Y_pred = logreg.predict(X_test)



logreg.score(X_train, Y_train)
# Support Vector Machines



svc = SVC()



svc.fit(X_train, Y_train)



# Y_pred = svc.predict(X_test)



svc.score(X_train, Y_train)



clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, Y_train)

clf.score(X_train,Y_train)

#Y_pred=clf.predict(X_test)
#Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train,Y_train)

random_forest.score(X_train,Y_train)

Y_pred=random_forest.predict(X_test)
len(Y_pred)
knn = KNeighborsClassifier(n_neighbors = 1)



knn.fit(X_train, Y_train)



#Y_pred = knn.predict(X_test)



knn.score(X_train, Y_train)
test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

submission = pd.DataFrame({

        "PassengerId": test_df['PassengerId'],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)