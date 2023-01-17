#First step is importing the required Libraries,these four libaries are useful for our intial process so I make it a habit to do this as a first step.

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns
df=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")

print(df.shape)

df.head() #After importing into dataframe use head fucntion to have a look at dataset and columns.
print(test.shape)

test.head()
# First Calculate the percent of missing values of each attribute in Training and Test sets

a=df.isnull().sum().sort_values(ascending=False)

percent=(df.isnull().sum()/df.isnull().count())*100

a=pd.concat([a,percent],axis=1)

a.head()
a=test.isnull().sum().sort_values(ascending=False)

percent=(test.isnull().sum()/test.isnull().count())*100

a=pd.concat([a,percent],axis=1)

a.head()
main_df=test["PassengerId"]# For Submission file we will need Id 
datasets=[df,test]  

for dataset in datasets:

    dataset.drop(["Cabin","PassengerId","Ticket","Fare"],axis=1,inplace=True)

    dataset["Age"].fillna(dataset["Age"].mode()[0],inplace=True)

    dataset["Embarked"].fillna(dataset["Embarked"].mode()[0],inplace=True)
df.head()# have a look at our remaining features
sns.countplot(x="Pclass",hue="Survived",data=df)

# from below Countplot we can see that Pclass plays a role in prediction of Survived,so we will keep that feature
ax=sns.barplot(x="Pclass",y="Survived",data=df)

ax.set_ylabel("Survival Probability")
sns.countplot(x="Sex",hue="Survived",data=df) # sex also plays a role,we will keep that 
sns.barplot(x="Sex",y="Survived",data=df)
fig = plt.figure(figsize=(10,8),)

a=sns.kdeplot(df.loc[(df["Survived"]==0),"Age"],color="g",shade=True,label="Not Survived")

a=sns.kdeplot(df.loc[(df["Survived"]==1),"Age"],color="b",shade=True,label="Survived")

plt.title('Age Distribution - Surviver V.S. Non Survivors', fontsize = 20)

plt.xlabel("Passenger Age", fontsize = 12)

plt.ylabel('Frequency', fontsize = 12);
sns.lmplot('Age','Survived',data=df)



# We can also say that the older the passenger the lesser the chance of survival
ax=sns.countplot(x="SibSp",hue="Survived",data=df)
sns.factorplot(x="SibSp",y="Survived",data=df,kind="bar")
sns.countplot(x="Parch",hue="Survived",data=df)
sns.factorplot(x="Parch",y="Survived",data=df,kind="bar")
train_test_data = [df, test]

for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False) 

for dataset in train_test_data:

    dataset["Title"]=dataset["Title"].map({"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 })

df.drop("Name",axis=1,inplace=True)

test.drop("Name",axis=1,inplace=True)    
df["FamilySize"]=df["SibSp"]+df["Parch"]

test["FamilySize"]=test["SibSp"]+test["Parch"]

labels=['SibSp', 'Parch']

df.drop(labels,axis=1,inplace=True)

test.drop(labels,axis=1,inplace=True)
test.head()
# for Training set

s=pd.get_dummies(df,drop_first=True)# Dont Forget to add drop_first=True(used to avoid Dummy Variable Trap)

target=s["Survived"]

train_data=s.iloc[:,1:]

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

train_data=sc.fit_transform(train_data)
# for Test set

s=pd.get_dummies(test,drop_first=True)

sc2=StandardScaler()

test=sc.fit_transform(s)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
from sklearn.neighbors import KNeighborsClassifier

clf=KNeighborsClassifier()

score=cross_val_score(clf,train_data,target,cv=k_fold)

print(score)
round(np.mean(score)*100,2)
from sklearn.svm import SVC

clf=SVC(kernel="rbf")

score=cross_val_score(clf,train_data,target,cv=k_fold)

print(score)

round(np.mean(score)*100,2)
from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()

score=cross_val_score(clf,train_data,target,cv=k_fold)

print(score)
round(np.mean(score)*100,2)
from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(criterion="entropy")

score=cross_val_score(clf,train_data,target,cv=k_fold)

print(score)
round(np.mean(score)*100,2)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=300,criterion="entropy")

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100,2)
from sklearn.ensemble import GradientBoostingClassifier

clf=GradientBoostingClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

round(np.mean(score)*100,2)
model=GradientBoostingClassifier()

model.fit(train_data,target)
y_hat=model.predict(test)

main_df=pd.DataFrame(main_df)

main_df["Survived"]=y_hat

main_df.to_csv("Final Submission.csv",index=False)