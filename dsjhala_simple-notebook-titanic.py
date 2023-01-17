#importing libraries 

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

#load the training data

train_data = pd.read_csv("../input/train.csv")

#lets see few rows of our data

train_data.head(6)

#lets what features we have

train_data.info()
# PassengerID is just a sequence number so we can delete it which do not have any impact on Survival

# Lets delete passengerid

del train_data["PassengerId"]
# Pclass - Its is a numerical catorgircal feature with order, lets plot graph & see its relevance 

sns.factorplot(x="Pclass",y='Survived',data=train_data)
# Name - well name shouldn't affect the survival of the passenger but it can be an important feature 

# what extra information i can get from passenger's name ???

train_data['Name'].head()

#hmm!! we can see below that we can fetch the family Names of passengers it could be a usefull feature to answer other

# questions like "Ethnicity" of the passengers survived. since it is not usufull in our main prediction task 

# i am leaving it for now . 

del train_data['Name']
# Sex- it would be interseting to see this feature's relation with 'Survived' feature

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))

sns.barplot(x='Sex',y='Survived',data=train_data,ax=ax2)

sns.countplot(train_data["Sex"],ax=ax1)
# age- Lets what how age impacts the chances of survival

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,5))

sns.factorplot(x="Survived",y="Age",data=train_data,ax=ax1)

sns.boxplot(x="Survived",y="Age",data=train_data,ax=ax2)

sns.regplot(x='Age',y='Survived',data=train_data,ax=ax3)

plt.close(2)
# we can add these features to create new feature called "Fam_Size"

train_data['Fam_Size']= train_data['SibSp'] + train_data['Parch']

fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,8))

sns.factorplot(x="SibSp",y="Survived",data=train_data,ax=ax1)

sns.factorplot(x="Parch",y="Survived",data=train_data,ax=ax2)

sns.factorplot(x="Fam_Size",y="Survived",data=train_data,ax=ax3)



plt.close(2)

plt.close(3)

 
# since we are goint to work with Fam_Size lets delete Sibsp & Parch

del train_data["SibSp"]

del train_data['Parch']
train_data["Ticket"].head(10)
del train_data["Ticket"]
# IT would be interesting to see Fare & Survival realtion-ship

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,5))

sns.boxplot(x='Survived',y='Fare',data=train_data,ax=ax1)

sns.factorplot(x='Survived',y='Fare',data=train_data,ax=ax2)

sns.regplot(x='Fare',y='Survived',data=train_data,ax=ax3)

plt.close(2)

# This feature has very few (only 2) missing values lets have a visualize this feature

#sns.countplot(train_data['Embarked'])

sns.factorplot(x='Embarked',y='Survived',data=train_data)
# Lets look at our dataframe now

train_data.head()
#only 2 missing values lets print rows with missing values

train_data[train_data["Embarked"].isnull()]
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))

sns.factorplot(x='Embarked',y='Survived',data=train_data,ax=ax1)

sns.boxplot(x='Embarked',y='Fare',data=train_data,ax=ax2)

plt.close(2)
train_data['Embarked']=train_data['Embarked'].fillna('C')
#687 missing values , lets fill them with the help of other features

train_data['Cabin'].head(5)
# As we can see above that cabin is alphnumeric , here we are not interted in the end digit of cabin lets just filter out

#first letter of cabin that way we will have a categorical variable(easy to analyze).

train_data['Cabin_Id'] = train_data['Cabin'].str[0]

#now we have our required data in cabin_id column we don't need 'Cabin' anymore

del train_data['Cabin']

train_data.head()
new_train=train_data[train_data['Cabin_Id'].notnull()]

new_train.head()
# depedent variables

X = new_train.iloc[:,[0,1,4,6]].values

y = new_train.iloc[:,7].values

from sklearn.model_selection import cross_val_score, ShuffleSplit

from sklearn.svm import SVC

clf = SVC(kernel='linear', C=1)

clf.fit(X,y)

cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

scores = cross_val_score(clf, X, y, cv=4)

scores.mean()                                              

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X,y)

cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

scores = cross_val_score(clf, X, y, cv=4)

scores.mean()                                              

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=10,random_state=0)

clf.fit(X,y)

cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

scores = cross_val_score(clf, X, y, cv=4)

scores.mean() 
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=1)

clf.fit(X,y)

cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

scores = cross_val_score(clf, X, y, cv=4)

scores.mean() 
# here new_train will store all the data with Nan In Cabin

df=train_data[train_data['Cabin_Id'].isnull()]

df.head()

k = df.iloc[:,[0,1,4,6]].values

pred = clf.predict(k)
#now add these predicted values of Cabin_Id to datafraem

df['Cabin_Id'] = pred

train_data = new_train.append(df)

train_data.head()
# new dataframe- Contains the rows with known age

new_train = train_data[train_data["Age"].notnull()]

new_train.head()

# features

X = new_train.iloc[:,[0,1,4,6]].values

# Age 

y = new_train.iloc[:,3].values
from sklearn import linear_model

from sklearn.cross_validation import train_test_split

from sklearn.metrics import mean_squared_error

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.20,random_state=0)

reg = linear_model.Lasso (alpha = 0.1)

reg.fit(X_train,y_train)

y_pred=reg.predict(X_val)

# Calculating Mean Square Error

mean_squared_error(y_val, y_pred) 
from sklearn import linear_model

from sklearn.cross_validation import train_test_split

from sklearn.metrics import mean_squared_error

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.20,random_state=0)

reg = linear_model.Ridge (alpha = 0.1)

reg.fit(X_train,y_train)

y_pred=reg.predict(X_val)

# Calculating Mean Square Error

mean_squared_error(y_val, y_pred) 
#Compaer it with mean

y_mean =np.empty(143)

y_mean.fill(y_pred.sum()/len(y_pred))

mean_squared_error(y_val, y_mean) 
df=train_data[train_data["Age"].isnull()]

# features

X = df.iloc[:,[0,1,4,6]].values

# Age 

y = reg.predict(X)

df['Age']=y

train_data = new_train.append(df)
train_data.info()
train_data.head()

#We have Sex, Embarked, Cabin_id as categorical features
Embark_dummy = pd.get_dummies(train_data["Embarked"])

Embark_dummy.head(5)
# we do not need to do for featre "SEX" because it is anyways in 2 categoreis but just for understansding lets do it.

Sex_dummy = pd.get_dummies(train_data["Sex"])

Sex_dummy.head(5)
Cabin_dummy = pd.get_dummies(train_data["Cabin_Id"])

Cabin_dummy.head(5)
del Embark_dummy['S']

del Cabin_dummy['T']

del Sex_dummy['female']
train_data['Sex'] = Sex_dummy['male']

train_data['Embark_C'] = Embark_dummy['C']

train_data['Embark_Q'] = Embark_dummy['Q']

train_data['Cabin_A'] = Cabin_dummy['A']

train_data['Cabin_B'] = Cabin_dummy['B']

train_data['Cabin_C'] = Cabin_dummy['C']

train_data['Cabin_D'] = Cabin_dummy['D']

train_data['Cabin_E'] = Cabin_dummy['E']

train_data['Cabin_F'] = Cabin_dummy['F']

train_data['Cabin_G'] = Cabin_dummy['G']

del train_data['Sex']

del train_data['Embarked']

del train_data['Cabin_Id']
train_data.head(10)
corr = train_data.corr()

f, ax = plt.subplots(figsize=(25,16))

sns.plt.yticks(fontsize=18)

sns.plt.xticks(fontsize=18)



sns.heatmap(corr, cmap='inferno', linewidths=0.1,vmax=1.0, square=True, annot=True)
# features

X = train_data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]].values

# dependent variable

y = train_data.iloc[:,0].values
from sklearn.model_selection import cross_val_score, ShuffleSplit

from sklearn.svm import SVC



clf = SVC(kernel='linear', C=1)

clf.fit(X,y)

cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

scores = cross_val_score(clf, X, y, cv=4)

scores.mean()                                              



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=15,random_state=0)

clf.fit(X,y)

cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

scores = cross_val_score(clf, X, y, cv=4)

scores.mean() 
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=1)

clf.fit(X,y)

cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

scores = cross_val_score(clf, X, y, cv=4)

scores.mean() 