import numpy as np 

import pandas as pd 



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import make_blobs

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import xgboost as xgb



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.head()
train_df.describe(include="all")
test_df.head()
test_df.describe(include="all")
train_df.dtypes
#first look at age and see how to best deal with missing values



train_df['Age'].hist()
#the data is right skewed. Let's take a  look at survival rates by age to see if age is a big factor.

#If it is not, then we can just impute median. If it is, however, then we will have to find another way.



#let's first group age into 10 bins



train_df['AgeBins'] = pd.qcut(train_df['Age'], 7)

sns.barplot(x="AgeBins", y="Survived", data=train_df)



#we find that babies, and surprisingly 20-36 year olds, have a high survival rate
#We can also take a look at survival rates based on name title

train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

sns.barplot(x="Title", y="Survived", data=train_df)
np.unique(train_df['Title'])



#Since survival rate varies quite a bit by Title, let's impute avg age by title
sns.barplot(x="Title", y="Age", data=train_df)
train_df['Title'].value_counts()



#However, looking at this list of ages below, pretty much everyone is either a Mr, Miss, Mrs, or Master.

#So now we know that title is quite predictive, but it doesn't really help us determine age.

#So let's impute median age for the missing values. As we saw from the summary up top, 28 is median age.
train_df.loc[(train_df.Age.isnull()),'Age'] = train_df["Age"].median()

test_df.loc[(test_df.Age.isnull()),'Age'] = test_df["Age"].median()
#Let's fill out the rest of the missing values in the train 



train_df['Cabin'].value_counts()



#Since cabin is pretty much unique to the person, we can just drop it from the dataset



train_df = train_df.drop(['Cabin'], axis = 1)

test_df = test_df.drop(['Cabin'], axis = 1)
train_df['Embarked'].value_counts()



#Since S is most common and there's only two missing, let's bring S for the two missing values

#Only one missing fare (in test dataset), so replace that with mean test fare

train_df['Embarked'].fillna("S", inplace=True)

test_df['Fare'].fillna(35.6, inplace=True)



#Let's also drop the AgeBins variable since it doesn't tell us anything more than what age already does

train_df = train_df.drop(['AgeBins'], axis = 1)
train_df.describe(include="all")
test_df.describe(include="all")
#Let's see how each variable correlates to survival rates.

#But first, let's see which variables are unique to passenger

#We can drop Name since it's unique to each passenger and we have already pulled out Title

#Let's see if ticket is unique



test_df.describe(include="all")
train_df[train_df.Ticket == "347082"]



#looks like each famiy has the same ticket. So let's keep it in the model for now, though it seems like

#SibSp and Parch give us the same information.
#So the only variables we'll drop for now is Name



train_df = train_df.drop(['Name'], axis = 1)

test_df = test_df.drop(['Name'], axis = 1)
test_df.describe(include="all")
#features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']

#categorical = ['Sex','Embarked','Title']



features = ['Pclass','Sex','Age','SibSp','Parch','Fare']

categorical = ['Sex']



y = ['Survived']
le = LabelEncoder()

train_df['Sex'] = le.fit_transform(train_df['Sex'])



le = LabelEncoder()

test_df['Sex'] = le.fit_transform(test_df['Sex'])
train_x = train_df[features].as_matrix()

test_x = test_df[features].as_matrix()

train_y = train_df['Survived']
train_x
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=250, learning_rate=0.05).fit(train_x, train_y)

predictions = gbm.predict(test_x)
submit_preds = pd.DataFrame({ 'PassengerId': test_df['PassengerId'], 'Survived': predictions })

submit_preds.to_csv("submit_preds.csv", index=False)