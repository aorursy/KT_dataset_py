import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import set_matplotlib_formats

import warnings

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from sklearn.metrics import roc_curve, classification_report

from sklearn.model_selection import RepeatedKFold, train_test_split



# %matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train['FamilySize'] = train ['SibSp'] + train['Parch'] + 1



train['IsAlone'] = 1 #initialize to yes/1 is alone

train['IsAlone'].loc[train['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
test['FamilySize'] = test ['SibSp'] + test['Parch'] + 1



test['IsAlone'] = 1 #initialize to yes/1 is alone

test['IsAlone'].loc[test['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
train.head()
test.head()
train.info()
train.describe()
sns.pairplot(train)
sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=train)
#graph distribution of quantitative data

plt.figure(figsize=[16,12])



plt.subplot(3,3,1)

plt.boxplot(x=train['Fare'], showmeans = True, meanline = True)

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(3,3,2)

plt.boxplot(train['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(3,3,3)

plt.boxplot(train['FamilySize'], showmeans = True, meanline = True)

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')



plt.subplot(3,3,4)

plt.hist(x = [train[train['Survived']==1]['Fare'], train[train['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(3,3,5)

plt.hist(x = [train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(3,3,6)

plt.hist(x = [train[train['Survived']==1]['FamilySize'], train[train['Survived']==0]['FamilySize']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(3,3,7)

plt.hist(x = [train[train['Survived']==1]['Sex'], train[train['Survived']==0]['Sex']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Sex Histogram by Survival')

plt.xlabel('Sex')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(3,3,8)

plt.hist(x = [train[train['Survived']==1]['Pclass'], train[train['Survived']==0]['Pclass']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Pclass Histogram by Survival')

plt.xlabel('Pclass')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(3,3,9)

plt.hist(x = [train[train['Survived']==1]['IsAlone'], train[train['Survived']==0]['IsAlone']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('IsAlone Histogram by Survival')

plt.xlabel('IsAlone')

plt.ylabel('# of Passengers')

plt.legend()



plt.tight_layout()
#Checking the number of missing values in train data

train.isna().sum()
fig, ax = plt.subplots(figsize=[10,5])

sns.heatmap(train.isna(), ax=ax, cbar=False, yticklabels=False)

ax.set_title("NaN in each label for train set");

fig2, ax2 = plt.subplots(figsize=[10,5])

sns.heatmap(test.isna(), ax=ax2, cbar=False, yticklabels=False)

ax2.set_title("NaN in each label for test set");
#checking the row where we have missing embarked data

train[train['Embarked'].isnull()]
train["Embarked"].value_counts()
#Calculating and visualizing the mean fare across the 3 class of embarked

train.pivot_table(index='Embarked', values='Fare', aggfunc=np.mean)
#Calculating and visualizing the mean Pclass across the 3 class of embarked

train.pivot_table(index='Embarked', values='Pclass', aggfunc=np.mean)
train['Embarked'].fillna('C', inplace=True)
# Chceking if we have any row with fare value as 0. 

train[train["Fare"] == 0]
# Defined a function to replace the 0 fare value with the average fare value with their respective pclas and embarked column.

def fill_fare_train(cols):

    fare = cols[0]

    pclass = cols[1]

    embarked = cols[2]

    if fare==0 :

        return train[train['Pclass']==pclass][train['Embarked']==embarked]['Fare'].mean()

    else:

        return fare
train['Fare'] = train[['Fare', 'Pclass','Embarked']].apply(fill_fare_train, axis=1)
train.pivot_table(index='Survived', values='Age', aggfunc=np.mean)
new_data = train[train['Age'].notna()]



plt.figure(figsize=(10,20))

sns.catplot(x="Embarked", y="Age", hue = "Pclass",kind="swarm", data=new_data)

plt.show()
# Defined a function to replace the nan age value with the average age value with their respective pclas and embarked column.

def fill_age_train(cols):

    age = cols[0]

    pclass = cols[1]

    embarked = cols[2]

    if pd.isna(age):

        return train[train['Pclass']==pclass][train['Embarked']==embarked]['Age'].mean()

    else:

        return age

    

train['Age'] = train[['Age', 'Pclass','Embarked']].apply(fill_age_train, axis=1)
def fill_age_test(cols):

    age = cols[0]

    pclass = cols[1]

    embarked = cols[2]

    if pd.isna(age):

        return test[test['Pclass']==pclass][test['Embarked']==embarked]['Age'].mean()

    else:

        return age

    

test['Age'] = test[['Age', 'Pclass','Embarked']].apply(fill_age_test, axis=1)
test[test["Fare"].isna()]
#Replacing the nan fare value with the average fare value of pclass 3 and embarked class S.

mean = test[test['Pclass']==3][test['Embarked']=='S']['Fare'].mean()

test['Fare'].fillna(mean, inplace=True)
X=train.drop(["Survived","Name","Ticket","Cabin","SibSp", "Parch"],axis=1)

y=train["Survived"]



test=test.drop(["Name","Ticket","Cabin","SibSp", "Parch"],axis=1)
X = pd.get_dummies(X,drop_first=True)

test = pd.get_dummies(test,drop_first=True)
Xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



model=RandomForestClassifier(criterion= "entropy", max_depth= 6, n_estimators= 100, oob_score= True, random_state= 0)

model.fit(Xtrain,ytrain)
model.score(Xtrain,ytrain)
model.score(xtest,ytest)
from xgboost import XGBClassifier

model = XGBClassifier(learning_rate= 0.01, max_depth= 4, n_estimators= 300, seed= 0)

model.fit(Xtrain,ytrain)
print("Train accuracy",model.score(Xtrain,ytrain))

print("Train accuracy",model.score(xtest,ytest))
from sklearn import metrics

pred = model.predict(test)
#Displaying 1st 10 predicted values

pred[:10]
Submission = pd.DataFrame({ 'PassengerId': test["PassengerId"],

                            'Survived': pred })

Submission.to_csv("mySubmission.csv", index=False)