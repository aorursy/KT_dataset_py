import pandas as pd 
train_df=pd.read_csv('../input/titanic/train.csv')

test_df=pd.read_csv('../input/titanic/test.csv')
train_df.head()
train_df.isnull().sum()
test_df.isnull().sum()
#relationship between Pclass and Survived 

train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#find Pclass is correlate with Survived

#Pclass=1 had higher survival rate

#keep this varible
#relationship between Sex and Survived

train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#find Sex and Survived variables are correlated

#male had less survival rate
#relationship between SibSp and Survived 

train_df[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#they had no correlation

#will drop this variable

train_df=train_df.drop(['SibSp'], axis=1)

test_df=test_df.drop(['SibSp'], axis=1)
#relationship between SibSp and Survived 

train_df[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#they had no correlation

#will drop this variable
train_df=train_df.drop(['Parch'], axis=1)

test_df=test_df.drop(['Parch'], axis=1)
#relationship between Cabine and Survived 

train_df[['Cabin','Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#they had no correlation

#will drop this variable
train_df=train_df.drop(['Cabin'], axis=1)

test_df=test_df.drop(['Cabin'], axis=1)
#also Name, PassengerId and Ticket are not relate to target varible

#we are going to drop them all 

train_df=train_df.drop(['Name', 'PassengerId', 'Ticket'], axis=1)

test_df=test_df.drop(['Name','Ticket'], axis=1)
import seaborn as sns

import matplotlib.pyplot as plt
#1. plot Age and Survived

#2. plot Sex and Survied 

#3. plot Fare and Survied 

#4. plot Embarked and Survied
#1.

#Age<4 had high survival rate 

#large number of 15-30 year olds didn't survive. 

age_g=sns.FacetGrid(train_df, col='Survived')

age_g.map(plt.hist, 'Age', bins=20, rwidth=0.8)

age_g.add_legend()

#2.

#female had higher survival rate than male

sex_g=sns.FacetGrid(train_df, col='Survived', aspect=1.5)

sex_g.map(plt.hist, 'Sex')

sex_g.add_legend()
#3. 

#higher fare had bigger number of suvived passengers

fare_g=sns.FacetGrid(train_df, col='Survived', aspect=1.5)

fare_g.map(plt.hist ,'Fare', bins=3, rwidth=0.8)

fare_g.add_legend()
#find median age to fill in the null values

age_median=train_df.Age.dropna().median()

train_df['Age']=train_df['Age'].fillna(age_median)

test_df['Age']=test_df['Age'].fillna(age_median)
#find median fare to fill in the null values

fare_median=test_df.Fare.dropna().median()

test_df['Fare']=test_df['Fare'].fillna(fare_median)

fare_median

train_df.head()
#find mode Embarked to fill in the null values



train_df['Embarked']=train_df['Embarked'].fillna('S')

train_df.isnull().sum()
#covert Sex into dummy variable 

#male=0, female=1

genders={'male':0, 'female':1}

data=[train_df, test_df]

for dataset in data: 

    dataset['Sex']=dataset['Sex'].map(genders)

#covert Embarked into three numerical features

#Q=0, S=1, C=2 

ports={'Q':0,'S':1, 'C':2}

for dataset in data: 

    dataset['Embarked']=dataset['Embarked'].map(ports)
#group age into 5 groups 

agegroup=pd.cut(train_df['Age'], 5)

agegroup
for dataset in data: 

    dataset.loc[dataset['Age']<=16, 'Age']=0

    dataset.loc[(dataset['Age']>16) & (dataset['Age'] <=32), 'Age']=1

    dataset.loc[(dataset['Age']>32) & (dataset['Age'] <=48), 'Age']=2

    dataset.loc[(dataset['Age']>48) & (dataset['Age'] <=64), 'Age']=3

    dataset.loc[ dataset['Age'] > 64, 'Age']=4

#group fare into 4 groups 

faregroup=pd.qcut(train_df['Fare'], 4)

faregroup.unique()
for dataset in data: 

    dataset.loc[dataset['Fare']<=7.91, 'Fare']=0

    dataset.loc[(dataset['Fare']>7.91) & (dataset['Fare'] <=14.454), 'Fare']=1

    dataset.loc[(dataset['Fare']>14.454) & (dataset['Fare'] <=31), 'Fare']=2

    dataset.loc[(dataset['Fare']>31), 'Fare']=3

    

dataset['Fare'] = dataset['Fare'].astype(int)

corr=train_df.corr()

sns.heatmap(corr)

corr
from sklearn.model_selection import train_test_split

X=train_df.drop('Survived', axis=1)

y=train_df['Survived']

x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0 )

from sklearn.linear_model import LogisticRegression 
logreg=LogisticRegression()

logreg.fit(x_train, y_train)

Y_pred=logreg.predict(x_test)

acc_log=round(logreg.score(x_train, y_train)*100, 2)

acc_log 
from sklearn.svm import SVC, LinearSVC
svc = SVC()

p2=svc.fit(x_train, y_train)

Y_pred = svc.predict(x_test)

acc_svc = round(svc.score(x_train, y_train) * 100, 2)

acc_svc
linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

Y_pred = linear_svc.predict(x_test)

acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)

acc_linear_svc
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train, y_train)

Y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

acc_decision_tree
models = pd.DataFrame({

    'Model': ['Support Vector Machines',  'Logistic Regression', 

              'Linear SVC', 'Decision Tree'],

    'Score': [acc_svc,  acc_log, 

             acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
X_test=test_df.drop('PassengerId', axis=1).copy()



decision_tree.fit(x_train, y_train)

y_pred = decision_tree.predict(X_test)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": y_pred

    })
submission