# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

from matplotlib import style



# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score
!ls
# getting the data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.info()
train_df.describe()
# missing values in Age`
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
missing = train_df.isnull().sum().sort_values(ascending=False)
total = train_df.isnull().sum() / train_df.isnull().count() * 100

final = round(total,1).sort_values(ascending=False)

missing_data = pd.concat([missing, final], axis=1, keys=['Total', '%'],sort=True).head(5)

missing_data
train_df.columns.values
sns.set_style('whitegrid')
sns.countplot('Survived',data=train_df,hue='Sex',palette='RdBu_r')
sns.countplot('Sex',data=train_df)
train_df.groupby(by='Sex',as_index=False)['Survived'].mean()
sns.barplot(x='Pclass',y='Survived',data=train_df)
sns.countplot(x='Survived',hue='Pclass',data=train_df)
train_df.groupby('Pclass',as_index=False)['Survived'].mean().sort_values(by='Survived',ascending=False)
#Comparing the Embarked feature against Survived

sns.barplot(x='Embarked',y='Survived',data=train_df)

train_df[['Embarked','Survived']].groupby('Embarked',as_index=False).mean().sort_values(by='Survived'

                                                                                        ,ascending=False)
FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

FacetGrid.add_legend()
sns.barplot(x='Parch',y='Survived',data=train_df)
sns.barplot(x='SibSp',y='Survived',data=train_df)
train_df.Age.hist(bins=20)

plt.xlabel('Age')

plt.ylabel('Count')
sns.lmplot(x='Age',y='Survived',data=train_df,palette='Set1')
sns.lmplot(x='Age',y='Survived',data=train_df,palette='Set1',hue='Sex')
sns.distplot(train_df['Age'].dropna())
#Checking for outliers in Age data

sns.boxplot(x='Sex',y='Age',data=train_df)



#getting the median age according to Sex

train_df.groupby('Sex',as_index=False)['Age'].median()
sns.countplot(x='Sex',data=train_df)
sns.countplot(x='Survived',hue='Sex',data=train_df)
train_df.describe(include='O')

women = train_df[train_df['Sex']=='female']

men = train_df[train_df['Sex']=='male']

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(10,4))

ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(),label='Survived',bins=20,ax=axes[0],kde=False)

ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(),label='Not Survived',bins=40,ax=axes[0],kde=False)

ax.set_title('Female')

ax.legend()

ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(),label='Survived',bins=20,ax=axes[1],kde=False)

ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(),label='Not Survived',bins=40,ax=axes[1],kde=False)

ax.set_title('Male')

ax.legend()
g = sns.FacetGrid(data=train_df,col='Survived',row='Pclass')

g.map(plt.hist,'Age',alpha=.5, bins=20)

g.add_legend()
train_df.Embarked.value_counts()
sns.barplot(x='Embarked',y='Survived',data=train_df)
sns.countplot('SibSp',data=train_df)
train_df['Fare'].hist(bins=40,figsize=(10,4))
import cufflinks as cf
cf.go_offline()

train_df['Fare'].iplot(kind='hist',bins=40)
plt.figure(figsize=(10,4))

sns.boxplot(x='Pclass',y='Age',data=train_df)
train_df.Embarked.describe()
train_df = train_df.drop(['PassengerId'],axis=1)
train_df.columns
test_passenger_id = pd.DataFrame(test_df.PassengerId)

test_passenger_id.head()
test_df=test_df.drop(['PassengerId'],axis=1)
train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)
train_df.Age.median()
train_df.Age.fillna(train_df.Age.median(),inplace=True)
train_df.Age.isnull().sum()
test_df.Age.fillna(test_df.Age.median(),inplace=True)
data = [train_df, test_df] # turning into list and adding 'relatives' column to dataframe

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset["IsAlone"] = np.where(dataset["relatives"] > 0, 0,1)

train_df['IsAlone'].value_counts()    
#dropping the Name,SibSP and Parch columns

for dataset in data:

    dataset.drop(['SibSp','Parch'],axis=1,inplace=True)
train_df.info()
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
top_value = 'S'

data = [train_df,test_df]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(top_value)
train_df.info()
### Fare

data = [train_df, test_df]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train_df,test_df]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir',

                                                 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)
sns.barplot(x='Title',y='Survived',data=train_df,)
gender = {'male':0,'female':1}

data = [train_df, test_df]

for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(gender)
train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)
train_df['Embarked'].value_counts()
ports = {'S':0,'C':1,'Q':77}

data = [train_df, test_df]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
## Converting Age from float to integer

train_df['Age'] = train_df['Age'].astype(int)
train_df['AgeGroup'] = pd.qcut(train_df.Age,6,labels=False)

test_df['AgeGroup'] = pd.qcut(test_df.Age,6,labels=False)
train_df['Fare'] = pd.qcut(train_df.Fare,5,labels=False)

test_df['Fare'] = pd.qcut(test_df.Fare,5,labels=False)
train_df.head(5)
#Splitting out training data into X: features and y: target

X = train_df.drop("Survived", axis=1)

y = train_df["Survived"]
#splitting our training data again in train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=123)
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

acc_logreg = round(accuracy_score(y_pred,y_test)*100,2)
cv_scores = cross_val_score(logreg,X,y,cv=5)

np.mean(cv_scores)*100
rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train,y_train)
accu_rf = rf.score(X_train,y_train)

accu_rf = round(accu_rf*100,2)

accu_rf
# Displaying the important features

imp_features = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rf.feature_importances_,3)})

imp_features = imp_features.sort_values('importance',ascending=False).set_index('feature')

imp_features.head(10)
imp_features.plot.bar()
dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

accu_dt = dt.score(X_train,y_train)

accu_dt = round(accu_dt*100,2)

accu_dt
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

accu_knn = knn.score(X_train,y_train)

accu_knn = round(accu_knn*100,2)

accu_knn
nb = GaussianNB()

nb.fit(X_train,y_train)

accu_nb = nb.score(X_train,y_train)

accu_nb = round(accu_nb*100,2)

accu_nb
results = pd.DataFrame({

    'Model':['Linear Reg.','Random Forest','Decision Trees','K-Nearest Neighbours','Naive Bayes'],

    'Accuracy':[acc_logreg,accu_rf,accu_dt,accu_knn,accu_nb]

})

results.sort_values(by='Accuracy',ascending=False)
y_final = rf.predict(test_df)

submission = pd.DataFrame({

    'PassengerId': test_passenger_id['PassengerId'],

    'Survived': y_final

})

submission.head()

submission.to_csv('titanic_rf.csv', index=False)