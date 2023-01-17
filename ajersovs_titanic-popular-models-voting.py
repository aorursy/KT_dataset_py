import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

combine = [train, test]
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

# sns.countplot(x='Survived',data=train,palette='RdBu_r')

train.columns
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
for df in combine:

    df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
for df in combine:

    df['Title'] = df['Name'].apply(lambda x:x.split(',')[1])

    df['Title'] = df['Title'].apply(lambda x:x.split()[0])

    df['Title'] = df['Title'].map(lambda x: x.replace('.',''))

train.Title.value_counts()[:6]
for df in combine:

    df['Title'] = df['Title'].replace(['Don', 'Rev', 'Dr', 'Mme',\

       'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the',\

       'Jonkheer'], 'Rare')



    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for df in combine:

    df['Title'] = df['Title'].map(titles)

    df['Title'] = df['Title'].fillna(0)
for df in combine:

    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
embark_train = pd.get_dummies(train['Embarked'],drop_first=True)

embark_test = pd.get_dummies(test['Embarked'],drop_first=True)
for df in combine:

    df.drop(['Embarked','Ticket','Name'],axis=1, inplace=True)

train = pd.concat([train,embark_train],axis=1)

test = pd.concat([test,embark_test],axis=1)
test.Fare = test.Fare.replace(np.nan, test['Fare'].median())
test.head()
train['Age_int'] =pd.cut(train['Age'], 5)

train['Age_int'].value_counts()
frames = [train,test]

for df in frames:    

    df.loc[ df['Age'] <= 16, 'Age'] = 0

    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

    df.loc[ df['Age'] > 64, 'Age'] = 4
train['Fare_int'] =pd.cut(train['Fare'], 4)

train['Fare_int'].value_counts()
for df in frames:    

    df.loc[ df['Fare'] <= 128, 'Fare'] = 0

    df.loc[(df['Fare'] > 128) & (df['Fare'] <= 256), 'Fare'] = 1

    df.loc[(df['Fare'] > 256) & (df['Fare'] <= 384), 'Fare'] = 2

    df.loc[ df['Fare'] > 384, 'Fare'] = 3

cols = ['PassengerId','Age_int','Fare_int']

train = train.drop(cols, axis=1)

test2 = test.copy()

test = test.drop('PassengerId', axis=1)
test.head()
train['FamSize'] = train['SibSp'] + train['Parch'] + 1

test['FamSize'] = test['SibSp'] + test['Parch'] + 1
train['IsAlone'] = train['FamSize']

test['IsAlone'] = test['FamSize']
train['IsAlone'].unique()
train['IsAlone'] = train['IsAlone'].replace([ 2,  5,  3,  7,  6,  4,  8, 11], 0)

test['IsAlone'] = test['IsAlone'].replace([ 2,  5,  3,  7,  6,  4,  8, 11], 0)
plt.figure(figsize=(12,6))

sns.heatmap(train.corr())
train = train.drop(['SibSp','Parch'], axis=1)

test = test.drop(['SibSp','Parch'], axis=1)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test) 

logreg = accuracy_score(y_test, predictions)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500, max_depth=10)
rfc.fit(X_train,y_train)
predictions2 = rfc.predict(X_test)
rfcmodel = accuracy_score(y_test, predictions2)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, y_train)
predictions3 = knn.predict(X_test)
knnmodel = accuracy_score(y_test, predictions3)
from sklearn.svm import SVC

model = SVC(probability=True)
model.fit(X_train,y_train)
predictions4 = model.predict(X_test)
svmmodel = accuracy_score(y_test, predictions4)
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()

#n_estimators =100, max_depth=3, min_samples_leaf =2,learning_rate=0.1
gbc.fit(X_train,y_train)
predictions5 = gbc.predict(X_test)
gbcmodel = accuracy_score(y_test, predictions5)
from sklearn.model_selection import GridSearchCV
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 9]}

# grid = GridSearchCV(SVC(), parameters, cv=5)

# grid.fit(X_train, y_train)

# print(grid.best_params_)
#No way to improve
# parameters = {'n_estimators':[50,100,200,300], 'max_depth':[2,3,4,5], 'min_samples_leaf':[2,3,4,5]}

# grid = GridSearchCV(gbc, parameters, cv=5)

# grid.fit(X_train, y_train)

# print(grid.best_params_)
gbc = GradientBoostingClassifier(max_depth=2,min_samples_leaf=3,n_estimators=200)

gbc.fit(X_train,y_train)

predictions5 = gbc.predict(X_test)

gbcmodel = accuracy_score(y_test, predictions5)
from sklearn.ensemble import VotingClassifier
eclf2 = VotingClassifier(estimators=[

('lr', logmodel), ('rf', rfc), ('knn', knn),('svc',model),('gbc',gbc)],

voting='soft',)

eclf2.fit(X_train, y_train)

predict_vote = eclf2.predict(X_test)

vote = accuracy_score(y_test, predict_vote)
labels = ['Logistic REgression','Random Forest','KNN','SVM', 'GBC','Vote']

models = [logreg, rfcmodel, knnmodel, svmmodel, gbcmodel, vote]
ev = pd.DataFrame(columns=['Model','Score%'])

for i in range(0,6):

    ev = ev.append({'Model': labels[i],'Score%':models[i]}, ignore_index=True)
ev.sort_values(by='Score%', ascending=False)
predictions = gbc.predict(test)
submission = pd.DataFrame({

        "PassengerId": test2["PassengerId"],

        "Survived": predictions})

submission.to_csv('gender_submission.csv', index=False)
print('Best result with tuned GBC on test df is 0.79425')