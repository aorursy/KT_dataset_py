# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/titanic/train.csv")
train.head()
test = pd.read_csv("../input/titanic/test.csv")
test.head()
print(train.info())
print(test.info())
print(train.shape,test.shape)
women = train.loc[train.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train.loc[train.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
sns.scatterplot(x='Age',y='Fare',hue='Pclass',data=train)
sns.catplot(x='Embarked',y='Survived',hue='Sex',col='Pclass',kind='bar',data=train,palette='rainbow')
sns.catplot(x='Embarked',y='Survived',hue='Sex',col='Pclass',kind='violin',data=train,palette='rainbow')
sns.catplot(x='Embarked',y='Age',hue='Survived',col='Pclass',kind='violin',data=train,palette='Spectral')
sns.countplot(x='Embarked',data=train,hue='Survived')
sns.countplot(x='Embarked',data=test)
sns.catplot(x='Parch',y='Survived',hue='Sex',kind='bar',data=train)
sns.catplot(x='SibSp',y='Survived',hue='Sex',kind='bar',data=train)
survived_ages = train[train.Survived == 1]["Age"]
not_survived_ages = train[train.Survived == 0]["Age"]
plt.subplot(1, 2, 1)
sns.distplot(survived_ages, kde=False)
plt.axis([0, 100, 0, 100])
plt.title("Survived")
plt.ylabel("Proportion")
plt.subplot(1, 2, 2)
sns.distplot(not_survived_ages, kde=False)
plt.axis([0, 100, 0, 100])
plt.title("Didn't Survive")
plt.show()
sns.pairplot(train)
g = sns.heatmap(train.corr(),annot=True,cmap='coolwarm')
figure, fare = plt.subplots(figsize=(10, 4.5))
sns.kdeplot(data=train.loc[(train['Survived'] == 0),'Fare'], kernel='gau', ax=fare, color="Red", shade=True, legend=True)

sns.kdeplot(data=train.loc[(train['Survived'] == 1),'Fare'], kernel='gau', ax=fare, color="Blue", shade=True, legend=True)


fare.set_xlabel("Fare")
fare.set_ylabel("Probability Density")
fare.legend(["Not Survived", "Survived"], loc='upper right')
fare.set_title("Graph for Fare")
figure, fare = plt.subplots(figsize=(10, 4.5))
sns.kdeplot(data=train.loc[(train['Survived'] == 0),'Age'], kernel='gau', ax=fare, color="Red", shade=True, legend=True)

sns.kdeplot(data=train.loc[(train['Survived'] == 1),'Age'], kernel='gau', ax=fare, color="Blue", shade=True, legend=True)


fare.set_xlabel("Age")
fare.set_ylabel("Probability Density")
fare.legend(["Not Survived", "Survived"], loc='upper right')
fare.set_title("Graph for Age")
sns.countplot(x='Parch',data=train,hue='Survived')
sns.countplot(x='SibSp',data=train,hue='Survived')
print(train.Embarked.unique(),test.Embarked.unique())
data_combined = [train,test]
for data in data_combined:
    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
print(train['Title'].unique())
for data in data_combined:
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
train[['Title','Survived']].groupby(['Title'],as_index=False).mean()
train.Title.unique()
for data in data_combined:
    for sex in data.Sex.unique():
        for pclass in data.Pclass.unique():
            age_data = data[(data['Pclass'] == pclass) & (data['Sex']==sex)]['Age'].dropna()
            data.loc[(data.Age.isnull()) & (data.Sex == sex) & (data.Pclass == pclass),'Age'] = age_data.mean()
train['AgeBand'] = pd.cut(train['Age'],5)
print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))
train.head()

for data in data_combined:
    data.loc[data['Age'] <= 16,'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32),'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48),'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64),'Age'] = 3
    data.loc[(data['Age'] > 64),'Age'] = 4

train = train.drop('AgeBand',axis=1)
print(train.head())

data_combined = [train,test] #Reinitializing because of the above mess :(
#From the graph 'S' is the most frequent port, filling NA in Embarked with 'S'
for data in data_combined:
    data['Embarked'] = data['Embarked'].fillna('S')
train[['Embarked','Survived']].groupby('Embarked',as_index=False).mean().sort_values(by='Survived',ascending=False)

for data in data_combined:
    data['Embarked'] = data['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
    
test['Fare'].fillna(test['Fare'].dropna().median(),inplace=True)


train['FareBand'] = pd.cut(train['Fare'],4)
train[['FareBand','Survived']].groupby('FareBand',as_index=False).mean().sort_values(by='FareBand', ascending=True)
for data in data_combined:
    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
data_combined = [train, test]
for data in data_combined:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1,'IsAlone'] = 1
    data['Age*Class'] = data.Age * data.Pclass

    
test.head()
for data in data_combined:
    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
for data in data_combined:
    data['Title'] = data['Title'].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Rare':4})
    
train.head()
#Features you want to use in the model
features = ['Pclass','Sex','Fare','Age','Embarked','IsAlone','FamilySize']
#Import stuff for encoding and preprocessing pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

X_train = train.copy()
X_test = test.copy()
y_train = X_train.pop('Survived')
X_train = X_train[features]
X_test = X_test[features]
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

#Random forest algorithm
# random_forest = RandomForestClassifier(n_estimators=150)
# random_forest.fit(X_train, y_train)
# Y_pred = random_forest.predict(X_test)
# random_forest.score(X_train, y_train)
# acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
# print(acc_random_forest)
# Y_pred
 

#XGBoost Classifier
xgb = XGBClassifier(random_state = 42)

param_grid = {'n_estimators': [15, 25, 50, 100],
              'colsample_bytree': [0.65, 0.75, 0.80],
              'reg_alpha': [1],
              'reg_lambda': [1, 2, 5],
              'subsample': [0.50, 0.75, 1.00],
              'learning_rate': [0.01, 0.1, 0.5],
              'gamma': [0.5, 1, 2, 5],
              'min_child_weight': [0.01],
              'sampling_method': ['uniform']}


def clf_performance(classifier, model_name):
    print(model_name)
    print('-------------------------------')
    print('   Best Score: ' + str(classifier.best_score_))
    print('   Best Parameters: ' + str(classifier.best_params_))

clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 3, verbose = True, n_jobs = 4)
best_clf_xgb = clf_xgb.fit(X_train, y_train)
clf_performance(best_clf_xgb, 'XGB')

Y_pred = best_clf_xgb.predict(X_test)
Y_pred
predictions = Y_pred
print(type(test.PassengerId))

output = pd.concat([test['PassengerId'],pd.DataFrame( np.round(predictions),columns=['Survived'])],axis=1)

output.to_csv('ja_submission.csv', index=False)
print("Your submission was successfully saved!")