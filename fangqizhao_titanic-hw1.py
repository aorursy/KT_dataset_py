import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head(10)
train['Pclass'].value_counts()
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print (train[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean())
import seaborn as sns
sns.set(style="whitegrid")

# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
train['Age'].hist(bins=60)
import matplotlib.pyplot as plt
f = sns.FacetGrid(train, col='Survived')
f.map(plt.hist, 'Age', bins=20)
train['SibSp'].value_counts()
print (train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())
sns.catplot( x = 'SibSp', y = 'Survived',order=[0,1,2,3,4,5,6], height=4, kind = "point", data = train)
train['Parch'].value_counts()
print (train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())
sns.catplot( x = 'Parch', y = 'Survived',order=[0,1,2,3,4,5,6], height=4, kind = "point", data = train)
train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
train['Embarked'].value_counts()
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
sns.set(style="whitegrid")
h = sns.catplot(x="Embarked", y="Survived", data=train,
                height=4, kind="bar", palette="muted")
h.despine(left=True)
h.set_ylabels("survival probability")
train = train.drop(['Name', 'Ticket', 'Cabin'],axis=1)
test = test.drop(['Name', 'Ticket','Cabin'],axis=1)
train.head(10)
for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')
for df in [train,test]:
    df['Embarked_spots']=df['Embarked'].map({'S':0,'C':1, 'Q':2})
import numpy as np

average_age_train = train["Age"].mean()
std_age_train = train["Age"].std()
count_nan_age_train = train["Age"].isnull().sum()

average_age_test = test["Age"].mean()
std_age_test = test["Age"].std()
count_nan_age_test = test["Age"].isnull().sum()

rand_1 = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

train["Age"][np.isnan(train["Age"])] = rand_1
test["Age"][np.isnan(test["Age"])] = rand_2

train['Age'] = train['Age'].astype(int)
test['Age'] = test['Age'].astype(int)
test["Fare"].fillna(test["Fare"].median(), inplace=True)
train.head(10)
test.head(10)
features = ['Pclass','Age','Sex_binary','SibSp','Parch','Fare', 'Embarked_spots']
target = 'Survived'
train[features].head(5)
train[target].head(3).values
from sklearn.tree import DecisionTreeClassifier

clf= DecisionTreeClassifier(max_depth=3,min_samples_leaf=2) 
clf.fit(train[features], train[target])

predictions = clf.predict(test[features])
predictions
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})
submission.head()
filename = 'Titanic Predictions 8.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)