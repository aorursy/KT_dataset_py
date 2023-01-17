# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')
testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')
traindf.info()
traindf.head()
traindf.isnull().sum().sort_values(ascending=False)
sns.barplot(x='Pclass', y='Survived', data=traindf)
sns.barplot(x='SibSp', y='Survived', data=traindf)
sns.barplot(x='Parch', y='Survived', data=traindf)
sns.barplot(x='Sex', y='Survived', data=traindf)
sns.barplot(x='Embarked', y='Survived', data=traindf)
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = traindf[traindf['Sex']=='female']
men = traindf[traindf['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')
FacetGrid = sns.FacetGrid(traindf, row='Embarked', height=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()
data = [traindf, testdf]

for dataset in data:
    mean = traindf["Age"].mean()
    std = traindf["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    print(age_slice)
    dataset["Age"] = dataset["Age"].astype(int)
traindf["Age"].isnull().sum()
common_value = 'S'
data = [traindf, testdf]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
le = preprocessing.LabelEncoder()
le.fit(traindf['Embarked'])
traindf['Embarked']=le.transform((traindf['Embarked']))
testdf['Embarked']=le.transform((testdf['Embarked']))
genders = {"male": 0, "female": 1}
data = [traindf, testdf]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
data = [traindf, testdf]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
sns.barplot(x='Title', y='Survived', data=traindf)
data = [traindf, testdf]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 10, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 15), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 20), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 28), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 35), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

# let's see how it's distributed 
traindf['Age'].value_counts()
traindf = traindf.drop(columns=['Name','Ticket','Cabin','Fare'])
testdf= testdf.drop(columns=['Name','Ticket','Cabin','Fare'])
traindf.isnull().sum().sort_values(ascending=False)
X_train = traindf.drop("Survived", axis=1)
Y_train = traindf["Survived"]
X_test  = testdf
svcmodel = SVC(C=0.1, gamma=1, kernel='poly')
svcmodel.fit(X_train,Y_train)
Randommodel = RandomForestClassifier()
Randommodel.fit(X_train,Y_train)
Y_predsvc = svcmodel.predict(X_test)
Y_predR = Randommodel.predict(X_test)
#Use cross validation for evaluating the model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
score = cross_val_score(svcmodel,X_train, Y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
print("SVC score:{}".format(score.mean()))
score1 = cross_val_score(Randommodel,X_train, Y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
print("Random forst score {}:".format(score1.mean()))



submission1 = pd.DataFrame(columns = ['PassengerId','Survived'])
submission1['PassengerId'] = testdf.index
submission1['Survived'] = Y_predsvc
submission1.to_csv('SVM.csv',index = False)
submission2 = pd.DataFrame(columns = ['PassengerId','Survived'])
submission2['PassengerId'] = testdf.index
submission2['Survived'] = Y_predR
submission2.to_csv('RF.csv',index = False)