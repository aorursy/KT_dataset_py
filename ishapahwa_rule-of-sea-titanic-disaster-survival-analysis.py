# import necessary packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# machine learning packages

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
# Train and Test datasets

input_train = '../input/train.csv'

input_test = '../input/test.csv'
# Reading the datasets

df_test = pd.read_csv(input_test)

df_train = pd.read_csv(input_train)

combine = [df_train, df_test]

df_train.head()
df_test.head()
## Checking for missing data

df_train.isna().sum()
# some initial analysis 

len(df_train[df_train["Sex"]=='male'])

len(df_train[df_train['Age']>60])

set(df_train['Survived'])

len(df_train[df_train["Survived"]==1])

len(df_train[df_train["Pclass"]==3])

len(df_train[df_train["SibSp"]>0])

len(df_train[df_train["Parch"]>0])

set(df_train["Fare"])
df_train.describe()
df_train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False)
df_train[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)
df_train[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)
df_train[['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)
grid = sns.FacetGrid(df_train, col='Survived')

grid.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
#len(df_train[df_train['Pclass']==1])

len(df_train[df_train['Pclass']==2])

#len(df_train[df_train['Pclass']==3])

grid = sns.catplot(x='Embarked',y='Survived',kind='point',data=df_train)

grid.add_legend();
f,ax=plt.subplots(2,2,figsize=(15,15))

sns.countplot('Embarked',data=df_train,ax=ax[0,0])

ax[0,0].set_title('No. Of Passengers Boarded')

sns.countplot('Embarked',hue='Sex',data=df_train,ax=ax[0,1])

ax[0,1].set_title('Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Survived',data=df_train,ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked',hue='Pclass',data=df_train,ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()
sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=df_train)

plt.show()
grid = sns.FacetGrid(df_train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Fare', alpha=.5, ci=None)

grid.add_legend()
print("Before dropping", df_train.shape, df_test.shape, combine[0].shape, combine[1].shape)



df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)

df_test = df_test.drop(['Ticket', 'Cabin'], axis=1)

combine = [df_train, df_test]



"After dropping", df_train.shape, df_test.shape, combine[0].shape, combine[1].shape
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-z]+)\.', expand=False)



pd.crosstab(df_train['Title'], df_train['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
dataset['Title']
all_titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(all_titles)

    dataset['Title'] = dataset['Title'].fillna(0)



df_train.head()
df_train = df_train.drop(['Name', 'PassengerId'], axis=1)

df_test = df_test.drop(['Name'], axis=1)

combine = [df_train, df_test]

df_train.shape, df_test.shape
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
grid = sns.FacetGrid(df_train, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
missing_ages = np.zeros((2,3))

missing_ages
gender=['male','female']

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == gender[i]) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            #print(guess_df)

            age_guess = guess_df.median()

            #print(age_guess)

            # Convert random age float to nearest .5 age

            missing_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == gender[i]) & (dataset.Pclass == j+1),\

                    'Age'] = missing_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



df_train.head()
df_train['AgeBand'] = pd.cut(df_train['Age'], 5)

df_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

df_train.head()
df_train = df_train.drop(['AgeBand'], axis=1)

combine = [df_train, df_test]

df_train.head()
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



df_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
df_train = df_train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

df_test = df_test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [df_train, df_test]



df_train.head()
freq_port = df_train.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)

df_test.head()
df_train['FareBand'] = pd.qcut(df_train['Fare'], 4)

df_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



df_train = df_train.drop(['FareBand'], axis=1)

combine = [df_train, df_test]

    

df_train.head(10)
df_test.head(10)
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



df_train.head()
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



df_train.head()
def split_train_test(data,size=0.3):

        arr = np.arange(len(data))

        np.random.shuffle(arr)

        train = data.iloc[arr[0:int(len(data)*(1-size))]]

        test = data.iloc[arr[int(len(data)*(1-size)):len(data)]]

        return train,test
dtrain,dtest=split_train_test(df_train,size = 0.3)

X_train=dtrain[dtrain.columns[1:]]

Y_train=dtrain[dtrain.columns[:1]]

X_test=dtest[dtest.columns[1:]]

Y_test=dtest[dtest.columns[:1]]

XX=df_train[df_train.columns[1:]]

YY=df_train['Survived']

XT=df_test[df_test.columns[1:]]
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)

logreg_acc
coeff_df = pd.DataFrame(df_train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)

knn_acc
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)

svc_acc
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

gauss_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)

gauss_acc
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

perc_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)

perc_acc
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

linsvc_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)

linsvc_acc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

sgd_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)

sgd_acc
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

dtree_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)

dtree_acc
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

ran_forest_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)

ran_forest_acc
models = pd.DataFrame({

    'Model': ['Logistic Regression','KNN','Support Vector Machines', 

              'Naive Bayes','Perceptron','Linear SVC', 

              'Stochastic Gradient Decent','Decision Tree','Random Forest'],

    'Score': [logreg_acc,knn_acc, svc_acc,gauss_acc, 

              perc_acc,linsvc_acc,sgd_acc,dtree_acc,ran_forest_acc]})

models.sort_values(by='Score', ascending=False)
Y_pred = logreg.predict(XT)

submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)