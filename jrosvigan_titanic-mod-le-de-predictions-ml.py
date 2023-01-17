import pandas as pd

import pandas.plotting

from pandas.plotting import scatter_matrix

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

import squarify

from matplotlib import animation

from matplotlib.animation import FuncAnimation,FFMpegFileWriter

from mpl_toolkits.mplot3d import Axes3D
pd.set_option('display.max_row',111)

pd.set_option('display.max_column',111)
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
df=train.copy()

print(df.columns)
print(df.shape)
print(df.dtypes.value_counts())

df.dtypes.value_counts().plot.pie()
df.isna()

plt.figure(figsize=(20,10))

sns.heatmap(df.isna(), cbar=False)
(df.isna().sum()/df.shape[0]).sort_values(ascending=True)
df['Survived'].value_counts()
df['Survived'].value_counts(normalize=True)
df.select_dtypes('float').columns
for col in df.select_dtypes('float'):

    #print(col)

    plt.figure()

    sns.distplot(df[col])
df.select_dtypes('int64').columns
df.select_dtypes('object').columns
for col in df[['Sex','Embarked','Pclass', 'SibSp', 'Parch']]:

    plt.figure()

    df[col].value_counts().plot.pie()
survivant_df=df[df['Survived']==1]
decedes_df=df[df['Survived']==0]
for col in df.select_dtypes('float'):

    plt.figure()

    sns.distplot(survivant_df[col], label='survivant')

    sns.distplot(decedes_df[col], label='decedes')

    plt.legend()
plt.figure(figsize=(20,10))

sns.countplot(x='Age',hue='Survived', data=df)
facet = sns.FacetGrid(df, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, df['Age'].max()))

facet.add_legend()

plt.xlim(0)
sns.countplot(x='Pclass',hue='Survived', data=df)
sns.countplot(x='SibSp',hue='Survived', data=df)
sns.countplot(x='Parch',hue='Survived', data=df)
pd.crosstab(df['Survived'],df['Sex'])
pd.crosstab(df['Survived'],df['Embarked'])
for col in df[['Sex','Embarked','Pclass', 'SibSp', 'Parch']]:

        plt.figure()

        sns.heatmap(pd.crosstab(df['Survived'],df[col]),annot=True,fmt='d')
import researchpy as rp

corr_type, corr_matrix, corr_ps = rp.corr_case(df.select_dtypes('float'))

print(corr_type)
corr_matrix
corr_ps
df_train = train.copy()

df_test = test.copy()

df_train.head()
df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)# tout stringaccompagné de point(.)

df_test['Title'] = df_test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)# tout stringaccompagné de point(.)

df_train['Title'].value_counts()
def encodigingName(df):

    code = {"Mr": 0, 

                     "Miss": 1, 

                     "Mrs": 2, 

                     "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                     "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

    df['Title'] = df['Title'].map(code)

    df.drop('Name', axis=1, inplace=True)

    return df



encodigingName(df_train)

encodigingName(df_test)
sns.heatmap(pd.crosstab(df_train['Survived'],df_train['Title']),annot=True,fmt='d')
def encodigingSexe(df):

    code = {"male": 0, 

            "female": 1}

    df['Sex'] = df['Sex'].map(code)

    return df



encodigingSexe(df_test)

encodigingSexe(df_train)
sns.heatmap(pd.crosstab(df_train['Survived'],df_train['Sex']),annot=True,fmt='d')
df_train["Age"].fillna(df_train.groupby("Title")["Age"].transform("median"), inplace=True)

df_test["Age"].fillna(df_test.groupby("Title")["Age"].transform("median"), inplace=True)
facet = sns.FacetGrid(df_train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, df_train['Age'].max()))

facet.add_legend()

plt.show() 
def encodigingAge(dataset):

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,

    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,

    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,

    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4

    return dataset



encodigingAge(df_test)

encodigingAge(df_train)
sns.heatmap(pd.crosstab(df_train['Survived'],df_train['Age']),annot=True,fmt='d')
# Pour connaitre le dominant dans la colonne Embarked afin de remplir les valeurs manquantes

def ClasseEmbarked(df):

    Pclass1 = df[df['Pclass']==1]['Embarked'].value_counts()

    Pclass2 = df[df['Pclass']==2]['Embarked'].value_counts()

    Pclass3 = df[df['Pclass']==3]['Embarked'].value_counts()

    df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

    df.index = ['1st class','2nd class', '3rd class']

    df.plot(kind='bar',stacked=True, figsize=(10,5))

    

ClasseEmbarked(df_test)    

ClasseEmbarked(df_train)
# le "S" est le plus dominant donc on remplace les NAN par les S

df_train['Embarked'] = df_train['Embarked'].fillna('S')

df_test['Embarked'] = df_test['Embarked'].fillna('S')
def encodiging(df):

    code = {"S": 0, 

            "C": 1, 

            "Q": 2}

    df['Embarked'] = df['Embarked'].map(code)

    return df



encodiging(df_test)

encodiging(df_train)
df_train["Fare"].fillna(df_train.groupby("Pclass")["Fare"].transform("median"), inplace=True)

df_test["Fare"].fillna(df_test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
facet = sns.FacetGrid(df_train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, df_train['Fare'].max()))

facet.add_legend()

plt.show() 
def classeFare(dataset):

    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,

    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,

    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,

    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3

    return dataset



classeFare(df_test)

classeFare(df_train)
df_train['Cabin'].value_counts()
def extratCabin(df):

    df['Cabin'] = df['Cabin'].str[:1]

    return df



extratCabin(df_test)

extratCabin(df_train)
def classeCabin(df):

    Pclass1 = df[df['Pclass']==1]['Cabin'].value_counts()

    Pclass2 = df[df['Pclass']==2]['Cabin'].value_counts()

    Pclass3 = df[df['Pclass']==3]['Cabin'].value_counts()

    df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

    df.index = ['1st class','2nd class', '3rd class']

    df.plot(kind='bar',stacked=True, figsize=(10,5))



classeCabin(df_train)
def encodigingCabin(df):

    code = {"A": 0, 

            "B": 0.4, 

            "C": 0.8, 

            "D": 1.2, 

            "E": 1.6, 

            "F": 2, 

            "G": 2.4, 

            "T": 2.8}

    df['Cabin'] = df['Cabin'].map(code)

    return df 



encodigingCabin(df_test)

encodigingCabin(df_train)
df_train["Cabin"].fillna(df_train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

df_test["Cabin"].fillna(df_test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"] + 1

df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"] + 1
facet = sns.FacetGrid(df_train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'FamilySize',shade= True)

facet.set(xlim=(0, df_train['FamilySize'].max()))

facet.add_legend()

plt.xlim(0)
def encodigingFamily(df):

    code = {1: 0,

            2: 0.4, 

            3: 0.8, 

            4: 1.2, 

            5: 1.6, 

            6: 2, 

            7: 2.4, 

            8: 2.8, 

            9: 3.2, 

            10: 3.6, 

            11: 4}

    df['FamilySize'] = df['FamilySize'].map(code)

    return df

    

encodigingFamily(df_test)

encodigingFamily(df_train)
variables_drop = ['Ticket', 'SibSp', 'Parch']



df_train = df_train.drop(variables_drop, axis=1)

df_test = df_test.drop(variables_drop, axis=1)
df_train = df_train.drop(['PassengerId'], axis=1)

#X_test = df_test.drop("PassengerId", axis=1).copy()



#trainset en Xtrain et Ytrain

X_train= df_train.drop('Survived', axis=1)

y_train = df_train['Survived']



X_train.shape, y_train.shape
X_train.head()
X_test = df_test.drop("PassengerId", axis=1).copy()

X_test.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.model_selection import learning_curve



import numpy as np
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)



print("\nle score KNN moyenne est :" + str(round(np.mean(score)*100, 2)))

N, train_score, val_score = learning_curve(clf, X_train, y_train,cv=k_fold, scoring=scoring,train_sizes=np.linspace(0.1, 1, 10))

plt.figure(figsize=(12, 8))

plt.plot(N, train_score.mean(axis=1), label='train score')

plt.plot(N, val_score.mean(axis=1), label='validation score')

plt.legend()
clf = DecisionTreeClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

print("\nle score Decision Tree moyenne est :" + str(round(np.mean(score)*100, 2)))
N, train_score, val_score = learning_curve(clf, X_train, y_train,cv=k_fold, scoring=scoring,train_sizes=np.linspace(0.1, 1, 10))

plt.figure(figsize=(12, 8))

plt.plot(N, train_score.mean(axis=1), label='train score')

plt.plot(N, val_score.mean(axis=1), label='validation score')

plt.legend()
clf = RandomForestClassifier(n_estimators=13)

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

print("\n le score Ramdom Forest moyenne est :" + str(round(np.mean(score)*100, 2)))
N, train_score, val_score = learning_curve(clf, X_train, y_train,cv=k_fold, scoring=scoring,train_sizes=np.linspace(0.1, 1, 10))

plt.figure(figsize=(12, 8))

plt.plot(N, train_score.mean(axis=1), label='train score')

plt.plot(N, val_score.mean(axis=1), label='validation score')

plt.legend()
clf = GaussianNB()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

print("\n le score Naive Bayes moyenne est :" + str(round(np.mean(score)*100, 2)))
N, train_score, val_score = learning_curve(clf, X_train, y_train,cv=k_fold, scoring=scoring,train_sizes=np.linspace(0.1, 1, 10))

plt.figure(figsize=(12, 8))

plt.plot(N, train_score.mean(axis=1), label='train score')

plt.plot(N, val_score.mean(axis=1), label='validation score')

plt.legend()
clf = SVC()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

print("\n le score SVM moyenne est :" + str(round(np.mean(score)*100, 2)))
N, train_score, val_score = learning_curve(clf, X_train, y_train,cv=k_fold, scoring=scoring,train_sizes=np.linspace(0.1, 1, 10))

plt.figure(figsize=(12, 8))

plt.plot(N, train_score.mean(axis=1), label='train score')

plt.plot(N, val_score.mean(axis=1), label='validation score')

plt.legend()
clf = SVC()

clf.fit(X_train, y_train)

ypred = clf.predict(X_test)

ypred
resultat = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": ypred

    })



resultat.head()
resultat.to_csv('resultat.csv', index=False)