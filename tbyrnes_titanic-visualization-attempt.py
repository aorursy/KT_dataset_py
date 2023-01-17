



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
titanic_test = pd.read_csv('../input/test.csv')

titanic_train = pd.read_csv('../input/train.csv')



data_full = [titanic_train, titanic_test]
titanic_test.head()
titanic_train['Title'] = titanic_train['Name'].apply( lambda x: x.split(',')[1].split('.')[0])
plt.figure(figsize=(10, 5))

sns.heatmap(titanic_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#drop Cabin detail in train and test

for data in data_full:

    data.drop('Cabin',axis=1,inplace=True)
df = titanic_train.dropna()



g = sns.FacetGrid(df, col="Pclass")

order = sorted(df.Parch.unique())

g.map(sns.boxplot, "Parch", "Age", color=".3", order=order )
g = sns.FacetGrid(df, col="Sex")



order = sorted(df.SibSp.unique())

g.map(sns.boxplot, "SibSp", "Age", order = order )


titanic_complete = pd.concat([titanic_train, titanic_test], ignore_index=True)

ageGrouped = titanic_complete.groupby(['Pclass','Parch','SibSp','Sex'])['Age'].mean()

ageGrouped
age_corr = titanic_train[['Age','Parch','Pclass','SibSp']]
plt.figure(figsize=(8, 5))

sns.heatmap(age_corr.corr(),cmap='viridis', annot=True)

def populateAges(df, grouped):

    d = []

    for i, row in df.iterrows():

        if pd.isnull(row.Age): 

            Parch = row.Parch

#             print(Parch)

            SibSp = row.SibSp

#             print(SibSp)

            Pclass = row.Pclass

            Sex = row.Sex

            newAge = grouped.get_value((Pclass, Parch, SibSp, Sex))

            if pd.isnull(newAge):

                if Pclass == 1:

                    newAge = 37

                elif Pclass == 2:

                    newAge = 29

                else:

                    newAge = 24

        else:

            newAge = row.Age

        d.append({'Age' : newAge})

    return pd.DataFrame(d)
age_test = titanic_train.dropna(subset=['Age'])

age_test = age_test.reset_index(drop=True)



age_reproduced = age_test.copy()

age_reproduced['Age'] = np.nan

df_age = populateAges(age_reproduced, ageGrouped)

age_reproduced['Age'] = df_age



df_age['true'] = age_test['Age']

df_age.describe()
fig, ax = plt.subplots(figsize=(10,8))

ax.hist([df_age['Age'], df_age['true']],bins=20 , label = ['Age','true'])

ax.legend()

plt.show()

# sns.distplot(df_age['true'], ax=ax, label = 'true age',, color='blue')


df_age['percent'] =  ( (abs(df_age['Age'] - df_age['true'])/ df_age['true'])) * 100

sns.boxplot(y='percent', data=df_age,showfliers=False)
sns.jointplot(x='true', y ='Age', data=df_age)
df_age['percent'].describe()
g = sns.FacetGrid(age_reproduced, col="Pclass")

order = sorted(age_reproduced.Parch.unique())

g.map(sns.boxplot, "Parch", "Age", color=".3", order = order )
titanic_test.head()
for data in data_full:

    data['Age']  = populateAges(data, ageGrouped) 

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

#     data['IsAlone'] = 0

#     data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

    data.drop('Ticket', axis=1, inplace=True)

    data['Fare'].fillna(data['Fare'].mean(), inplace=True)

        # Mapping Fare

    data.loc[ data['Fare'] <= 7.91, 'Fare']       = 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2

    data.loc[ data['Fare'] > 31, 'Fare']       = 3

    data['Fare'] = data['Fare'].astype(int)

    #map age into discrete ranges

#     data.loc[ data['Age'] <= 16, 'Age']         = 0

#     data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

#     data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age']  = 2

#     data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age']  = 3

#     data.loc[ data['Age'] > 64, 'Age']         = 4

#     data['Age'] = data['Age'].astype(int)

    

embarked = pd.get_dummies(titanic_train['Embarked'], drop_first=True)

sex = pd.get_dummies(titanic_train['Sex'], drop_first=True)

titanic_train.drop(['Embarked','Sex'], axis=1, inplace=True)

titanic_train = pd.concat([titanic_train,embarked,sex], axis=1)



embarked = pd.get_dummies(titanic_test['Embarked'], drop_first=True)

sex = pd.get_dummies(titanic_test['Sex'], drop_first=True)

titanic_test.drop(['Embarked','Sex'], axis=1, inplace=True)

titanic_test = pd.concat([titanic_test,embarked,sex], axis=1)
print(titanic_train[['Title','Survived']].groupby(['Title'], as_index=False).count())
titanic_test.head()
titanic_train['Title'] = titanic_train['Name'].apply( lambda x: x.split(',')[1].split('.')[0])

titanic_train = titanic_train.drop('Name', axis=1)

titanic_test['Title'] = titanic_test['Name'].apply( lambda x: x.split(',')[1].split('.')[0])

titanic_test = titanic_test.drop('Name', axis=1) 

titanic_train.head()

titanic_train['Title'] = titanic_train['Title'].replace( ['Capt','Col','Don','Dr','Jonkheer','Lady','Major','Rev','Sir'], 'Rare', regex = True) 

titanic_test['Title'] = titanic_test['Title'].replace(['Capt','Col','Don','Dr','Jonkheer','Lady','Major','Rev','Sir'], 'Rare', regex=True)



titanic_train['Title'] = titanic_train['Title'].replace( ['Ms','Mlle'], 'Miss', regex = True) 

titanic_test['Title'] = titanic_test['Title'].replace( ['Ms','Mlle'], 'Miss', regex = True) 



titanic_train['Title'] = titanic_train['Title'].replace('Mme', 'Mrs', regex=True)

titanic_test['Title'] = titanic_test['Title'].replace('Mme', 'Mrs', regex=True)



# titanic_train[['Age', 'Survived']].groupby(titanic_train['Age']).sum()
#create dummy values for PCA

titanic_train['Title'] = titanic_train['Title'].astype('category')

titanic_train['Title'] = titanic_train['Title'].cat.codes

titanic_test['Title'] = titanic_test['Title'].astype('category')

titanic_test['Title'] = titanic_test['Title'].cat.codes
titanic_train.drop(['PassengerId'], axis=1, inplace=True)

PassengerTest = titanic_test['PassengerId']

titanic_test.drop(['PassengerId'], axis=1, inplace=True)

titanic_test.head()
titanic_train.groupby(['Title']).sum()
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

df = titanic_train.drop('Survived', axis=1)

scaler = StandardScaler()

# scaler.fit(df)

# scaled_data = scaler.transform(df)



pca =PCA(n_components = 2)

pca.fit(df)

# pca.fit(scaled_data)

x_pca = pca.transform(df)
x_pca
pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum()
plt.figure(figsize=(8,6))

plt.scatter(x=x_pca[:,0], y=x_pca[:,1], c=titanic_train['Survived'])
titanic_test.info()
from mpl_toolkits.mplot3d import Axes3D



fig  = plt.figure(figsize=(6,6))

ax = plt.axes(projection='3d')

ax.scatter(x_pca[:,0],x_pca[:,1],x_pca[:,2],zdir='y', c=titanic_train['Survived'])

# Axes3D.scatter(x_pca[:,0],x_pca[:,1],x_pca[:,2])
# X = titanic_train.loc[:, titanic_train.columns !='Survived']

y = titanic_train['Survived']



X = x_pca
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import cross_val_score



classifiers = [

    KNeighborsClassifier(2),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

	AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



for model in classifiers:

    score = cross_val_score(model, X, y, cv=10).mean()

    print(model.__class__.__name__, score)
titanic_test.head()
# X = titanic_train.loc[:, titanic_train.columns !='Survived']

y = titanic_train['Survived']

X = x_pca



log = LogisticRegression()

# GBC =  GradientBoostingClassifier()

# SVC =  SVC(probability=True)

log.fit(X, y )



# PCA on test data

df = titanic_test

scaler = StandardScaler()

scaler.fit(df)

scaled_data = scaler.transform(df)

pca = PCA(n_components = 2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)



predictions = log.predict(x_pca)



titanic_test['Survived'] = predictions

titanic_test['PassengerId'] = PassengerTest
# pca.explained_variance_ratio_.sum()
titanic_test
# from sklearn.linear_model import LogisticRegression



# logmodel = LogisticRegression()

# logmodel.fit(titanic_train.drop(['Survived'], axis=1), titanic_train['Survived'])



# predictions = logmodel.predict(titanic_test)

# titanic_test["Survived"] = predictions



# titanic_train.head()
titanic_test
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)