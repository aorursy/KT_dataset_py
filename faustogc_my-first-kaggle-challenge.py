import numpy as np

import scipy as sp

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from tabulate import tabulate
df = pd.read_csv("../input/train.csv")
df.shape
df.head()
df.dtypes
test_df = pd.read_csv("../input/test.csv")

test_df.head()
props = []



for col in 'Pclass SibSp Parch Ticket Cabin Embarked'.split():

    props.append([col, df[col].nunique()])

    

print(tabulate(props, headers=['Column', 'nunique()']))
df['Pclass'].value_counts()
df['Embarked'].value_counts()
df['Age'].hist()

plt.show()
df['Fare'].hist()

plt.show()
df['SibSp'].hist()

plt.show()
df['Parch'].hist()

plt.show()
sns.heatmap(df.corr())

plt.show()
g = sns.FacetGrid(df, col="Survived")

g.map(plt.hist, "Fare")

plt.show()
g = sns.FacetGrid(df, col="Survived")

g.map(plt.hist, "Parch")

plt.show()
g = sns.FacetGrid(df, col="Survived")

g.map(plt.hist, "Age")

plt.show()
sns.barplot(x="Pclass", y="Survived", data=df)

plt.show()
sns.barplot(x="Sex", y="Survived", data=df)

plt.show()
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=df)

plt.show()
target = 'Survived'



X = df.drop(target, axis=1)

y = df[target]



del df
X.isnull().sum()
embarked_mode = X['Embarked'].mode()[0]



def imput_embarked(X, val):

    X['Embarked'].fillna(embarked_mode, inplace=True)



imput_embarked(X, embarked_mode)
median_age_sex_pclass = X[['Sex', 'Pclass', 'Age']].groupby(['Sex', 'Pclass']).agg(np.median).reset_index()

median_age_sex_pclass
X['AgeFill'] = X['Age']

for csex in ['male', 'female']:

    for cpclass in [1, 2, 3]:

        new_val = median_age_sex_pclass[(median_age_sex_pclass['Sex'] == csex) &

                                        (median_age_sex_pclass['Pclass'] == cpclass)]['Age'].values[0]

        

        X.loc[(X['Sex'] == csex) &

              (X['Pclass'] == cpclass), 'AgeFill'] = X[(X['Sex'] == csex) & 

                                                       (X['Pclass'] == cpclass)]['AgeFill'].fillna(new_val)
def create_family(X):

    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
def create_title(X):

    X['Title'] = X['Name'].apply(lambda s: s.split(', ')[1].split('.')[0])

    

    print(X['Title'].value_counts())

    print('\n')

    

    X['Title'] = X['Title'].replace('Master Dr Rev Mlle Col Major Jonkheer Don Capt Sir'.split(),

                                     'Rare')

    X['Title'] = X['Title'].replace(['Ms', 'Lady', 'Mme', 'the Countess'], 'Miss')

    

    print(X['Title'].value_counts())
create_family(X)
create_title(X)
X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})



X['Title'] = X['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Rare': 3})
X['Embarked'] = X['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
def clean_df(X):

    X.drop(['PassengerId', 'Name', 'Ticket', 'Age', 'Cabin', 'Parch', 'SibSp'], axis=1, inplace=True)
clean_df(X)
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()



X_scaled = pd.DataFrame(std_scaler.fit_transform(X), columns=X.columns)
X_scaled.head()
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
res = []

for k in range(2, 33):

    knn = KNeighborsClassifier(n_neighbors=k)

    

    score = cross_val_score(knn, X, y, cv=10)

    

    res.append([k, np.mean(score), np.std(score)])
print(tabulate(res))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.get_params()
random_params = {

    'max_leaf_nodes': sp.stats.randint(6, 50),

    'n_estimators': sp.stats.randint(50, 500),

    'max_features': sp.stats.uniform(0.3, 0.8)

}
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(rf, random_params, n_iter=50, n_jobs=-1, cv=3)
rand1 = random_search.fit(X, y)
rand1.best_estimator_