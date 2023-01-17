import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

combine = [train_df, test_df]

train_df.head()
ftr = train_df.columns.values

ftr_cat = ['Survived', 'Sex', 'Cabin', 'Embarked']  # Categorical features

ftr_ord = ['Pclass']                                # Ordinal     features

ftr_dis = ['SibSp', 'Parch']                        # Discrete    features

ftr_con = ['Age', 'Fare']                           # Continous   features
train_df.info()

print('_'*40)

test_df.info()
# Numerical Description



train_df.describe()



# Around 38% samples survived representative of the actual survival rate at 32%.

# Most passengers (> 75%) did not travel with parents or children.

# Nearly 30% of the passengers had siblings and/or spouse aboard.

# Fares varied significantly with few passengers (<1%) paying as high as $512.

# Few elderly passengers (<1%) within age range 65-80.
# Categorical Description



train_df.describe(include=['O'])   # 'O' stands for object



# Names are unique across the dataset (count=unique=891)

# Sex variable as two possible values with 65% male (top=male, freq=577/count=891).

# Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.

# Embarked takes three possible values. S port used by most passengers (top=S)

# Ticket feature has high ratio (22%) of duplicate values (unique=681).
# Analyze by pivoting features



for _f in ['Pclass', 'Sex', 'SibSp', 'Parch']:

    print(train_df[[_f, 'Survived']].groupby([_f], as_index=False).mean().sort_values(by='Survived', ascending=False))

    print('--------------------')
# Correlating numerical features



g1 = sns.FacetGrid(train_df, col='Survived')

g1.map(plt.hist, 'Age', bins=20)
# Correlating numerical and ordinal features



g2 = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)

g2.map(plt.hist, 'Age', alpha=.5, bins=20)

g2.add_legend();
# Correlating categorical features



g3 = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)

g3.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order=[1, 2, 3], hue_order=['male', 'female'])

g3.add_legend()
# Correlating categorical and numerical features



g4 = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)

g4.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None, order=['male', 'female'])

g4.add_legend()
# Drop Ticket & Cabin



train_df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)

test_df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
# Extract Title from Name and drop Name

for ds in combine:

    ds['Title'] = ds.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



print(pd.crosstab(train_df['Title'], train_df['Sex']))



for ds in combine:

    ds['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', inplace=True)

    ds['Title'].replace('Mlle', 'Miss', inplace=True)

    ds['Title'].replace('Ms', 'Miss', inplace=True)

    ds['Title'].replace('Mme', 'Mrs', inplace=True)



print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

title_mapping_rev = {1: "Mr", 2: "Miss", 3: "Mrs", 4: "Master", 5: "Rare"}

for ds in combine:

    ds['Title'] = ds['Title'].map(title_mapping)

    ds['Title'] = ds['Title'].fillna(0)



train_df.drop(['Name', 'PassengerId'], axis=1, inplace=True)

test_df.drop(['Name'], axis=1, inplace=True)
# Sex



for ds in combine:

    ds['Sex'] = ds['Sex'].map({'male': 0, 'female': 1}).astype(int)
# Age: Fill missing values according to Sex and Pclass



g5 = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)

g5.map(plt.hist, 'Age', alpha=.5, bins=20)

g5.add_legend()



age_lookup = np.zeros((2,3))   # Guess age by Sex and Pclass



for ds in combine:

    for sex in range(2):

        for pclass in range(3):

            temp_df = ds[(ds.Sex == sex) & (ds.Pclass == pclass + 1)]['Age'].dropna()

            

            # Guess randomly:

            #   age_mean = temp_df.mean()

            #   age_std = temp_df.std()

            #   age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            # Or use median:

            age_guess = temp_df.median()



            age_lookup[sex, pclass] = int(round(age_guess))

    

    for sex in range(2):

        for pclass in range(3):

            ds.loc[(ds.Age.isnull()) & (ds.Sex == sex) & (ds.Pclass == pclass + 1), 'Age'] = age_lookup[sex, pclass]

    ds['Age'] = ds['Age'].astype(int)
# Convert Age to ordinal



train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

train_df.drop(['AgeBand'], axis=1, inplace=True)



for dataset in combine:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()
# Create new feature: FamilySize



for ds in combine:

    ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1

    ds['IsAlone'] = ds['FamilySize'] == 1

    ds['IsAlone'] = ds['IsAlone'].astype(int)



for _f in ['FamilySize', 'IsAlone']:

    print(train_df[[_f, 'Survived']].groupby([_f], as_index=False).mean().sort_values(by='Survived', ascending=False))

    print('------------------------')



for ds in combine:

    ds.drop(['Parch', 'SibSp', 'FamilySize'], axis=1, inplace=True)



train_df.head()
# Create new feature: Age*Class



for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df[['Age*Class', 'Age', 'Pclass']].head(10)
# Embarked: Fill missing values by most frequent



freq_port = train_df.Embarked.dropna().mode()[0]    # S

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port).map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Fare



test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)



for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df.drop(['FareBand'], axis=1, inplace=True)
# Dummies



train_df = pd.get_dummies(train_df, prefix=['Title', 'Embarked'], columns=['Title', 'Embarked'])

test_df = pd.get_dummies(test_df, prefix=['Title', 'Embarked'], columns=['Title', 'Embarked'])

combine = [train_df, test_df]
train_df.head(10)
test_df.head(10)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.utils.testing import ignore_warnings

from sklearn.exceptions import ConvergenceWarning



@ignore_warnings(category=ConvergenceWarning)

def classification_report(cls):

    cls.fit(X_train, Y_train)

    y_pred = cls.predict(X_test)

    train_acc = round(cls.score(X_train, Y_train) * 100, 2)

    train_cv_acc = round(cross_val_score(cls, X_train, Y_train, cv=20).mean() * 100, 2)

    return train_acc, train_cv_acc, y_pred
classifiers = {

    'Logistic Regression': LogisticRegression(solver='liblinear'),

    'Support Vector Machine': SVC(gamma='auto'),

    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),

    'Gaussian Naive Bayes': GaussianNB(),

    'Perceptron': Perceptron(),

    'Linear SVC': LinearSVC(),

    'Stochastic Gradient Descent': SGDClassifier(),

    'Decision Tree': DecisionTreeClassifier(),

    'Random Forest': RandomForestClassifier(n_estimators=100),

    'MLP': MLPClassifier(hidden_layer_sizes=(20,)),

}



scores = [classification_report(cls) for cls in list(classifiers.values())]

preds = dict(zip(classifiers.keys(), [x[2] for x in scores]))



models = pd.DataFrame({

    'Models': list(classifiers.keys()),

    'Train Score': [x[0] for x in scores],

    'CV Score': [x[1] for x in scores],

})

models.sort_values(by='CV Score', ascending=False)
svm = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": preds['Support Vector Machine']})

dt = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": preds['Decision Tree']})

rf = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": preds['Random Forest']})

mlp = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": preds['MLP']})



svm.to_csv('svm.csv', index=False)

dt.to_csv('dt.csv', index=False)

rf.to_csv('rf.csv', index=False)

mlp.to_csv('mlp.csv', index=False)



print('Submission files created')