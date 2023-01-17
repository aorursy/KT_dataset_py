# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame, get_dummies



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.model_selection import train_test_split,StratifiedKFold,RandomizedSearchCV, GridSearchCV

from sklearn.metrics import accuracy_score



from sklearn.feature_extraction import DictVectorizer

from scipy.stats import randint as sp_randint



from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# preview na value

titanic_df.isnull().sum()
titanic_df.info()
# How many people have survived?

titanic_df['Survived'].describe()

# answer : 38%
test_df.isnull().sum()
titanic_df.groupby('Embarked').count()
# First of all, drop unnecessary columns, these columns won't be useful in analysis and prediction

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

test_df    = test_df.drop(['Name','Ticket'], axis=1)
# Looking at distribution of cabin and age



titanic_df['Age'].hist()
# Fill the missing Age

totalNa = titanic_df['Age'].isnull().sum()

ageDist = titanic_df['Age'].dropna().tolist()

ageDist.sort()



fig, axes = plt.subplots(nrows=2, ncols=1)

print(axes)

titanic_df['Age'][titanic_df['Age'].isnull() == False].plot(kind='hist', bins=50, ax=axes[0])

axes[0].set_title('Old Age')



np.random.seed(71) # make the random reproductable

ages = np.random.choice(ageDist, totalNa)



newAgeColumn = 'FilledAge'



titanic_df[newAgeColumn] = titanic_df['Age']



titanic_df.loc[titanic_df['Age'].isnull(), newAgeColumn ] = ages

titanic_df[newAgeColumn].plot(kind='hist', bins=50, ax=axes[1])

axes[1].set_title(newAgeColumn)





totalNa = test_df['Age'].isnull().sum()

ages = np.random.choice(ageDist, totalNa)



test_df[newAgeColumn] = test_df['Age']

test_df.loc[test_df['Age'].isnull(), newAgeColumn ] = ages
# Missnig Fare

totalNa = test_df['Fare'].isnull().sum()

fareDist = titanic_df['Fare'].dropna().tolist()

fares = np.random.choice(fareDist, 1) 

test_df.loc[ test_df['Fare'].isnull(), 'Fare' ] = fares
# Filling Embarked

embarkedDist = titanic_df['Embarked'].dropna().tolist()

totalMissing = titanic_df['Embarked'].isnull().sum()



np.random.seed(71)



embarked = np.random.choice( embarkedDist, totalMissing )



titanic_df['FilledEmbarked'] = titanic_df['Embarked']

test_df['FilledEmbarked'] = test_df['Embarked']



titanic_df.loc[titanic_df['Embarked'].isnull(), 'FilledEmbarked' ] = embarked

# no need to assign for test_df since there is no missing value there.



selectedColumns = ['Survived','Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'FilledAge', 'FilledEmbarked']

titanic_df = titanic_df[selectedColumns];

titanic_df.describe()
# Age

currColumn = 'FilledAge'

fig, axes = plt.subplots(nrows=2, ncols=1)



plt.tight_layout()



titanic_df[ titanic_df['Survived'] == 0 ][currColumn].hist(ax=axes[0],bins=50)

axes[0].set_title('Distribution of not survived and Age')

titanic_df[ titanic_df['Survived'] == 1 ][currColumn].hist(ax=axes[1],bins=50)

axes[1].set_title('Distribution of survived and Age')
titanic_df.groupby(['Survived','Sex']).count() / titanic_df.count()

# Here we can see that 50% of male died,

# so Sex might be a good feature to use in this problem.
titanic_df.groupby(['Survived','Pclass']).count() / titanic_df.count()

# Also, we find that 40% of the death is passengers in pclass=3
titanic_df.groupby(['Survived','FilledEmbarked']).count() / titanic_df.count()

# Passengers in Embarked `S` contributes significantly in the death (47%)
titanic_df.groupby(['Survived','Parch']).count() / titanic_df.count()
titanic_df.groupby(['Parch']).count() / titanic_df.count()
titanic_df.groupby(['Survived','SibSp']).count() / titanic_df.count()
# digitize non-numerical columsn ( sex, embarked )

def digitizeNonNumericalColumns(df):

    new_df = df.copy();

    new_df['Sex'] = df['Sex'].map( {'female': 0, 'male':1 }).astype(int)



    ddf = get_dummies(new_df['FilledEmbarked'],prefix='Embarked')

    return new_df.join(ddf).drop('FilledEmbarked',axis=1)



digitized_df = digitizeNonNumericalColumns(titanic_df)

digitized_df.info()

test_dig_df = digitizeNonNumericalColumns(test_df)

test_dig_df.info()
# split data

train_df, val_df = train_test_split(digitized_df, test_size=0.2,stratify=digitized_df['Survived'])
# random forest

clf = RandomForestClassifier(n_estimators = 200)



features = digitized_df.columns.tolist()

features.remove('Survived')



param_dist = {"max_depth": [3, None],

              "min_samples_split": sp_randint(2, 11),

             "min_samples_leaf": sp_randint(2, 11),

              "bootstrap": [True, False],

              "criterion": ["gini", "entropy"]}

n_iter_search = 10

random_search = RandomizedSearchCV(clf, param_distributions=param_dist,

                                   n_iter=n_iter_search)



# Fit the training data to the Survived labels and create the decision trees



random_search.fit(train_df[features],train_df['Survived'].tolist())



# Take the same decision trees and run it on the test data

best = random_search.best_estimator_

pred = best.predict(val_df[features])

print("Accuracy for RandomForest : %.2f" % (accuracy_score(val_df['Survived'],pred)))
def writePred(df, filename ):

    df[['PassengerId','Survived']].to_csv(filename,header=True,index=False)

test_dig_df['Survived'] = best.predict(test_dig_df[features])

writePred(test_dig_df,'randomforest-model.csv')
# svm

clf = SVC()

#clf.fit(train_df[features],train_df['Survived'])



param_dist = { 

    "kernel": ['rbf', 'sigmoid'],

    'C': [0.1,0.5,1,10]

}

grid_search = GridSearchCV(clf, param_grid=param_dist,scoring='accuracy')



grid_search.fit(train_df[features],train_df['Survived'])

best = grid_search.best_estimator_

pred = best.predict(val_df[features])

print("Accuracy for SVM : %.2f" % (accuracy_score(val_df['Survived'],pred)))



test_dig_df['Survived'] = best.predict(test_dig_df[features])

writePred(test_dig_df,'svm-model.csv')
# PCA

pca = PCA()

pca.fit(train_df[features])

pca.explained_variance_ratio_

variances = pca.explained_variance_ratio_

totalVar = sum(variances)

# threshold PCA retrain > 95%

accVar = 0

for i in range(len(variances)):

    accVar = accVar + variances[i]

    if accVar*1.0/totalVar >= 0.95 :

        break



print("Suitable  no. of PCA components %d" % (i+1))



pca = PCA(n_components=i+1)



t = train_df[features].apply(lambda x: MinMaxScaler().fit_transform(x))

pca_train_df = pca.fit_transform(t)

pca_val_df   = pca.fit_transform(val_df[features].apply(lambda x: MinMaxScaler().fit_transform(x)))

pca_test_df  = pca.fit_transform(test_dig_df[features].apply(lambda x: MinMaxScaler().fit_transform(x)))



print(pca_train_df)
# Using SVM with PCA



survived = train_df['Survived'].map( lambda s : 'b' if s == 1 else 'r' )

plt.scatter( pca_train_df[:, 0], pca_train_df[:, 1], c=survived, alpha=0.8)

plt.show()



clf = SVC()

clf.fit(pca_train_df,train_df['Survived'] )



pred = clf.predict(pca_val_df)

print("Accuracy for SVM & PCA : %.2f" % (accuracy_score(val_df['Survived'],pred)))



test_dig_df['Survived'] = clf.predict(pca_test_df)

writePred(test_dig_df,'svm-pca-model.csv')