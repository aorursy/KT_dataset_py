#useful libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns



#ML algorithms

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import RFE

from sklearn.feature_selection import RFECV



from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#Seaborn

sns.set_palette("deep", desat=.6)

sns.set_context(rc={"figure.figsize": (8, 4)})

import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output
train_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)



full = train_df.append( test_df , ignore_index = True )

titanic = full[ :891 ]



train_df.head()
test_df.head()
len(train_df)
train_df.isnull().sum().sort_values(ascending=False)
train_df.dtypes
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)

train_df.isnull().sum().sort_values(ascending=False)
train_df = train_df.drop(['Cabin'], axis=1)

train_df.head()
train_df.Embarked.unique()
train_df['Embarked'].value_counts()
train_df['Embarked'].fillna('S', inplace=True)

train_df['Embarked'].value_counts()
train_df.describe()
print('Distribution of target variable\n',train_df['Survived'].value_counts())

ax = sns.countplot(train_df['Survived'],palette="viridis")



for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/len(train_df)), (x.mean(), y), 

            ha='center', va='bottom') # set the alignment of the text
mean_correlation = train_df.corr()

plt.figure(figsize=(8,8))

sns.heatmap(mean_correlation,vmax=1,square=True,annot=True,cmap='Oranges')
sns.barplot(x='Pclass', y='Fare', hue='Survived', data = train_df, palette = 'viridis')

plt.title('Relationship between Class and Fare')
sns.barplot(x='Survived', y='Fare', data = train_df, palette = 'viridis')

plt.title('Relationship between Fare and be alive')
sns.barplot( x='Survived', y='Age', data = train_df, palette = 'viridis')

plt.title('Relationship between Survived and Age (mean)')
sns.countplot( x='Sex', hue='Survived', data = train_df, palette = 'viridis')

plt.title('Relationship between Survived and Sex')
def people(passenger):

    age, sex = passenger

    return 'child' if age < 15 else sex

    

train_df['People'] = train_df[['Age','Sex']].apply(people, axis=1)
sns.countplot( x='People', hue='Survived', data = train_df, palette = 'viridis')

plt.title('Relationship between Survived and People')
sns.barplot( x='People', y='Survived', data = train_df, palette = 'viridis')

plt.title('Relationship between Survived and People')
sns.countplot( x='Embarked', hue='Survived', data = train_df, palette = 'viridis')

plt.title('Relationship between Survived and Embarked')
sns.barplot( x='Embarked', y='Fare', hue='Survived', data = train_df, palette = 'viridis')

plt.title('Relationship between Fare and Embarked')
sns.countplot( x='SibSp', hue='Survived', data = train_df, palette = 'viridis')

plt.title('Relationship between Survived | Siblings and Spouses')
sns.countplot( x='Parch', hue='Survived', data = train_df, palette = 'viridis')

plt.title('Relationship between Survived and Sex')
train_df['family_members'] =  train_df['Parch'] + train_df['SibSp']

sns.countplot( x='family_members', hue='Survived', data = train_df, palette = 'viridis')

plt.title('Relationship between Survived and and Family members')
sns.barplot( x='family_members', y='Survived', data = train_df, palette = 'viridis')

plt.title('Relationship between Survived and Family members')
train_df['Lastname'], train_df['Name'] = train_df['Name'].str.split(',', 1).str
train_df.head()
train_df['Nclass'], train_df['Name'] = train_df['Name'].str.split('.', 1).str

train_df.head()
sns.countplot( x='Nclass', hue='Survived', data = train_df, palette = 'viridis')

plt.title('Relationship between Survived and Nclass')
train_df.Nclass.unique()
train_df['Nclass'] = train_df['Nclass'].map({' Jonkheer': 'Other', ' the Countess': 'Other', ' Col': 'Other', ' Rev': 'Other', ' Mlle': 'Mrs', ' Mme': 'Mrs', ' Capt': 'Other', ' Ms': 'Miss', ' Lady': 'Miss', ' Major': 'Other', ' Sir': 'Mr', ' Dr': 'Mr', ' Don': 'Mr', ' Master': 'Mr', ' Miss': 'Miss', ' Mrs': 'Mrs', ' Mr': 'Mr',})
train_df.Nclass.unique()
sns.countplot( x='Nclass', hue='Survived', data = train_df, palette = 'viridis')

plt.title('Relationship between Survived and Nclass')
train_df.head()
train_df = train_df.drop(['PassengerId'], axis=1)

train_df = train_df.drop(['Name'], axis=1)

train_df = train_df.drop(['Ticket'], axis=1)

train_df = train_df.drop(['Lastname'], axis=1)

#train_df = train_df.drop(['Sex'], axis=1)

train_df.head()
train_df.Embarked.unique()
train_df.Sex.unique()
categories = train_df.select_dtypes(include=['object'])

categories.columns
dummy_df = pd.get_dummies(train_df[categories.columns])

train_df = pd.concat([train_df, dummy_df], axis=1)

train_df = train_df.drop(categories, axis=1)
train_df.head()
# F - Fisher.

target = train_df['Survived']

k = 7  # Number of attributes

train = train_df.drop(['Survived'], axis=1)

atributes = list(train.columns.values)

selected = SelectKBest(f_classif, k=k).fit(train, target)

atrib = selected.get_support()

final = [atributes[i] for i in list(atrib.nonzero()[0])]

final
# F - Fisher.

target = train_df['Survived']

k = 6  # Number of attributes

train = train_df.drop(['Survived'], axis=1)

atributes = list(train.columns.values)

selected = SelectKBest(f_classif, k=k).fit(train, target)

atrib = selected.get_support()

final = [atributes[i] for i in list(atrib.nonzero()[0])]

final
# F - Fisher.

target = train_df['Survived']

k = 5  # Number of attributes

train = train_df.drop(['Survived'], axis=1)

atributes = list(train.columns.values)

selected = SelectKBest(f_classif, k=k).fit(train, target)

atrib = selected.get_support()

final = [atributes[i] for i in list(atrib.nonzero()[0])]

final
# ExtraTrees

model = ExtraTreesClassifier()

era = RFE(model, 7)  # Number of attributes

era = era.fit(train, target)

atrib = era.support_

final = [atributes[i] for i in list(atrib.nonzero()[0])]

final
# ExtraTrees

model = ExtraTreesClassifier()

era = RFE(model, 6)  # Number of attributes

era = era.fit(train, target)

atrib = era.support_

final = [atributes[i] for i in list(atrib.nonzero()[0])]

final
# ExtraTrees

model = ExtraTreesClassifier()

era = RFE(model, 5)  # Number of attributes

era = era.fit(train, target)

atrib = era.support_

final = [atributes[i] for i in list(atrib.nonzero()[0])]

final
X = train_df.drop(['Survived'],axis = 1 )

y = train_df.Survived



X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3)
rf = RandomForestClassifier() 

rfecv = RFECV(estimator=rf, step=1, cv=10, scoring='accuracy')   #10-fold cross-validation

rfecv = rfecv.fit(X_train, y_train)



print('Optimal number of features :', rfecv.n_features_)

print('Best features :', X_train.columns[rfecv.support_])
from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio='minority')

X_train_res, y_train_res = smote.fit_sample(X_train, y_train)
X_train_res.shape
classifiers = [

    KNeighborsClassifier(),

    SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LogisticRegression(),

    LinearSVC()]



for clf in classifiers:

    name = clf.__class__.__name__

    clf.fit(X_train_res, y_train_res)

    print(clf.score(X_test,y_test), name)
params = {

    'eta': 1,

    'max_depth': 15,

    'objective': 'binary:logistic',

    'lambda' : 3,

    'alpha' : 3

}    

model = xgb.train(params, xgb.DMatrix(X_train, y_train), 100, verbose_eval=50)

predictions = model.predict(xgb.DMatrix(X_test))

survived = [int(round(value)) for value in predictions]

accuracy = accuracy_score(survived, y_test)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
xgb.plot_importance(booster=model)

plt.show()
print(classification_report(survived, y_test))

print('\n')

print(confusion_matrix(survived, y_test))
test_df.isnull().sum().sort_values(ascending=False)
test_df.head()
test_df['Lastname'], test_df['Name'] = test_df['Name'].str.split(',', 1).str

test_df['Nclass'], test_df['Name'] = test_df['Name'].str.split('.', 1).str
test_df = test_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Lastname'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)
#filling missed values in Age and Fare

test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)

test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)
test_df.head()
def people(passenger):

    age, sex = passenger

    return 'child' if age < 15 else sex

    

test_df['People'] = test_df[['Age','Sex']].apply(people, axis=1)
test_df['Nclass'] = test_df['Nclass'].map({' Col': 'Other', ' Rev': 'Other', ' Ms': 'Miss', ' Dr': 'Mr', ' Dona': 'Mrs', ' Master': 'Mr', ' Miss': 'Miss', ' Mrs': 'Mrs', ' Mr': 'Mr',})
test_df.head()
#Add family members

test_df['family_members'] =  test_df['Parch'] + train_df['SibSp']
categories = test_df.select_dtypes(include=['object'])

categories.columns
dummy_df = pd.get_dummies(test_df[categories.columns])

test_df = pd.concat([test_df, dummy_df], axis=1)

test_df = test_df.drop(categories, axis=1)
test_df.isnull().sum().sort_values(ascending=False)
test_df.head()
#prediction with GradientBoostingClassifier

model = GradientBoostingClassifier()

model.fit(X, y)

# make prediction

prediction = model.predict(test_df)

test_df['Survived'] = prediction
test_df.head()
id = full[891:].PassengerId

test = pd.DataFrame( { 'PassengerId': id , 'Survived': prediction } )
test.head()