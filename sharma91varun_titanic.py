# Importing Libraries



%matplotlib inline

import pandas as pd

import numpy as np

import re as re
#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8





#Ignore Warnings

import warnings

warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_raw = pd.read_csv("/kaggle/input/titanic/train.csv")

train = data_raw.copy(deep = True)

test = pd.read_csv("/kaggle/input/titanic/test.csv")



fulldata = [train, test]



train.head()
train.info()
test.info()
train.sample(10)
# Cleaning the dataset



print('Null values in Train: \n', train.isnull().sum())

print("-"*20)



print('Null values in Test: \n', test.isnull().sum())

print("-"*20)
train.describe(include = 'all')
print (train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean())
print (train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean())
#Feature Enginnering



for dataset in fulldata:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']+1

    

print (train[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in fulldata:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

print (train[['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean())
print (train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).count())
for dataset in fulldata:

    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])

    

print (train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).count())
for dataset in fulldata:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    dataset['CategoricalFare'] = pd.qcut(dataset['Fare'], 4)



print (train[['CategoricalFare','Survived']].groupby(['CategoricalFare'], as_index=False).mean())
for dataset in fulldata:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)

    

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

    dataset['CategoricalAge'] = pd.cut(dataset['Age'], 5)



print (train[['CategoricalAge','Survived']].groupby(['CategoricalAge'], as_index=False).mean())
def get_title(name):

    title_search = re.search('([A-Za-z]+)\.', name)

    

    if title_search:

        return title_search.group(1)

    return ""



for dataset in fulldata:

    dataset['Title'] = dataset['Name'].apply(get_title)

    

print (pd.crosstab(train['Title'], train['Sex']))
for dataset in fulldata:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \

                    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

print (train[['Title','Survived']].groupby(['Title'], as_index=False).mean())
for dataset in fulldata:

    dataset['Cabin'] = dataset['Cabin'].fillna(0)

    

for dataset in fulldata:

    dataset['HasCabin'] = 0

    dataset.loc[dataset['Cabin'] == 0, 'HasCabin'] = 1

    

print (train[['HasCabin','Survived']].groupby(['HasCabin'], as_index=False).mean())
train.info()

print("-"*50)

test.info()
train.head()
# for future revisions



# from sklearn.preprocessing import LabelEncoder



# label = LabelEncoder()



# for dataset in fulldata:

#     dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

#     dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

#     dataset['Title_Code'] = label.fit_transform(dataset['Title'])

#     dataset['Fare_Code'] = label.fit_transform(dataset['CategoricalFare'])

#     dataset['Age_Code'] = label.fit_transform(dataset['CategoricalAge'])
col = ['Pclass', 'Sex', 'CategoricalAge', 'CategoricalFare', 'Title', 'HasCabin']

train_data = pd.get_dummies(train[col], drop_first = True)

test_data = pd.get_dummies(test[col], drop_first = True)
train_data_col = train_data.columns.tolist()

test_data_col = test_data.columns.tolist()
y_train_data = train['Survived']
train_data_col
X_train_data = train_data.values

X_test_data = test_data.values

y_train_data = y_train_data.values
X_train_data
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train_data = sc_X.fit_transform(X_train_data)
X_test_data = sc_X.transform(X_test_data)
X_train_data
# Below code beautifully shows accuracy across various methodologies 

# Borrowed from Sina's work at https://www.kaggle.com/sinakhorami/titanic-best-working-classifier





import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

	AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log 	 = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



X = X_train_data

y = y_train_data



acc_dict = {}



for train_index, test_index in sss.split(X, y):

	X_train, X_test = X[train_index], X[test_index]

	y_train, y_test = y[train_index], y[test_index]

	

	for clf in classifiers:

		name = clf.__class__.__name__

		clf.fit(X_train, y_train)

		train_predictions = clf.predict(X_test)

		acc = accuracy_score(y_test, train_predictions)

		if name in acc_dict:

			acc_dict[name] += acc

		else:

			acc_dict[name] = acc



for clf in acc_dict:

	acc_dict[clf] = acc_dict[clf] / 10.0

	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

	log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

candidate_classifier = RandomForestClassifier()

candidate_classifier.fit(X_train_data, y_train_data)



submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = candidate_classifier.predict(X_test_data)

submission.head()
#submission.to_csv("submit_2.csv", index = False)