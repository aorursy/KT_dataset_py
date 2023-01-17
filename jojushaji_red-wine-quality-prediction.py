# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Loading dataset

wine = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
#lets check the dataset

wine.head()

print(wine.info())

#no missing values
catgorical_feat=[features for features in wine.columns if wine[features].nunique()<10]

catgorical_feat

print(wine.quality.value_counts())

print(wine.quality.value_counts(normalize=True))

sns.countplot(x='quality',data=wine)


wine.hist(figsize = (10,10),color="b",bins=40,alpha=1)

wine.describe()
wine.columns
cont_feat=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol', 'quality']




wine[cont_feat].plot.box(figsize=(20,10))



plt.show()

#there are also some outliers also
sns.pairplot(wine,hue='quality')
plt.figure(figsize = (18,12))

sns.heatmap(wine.corr(), annot = True, cmap = "RdYlGn")



plt.show()
#Here we see that fixed acidity does not give any specification to classify the quality.

fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)
#Here we see that its quite a downing trend in the volatile acidity as we go higher the quality 

fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)
#Composition of citric acid go higher as we go higher in the quality of the wine

fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'citric acid', data = wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'residual sugar', data = wine)
#Composition of chloride also go down as we go higher in the quality of the wine

fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'chlorides', data = wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)
#Sulphates level goes higher with the quality of wine

fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'sulphates', data = wine)
#Alcohol level also goes higher as te quality of wine increases

fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'alcohol', data = wine)
#Making binary classificaion for the response variable.

#Dividing wine as good and bad by giving the limit for the quality

bins = (2, 6.5, 8)

group_names = ['bad', 'good']

wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Now lets assign a labels to our quality variable

label_quality = LabelEncoder()
#Bad becomes 0 and good becomes 1 

wine['quality'] = label_quality.fit_transform(wine['quality'])
wine['quality'].value_counts()
sns.countplot(wine['quality'])
#Now seperate the dataset as response variable and feature variabes

X = wine.drop('quality', axis = 1)

y = wine['quality']
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

%matplotlib inline

#Train and Test splitting of data 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#Applying Standard scaling to get optimized result

sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

clf = LogisticRegression()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
from sklearn.model_selection import KFold

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 15)

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# kNN Score

round(np.mean(score)*100, 2)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# decision tree Score

round(np.mean(score)*100, 2)
from sklearn.ensemble import RandomForestClassifier

rnd = RandomForestClassifier(n_estimators=45)

scoring = 'accuracy'

score = cross_val_score(rnd, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
from sklearn.svm import SVC

clf = SVC()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)

from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)

{'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
# Use the random grid to search for best hyperparameters

# First create the base model to tune

#rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

#rf_random.fit(X_train,y_train)
#rf_random.best_params_
#clf = RandomForestClassifier(n_estimators=45)

#scoring = 'accuracy'

#score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

#print(score)
rnd=RandomForestClassifier(n_estimators=50)

rnd.fit(X_train,y_train)

#test_data = test.drop( "Loan_ID", axis=1).copy()

prediction = rnd.predict(X_test)
print(classification_report(y_test, prediction))
rfc_eval = cross_val_score(estimator = rnd, X = X_train, y = y_train, cv = 6)

rfc_eval.mean()