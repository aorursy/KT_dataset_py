import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



#data Preparations

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, log_loss



# machine learning modeling

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression, SGDClassifier



# Model Tunning

from sklearn.model_selection import RandomizedSearchCV
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')# train data 

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')# test data

full_data = [train_data + test_data]
print(train_data.Age.isna().value_counts(), '\n') 

print(train_data.Cabin.isna().value_counts(),'\n') 

print(train_data.Embarked.isna().value_counts(),' \n')
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean()
print(train_data[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean())
# Data Cleansing        
# train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data.head(10)
test_data.head(10)
train_data['Family_size'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['Family_size'] = test_data['SibSp'] + test_data['Parch'] + 1

print(train_data[['Family_size', 'Survived']].groupby(['Family_size'], as_index = False).mean())
# train_data.Embarked.value_counts()

# print('\n')

# train_data.Cabin.value_counts()

test_data.Embarked.value_counts()
train_data['Embarked'] = train_data['Embarked'].fillna('S')

test_data['Embarked'] = test_data['Embarked'].fillna('S')

# Filing mising value by random

avg_age = train_data['Age'].mean()

age_std = train_data['Age'].std()

age_null_count = train_data['Age'].isnull().sum()



age_null_random_list = np.random.randint(avg_age - age_std, avg_age + age_std, size = age_null_count)



train_data['Age'] [np.isnan(train_data['Age'])] = age_null_random_list

train_data['Age'] = train_data['Age'].astype(int)
# Filing mising value by random

avg_age = test_data['Age'].mean()

age_std = test_data['Age'].std()

age_null_count = test_data['Age'].isnull().sum()



age_null_random_list = np.random.randint(avg_age - age_std, avg_age + age_std, size = age_null_count)



test_data['Age'] [np.isnan(test_data['Age'])] = age_null_random_list

test_data['Age'] = test_data['Age'].astype(int)
train_data['CategoricalAge'] = pd.cut(train_data['Age'], 5)
test_data['CategoricalAge'] = pd.cut(test_data['Age'], 5)
print(train_data[['CategoricalAge', 'Survived']].groupby(train_data['CategoricalAge'], as_index = False).mean())
# getting title

def getTitle(name):

    title_serach = re.search(' ([A-Za-z]+)\.', name)

    if title_serach:

        return title_serach.group(1)

    return ''



train_data['Title'] = train_data['Name'].apply(getTitle)

print()

    

print(pd.crosstab(train_data.Title, train_data.Sex))
test_data['Title'] = test_data['Name'].apply(getTitle)

train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')

train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')

train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')



print (train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
test_data['Title'] = test_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')

test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')

test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')

# handeling the rest of missing value in the datset

train_data.isnull().any()

print('\n')

train_data.info()

        
# train_data.Fare.value_counts()

train_data['CategoricalFare'] = pd.cut(train_data['Fare'], 4)



print(train_data[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index = False).mean())
test_data['CategoricalFare'] = pd.cut(test_data['Fare'], 4)
drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','CategoricalAge', 'CategoricalFare']

train_data = train_data.drop(drop_columns, axis=1)
test_data = test_data.drop(drop_columns, axis = 1)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
def Encoder(data):

    for col in data:

        if data[col].dtypes == 'object':

            data[col] = le.fit_transform(data[col])

    return data
train_data = Encoder(train_data)
test_data = Encoder(test_data)

# test_data.head(10)
colormap = plt.cm.RdBu

plt.figure(figsize = (16,14))

plt.title('Pearson Correlation of Features', y = 1.05, size = 14)

sns.heatmap(train_data.astype(float).corr(), linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
distributions = sns.pairplot(train_data[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked', u'Family_size', u'Title']], 

                             hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
#Modeling 
X = train_data.iloc[:, 1:]

y = train_data.iloc[:, :1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Hyperparameter Grid search for the models
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

print(random_grid)
# # Use the random grid to search for best hyperparameters

# # First create the base model to tune

# rf = RandomForestClassifier()

# # Random search of parameters, using 5 fold cross validation, 

# # search across 100 different combinations, and use all available cores

# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

# # Fit the random search model

# rf_random.fit(X_train, y_train)

# # print(rf_random.best_params_)
# def evaluate(model, test_features, test_labels):

#     predictions = model.predict(test_features)

#     errors = abs(predictions - test_labels)

#     mape = 100 * np.mean(errors / test_labels)

#     accuracy = 100 - mape

#     print('Model Performance')

#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

#     print('Accuracy = {:0.2f}%.'.format(accuracy))

    

#     return accuracy

# base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)

# base_model.fit(X_train, y_train)

# base_accuracy = evaluate(base_model, X_test, y_test)
classifiers = [KNeighborsClassifier(5),

               SVC( kernel = 'linear',C = 0.025),

               DecisionTreeClassifier(),

               RandomForestClassifier(n_estimators= 800,  min_samples_split = 10, min_samples_leaf = 4, max_features = 'sqrt', max_depth = 100, bootstrap = True),

               AdaBoostClassifier(n_estimators = 500, learning_rate = 0.75),

               GradientBoostingClassifier(n_estimators = 500, 

                                        #'max_features': 0.2,

                                        max_depth =  5,

                                        min_samples_leaf = 2,

                                        verbose = 0),

               SGDClassifier(max_iter=1000, tol=1e-3),

               xgb.XGBClassifier(max_depth=3, n_estimators=500, learning_rate=0.05),

               GaussianNB(),

               MLPClassifier(),

               ExtraTreesClassifier(n_jobs = -1, n_estimators = 500, max_features = 0.5, max_depth = 8, min_samples_leaf = 2, verbose = 0),

               LinearDiscriminantAnalysis(),#LDA, A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.The fitted model can also be used to reduce the dimensionality of the input by projecting it to the most discriminative directions.

               QuadraticDiscriminantAnalysis(),#A classifier with a quadratic decision boundary

               LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)
accuracy_dict = {}



for clf in classifiers:

        name = clf.__class__.__name__

        clf.fit(X_train, y_train)

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        

        if name in accuracy_dict:

            accuracy_dict[name] += acc

        else:

            accuracy_dict[name] = acc

            

for clf in accuracy_dict:

    accuracy_dict[clf] = accuracy_dict[clf] 

    log_entry = pd.DataFrame([[clf, accuracy_dict[clf]]], columns = log_cols)

    log = log.append(log_entry)

    

    

plt.xlabel('Accuracy')

plt.title('Classifier_Accuracy')

sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
# #Random Forest

# rf_params = {'n_estimators': 800, 

#              'min_samples_split': 10, 

#              'min_samples_leaf': 4, 

#              'max_features': 'sqrt', 

#              'max_depth': 100, 

#              'bootstrap': True}



# # Extra Trees Parameters

# et_params = {

#             'n_jobs': -1,

#             'n_estimators':500,

#             #'max_features': 0.5,

#             'max_depth': 8,

#             'min_samples_leaf': 2,

#             'verbose': 0}



# # AdaBoost parameters

# ada_params = {

#             'n_estimators': 500,

#             'learning_rate' : 0.75}



# # Gradient Boosting parameters

# gb_params = {

#             'n_estimators': 500,

#              #'max_features': 0.2,

#             'max_depth': 5,

#             'min_samples_leaf': 2,

#             'verbose': 0}



# # Support Vector Classifier parameters 

# svc_params = {

#             'kernel' : 'linear',

#             'C' : 0.025}
# Create 5 objects that represent our 4 models

# rf = Classifier(clf=RandomForestClassifier, seed=SEED, params=rf_params)

# et = Classifier(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

# ada = Classifier(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

# gb = Classifier(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

# svc = Classifier(clf=SVC, seed=SEED, params=svc_params)
test_data['Fare'] = test_data['Fare'].fillna(7.75)
#Predition

selected_classifier = SVC( kernel = 'linear',C = 0.025)

selected_classifier.fit(X_train, y_train)

prediction = selected_classifier.predict(test_data)
full_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')# test data

output = pd.DataFrame({'PassengerId': full_test_data.PassengerId, 'Survived': prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
# test_data.info()