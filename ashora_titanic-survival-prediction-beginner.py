# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import numpy as np     numpy is already imported

# import pandas as pd    pandas is already imported



# import seaborn to visualize data that built on top of matplotlib

import seaborn as sns



# import pipeline to make machine learning pipeline to overcome data leakage problem

from sklearn.pipeline import Pipeline



# import StandardScaler to Column Standardize the data

# many algorithm assumes data to be Standardized

from sklearn.preprocessing import StandardScaler



# train_test_split is used to split the data into train and test set of given data

from sklearn.model_selection import train_test_split



# KFold is used for defining the no.of folds for Cross Validation

from sklearn.model_selection import KFold



# cross_val_score is used to find the score on given model and KFlod

from sklearn.model_selection import cross_val_score



# used for Hyper-parameter

from sklearn.model_selection import GridSearchCV



# classification report show the classification report

# precision, recall, f1-score

from sklearn.metrics import classification_report



# accuracy score is also a metrics to judge the model

# mostly used for Balanced dataset

# not better for Imabalanced dataset

from sklearn.metrics import accuracy_score



# confusion matrix show the comparision between actual label and predicted label of data

# mostly used for Binary classification

from sklearn.metrics import confusion_matrix



# importing different algorithms to train our data and find better model among all algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier



# KNN is slow algorithm for runtime b'z it doesn't learn anything at the time of fitting the model

# KNN just stores every datapoint and find K Nearest Neighbor and

# among all nearest neighbor whichever has high no.of points that is the label of that query point

# Because it store every datapoint in memory to predict the label in runtime, It is not better for large data

# KNN is less used in industry because it is not good for Low "latency"(time to predict the label of given query point) systems like SearchEngines

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



# Ensembles generally group more than one model to give better model

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



# import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.shape
train.head()
train.tail()
train.describe().transpose()
train.groupby('Survived').size()
train.info()
train.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')
# setting the style of axes(plotting area) as 'whitegrid'.

sns.set_style('whitegrid')



# let's count the #person survived

sns.countplot(x='Survived',data= train, palette='RdBu_r')
# count # survived person catergorised by 'Sex'

sns.countplot(x='Survived', hue='Sex',data= train, palette='RdBu_r')
sns.countplot(x='Survived', hue='Pclass',data= train, palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=False,bins=30,color='darkred')
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=train)
sns.countplot(x='Survived',hue='SibSp',data=train)
train['Fare'].hist(color='green',bins=35)
sns.boxplot(x='Pclass', y='Age',data=train,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)



# we have also to clean the test data

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.info()
train.drop('Cabin',axis=1,inplace=True)

# drop Cabin from test data also

test.drop('Cabin',axis=1,inplace=True)
train.head()
train.isnull().sum()
train.dropna(inplace=True)

test.isnull().sum()
test[test['Fare'].isnull()]
# set value of 'Fare' at index Location 152 as 50

# we can't delete the row because it we have to submit my prediction values and that should be equal to rows that is being given

test.set_value(152,'Fare',50)
train.isnull().sum()

test.isnull().sum()
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True) # getting dummy of 'Sex' column

embark = pd.get_dummies(train['Embarked'],drop_first=True) # getting dummy of 'Embarked'
# for test data

sex_test = pd.get_dummies(test['Sex'],drop_first=True) # getting dummy of 'Sex' column

embark_test = pd.get_dummies(test['Embarked'],drop_first=True) # getting dummy of 'Embarked'
# drop columns: 'Sex', 'Embarked', 'Name','Ticket','PassengerId'

train.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)



# for test

test.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)

# for train

train = pd.concat([train,sex,embark],axis=1)

# for test

test = pd.concat([test,sex_test,embark_test],axis=1)
train.head()
# let's also see test data header part

test.head()
predictors = train.drop(['Survived'],axis=1)
predictors.head() # Now It has no label, It is pure training data without labels
target = train['Survived']
target.head()
# Create a Validation dataset

# .values returns numpy.ndarray

# we are using array instead of dataframe to train model because

# array is faster to compute instead of dataframe



X = predictors.values

Y = target.values

validation_size = 0.20

seed = 42 

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=validation_size,random_state=seed)
type(X)
# code source: machinelearningmastery.com

# Spot-check Algorithms

models = []



# In LogisticRegression set: solver='lbfgs',multi_class ='auto', max_iter=10000 to overcome warning

models.append(('LR',LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=10000)))

models.append(('LDA',LinearDiscriminantAnalysis()))

models.append(('KNN',KNeighborsClassifier()))

models.append(('CART',DecisionTreeClassifier()))

models.append(('NB',GaussianNB()))

models.append(('SVM',SVC(gamma='scale')))



# evaluate each model in turn

results = []

names = []



for name, model in models:

    # initializing kfold by n_splits=10(no.of K)

    kfold = KFold(n_splits = 10, random_state=seed)

    

    # cross validation score of given model using cross-validation=kfold

    cv_results = cross_val_score(model,X_train,y_train,cv=kfold, scoring="accuracy")

    

    # appending cross validation result to results list

    results.append(cv_results)

    

    # appending name of algorithm to names list

    names.append(name)

    

    # printing cross_validation_result's mean and standard_deviation

    print(name, cv_results.mean()*100.0, "(",cv_results.std()*100.0,")")
#Let's Compare by plotting it

figure = plt.figure()

figure.suptitle('Algorithm Comparison')

ax = figure.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)
# test options and evaluation matrix

num_folds=10

seed=42

scoring='accuracy'
# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
# source of code: machinelearningmastery.com

# Standardize the dataset

pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=10000))])))

pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA',LinearDiscriminantAnalysis())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsClassifier())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeClassifier())])))

pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',GaussianNB())])))

pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC(gamma='scale'))])))

results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())

    print(msg)
# Compare Algorithms

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# Tune scaled SVM

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]

kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

param_grid = dict(C=c_values, kernel=kernel_values)

model = SVC()

kfold = KFold(n_splits=num_folds, random_state=seed)

grid = GridSearchCV(estimator= model, param_grid=param_grid, scoring=scoring,cv=kfold)

grid_result = grid.fit(rescaledX,y_train)

print("Best: %f using %s"% (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds= grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']



for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r"%(mean,stdev,param))
# code source: https://www.machinelearningmastery.com by jason Brownlee

# ensembles

ensembles = []

ensembles.append(('AB', AdaBoostClassifier()))

ensembles.append(('GBM', GradientBoostingClassifier()))

ensembles.append(('RF', RandomForestClassifier()))

ensembles.append(('ET', ExtraTreesClassifier()))

results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)

    print(msg)
# Compare Algorithms

fig = plt.figure()

fig.suptitle('Ensemble Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# this accuracy score shows the accuracy of what we have splitted earlier

# this accuracy is not using test dataset, right.



# fitting the StandardScaler() to calculate mean and standard deviation

scaler = StandardScaler().fit(X_train)



# transform X_train according to calculated mean and standard deviation

rescaledX = scaler.transform(X_train)



# making model SVC(Support Vector Classifier)

model = SVC(C=2.0,kernel='rbf')



# fitting the model

model.fit(rescaledX,y_train)







# estimated accuracy on validation dataset

rescaledValidationX = scaler.transform(X_test)

predictions = model.predict(rescaledValidationX)

print(accuracy_score(y_test,predictions)*100)
# let's transform the test data

# let's again fit the model using complete train data.



scaler = StandardScaler().fit(X)



rescaledX = scaler.transform(X)



# create model SVC(Support Vector Classifier)

model = SVC(C=2.0,kernel='rbf')



# fit the model

model.fit(rescaledX,Y)



transformed_test = scaler.transform(test)
predictions = model.predict(transformed_test)
predictions
# importing XGBClassifier

from xgboost import XGBClassifier
new_model = XGBClassifier(n_estimators = 1000, learning_rate=0.05)
new_model.fit(X_train,y_train,

             early_stopping_rounds = 5,

             eval_set = [(X_test,y_test)],

             verbose = False)
xgb_predictions = new_model.predict(X_test)
xgb_predictions
accuracy_score(xgb_predictions,y_test)
# passing the array(using test.values, .values returns an array) not dataframe

test_predictions = new_model.predict(test.values)
test_predictions
xgb_submission= pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
xgb_submission.head(15)
xgb_submission['Survived'] = test_predictions
xgb_submission.head(15)
xgb_submission.to_csv('gender_submission.csv',index=False)