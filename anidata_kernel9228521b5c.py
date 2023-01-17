# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.info()
test.info()
train.describe()
#Print Data Shape

print ("Train data shape:", train.shape)

print ("Test data shape:", test.shape)
#Seperate Numerical and Categorical Feature



num_fea = train.dtypes[train.dtypes!='object'].index

print('Numerical features:', len(num_fea))





cat_fea = train.dtypes[train.dtypes=='object'].index

print('Categorical features:', len(cat_fea))
train.head(5)
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(x = "Pclass", hue = "Survived", data = train)
#check for NA values

print (train.isnull().sum())  

print (train.isnull().values.any())

print (train.isnull().values.sum())
# percent of missing "Age" 

print('Percent of missing "Age" records is %.3f%%' %((train['Age'].isnull().sum()/train.shape[0])*100))
ax = train["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)

train["Age"].plot(kind='density', color='teal')

ax.set(xlabel='Age')

plt.xlim(-10,85)

plt.show()
sns.barplot(x='Pclass', y='Survived', data=train)
sns.barplot(x='Sex', y='Survived', data=train)

plt.show()
train['Pclass'].value_counts()
train = train.drop(['Ticket','Name','Cabin'],axis=1)

test = test.drop(['Ticket','Name','Cabin'],axis=1)
#check for NA values

print (train.isnull().sum())  

print (train.isnull().values.any())

print (train.isnull().values.sum())
train['Age'] = train['Age'].fillna(train['Age'].mode()[0])

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
#check for NA values

print (train.isnull().sum())  

print (train.isnull().values.any())

print (train.isnull().values.sum())
from sklearn.model_selection import train_test_split
# Putting feature variable to X

X = train.drop(['Survived','PassengerId'], axis=1)



X.head()
# Putting response variable to y

y = train['Survived']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X_train[['Age','SibSp','Parch','Fare']] = scaler.fit_transform(X_train[['Age','SibSp','Parch','Fare']])



X_train.head()
### Checking the Churn Rate

survive = (sum(train['Survived'])/len(train['Survived'].index))*100

survive
# Importing matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Let's see the correlation matrix 

plt.figure(figsize = (20,10))        # Size of the figure

sns.heatmap(train.corr(),annot = True)

plt.show()
from sklearn import preprocessing





# encode categorical variables using Label Encoder



# select all categorical variables

train_categorical = train.select_dtypes(include=['object'])

train_categorical.head()
# apply Label encoder to df_categorical



le = preprocessing.LabelEncoder()

train_categorical = train_categorical.apply(le.fit_transform)

train_categorical.head()
# concat df_categorical with original df

train = train.drop(train_categorical.columns, axis=1)

train = pd.concat([train, train_categorical], axis=1)

train.head()
# look at column types

train.info()
# convert target variable income to categorical

train['Survived'] = train['Survived'].astype('category')
# Importing train-test-split 

from sklearn.model_selection import train_test_split
# Putting feature variable to X

X = train.drop('Survived',axis=1)



# Putting response variable to y

y = train['Survived']
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.30, 

                                                    random_state = 99)

X_train.head()
# Importing decision tree classifier from sklearn library

from sklearn.tree import DecisionTreeClassifier



# Fitting the decision tree with default hyperparameters, apart from

# max_depth which is 5 so that we can plot and read the tree.

dt_default = DecisionTreeClassifier(max_depth=5)

dt_default.fit(X_train, y_train)
# Let's check the evaluation metrics of our default model



# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Making predictions

y_pred_default = dt_default.predict(X_test)



# Printing classification report

print(classification_report(y_test, y_pred_default))
# Printing confusion matrix and accuracy

print(confusion_matrix(y_test,y_pred_default))

print(accuracy_score(y_test,y_pred_default))
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(1, 40)}



# instantiate the model

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(5, 200, 20)}



# instantiate the model

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# GridSearchCV to find optimal min_samples_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_split': range(5, 200, 20)}



# instantiate the model

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

tree.fit(X_train, y_train)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# Create the parameter grid 

param_grid = {

    'max_depth': range(3,5,10),

    'min_samples_leaf': range(30,50,100),

    'min_samples_split': range(50, 100, 150),

    'criterion': ["entropy", "gini"]

}



n_folds = 5



# Instantiate the grid search model

dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 

                          cv = n_folds, verbose = 1)



# Fit the grid search to the data

grid_search.fit(X_train,y_train)
# cv results

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results
# printing the optimal accuracy score and hyperparameters

print("best accuracy", grid_search.best_score_)

print(grid_search.best_estimator_)
# model with optimal hyperparameters

clf_gini = DecisionTreeClassifier(criterion = "gini", 

                                  random_state = 100,

                                  max_depth=3, 

                                  min_samples_leaf=30,

                                  min_samples_split=50)

clf_gini.fit(X_train, y_train)
# accuracy score

clf_gini.score(X_test,y_test)
# classification metrics

from sklearn.metrics import classification_report,confusion_matrix

y_pred = clf_gini.predict(X_test)

print(classification_report(y_test, y_pred))
# confusion matrix

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))
# model with optimal hyperparameters

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(criterion = "gini", 

                                  random_state = 100,

                                  max_depth=7, 

                                  min_samples_leaf=30,

                                  min_samples_split=50,

                                  n_estimators=500)

rfc.fit(X_train, y_train)
# accuracy score

rfc.score(X_test,y_test)
# classification metrics

from sklearn.metrics import classification_report,confusion_matrix

y_pred = rfc.predict(X_test)

print(classification_report(y_test, y_pred))
# confusion matrix

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))
# GridSearchCV to find optimal n_estimators

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5

# parameters to build the model on

parameters = {'max_depth': range(2, 20, 5)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",return_train_score=True)

rf.fit(X_train, y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head(10)
# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_max_depth"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# GridSearchCV to find optimal n_estimators

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'n_estimators': range(100, 1500, 400)}



# instantiate the model (note we are specifying a max_depth)

rf = RandomForestClassifier(max_depth=4)





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",return_train_score=True)

rf.fit(X_train, y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with n_estimators

plt.figure()

plt.plot(scores["param_n_estimators"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_n_estimators"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("n_estimators")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [4,8,10],

    'min_samples_leaf': range(100, 400, 200),

    'min_samples_split': range(200, 500, 300),

    'n_estimators': [100,200, 500], 

    'max_features': [2,5]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1,verbose = 1)
# Fit the grid search to the data

grid_search.fit(X_train, y_train)
# printing the optimal accuracy score and hyperparameters

print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
# model with the best hyperparameters

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(bootstrap=True,

                             max_depth=7,

                             min_samples_leaf=100, 

                             min_samples_split=200,

                             max_features=5,

                             n_estimators=500)
# fit

rfc.fit(X_train,y_train)
# predict

predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))
from sklearn.ensemble import ExtraTreesClassifier

model1 = ExtraTreesClassifier()

model1.fit(X_train, y_train)



y_pred = model1.predict(X_test)



print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))
# Cross validate model with Kfold stratified cross val

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms 

random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING



# Adaboost

DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsadaDTC.fit(X_train,y_train)



ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_
#ExtraTrees 

ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3,5],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsExtC.fit(X_train,y_train)



ExtC_best = gsExtC.best_estimator_



# Best score

gsExtC.best_score_
# RFC Parameters tunning 

RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 5],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsRFC.fit(X_train,y_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
RFC_best
# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1]

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsGBC.fit(X_train,y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
### SVC classifier

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsSVMC.fit(X_train,y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
# model with the best hyperparameters

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(bootstrap=False,

                             

                             min_samples_leaf=3, 

                             min_samples_split=2,

                             max_features=3,

                             n_estimators=300)
# fit

rfc.fit(X_train,y_train)



# predict

predictions = rfc.predict(X_test)

print(classification_report(y_test,predictions))



print(accuracy_score(y_test,predictions))
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,y_train,cv=kfold)
votingC = VotingClassifier(estimators=[('rfc', RFC_best),('extc', ExtC_best),('gbc',GBC_best)], voting='soft', n_jobs=-1)



votingC = votingC.fit(X_train, y_train)
predictions = votingC.predict(X_test)



print(classification_report(y_test,predictions))



print(accuracy_score(y_test,predictions))
# apply Label encoder to df_categorical



from sklearn import preprocessing





# encode categorical variables using Label Encoder



# select all categorical variables

test_categorical = test.select_dtypes(include=['object'])

test_categorical.head()



le_test = preprocessing.LabelEncoder()

test_categorical = test_categorical.apply(le_test.fit_transform)

test_categorical.head()
# concat df_categorical with original df

test = test.drop(test_categorical.columns, axis=1)

test = pd.concat([test, test_categorical], axis=1)

test.head()
test['Age'] = test['Age'].fillna(test['Age'].mode()[0])

test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])

test['Fare'] = test['Fare'].fillna(test['Fare'].mode()[0])
test.isnull().sum()
test.head()
#Predict in Actual Test datasets(test.csv) with Final Trained Model

result = rfc.predict(test)
result
# IF we want to submit the model in Kaggle we can generate the below dataframe With Label and ImageID



submission = pd.Series(result,name="Survived")


submission
#Map Image Label with test datasets

submission = pd.concat([pd.Series(range(892,1310),name = "PassengerId"),submission],axis = 1)

submission.head()
#converting into CSV file 



submission.to_csv("final_submission.csv",index=False)
#Rechecking if the generated csv file is mathcing with submission Dataframe

final_submission = pd.read_csv("final_submission.csv")
# Check the finally submitted dataset

final_submission