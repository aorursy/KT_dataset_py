"""

Importing Libraries

"""

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import matplotlib as mp
"""

Importing Machine Learning Libraries

"""

from sklearn.metrics import cohen_kappa_score as MC

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold

from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.metrics import f1_score

from sklearn.naive_bayes import GaussianNB

from sklearn import svm, datasets

from sklearn import linear_model

from sklearn import preprocessing, model_selection

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.model_selection import cross_val_predict

from sklearn import metrics 
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")
train_data.head() # first five columns of training dataset
plt.figure(figsize = (10,5))



corrMatrix = train_data.corr()



sns.heatmap(corrMatrix, annot=True,cmap="Reds")

plt.show()
# Extract Title from Name, store in column and plot barplot

import re



train_data['Title'] = train_data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

sns.countplot(x='Title', data=train_data);

plt.xticks(rotation=45);



"""

Replacing Title Feature with smaller amount of titles

"""

train_data['Title'] = train_data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})

train_data['Title'] = train_data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',

                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')

sns.countplot(x='Title', data=train_data);

plt.xticks(rotation=45);
# Did they have a Cabin?

train_data['Has_Cabin'] = ~train_data.Cabin.isnull()
"""

dropping categorical column that is not needed

"""

classes = train_data['Survived']



"""

Convert features into numeric values

"""

train_data['Sex'] = pd.factorize(train_data['Sex'])[0]

train_data['Title'] = pd.factorize(train_data['Title'])[0]

train_data['Embarked'] = pd.factorize(train_data['Embarked'])[0]



    

new_data = train_data.drop(['Cabin','Ticket','Survived','Name'],axis = 1)



new_data #showing the data after deletion of columns
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

new_data = my_imputer.fit_transform(new_data)
"""

Splitting the Data for Testing and Training

"""

X_train, X_test, y_train, y_test = train_test_split(new_data, classes, test_size=0.25, stratify=classes,random_state=5)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
"""

Linear Support Vector Machines 

"""

svc = Pipeline([("scaler",StandardScaler()),("linear_svc",LinearSVC(C=1, loss= 'hinge'))])



svc.fit(X_train,y_train) # fitting linear SVM to dataset



y_pred_svc = svc.predict(X_test) # making predictions with linear SVM



print(classification_report(y_test,y_pred_svc)) 
"""

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



print(random_grid)



"""
"""

# Using the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()



# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available scores



rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)



# Fit the random search model

rf_random.fit(X_train,y_train)

"""
"""



# view the best parameters from fitting the random search



rf_random.best_params_

"""
"""

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [70, 80, 90, 100],

    'max_features': [2, 3,'auto','sqrt'],

    'min_samples_leaf': [1, 2, 3],

    'min_samples_split': [1, 2, 3],

    'n_estimators': [600, 800, 1000, 1500]

}



# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



# Fit the grid search to the data

grid_search.fit(X_train,y_train)



grid_search.best_params_

"""
"""

Random Forest Classifier

"""

from sklearn.ensemble import RandomForestClassifier 



rf = RandomForestClassifier(max_depth=70,n_estimators = 600,min_samples_split = 3,min_samples_leaf = 2,

                           max_features = 'sqrt',bootstrap = True)

rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)



print(classification_report(y_test,y_pred_rf))
"""

Gradient Boosting Classifier

"""

from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier(random_state=0)

gbc.fit(X_train,y_train)

y_pred_gbc = gbc.predict(X_test)



print(classification_report(y_test,y_pred_gbc))
"""

XGBoost

"""

import xgboost as xgb



xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

xgb_model.fit(X_train, y_train)



y_pred_xgb = xgb_model.predict(X_test)



print(classification_report(y_test,y_pred_xgb))
"""

k-Nearest Neighbor

"""

from sklearn.neighbors import KNeighborsClassifier 



knn = KNeighborsClassifier(n_neighbors = 3)

knn = knn.fit(X_train,y_train)

y_pred_knn = knn.predict(X_test)



print(classification_report(y_test,y_pred_knn))
"""

Gaussian Naive Bayes

"""

gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred_gnb = gnb.predict(X_test)



print(classification_report(y_test,y_pred_gnb))
"""

Linear Discriminant Analysis

"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 



lda=LDA(n_components=None)

fit = lda.fit(X_train,y_train) # fitting LDA to dataset

y_pred_lda=lda.predict(X_test) # predicting with LDA 



print(classification_report(y_test,y_pred_lda))
"""

Decision Tree

"""

trees = tree.DecisionTreeClassifier(max_depth = 3)

trees = trees.fit(X_train, y_train)

y_pred_tree = trees.predict(X_test) 



print(classification_report(y_test,y_pred_tree))
test_data.head() # first 5 rows of test dataset
test_data['Title'] = test_data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

sns.countplot(x='Title', data=test_data);

plt.xticks(rotation=45);



"""

Replacing Title Feature with smaller amount of titles

"""

test_data['Title'] = test_data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})

test_data['Title'] = test_data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',

                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')

sns.countplot(x='Title', data=test_data);

plt.xticks(rotation=45);
"""

Convert features into numeric values

"""

test_data['Sex'] = pd.factorize(test_data['Sex'])[0]

test_data['Title'] = pd.factorize(test_data['Title'])[0]

test_data['Embarked'] = pd.factorize(test_data['Embarked'])[0]



# Did they have a Cabin?

test_data['Has_Cabin'] = ~test_data.Cabin.isnull()



new_test_data = test_data.drop(['Cabin','Ticket','Name'],axis = 1)



new_test_data # Output Dataframe
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

new_test_data = my_imputer.fit_transform(new_test_data)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

new_test_data = sc.fit_transform(new_test_data)
y_pred_final = rf.predict(new_test_data)
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": y_pred_final

    })



# submission.to_csv('submission8.csv', index=False)