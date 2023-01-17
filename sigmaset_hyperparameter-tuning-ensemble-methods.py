import numpy as np

import pandas as pd

import matplotlib.pyplot as plt; plt.rcdefaults()



# Feature Engineering

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split



# Classification Models

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier



# Hyperparameter Tuning

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold



# Performance Measures

from sklearn.metrics import accuracy_score



# Global Variables

rnd_state = 42

skfold = StratifiedKFold(n_splits=5)
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

train_data.head()
# Stratified sampling is adopted from https://github.com/ageron/handson-ml



train_data["Age_categories"] = np.ceil(train_data["Age"]/1.5) 

# Dividing the Fare by 1.5 to limit the number of categories and rounding up with ceil to have discrete categories

train_data["Age_categories"].where (train_data["Age_categories"] <2, 2.0, inplace=True)

train_data["Age_categories"].where (train_data["Age_categories"] >50, 50.0, inplace=True)

#keeping only the categories younger than 5o years and older than 2 years and merging the other categories into category 50 and 2



split = StratifiedShuffleSplit (n_splits=1, test_size=0.2, random_state= rnd_state)

for train_index, test_index in split.split(train_data, train_data["Age_categories"]):

    strat_train_set = train_data.loc[train_index]

    strat_test_set = train_data.loc[test_index]



# Now remove the Age_categories so that the data is back to its original state

for set_ in (strat_train_set, strat_test_set):

    set_.drop("Age_categories", axis=1, inplace=True)  



# We are interested in predicting the Survival, which means Survival is the target feature and needs to be dropped from the validation set 

strat_train_set_p = strat_train_set.drop ("Survived", axis=1)

survival_labels = strat_train_set["Survived"].copy()



strat_test_set_p = strat_test_set.drop ("Survived", axis=1)

validation_labels = strat_test_set["Survived"].copy()
strat_train_set_p.info() 
def extract_title (dataset):

    dataset['Title'] = dataset.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

    # normalize the titles

    normalized_titles = {

        "Capt":       "other",

        "Col":        "other",

        "Major":      "other",

        "Jonkheer":   "other",

        "Don":        "Mr",

        "Sir" :       "Mr",

        "Dr":         "other",

        "Rev":        "other",

        "the Countess":"other",

        "Dona":       "Mrs",

        "Mme":        "Mrs",

        "Mlle":       "Miss",

        "Ms":         "Mrs",

        "Mr" :        "Mr",

        "Mrs" :       "Mrs",

        "Miss" :      "Miss",

        "Master" :    "Master",

        "Lady" :      "Mrs",

        "NaN" :       "other"  

    }

    # map the titles to new categories 

    dataset.Title = dataset.Title.map(normalized_titles)

    return dataset
def data_cleaner (dataset):

    # you can add a list of feature engineering functions here

    # For example, I created 6 age buckets for my age values and used the age as a categorical attribute but it didnt help my accuracy much so I dropped it from this notebook.

    

    # Extracting titles from the name

    dataset = extract_title (dataset)

    return dataset
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_familySize = False): # no *args or **kargs

        self.add_familySize = add_familySize

    def fit(self, X, y=None):

        return self  # nothing else to do

    def transform(self, X, y=None):

        familySize = X[:, Sibsp_ix] + X[:, Parch_ix]

        if self.add_familySize:

            return np.c_[X, familySize]

        else:

            return np.c_[X]

        

Sibsp_ix, Parch_ix = 2, 3 # column index from the dataset, index starts from 0



# I created one preprocessing pipelines for processing both numeric and categorical data.

numeric_features = ['Age','Fare', 'SibSp', 'Parch']

numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('attribs_adder', CombinedAttributesAdder(add_familySize = False)),

    ('scaler', StandardScaler())

])



categorical_features = ['Title', 'Sex', 'Pclass', 'Embarked']

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='S')),

    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)

    ])
train_set_prepared = preprocessor.fit_transform(data_cleaner(strat_train_set_p))



validation_set_prepared = preprocessor.fit_transform(data_cleaner(strat_test_set_p))
train_set_prepared.shape
log_clf = LogisticRegression(random_state=rnd_state, solver='lbfgs')

log_clf.fit(train_set_prepared, survival_labels)
log_pred = log_clf.predict(validation_set_prepared).astype(int)
log_cross_scores = cross_val_score(log_clf, train_set_prepared, survival_labels, cv=skfold, scoring="accuracy")

log_scores = (log_cross_scores.mean() + accuracy_score (log_pred, validation_labels))/2

log_scores
sgd_clf = SGDClassifier(max_iter=60, penalty = None, eta0=0.1, random_state=rnd_state, tol =1e-3)

sgd_clf.fit(train_set_prepared, survival_labels)
sgd_cross_scores = cross_val_score(sgd_clf, train_set_prepared, survival_labels, cv=skfold, scoring="accuracy")

sgd_pred = sgd_clf.predict(validation_set_prepared).astype(int)

sgd_scores = (sgd_cross_scores.mean() + accuracy_score (sgd_pred, validation_labels))/2

sgd_scores
#svm_clf = LinearSVC(loss="hinge", random_state=42)

#svm_clf = SVC (kernel="poly", degree=3, coef0=1, C=5)



svm_clf = SVC (gamma='auto')

svm_clf.fit (train_set_prepared, survival_labels)
svm_cross_scores = cross_val_score(svm_clf, train_set_prepared, survival_labels, cv=skfold, scoring="accuracy")

svm_pred = svm_clf.predict(validation_set_prepared).astype(int)

svm_scores = (svm_cross_scores.mean() + accuracy_score (svm_pred, validation_labels))/2

svm_scores
# # This takes some time to run, uncomment the section to run

# param_grid = {

#     'C':[1,10,100,1000],

#     'gamma':[1,0.1,0.001,0.0001], 

#     'kernel':['linear','rbf']}



# grid_search_svm = GridSearchCV(SVC(), param_grid, refit = True, verbose=2)



# grid_search_svm.fit (titanic_train_prepared, Survival_labels)
# grid_search_svm.best_params_
svm_grid_clf = SVC (kernel="rbf", gamma=0.1, C=10) #best parameters after grid search

svm_grid_clf.fit (train_set_prepared, survival_labels)



svm_grid_cross_scores = cross_val_score(svm_grid_clf, train_set_prepared, survival_labels, cv=skfold, scoring="accuracy")

svm_pred = svm_grid_clf.predict(validation_set_prepared).astype(int)

svm_grid_scores = (svm_grid_cross_scores.mean() + accuracy_score (svm_pred, validation_labels))/2

svm_grid_scores
forest_clf = RandomForestClassifier(random_state=rnd_state, n_estimators=10)

forest_clf.fit (train_set_prepared, survival_labels)
forest_clf.feature_importances_
param_grid = [

    # try 12 (3×4) combinations of hyperparameters

    {'n_estimators': [200,  250], 'max_features': [10, 17]},

    # then try 6 (2×3) combinations with bootstrap set as False

    {'bootstrap': [False], 'n_estimators': [200, 250], 'max_features': [10, 12, 17]},

  ]



# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 

grid_forest_clf = GridSearchCV(forest_clf, param_grid, cv=skfold,

                           scoring='accuracy', return_train_score=True)



grid_forest_clf.fit(train_set_prepared, survival_labels)
grid_forest_clf.best_params_
grid_forest_pred = grid_forest_clf.predict(validation_set_prepared).astype(int)

grid_forest_scores = (grid_forest_clf.best_score_ + accuracy_score (grid_forest_pred, validation_labels))/2

grid_forest_scores
from scipy.stats import randint

param_distribs = {

        'n_estimators': randint(low=100, high=300),

        'max_features': randint(low=8, high=17),

    }



rnd_forest_clf = RandomizedSearchCV(forest_clf, param_distributions=param_distribs,

                                n_iter=10, cv=skfold, scoring='accuracy', random_state=rnd_state)

rnd_forest_clf.fit(train_set_prepared, survival_labels)
rnd_forest_clf.best_params_
rnd_forest_pred = rnd_forest_clf.predict(validation_set_prepared).astype(int)

rnd_forest_scores = (rnd_forest_clf.best_score_ + accuracy_score (rnd_forest_pred, validation_labels))/2

rnd_forest_scores
knn_clf = KNeighborsClassifier()

knn_clf.fit(train_set_prepared, survival_labels) 



knn_cross_scores = cross_val_score(knn_clf, train_set_prepared, survival_labels, cv=skfold, scoring="accuracy")

knn_pred = knn_clf.predict(validation_set_prepared).astype(int)

knn_scores = (knn_cross_scores.mean() + accuracy_score (knn_pred, validation_labels))/2

knn_scores
param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]



grid_knn_clf = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)

grid_knn_clf.fit(train_set_prepared, survival_labels)

grid_knn_clf.best_params_
grid_knn_cross_scores = cross_val_score(grid_knn_clf, train_set_prepared, survival_labels, cv=skfold, scoring="accuracy")

grid_knn_pred = grid_knn_clf.predict(validation_set_prepared).astype(int)

grid_knn_scores = (grid_knn_cross_scores.mean() + accuracy_score (grid_knn_pred, validation_labels))/2

grid_knn_scores
ext_clf = ExtraTreesClassifier()



param_grid = {"max_depth": [None],

              "max_features": [10, 17],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False, True],

              "n_estimators" :[50,100,200],

              "criterion": ["gini"]}



# Cross validate model with Kfold stratified cross val

#kfold = StratifiedKFold(n_splits=10)

grid_ext_clf = GridSearchCV(ext_clf,param_grid, cv=skfold, scoring="accuracy", n_jobs= 4, verbose = 1)

grid_ext_clf.fit(train_set_prepared, survival_labels)



grid_ext_clf.best_params_
grid_ext_pred = grid_ext_clf.predict(validation_set_prepared).astype(int)



grid_ext_scores = (grid_ext_clf.best_score_ + accuracy_score (grid_ext_pred, validation_labels))/2

grid_ext_scores
ada_clf = AdaBoostClassifier(

    DecisionTreeClassifier(random_state=rnd_state, max_depth=2),

    random_state = rnd_state)



param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[10, 100, 200, 250],

              "learning_rate":  [0.05, 0.5, 1.5, 2.5]}



grid_ada_clf = GridSearchCV(ada_clf, param_grid, cv=skfold, scoring="accuracy", n_jobs= -1, verbose = 1)

grid_ada_clf.fit(train_set_prepared, survival_labels)

grid_ada_clf.best_params_



ada_pred = grid_ada_clf.predict(validation_set_prepared).astype(int)

grid_ada_scores = (grid_ada_clf.best_score_ + accuracy_score (ada_pred, validation_labels))/2

grid_ada_scores
gb_clf = GradientBoostingClassifier(random_state=rnd_state)



param_grid = {

              'n_estimators' : [25, 50 ,75, 100, 200],

              'learning_rate': [0.005 ,0.05, 0.5, 1.5],

              'max_depth': [2, 4, 6, 8],

              'max_features': [10, 12, 17] 

              }

grid_gb_clf = GridSearchCV(gb_clf, param_grid, cv=skfold, scoring="accuracy", n_jobs= -1, verbose = 1)

grid_gb_clf.fit(train_set_prepared, survival_labels)



gb_pred = grid_gb_clf.predict(validation_set_prepared).astype(int)

grid_gb_scores = (grid_gb_clf.best_score_ + accuracy_score (gb_pred, validation_labels))/2
grid_gb_clf.best_params_
%matplotlib inline

plt.figure(figsize=(8, 4))

plt.plot(['Logistic', 'SGD', 'SVM', 'Random Forest', 'KNN', 'Extra-Tree', 'AdaBoost', 'Gradient Boost'], [log_scores, sgd_scores, svm_scores, rnd_forest_scores, knn_scores, grid_ext_scores, grid_ada_scores, grid_gb_scores], 'ro')



plt.show()
voting_clf = VotingClassifier(

    estimators=[('rf', rnd_forest_clf), ('svc', svm_clf), ('knn', grid_knn_clf)],

    voting='hard')



voting_clf.fit (train_set_prepared, survival_labels)
voting_cross_scores = cross_val_score(voting_clf, train_set_prepared, survival_labels, cv=skfold, scoring="accuracy")

voting_pred = voting_clf.predict(validation_set_prepared).astype(int)

voting_scores = (voting_cross_scores.mean() + accuracy_score (voting_pred, validation_labels))/2

voting_scores 
%matplotlib inline

plt.figure(figsize=(8, 4))

plt.plot(['SVM', 'Random Forest', 'KNN', 'Ensemble'], [svm_scores, rnd_forest_scores, grid_knn_scores, voting_scores], 'ro')



plt.show()
test_set_prepared = preprocessor.fit_transform(data_cleaner(test_data))

predictions = voting_clf.predict(test_set_prepared).astype(int)
#set ids as PassengerId and predict survival 

ids = test_data['PassengerId']

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)