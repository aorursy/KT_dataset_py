!pip install hypopt
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats

import re

from xgboost import XGBClassifier

from xgboost import plot_importance

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from sklearn.svm import SVC

from sklearn import model_selection

from sklearn.model_selection import KFold

from sklearn.model_selection import PredefinedSplit

from hypopt import GridSearch

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import feature_selection

from collections import Counter

pd.set_option("display.max_rows", 50, "display.max_columns", 50)
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')

PassengerId = df_test['PassengerId']

sample_subm = pd.read_csv('../input/titanic/gender_submission.csv')
# Outlier detection 



def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(df_train,2,["Age","SibSp","Parch","Fare"])

# Drop outliers

df_train = df_train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
plt.figure(figsize=(6,3))

df_train.Survived.value_counts().plot(kind='bar', rot=360)

plt.show()
full_data = [df_train, df_test]



# Feature that tells whether a passenger had a cabin on the Titanic

df_train['Has_Cabin'] = df_train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

df_test['Has_Cabin'] = df_test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Feature engineering steps taken from Sina

# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    

    

# Remove all NULLS in the Fare column and create a new feature CategoricalFare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(df_train['Fare'].median())

df_train['CategoricalFare'] = pd.qcut(df_train['Fare'], 4)

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

df_train['CategoricalAge'] = pd.cut(df_train['Age'], 5)

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

df_train = df_train.drop(drop_elements, axis = 1)

df_train = df_train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

df_test  = df_test.drop(drop_elements, axis = 1)
#features = ['Pclass', 'Sex', 'Embarked']

features = df_train.columns[1:]

fig = plt.figure(figsize=(15, 13))

for i in range(len(features)):

    fig.add_subplot(4, 3, i+1)

    sns.countplot(x=features[i], hue="Survived", data=df_train)

plt.show()
#df_train = pd.get_dummies(df_train, columns=['Title', 'FamilySize', 'Embarked'])
dt = DecisionTreeClassifier(random_state = 0)

dt_rfe = feature_selection.RFECV(dt, step = 1, scoring = 'accuracy', cv = 5)

dt_rfe.fit(df_train[features], df_train.Survived)



#transform x&y to reduced features and fit new model

#alternative: can use pipeline to reduce fit and transform steps: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

rfe_features = df_train[features].columns.values[dt_rfe.get_support()]

rfe_results = model_selection.cross_validate(dt, df_train[rfe_features], df_train.Survived, cv  = 5)

print(rfe_features)
y_train = df_train.Survived

X_train = df_train.loc[:,features].values

X_train_rfe = df_train.loc[:,rfe_features].values

X_test = df_test.values

X_test_rfe = df_test.loc[:,rfe_features].values
xgb_params = {

#    'tree_method': ['gpu_hist'],

    'criterion':['gini', 'entropy'],

    'n_estimators': [100,500,700],

    'n_jobs': [1],

    'scale_pos_weight': [1],

    'base_score': [0.5],

    'learning_rate': [0.01,0.1],

    'reg_alpha': [0,5,10],

    'reg_lambda': [0],

    'gamma': [1,5],

    'max_delta_step': [0],

    'max_depth': [100],

    'objective': ['binary:logistic'],

    'random_state': [0]

    }



lr_params = {

    'penalty':['l1','l2'], 

    'dual':[False], 

    'tol':[0.0001], 

    'C':[0.5,1,5], 

    'fit_intercept':[True], 

    'intercept_scaling':[1], 

    'class_weight':[None], 

    'random_state':[0], 

    'solver':['lbfgs'], 

    'max_iter':[100,1000], 

    'multi_class':['auto'], 

    'verbose':[0], 

    'warm_start':[False], 

    'n_jobs':[None], 

    'l1_ratio':[None]

    }



dt_params = {

    'criterion':['gini', 'entropy'],

    'random_state': [0],

    'max_depth':[50,100],

    'max_leaf_nodes':[5,10,100],

    'min_impurity_decrease':[0.0, 0.1,1], 

    'min_impurity_split':[None],

    'min_samples_split':[2]

    }



rf_params = {

    'n_estimators':[100,500], 

    'criterion':['gini', 'entropy'], 

    'max_depth':[None], 

    'min_samples_split':[2], 

    'min_samples_leaf':[1], 

    'min_weight_fraction_leaf':[0.0], 

    'max_features':['auto'], 

    'max_leaf_nodes':[None], 

    'min_impurity_decrease':[0.0, 0.1,1], 

    'min_impurity_split':[None], 

    'bootstrap':[True], 

    'oob_score':[False], 

    'n_jobs':[None], 

    'random_state':[0], 

    'verbose':[0], 

    'warm_start':[False], 

    'class_weight':[None]

    }



lg_params = {

    'application': ['binary'], # for binary classification

#     'num_class' : 1, # used for multi-classes

    'boosting': ['gbdt'], # traditional gradient boosting decision tree

    'criterion': ['gini', 'entropy'],

#    'num_iterations': [100,500], 

    'learning_rate': [0.05],

    'num_leaves': [62],

    'device': ['cpu'], # you can use GPU to achieve faster learning

    'max_depth': [-1], # <0 means no limit

    'max_bin': [200], # Small number of bins may reduce training accuracy but can deal with over-fitting

    'lambda_l1': [0,1], # L1 regularization

    'lambda_l2': [0,5], # L2 regularization

    'metric' : ['binary_error'],

    'subsample_for_bin': [100], # number of samples for constructing bins

    'subsample': [1], # subsample ratio of the training instance

    'colsample_bytree': [0.5], # subsample ratio of columns when constructing the tree

    'min_split_gain': [0.5], # minimum loss reduction required to make further partition on a leaf node of the tree

    'min_child_weight': [1], # minimum sum of instance weight (hessian) needed in a leaf

    'min_child_samples': [5]# minimum number of data needed in a leaf

    }



svc_params = {

    'C':[0.1, 1.0, 5],

    'gamma':['auto'],

    'random_state':[0],

    'kernel':['rbf'], 

    'degree':[2,3],

    'coef0':[0.0], 

    'shrinking':[True], 

    'tol':[0.001], 

    'cache_size':[200], 

    'class_weight':[None], 

    'verbose':[False], 

    'max_iter':[-1], 

    'decision_function_shape':['ovr']

    }



et_params = {

    'n_estimators':[100,500],

    'criterion': ['gini', 'entropy'],

    'random_state': [0],

    'max_depth':[None], 

    'min_samples_split':[2], 

    'min_samples_leaf':[1], 

    'min_weight_fraction_leaf':[0.0], 

    'max_features':['auto'], 

    'max_leaf_nodes':[None], 

    'min_impurity_decrease':[0.0], 

    'min_impurity_split':[None], 

    'bootstrap':[False], 

    'oob_score':[False], 

    'n_jobs':[None], 

    'warm_start':[False], 

    'class_weight':[None]    

    }



kn_params = {

   'n_neighbors': [1,2,5,7], #default: 5

    'weights': ['uniform', 'distance'], #default = ‘uniform’

    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

    }
# Some useful parameters which will come in handy later on

ntrain = df_train.shape[0]

ntest = df_test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(n_splits=5, random_state = SEED)



class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)
def gridSearchCLF(estimator, params, X_train, y_train, X_val, y_val):

    clf = GridSearch(model=estimator, param_grid=params)

    clf_fitted = clf.fit(X_train, y_train, X_val, y_val, scoring='f1')

    return clf_fitted
def get_oof(clf, params, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        y_te = y_train[test_index]

        

        #ps_x_val = PredefinedSplit(test_fold=x_te)

        clf_fitted = gridSearchCLF(clf, params, x_tr, y_tr, x_te, y_te)



        #clf.train(x_tr, y_tr)



        oof_train[test_index] = clf_fitted.predict_proba(x_te)[:,1]

        #oof_test_skf[i, :] = clf_fitted.predict(x_test)

        oof_test_skf[i, :] = clf_fitted.predict_proba(x_test)[:,1]

    

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
lr = LogisticRegression()

dt = DecisionTreeClassifier()

rf = RandomForestClassifier()

xgb = XGBClassifier()

svc = SVC(probability=True)

et = ExtraTreesClassifier()

kn = KNeighborsClassifier()

lg = LGBMClassifier()
lr_oof_train, lr_oof_test = get_oof(lr, lr_params, X_train, y_train, X_test) 

dt_oof_train, dt_oof_test = get_oof(dt, dt_params, X_train, y_train, X_test) 

rf_oof_train, rf_oof_test = get_oof(rf, rf_params, X_train, y_train, X_test) 

#xgb_oof_train, xgb_oof_test = get_oof(xgb, xgb_params, X_train, y_train, X_test)

#lg_oof_train, lg_oof_test = get_oof(lg, lg_params, X_train, y_train, X_test)

svc_oof_train, svc_oof_test = get_oof(svc, svc_params, X_train, y_train, X_test)

et_oof_train, et_oof_test = get_oof(et, et_params, X_train, y_train, X_test)

#kn_oof_train, kn_oof_test = get_oof(kn, kn_params, X_train, y_train, X_test)
x_train = np.concatenate((lr_oof_train, dt_oof_train, rf_oof_train, svc_oof_train, et_oof_train), axis=1)

x_test = np.concatenate((lr_oof_test, dt_oof_test, rf_oof_test, svc_oof_test, et_oof_test), axis=1)
def gridSearchCLF_final(estimator, params, X_train, y_train):

    clf = GridSearchCV(estimator=estimator, param_grid=params, cv=5, scoring='accuracy')

    clf_fitted = clf.fit(X_train, y_train)

    mean_score = clf.cv_results_['mean_test_score'][0]

    results = f"{str(clf.best_estimator_).split('(')[0]} has the best mean score: {clf.best_score_} \n"

    return clf_fitted, results
clf_fitted_final, results = gridSearchCLF_final(xgb, xgb_params, x_train, y_train)

y_pred = clf_fitted_final.predict(x_test)

submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': y_pred})

submission.to_csv("submission.csv", index=False)
print(results)
#fig, ax = plt.subplots(figsize=(6,6))

#feature_importance = plot_importance(xgb_clf.best_estimator_, ax=ax)

#feature_importance

#plt.show()