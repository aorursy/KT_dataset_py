import os

import re

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.feature_extraction.dict_vectorizer import DictVectorizer



from sklearn.metrics.classification import classification_report, accuracy_score, confusion_matrix





from sklearn.utils.testing import all_estimators



from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



from sklearn.model_selection import cross_val_score, train_test_split, KFold, RandomizedSearchCV



from xgboost import XGBClassifier



from sklearn.preprocessing import LabelEncoder, MinMaxScaler



INPUT_DIR = '../input'



N_FOLDS = 4

N_ITER = 50

SEED = 32
""" -------------------------------------- Data loading -------------------------------------- """



# load dataframes

df_train_raw = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))

df_test_raw = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
""" -------------------------------------- Null data -------------------------------------- """



print(f'Train data consist of {df_train_raw.shape[0]} rows, with null values:\n{df_train_raw.isnull().sum()}\n\n')

print(f'Train data consist of {df_test_raw.shape[0]} rows, with null values:\n{df_test_raw.isnull().sum()}')



# print some statistics

df_train_raw.describe(include = 'all')
""" -------------------------------- Feature Engineering ------------------------------------- """





def get_title(name):

    """

    Define function to extract titles from passenger names

    """

    

    title_search = re.search(' ([A-Za-z]+)\.', name)

    

    # if the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""





# 

df_full = [df_train_raw.copy(), df_test_raw.copy()]



for dataset in df_full:

    

    # Create new feature Last Name in order to group families

    dataset['LastName'] = dataset['Name'].apply(lambda x: str.split(x, ",")[0])

    dataset['LastName'] = dataset['LastName'].astype('category').cat.codes

#     dataset['LastName'] = dataset['LastName'].astype('category')

    

    dataset['Namelength'] = dataset['Name'].apply(len)



    # Create a new feature Title, containing the titles of passenger names

    dataset['Title'] = dataset['Name'].apply(get_title)



    # Group all non-common titles into one single grouping "Rare"

    dataset['Title'] = dataset['Title'].replace(

        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



    # Mapping titles

    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})

    dataset['Title'] = dataset['Title'].fillna(0)

#     dataset['Title'] = dataset['Title'].astype('category')



    # Feature that tells whether a passenger had a cabin on the Titanic

    dataset['HasCabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)#.astype('category')



    # Create new feature FamilySize as a combination of SibSp and Parch

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



    # Create new feature IsAlone from FamilySize

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    dataset['IsAlone'] = dataset['IsAlone'].astype('category')

    

    # Mapping Embarked

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})#.astype('category')



    # Remove all NULLS in the Fare column and create a new feature CategoricalFare

    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

    dataset['CatFare'] = pd.qcut(dataset['Fare'], q=5, labels=False)#.astype('category')

    

    # Fill null age rows with median value

    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

    dataset['CatAge'] = pd.qcut(dataset['Age'], q=4, labels=False)#.astype('category')

    

    # Create new feature

#     dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']



    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1})#.astype('category')



    drop_columns = ['PassengerId', 'Name', 'Cabin', 'Ticket']

#     drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

    dataset.drop(drop_columns, axis=1, inplace=True)



df_train, df_test = df_full



df_train.describe(include='all')
print(df_train._get_numeric_data().columns)
""" ---------------------------------- Feature preparation ----------------------------------- """





def min_max_scale(train_data, test_data, numeric_cols):

    """

    Define function to scale numeric variables to [0, 1] range

    """

    

    data = pd.concat([train_data, test_data])

    

    scaled_train_data, scaled_test_data = train_data.copy(), test_data.copy()



    for feature_name in numeric_cols:

        max_v = data[feature_name].max()

        min_v = data[feature_name].min()

        scaled_train_data[feature_name] = (train_data[feature_name] - min_v) / (max_v - min_v)

        scaled_test_data[feature_name] = (test_data[feature_name] - min_v) / (max_v - min_v)

        

    return scaled_train_data, scaled_test_data





def categorical_encode(train_data, test_data, categorical_cols):

    """

    Define function to perform one hot encoding of data

    """

    

    data = pd.concat([train_data, test_data])

    

    scaled_train_data, scaled_test_data = train_data.copy(), test_data.copy()

    result = df.copy()

    for feature_name in categorical_cols:

        max_v = df[feature_name].max()

        min_v = df[feature_name].min()

        result[feature_name] = (df[feature_name] - min_v) / (max_v - min_v)

    return result





label_column = 'Survived'

    

# get all column names

cols = list(df_train.columns.values)



# numeric columns

num_cols = [e for e in df_train.select_dtypes(include=[np.number]).columns.tolist() if e != label_column]



# categorical columns

cat_cols = [e for e in cols if e not in num_cols and e != label_column]



print(num_cols, cat_cols)



x_train, y_train = df_train.drop(label_column, axis=1), df_train[label_column].astype(int)

x_test = df_test



x_train, x_test = min_max_scale(x_train, x_test, num_cols)



x_train.describe(include = 'all')
""" -------------------------------- Correlation report ----------------------------------- """



plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(df_train.astype(float).corr(), linewidths=0.1, vmax=1.0, 

            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
""" ----------------------------- Grid params initialization ------------------------------ """



MODELS = {

    'lr': {

        'model': LogisticRegression,

        'params': {

            'fit_intercept': [True, False],

            'multi_class': ['ovr'],

            'penalty': ['l2'],

            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

            'tol': [0.01, 0.05, 0.1, 0.5, 1, 5],

            'random_state': [SEED],

        },

        'best_params': {'tol': 0.05, 'solver': 'newton-cg', 'random_state': 32, 'penalty': 'l2', 'multi_class': 'ovr', 'fit_intercept': True},

        'best score': 0.813692480359147,

    },

    'mlp': {

        'model': MLPClassifier,

        'params': {

            'activation' : ['identity', 'logistic', 'tanh', 'relu'],

            'solver' : ['lbfgs', 'adam'],

            'learning_rate' : ['constant', 'invscaling', 'adaptive'],

            'learning_rate_init': [.01, .05, .1, .2, .5, 1, 2],

            'random_state': [SEED],

        },

        'best_params': {'solver': 'lbfgs', 'random_state': 32, 'learning_rate_init': 2, 'learning_rate': 'adaptive', 'activation': 'identity'},

        'best_score': 0.8092031425364759,

    },

#     'knn': {

#         'model': KNeighborsClassifier,

#         'params': {

#             'n_neighbors' : range(1, 10),

#             'weights' : ['uniform', 'distance'],

#             'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],

#             'leaf_size' : range(10, 100, 10),

#         }

#     },

#     'lrcv': {

#         'model': LogisticRegressionCV,

#         'params': {

#             'Cs': [1, 2, 4, 8, 16, 32],

#             'fit_intercept': [True, False],

#             'refit': [True, False],

#             'multi_class': ['ovr'],

#             'penalty': ['l2'],

#             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

#             'tol': [0.01, 0.05, 0.1, 0.5, 1, 5],

#             'cv': [cv]

#         },

#         'best_params': {'tol': 0.05, 'solver': 'newton-cg', 'refit': True, 'penalty': 'l2', 'multi_class': 'ovr', 'fit_intercept': False, 'cv': 4, 'Cs': 2},

#         'best_score': 0.8428731762065096

#     },

    'dt': {

        'model': DecisionTreeClassifier,

        'params': {

            'criterion': ['gini', 'entropy'],

            'max_depth': range(6, 10),

            'max_features': ['auto', 'sqrt', 'log2', None],

            'min_samples_split': [2, 5, 10], # Minimum number of samples required to split a node

            'min_samples_leaf': [1, 2, 4], # Minimum number of samples required at each leaf node

        },

        'best_params': {'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': None, 'max_depth': 6, 'criterion': 'gini'},

        'best_score': 0.8181818181818182,

    },

    'svc': {

        'model': SVC,

        'params': {

            'C': [0.1, 0.5, 1., 2., 4.],

            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

            'gamma': ['auto', 'scale'],

            'degree': range(5),

            'tol': [0.1, 0.5, 1, 5],

        },

        'best_params': {'tol': 1, 'shrinking': False, 'probability': False, 'kernel': 'rbf', 'gamma': 'scale', 'degree': 4, 'C': 2.0},

        'best_score': 0.8428731762065096

    },

    'rf': {

        'model': RandomForestClassifier,

        'params': {

            'n_estimators': range(10, 251, 20),

            'max_features': ['auto', 'sqrt', 'log2', None],

            'max_depth': range(5, 20),

            'min_samples_split': range(2, 10), # Minimum number of samples required to split a node

            'min_samples_leaf': range(1, 10), # Minimum number of samples required at each leaf node

            'bootstrap': [True, False], # Method of selecting samples for training each tree,

            'random_state': [SEED],

        },

        'best_params': {'random_state': 32, 'n_jobs': -1, 'n_estimators': 70, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': 17, 'bootstrap': True},

        'best_score': 0.8417508417508418

    },

    'ada': {

        'model': AdaBoostClassifier,

        'params': {

            'n_estimators': range(10, 251, 20),

            'learning_rate': [.01, .05, .1, .2, .5, 1, 2],

            'algorithm': ['SAMME', 'SAMME.R'],

            'random_state': [SEED],

        },

        'best_params': {'random_state': 32, 'n_estimators': 170, 'learning_rate': 1, 'algorithm': 'SAMME.R'},

        'best_score': 0.8237934904601572

    },

    'et': {

        'model': ExtraTreesClassifier,

        'params': {

            'n_estimators': range(10, 251, 20),

            'max_features': ['auto', 'sqrt', 'log2', None],

            'max_depth': range(5, 20),

            'min_samples_split': range(2, 10), # Minimum number of samples required to split a node

            'min_samples_leaf': range(1, 10), # Minimum number of samples required at each leaf node

            'bootstrap': [True, False], # Method of selecting samples for training each tree,

            'random_state': [SEED],

        },

        'best_params': {'random_state': 32, 'n_jobs': -1, 'n_estimators': 70, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': 11, 'bootstrap': True},

        'best_score': 0.8294051627384961

    },

    'gb': {

        'model': GradientBoostingClassifier,

        'params': {

            'n_estimators': range(10, 251, 20),

            'max_depth': range(5, 20),

            'loss': ['deviance', 'exponential'],

            'learning_rate': [.01, .05, .1, .2, .5, 1, 2],                      

            'subsample': [.25, .5, .8, 1.],

            'min_samples_split': range(2, 10), # Minimum number of samples required to split a node

            'min_samples_leaf': range(1, 10), # Minimum number of samples required at each leaf node

            'random_state': [SEED],

        },

        'best_params': {'subsample': 0.5, 'random_state': 32, 'n_estimators': 150, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 13, 'loss': 'exponential', 'learning_rate': 1},

        'best_score': 0.8361391694725028

    }

#     'xgb': {

#         'model': XGBClassifier,

#         'params': {

#             'n_estimators': range(8, 20),

#             'max_depth': range(5, 20),

#             'learning_rate': [.01, .05, .1, .2, .5, 1, 2],

#             'colsample_bytree': [.6, .7, .8, .9, 1]

#         }

#     }

}
""" ---------------------------- Best models configuration search --------------------------- """



FIT_FROM_SCRATCH = True



for name, model in MODELS.items():

    

    if 'best_score' in model and not FIT_FROM_SCRATCH:

        

        # Initialize with best parameters & fit to data

        print(f'Fitting {name}...')

        

        model['best_estimator'] = model['model'](**model['best_params']).fit(x_train, y_train)

        

        scores = cross_val_score(model['best_estimator'], x_train, y_train, cv=N_FOLDS)

        score = sum(scores) / len(scores)

        diff = score - model['best_score']

        

        if diff > 0:

            print(f'Accuracy of model {name}: {score} (BIGGER for {diff})')

        elif diff < 0:

            print(f'Accuracy of model {name}: {score} (SMALLER for {-diff})')

        else:

            print(f'Accuracy of model {name}: {score} (SAME)')

    else:

        # Perform random search

        searcher = RandomizedSearchCV(param_distributions=model['params'],

                                      estimator=model['model'](), scoring="accuracy",

                                      verbose=1, n_iter=N_ITER, cv=N_FOLDS)

        # Fit to data

        print(f'Fitting {name}...')

        

        searcher.fit(x_train, y_train)



        # Print the best parameters and best accuracy

        print(f'Best parameters found for {name}: {searcher.best_params_}')

        print(f'Best accuracy found {name}: {searcher.best_score_}')



        model['best_estimator'] = searcher.best_estimator_

        model['best_params'] = searcher.best_params_

        model['best_score'] = searcher.best_score_
""" ---------------------------------- Preparing 2nd level features ------------------------------------ """



df = pd.DataFrame()



X_train, X_test = {}, {}



for name, model in MODELS.items():

    vtrain = MODELS[name]['best_estimator'].predict(x_train)

    vtest = MODELS[name]['best_estimator'].predict(x_test)

    

    df[name] = np.reshape(vtrain, [-1])

    

    X_train[name] = vtrain

    X_test[name] = vtest
""" -------------------------------- Correlation report ----------------------------------- """



plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0, 

            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
""" --------------------------- Choosing only low correlated ones ---------------------------- """



low_correlated_models = ['lr', 'dt', 'svc', 'gb']



X_train = np.vstack([v for k, v in X_train.items()]).T

X_test = np.vstack([v for k, v in X_test.items()]).T



print(X_train.shape, X_test.shape, y_train.shape)
# """ ----------------------------------- Fitting XGBoost classifier ------------------------------------- """



# xgb_params = {

#     'n_estimators': range(20, 501, 20),

#     'max_depth': range(4, 21, 4),

#     'learning_rate': [.01, .05, .1, .2, .5, 1, 2],

#     'colsample_bytree': [.6, .7, .8, .9, 1]

# }

# # xgb = XGBClassifier(**{'n_estimators': 20, 'max_depth': 4, 'learning_rate': 0.05, 'colsample_bytree': 0.8})



# # Perform random search

# xgb = RandomizedSearchCV(param_distributions=xgb_params,

#                               estimator=XGBClassifier(), scoring="accuracy",

#                               verbose=1, n_iter=N_ITER, cv=N_FOLDS)

# # Fit to data

# print(f'Fitting xgb...')    

# xgb.fit(X_train, y_train)



# print(f'Best parameters found for {name}: {xgb.best_params_}')

# print(f'Best accuracy found {name}: {xgb.best_score_}')



# pred = xgb.predict(X_test)
pred = MODELS['svc']['best_estimator'].predict(x_test)
# pred = MODELS[max(MODELS, key=lambda k: MODELS[k]['best_score'])]['best_estimator'].predict(x_test)

submission = pd.DataFrame({'PassengerId': df_test_raw['PassengerId'], 'Survived': pred})

submission.to_csv('submission.csv', index=False)