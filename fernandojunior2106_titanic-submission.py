# Basic libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

import time



# Classification Models

from sklearn.linear_model import LogisticRegression # Logistic Regression

from sklearn.neighbors import KNeighborsClassifier # KNN

from sklearn.svm import SVC # SVM

from sklearn.ensemble import RandomForestClassifier # RandomForest

from sklearn.naive_bayes import GaussianNB # Naive Bayes

from xgboost import XGBClassifier # Xgboost



# Regressor models

from sklearn.ensemble import RandomForestRegressor



# Pre Processing Functions

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer # handle missing values (naive approach)

from sklearn.impute import KNNImputer # handle missing values (better approach)

from sklearn.preprocessing import OneHotEncoder # enconding cat variables

from sklearn.preprocessing import MinMaxScaler # normale (values betwenn 0 and 1)

from sklearn.preprocessing import StandardScaler # standardize (center the data around 0)

from sklearn.decomposition import PCA # Principal Components Analysis



# Evaluate classification models

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, make_scorer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_validate



# Tune models

from sklearn.model_selection import GridSearchCV 

from sklearn.model_selection import RandomizedSearchCV



# Statistics functions

from scipy.stats import randint

from scipy.stats import uniform
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read Data Sets

df = pd.read_csv("/kaggle/input/titanic/train.csv")

submission_df = pd.read_csv("/kaggle/input/titanic/test.csv")

ids = pd.read_csv("/kaggle/input/titanic/test.csv")
# Look at data

df.head()
# Features data type

df.dtypes
# Numeric Features description

df.describe(include=['int64', 'float64']).T
# Categorical Features description

df.describe(include=['object']).T
# Vizualizing null values

plt.figure(figsize=(12,8))

sns.heatmap(df.isnull(), cbar=False)

plt.show()
# Total null values per feature

print(df.isnull().sum())
# Analyzing age

f, axes = plt.subplots(5, figsize=(8,24))

sns.boxplot(x='Sex',y='Age',data=df,ax=axes[0]) # has no effect

sns.boxplot(x='Pclass',y='Age',data=df,ax=axes[1]) # different values

sns.boxplot(x='SibSp',y='Age',data=df,ax=axes[2]) # different values

sns.boxplot(x='Parch',y='Age',data=df,ax=axes[3]) # different values

sns.boxplot(x='Embarked',y='Age',data=df,ax=axes[4]) # has no effect

plt.show()
# Inferring age based on previus step

age_model = RandomForestRegressor()

age_model.fit(df.dropna()[['Pclass', 'SibSp', 'Parch']], df.dropna()[['Age']])

df.loc[df['Age'].isna(), 'Age'] = age_model.predict(df[df['Age'].isna()][['Pclass', 'SibSp', 'Parch']])
print(df.isnull().sum())
# Adjust columns



# Define size of the family

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1



# Create Name based columns

df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df['Family_Name'] = df.Name.str.extract('([A-Z]\w+,)', expand=False)



# Cabin based features

df['cabin_number'] = df.Cabin.str.extract('(\d+)')

df['cabin_sector'] = df.Cabin.str.extract('([a-zA-Z]+)')



df = df.drop(columns=['Ticket', 'Cabin', 'PassengerId', 'Name'])





# Define size of the family

submission_df['FamilySize'] = submission_df['SibSp'] + submission_df['Parch'] + 1



# Create Name based columns

submission_df['Title'] = submission_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

submission_df['Family_Name'] = submission_df.Name.str.extract('([A-Z]\w+,),', expand=False)



# Cabin based features

submission_df['cabin_number'] = submission_df.Cabin.str.extract('(\d+)')

submission_df['cabin_sector'] = submission_df.Cabin.str.extract('([a-zA-Z]+)')



submission_df = submission_df.drop(columns=['Ticket', 'Cabin', 'PassengerId', 'Name'])
# Analysing Titles

df.Title.value_counts()
# Adjust Title

df.loc[df.Title.str.contains('^(?!.*(Mr|Miss|Mrs)).*$'), 'Title'] = 'Master'

submission_df.loc[submission_df.Title.str.contains('^(?!.*(Mr|Miss|Mrs)).*$'), 'Title'] = 'Master'

df.Title.value_counts()
# Analysing Family_Name

names = df.Family_Name.value_counts()[df.Family_Name.value_counts()<=2]

names = names.index.tolist()
# Adjust family name

df.loc[df.Family_Name.isin(names), 'Family_Name'] = 'Other'

submission_df.loc[submission_df.Family_Name.isin(names), 'Family_Name'] = 'Other'

df.Family_Name.value_counts()
# Analysing cabin_sector

df.loc[df.cabin_sector.isna(), 'cabin_sector'] = 'other_sector'

df.loc[df.cabin_sector.str.contains('G|T'), 'Title'] = 'other_sector'

submission_df.loc[submission_df.cabin_sector.isna(), 'cabin_sector'] = 'other_sector'

submission_df.loc[submission_df.cabin_sector.str.contains('G|T'), 'Title'] = 'other_sector'

df.cabin_sector.value_counts()
# Analysing cabin_sector

df.loc[df.cabin_number.isna(), 'cabin_number'] = 0

submission_df.loc[submission_df.cabin_number.isna(), 'cabin_number'] = 0
df['cabin_number'] = pd.to_numeric(df['cabin_number'])

submission_df['cabin_number'] = pd.to_numeric(submission_df['cabin_number'])
# Split into train and test

X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis = 1), df.Survived, test_size=0.30, random_state=21)
# Creating pipelines



# Numeric features

num_features = list(X_test.select_dtypes(include=['int64', 'float64']).columns)

numeric_transformer = Pipeline(steps=[

    # ('imputer', SimpleImputer(strategy='median')),

    ('imputer', KNNImputer(n_neighbors=4, weights="uniform")),

    ('center', StandardScaler()),

    ('scale', MinMaxScaler())])



# Categorical features

cat_features = list(X_test.select_dtypes(include=['object']).columns)

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



# Features Pipeline

features_preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, num_features),

        ('cat', categorical_transformer, cat_features)])



# Logistic Regression

logreg_model = Pipeline(steps=[

    ('features_preprocessor', features_preprocessor),

    ('logreg', LogisticRegression())

])



# Xgboost

xgb_model = Pipeline(steps=[

    ('features_preprocessor', features_preprocessor),

    ('xgb', XGBClassifier(n_jobs=-1,

                          tree_method = "gpu_hist"))

])



# KNN

knn_model = Pipeline(steps=[

    ('features_preprocessor', features_preprocessor),

    ('knn', KNeighborsClassifier())

])



# SVM

svm_model = Pipeline(steps=[

    ('features_preprocessor', features_preprocessor),

    ('svm', SVC())

])



# RandomForest

rf_model = Pipeline(steps=[

    ('features_preprocessor', features_preprocessor),

    ('rf', RandomForestClassifier())

])



# Naive Bayes

nb_model = Pipeline(steps=[

    ('features_preprocessor', features_preprocessor),

    ('nb', GaussianNB())

])



# Hyper parameters grid



# Logistic Regression

logreg_grid = {

        'logreg__solver': ['newton-cg', 'lbfgs', 'liblinear'],

        'logreg__penalty': ['l2'],

        'logreg__C': [100, 10, 1.0, 0.1, 0.01]

}



# Xgboost

xgb_grid = {

    "xgb__learning_rate": [0.1, 0.01, 0.001],

    "xgb__colsample_bytree": [0.6, 0.8, 1.0],

    "xgb__subsample": [0.6, 0.8, 1.0],

    "xgb__max_depth": [2, 4, 6],

    "xgb__n_estimators": [100, 200, 300, 400],

    "xgb__reg_lambda": [1, 1.5, 2],

    "xgb__gamma": [0, 0.1, 0.3],

}



# KNN

knn_grid = {

        'knn__n_neighbors': range(1, 21, 2),

        'knn__weights': ['uniform', 'distance'],

        'knn__metric': ['euclidean', 'manhattan', 'minkowski']

}



# SVM

svm_grid = {

        'svm__kernel': ['poly', 'rbf', 'sigmoid'],

        'svm__C': [50, 10, 1.0, 0.1, 0.01],

        'svm__gamma': ['scale']

}



# RandomForest

rf_grid = {

        'rf__n_estimators': [10, 100, 1000],

        'rf__max_features': ['sqrt', 'log2']

}

# Function to test all pipelines we have defined

def run_exps(X_train: pd.DataFrame , y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:

    '''

    Lightweight script to test many models and find winners

    :param X_train: training split

    :param y_train: training target vector

    :param X_test: test split

    :param y_test: test target vector

    :return: DataFrame of predictions

    '''

    

    dfs = []

    models = [

              ('Logistic Regression', logreg_model, logreg_grid),

              ('XGBoost', xgb_model, xgb_grid),

              ('KNN', knn_model, knn_grid),

              ('SVM', svm_model, svm_grid),

              ('Random Forest', rf_model, rf_grid),]



    results = []

    names = []

    # gsearch = []

    # scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']

    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

    target_names = list(map(str, y_train.unique()))



    for name, model, grid in models:

            # 5 k-fold cross validation

            kfold = StratifiedKFold(n_splits=5)

            cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)

            random_search_results = RandomizedSearchCV(estimator=model, 

                               param_distributions=grid, 

                               scoring=scoring, 

                               n_jobs=-1,

                               n_iter=50,

                               refit="AUC"

                               ) 

            clf = random_search_results.fit(X_train, y_train)

            y_pred = random_search_results.predict(X_test)

            print(name)

            print(random_search_results.best_params_)

            print(classification_report(y_test, y_pred, target_names=target_names))



    results.append(cv_results)

    names.append(name)

    # gsearch.append(random_search_results)

    

    this_df = pd.DataFrame(cv_results)

    this_df['model'] = name

    dfs.append(this_df)



    final = pd.concat(dfs, ignore_index=True)

    return final
models_results = run_exps(X_train, y_train, X_test, y_test)
# running the best model from previus step

def best_model_submission(X_train: pd.DataFrame , y_train: pd.DataFrame, X_submission: pd.DataFrame, best_model: list):

  

    '''

    Run fit function to the best model

    :param X_train: training split

    :param y_train: training target vector

    :param X_submission: test split

    :param best_model: list with name and classifier

    :return: submission predictions

    '''



    for name, model in best_model:

            # 5 k-fold cross validation

            clf = model.fit(X_train, y_train)

            y_pred = clf.predict(X_submission)



    return y_pred
# defining best hypeparameters from previus step

xgb_model = Pipeline(steps=[

    ('features_preprocessor', features_preprocessor),

    ('xgb', XGBClassifier(n_jobs = -1,

                          subsample = 0.8,

                          reg_lambda = 1,

                          n_estimators = 300,

                          max_depth = 6,

                          learning_rate = 0.01,

                          gamma = 0.3,

                          colsample_bytree = 1.0))

])



# runing the best model

model = [('XGBoost', xgb_model)]

y_pred = best_model_submission(pd.concat([X_train,X_test],ignore_index=True), 

                        pd.concat([y_train,y_test],ignore_index=True), 

                        submission_df,

                        model)
results = pd.DataFrame()

results['PassengerId'] = ids.PassengerId

results['Survived'] = y_pred
results
results.to_csv("gender_submission.csv", index=False)