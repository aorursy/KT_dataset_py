# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling # library for automatic EDA

import matplotlib.pyplot as plt # plotting library

from scipy import stats

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from category_encoders import OneHotEncoder

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier, plot_importance as plot_importance_xgb

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv', index_col=0)

train_data.head()
pd_report = pandas_profiling.ProfileReport(train_data)
display(pd_report)
survivors = train_data[train_data['Survived'] == 1]

nonsurvivors = train_data[train_data['Survived'] == 0]
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,8))



ax[0][0].boxplot([survivors['Age'].dropna(), nonsurvivors['Age'].dropna()], labels=['survivors', 'nonsurvivors'])

ax[0][0].set_title('Age')



ax[0][1].boxplot([survivors['Fare'], nonsurvivors['Fare']], labels=['survivors', 'nonsurvivors'])

ax[0][1].set_title('Fare')



ax[1][0].boxplot([survivors['SibSp'], nonsurvivors['SibSp']], labels=['survivors', 'nonsurvivors'])

ax[1][0].set_title('SibSp')



ax[1][1].boxplot([survivors['Parch'], nonsurvivors['Parch']], labels=['survivors', 'nonsurvivors'])

ax[1][1].set_title('Parch')
# Function for transforming high-diversity categorical features into numerical variables, using Weight_of_Evidence method.

# The returned values are standardized and the outliers are handeled by replacing with +3 or -3.



def Weight_of_Evidence(Feature, Target_var, Train_index):

    """

    Args:

        Feature: is the categorical feature or attribute we want to transform and is a "series"

        Target_var: is the binary target variable and also a "series'

        Train_index: index of training dataset

    """

    

    TY = Target_var.value_counts()[1]

    TN = Target_var.value_counts()[0]

    Target_var_Feature = pd.crosstab(Target_var, Feature.loc[Train_index], margins=True)



    # Replace zeros with small number 0.001 to prevent nan in WOE calculation

    Target_var_Feature.replace(0, 0.001, inplace=True)



    Feature_prep = []

    for i in Feature.index:

        if Target_var_Feature.get([Feature[i]]) is None:

            WOE = np.nan

        else:

            WOE = np.log((Target_var_Feature[Feature[i]][1] / TY) / (Target_var_Feature[Feature[i]][0] / TN))

        Feature_prep.append(WOE)

    Feature_prep = pd.DataFrame({Feature.name + '_prep': Feature_prep}, index=Feature.index)

    Feature_prep.fillna(np.mean(Feature_prep.loc[Train_index]))



    # Standardize variable

    Feature_prep = (Feature_prep - np.mean(Feature_prep.loc[Train_index])) / np.std(Feature_prep.loc[Train_index])



    # Deal with outliers

    Feature_prep.where(cond=Feature_prep[Feature_prep.columns[0]] < 3, other=3, inplace=True)

    Feature_prep.where(cond=Feature_prep[Feature_prep.columns[0]] > -3, other=-3, inplace=True)

    

    return Feature_prep
# Name

train_data['Title'] = train_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]



train_data['Title'] = train_data['Title'].replace(['Miss', 'Mrs','Ms'], 'Miss/Mrs/Ms')

train_data['Title'] = train_data['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev',

                                                   'Master', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Other')



train_data = pd.concat([train_data, pd.get_dummies(train_data['Title'], prefix='Title')], axis=1)



# Pclass

train_data = pd.concat([train_data, pd.get_dummies(train_data['Pclass'], prefix='Pclass')], axis=1)



# Sex

train_data = pd.concat([train_data, pd.get_dummies(train_data['Sex'])], axis=1)



# Age

train_data['Age_std'] = train_data['Age']

train_data['Age_std'].fillna(train_data['Age'].mean(), inplace=True)

train_data['Age_std'] = (train_data['Age_std'] - np.mean(train_data['Age'])) / np.std(train_data['Age'])

train_data['Age_std'] = train_data['Age_std'].where(cond=train_data['Age_std'] < 3, other=3)

train_data['Age_std'] = train_data['Age_std'].where(cond=train_data['Age_std'] > -3, other=-3)



# SibSp --> no preprocessing



# Parch --> no preprocessing



# Ticket

train_data = pd.concat([train_data, Weight_of_Evidence(train_data['Ticket'], train_data['Survived'], train_data.index)], axis=1)



# Cabin --> no use



# Fare

train_data['Fare_std'] = (train_data['Fare'] - np.mean(train_data['Fare'])) / np.std(train_data['Fare'])

train_data['Fare_std'] = train_data['Fare_std'].where(cond=train_data['Fare_std'] < 3, other=3)

train_data['Fare_std'] = train_data['Fare_std'].where(cond=train_data['Fare_std'] > -3, other=-3)



# Embarked

mode = stats.mode(train_data['Embarked'])[0][0]

train_data['Embarked'].fillna(mode, inplace=True)

train_data = pd.concat([train_data, pd.get_dummies(train_data['Embarked'], prefix='Embarked')], axis=1)
target_var = train_data['Survived']



feature_cols = ['SibSp', 'Parch', 'Title_Miss/Mrs/Ms', 'Title_Mr', 'Title_Other',

                'Pclass_1', 'Pclass_2', 'Pclass_3', 'female', 'male', 'Age_std', 'Ticket_prep', 'Fare_std',

                'Embarked_C', 'Embarked_Q', 'Embarked_S']



features = train_data[feature_cols]
# Build a forest and compute the impurity-based feature importances

forest = RandomForestClassifier(n_estimators=250, random_state=101)



forest.fit(features, target_var)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

indices = np.argsort(importances)[::-1]



# Plot the feature importances of the forest

plt.figure(figsize=(12,8))

plt.title("Feature importances")

plt.bar(range(features.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(features.shape[1]), features.columns[indices], rotation=45)

plt.subplots_adjust(bottom=0.3)

plt.show()
from sklearn.inspection import permutation_importance



result = permutation_importance(forest, features, target_var, n_repeats=10, random_state=101)



sorted_idx = result.importances_mean.argsort()



fig, ax = plt.subplots()

ax.boxplot(result.importances[sorted_idx].T,

           vert=False, labels=features.columns[sorted_idx])

ax.set_title("Permutation Importances (test set)")

fig.tight_layout()

plt.show()
train_data = pd.read_csv('/kaggle/input/titanic/train.csv', index_col=0)
# Name --> transform to categorical

train_data['Title'] = train_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]



train_data['Title'] = train_data['Title'].replace(['Miss', 'Mrs','Ms'], 'Miss/Mrs/Ms')

train_data['Title'] = train_data['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev',

                                                   'Master', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Other')



# Pclass --> dummy encodding



# Sex --> dummy encodding



# Age --> imputation and standardization



# SibSp --> no preprocessing



# Parch --> no preprocessing



# Ticket --> transform to numerical

train_data['Ticket_prep'] = train_data.groupby('Ticket')['Ticket'].transform('count')



# Cabin --> no use



# Fare --> imputation and standardization



# Embarked --> imputation and dummy encodding
target_var = train_data['Survived']



feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',

                   'Fare', 'Embarked', 'Title', 'Ticket_prep']



numeric_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'Ticket_prep']

categoric_cols = ['Pclass', 'Sex', 'Embarked', 'Title']



features = train_data[feature_cols]
# Creating Pipeline for fitting several different data preprocessing and models



def defineBestModelPipeline(features, target, categorical_columns, numeric_columns):    

    

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=100)



    # Numerical features preprocessing

    num_prep_1 = Pipeline(steps=[('imp', KNNImputer(n_neighbors=10)),

                                  ('scaler', MinMaxScaler())])

    

    num_prep_2 = Pipeline(steps=[('imp', KNNImputer(n_neighbors=10)),

                                  ('scaler', StandardScaler())])

    

    num_prep_3 = Pipeline(steps=[('imp', SimpleImputer(strategy='mean')),

                                  ('scaler', MinMaxScaler())])

    

    num_prep_4 = Pipeline(steps=[('imp', SimpleImputer(strategy='mean')),

                                  ('scaler', StandardScaler())])

    

    num_prep_5 = Pipeline(steps=[('imp', SimpleImputer(strategy='median')),

                                  ('scaler', MinMaxScaler())])

    

    num_prep_6 = Pipeline(steps=[('imp', SimpleImputer(strategy='median')),

                                  ('scaler', StandardScaler())])



    

    # Categorical features transformation

    cat_prep = Pipeline(steps=[('frequent', SimpleImputer(strategy='most_frequent')),

                                ('onehot', OneHotEncoder(use_cat_names=True))])

    

    

    # Combining both numerical and categorical preprocessing pipelines



    data_preprocessing_1 = ColumnTransformer(transformers=[('num', num_prep_1, numeric_cols),

                                                             ('cat', cat_prep, categoric_cols)])

    

    data_preprocessing_2 = ColumnTransformer(transformers=[('num', num_prep_2, numeric_cols),

                                                             ('cat', cat_prep, categoric_cols)])

    

    data_preprocessing_3 = ColumnTransformer(transformers=[('num', num_prep_3, numeric_cols),

                                                             ('cat', cat_prep, categoric_cols)])

    

    data_preprocessing_4 = ColumnTransformer(transformers=[('num', num_prep_4, numeric_cols),

                                                             ('cat', cat_prep, categoric_cols)])

    

    data_preprocessing_5 = ColumnTransformer(transformers=[('num', num_prep_5, numeric_cols),

                                                             ('cat', cat_prep, categoric_cols)])



    data_preprocessing_6 = ColumnTransformer(transformers=[('num', num_prep_6, numeric_cols),

                                                             ('cat', cat_prep, categoric_cols)])

    

    

    # Initialize a Pipeline object as the function being optimized in RandomSearchCV

    pipe = Pipeline(steps=[('data_transformations', data_preprocessing_1), # Initializing data preprocessing step

                           ('clf', SVC())]) # Initializing modeling step

    

    

    # Now, we define the grid of parameters that RandomSearchCV will use. 

    params_grid = [

                    {'data_transformations': [data_preprocessing_1, data_preprocessing_2, data_preprocessing_3, 

                                              data_preprocessing_4, data_preprocessing_5, data_preprocessing_6],

                     'clf': [KNeighborsClassifier()],

                     'clf__n_neighbors': np.random.RandomState(100).randint(2,100,10),

                     'clf__metric': ['minkowski', 'euclidean']},



                    {'data_transformations': [data_preprocessing_1, data_preprocessing_2, data_preprocessing_3, 

                                              data_preprocessing_4, data_preprocessing_5, data_preprocessing_6],

                     'clf': [LogisticRegression(random_state=5, max_iter=1000 , class_weight='balanced')],

                     'clf__C': np.random.RandomState(100).rand(10)},

        

                    {'data_transformations': [data_preprocessing_1, data_preprocessing_2, data_preprocessing_3, 

                                              data_preprocessing_4, data_preprocessing_5, data_preprocessing_6],

                     'clf': [SVC(random_state=5)],

                     'clf__C': np.random.RandomState(100).rand(10),

                     'clf__gamma': np.random.RandomState(100).rand(10)},

        

                    {'data_transformations': [data_preprocessing_1, data_preprocessing_2, data_preprocessing_3, 

                                              data_preprocessing_4, data_preprocessing_5, data_preprocessing_6],

                     'clf': [DecisionTreeClassifier()],

                     'clf__criterion': ['gini', 'entropy'],

                     'clf__max_features': [None, "auto", "log2"],

                     'clf__min_samples_leaf': np.random.RandomState(100).uniform(0.001, 0.05, 20)},

        

                    {'data_transformations': [data_preprocessing_1, data_preprocessing_2, data_preprocessing_3, 

                                              data_preprocessing_4, data_preprocessing_5, data_preprocessing_6],

                     'clf': [RandomForestClassifier(random_state=5, class_weight='balanced')],

                     'clf__n_estimators': np.random.RandomState(100).randint(100,300,20),

                     'clf__max_features': [None, "auto", "log2"],

                     'clf__max_depth': [None, stats.randint(1, 5)]},                    

        

                    {'data_transformations': [data_preprocessing_1, data_preprocessing_2, data_preprocessing_3, 

                                              data_preprocessing_4, data_preprocessing_5, data_preprocessing_6],

                     'clf': [ExtraTreesClassifier(random_state=5, class_weight='balanced')],

                     'clf__n_estimators': np.random.RandomState(100).randint(100,300,20),

                     'clf__max_features': [None, "auto", "log2"],

                     'clf__max_depth': [None, stats.randint(1, 6)]}

    ]

    

    # Fitting RandomSearchCV to search over the grid of parameters

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    

    best_model_pipeline = RandomizedSearchCV(pipe, params_grid, n_iter=100, 

                                             scoring=metrics, refit='accuracy', 

                                             n_jobs=-1, cv=5, random_state=101)

    best_model_pipeline.fit(x_train, y_train)

    

    return x_train, x_test, y_train, y_test, best_model_pipeline
# Calling the function above, returing train/test data and best model's pipeline

x_train, x_test, y_train, y_test, best_model_pipeline = defineBestModelPipeline(features, target_var, categoric_cols, numeric_cols)
print(best_model_pipeline.best_estimator_)

print(best_model_pipeline.best_score_)
# Function responsible for checking our model's performance on the test data

def Classifier_performance(classifier, x_test, y_test):

    predictions = classifier.predict(x_test)

    

    results = []

    f1 = np.round(f1_score(y_test, predictions), 2)

    precision = np.round(precision_score(y_test, predictions), 2)

    recall = np.round(recall_score(y_test, predictions), 2)

    roc_auc = np.round(roc_auc_score(y_test, predictions), 2)

    accuracy = np.round(accuracy_score(y_test, predictions), 2)

    

    results.append(f1)

    results.append(precision)

    results.append(recall)

    results.append(roc_auc)

    results.append(accuracy)

    

    print("\n\n#---------------- Test set results (Best Classifier) ----------------#\n")

    print("F1 score, Precision, Recall, ROC_AUC score, Accuracy:")

    print(results)

    

    return results
# Checking best model's performance on test data

test_set_results = Classifier_performance(best_model_pipeline, x_test, y_test)
pd.DataFrame(best_model_pipeline.cv_results_)
sample_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

sample_submission.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv', index_col=0)

test_data.head()
test_data['Title'] = test_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]



test_data['Title'] = test_data['Title'].replace(['Miss', 'Mrs','Ms'], 'Miss/Mrs/Ms')

test_data['Title'] = test_data['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev',

                                                   'Master', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Other')



test_data['Ticket_prep'] = test_data.groupby('Ticket')['Ticket'].transform('count')



features_test = test_data[feature_cols]



# Applying best model on the test data file

predictions = best_model_pipeline.predict(features_test)



# Creating submission file

test_data['Survived'] = predictions

submission_df = test_data.reset_index()[['PassengerId', 'Survived']]

submission_df
submission_df.to_csv('First_submission.csv', index=False)