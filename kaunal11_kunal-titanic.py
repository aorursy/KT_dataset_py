# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Import Dependencies

%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import missingno as msno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



# Machine learning

import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier, Pool, cv



# Let's ignore warnings for now

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import Train and Test Data

train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')

gender_submission=pd.read_csv('../input/titanic/gender_submission.csv')
# View the Training data

train.head()
# View the Test data

test.head()
# View the Gender Submission data

gender_submission.head()
train.describe()
train.dtypes
## Plot graphic of missing values

msno.matrix(train)
## Summarizing the missing values per column



train.isnull().sum()
df_bin = pd.DataFrame() # for discretised continuous variables

df_con = pd.DataFrame() # for continuous variables
def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):

    """

    Function to plot counts and distributions of a label variable and 

    target variable side by side.

    ::param_data:: = target dataframe

    ::param_bin_df:: = binned dataframe for countplot

    ::param_label_column:: = binary labelled column

    ::param_target_column:: = column you want to view counts and distributions

    ::param_figsize:: = size of figure (width, height)

    ::param_use_bin_df:: = whether or not to use the bin_df, default False

    """

    if use_bin_df: 

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=bin_df);

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], 

                     kde_kws={"label": "Survived","bw":0.1});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={"label": "Did not survive","bw":0.1});

    else:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=data);

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], 

                     kde_kws={"label": "Survived","bw":0.1});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={"label": "Did not survive","bw":0.1});
# How many people survived?

fig = plt.figure(figsize=(20,1))

sns.countplot(data=train,y='Survived')

train['Survived'].value_counts()
# Let's add this to our subset dataframes

df_bin['Survived'] = train['Survived']

df_con['Survived'] = train['Survived']
sns.distplot(train['Pclass'])
df_bin['Pclass'] = train['Pclass']

df_con['Pclass'] = train['Pclass']
train['Name'].value_counts()
# Distribution of males and females



fig = plt.figure(figsize=(20,1))

sns.countplot(data=train,y='Sex')

train['Sex'].value_counts()
# add Sex to the subset dataframes

df_bin['Sex'] = train['Sex']

df_bin['Sex'] = np.where(df_bin['Sex'] == 'female', 1, 0) # change sex to 0 for male and 1 for female



df_con['Sex'] = train['Sex']
# How does the Sex variable look compared to Survival?



# pie chart for male and female survivors

labels = ['Survived', 'Not survived']

male_sizes = [train[(train['Sex']=='male')&(train['Survived']==1)].count()[0],train[(train['Sex']=='male')&(train['Survived']==0)].count()[0]]

female_sizes = [train[(train['Sex']=='female')&(train['Survived']==1)].count()[0],train[(train['Sex']=='female')&(train['Survived']==0)].count()[0]]

         

# print(sizes) # adds up to 1433, which is the total number of participants

fig1, ax1 = plt.subplots()

ax1.pie(male_sizes, labels=labels, autopct='%1.1f%%', shadow=True)

ax1.axis('equal')

plt.title('Male Survivals')

plt.show()



fig2, ax2 = plt.subplots()

ax2.pie(female_sizes, labels=labels, autopct='%1.1f%%', shadow=True)

ax2.axis('equal')

plt.title('Female Survivals')

plt.show()
# Find missing values

train['Age'].isnull().sum()
train['SibSp'].value_counts()
# Add SibSp to subset dataframes

df_bin['SibSp'] = train['SibSp']

df_con['SibSp'] = train['SibSp']
# Visualise the counts of SibSp and the distribution of the values

# against Survived

plot_count_dist(train, 

                bin_df=df_bin, 

                label_column='Survived', 

                target_column='SibSp', 

                figsize=(20, 10))
# Add Parch to subset dataframes

df_bin['Parch'] = train['Parch']

df_con['Parch'] = train['Parch']
# Visualise the counts of Parch and the distribution of the values

# against Survived

plot_count_dist(train, 

                bin_df=df_bin,

                label_column='Survived', 

                target_column='Parch', 

                figsize=(20, 10))
# Add Fare to sub dataframes

df_con['Fare'] = train['Fare'] 

df_bin['Fare'] = pd.cut(train['Fare'], bins=5) # discretised
# Visualise the Fare bin counts as well as the Fare distribution versus Survived.

plot_count_dist(data=train,

                bin_df=df_bin,

                label_column='Survived', 

                target_column='Fare', 

                figsize=(20,10), 

                use_bin_df=True)
df_bin.head()
df_con.head()
# Visualise the Embarking Ports 

sns.countplot(y='Embarked',data=train)
# Add Embarked to sub dataframes

df_bin['Embarked'] = train['Embarked']

df_con['Embarked'] = train['Embarked']
# Remove Embarked rows which are missing values

print(len(df_con))

df_con = df_con.dropna(subset=['Embarked'])

df_bin = df_bin.dropna(subset=['Embarked'])

print(len(df_con))
df_con.head()
df_bin.head()
# One-hot encode binned variables

one_hot_cols = df_bin.columns.tolist()

one_hot_cols.remove('Survived')

df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)



df_bin_enc.head()
# One hot encode the categorical columns

df_embarked_one_hot = pd.get_dummies(df_con['Embarked'], 

                                     prefix='embarked')



df_sex_one_hot = pd.get_dummies(df_con['Sex'], 

                                prefix='sex')



df_plcass_one_hot = pd.get_dummies(df_con['Pclass'], 

                                   prefix='pclass')
# Combine the one hot encoded columns with df_con_enc

df_con_enc = pd.concat([df_con, 

                        df_embarked_one_hot, 

                        df_sex_one_hot, 

                        df_plcass_one_hot], axis=1)



# Drop the original categorical columns (because now they've been one hot encoded)

df_con_enc = df_con_enc.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
# Let's look at df_con_enc

df_con_enc.head()
# Split the dataframe into data and labels

X_train = df_con_enc.drop('Survived', axis=1) # data

y_train = df_con_enc.Survived # labels
# Function that runs the requested algorithm and returns the accuracy metrics

def fit_ml_algo(algo, X_train, y_train, cv):

    

    # One Pass

    model = algo.fit(X_train, y_train)

    acc = round(model.score(X_train, y_train) * 100, 2)

    

    # Cross Validation 

    train_pred = model_selection.cross_val_predict(algo, 

                                                  X_train, 

                                                  y_train, 

                                                  cv=cv, 

                                                  n_jobs = -1)

    # Cross-validation accuracy metric

    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    

    return train_pred, acc, acc_cv
# Logistic Regression



train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(), 

                                                               X_train, 

                                                               y_train, 

                                                                    10)



print("Accuracy: %s" % acc_log)

print("Accuracy CV 10-Fold: %s" % acc_cv_log)
# k-Nearest Neighbours



train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(), 

                                                  X_train, 

                                                  y_train, 

                                                  10)



print("Accuracy: %s" % acc_knn)

print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
# Gaussian Naive Bayes



train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(GaussianNB(), 

                                                                      X_train, 

                                                                      y_train, 

                                                                           10)



print("Accuracy: %s" % acc_gaussian)

print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)
# Linear SVC



train_pred_svc, acc_linear_svc, acc_cv_linear_svc = fit_ml_algo(LinearSVC(),

                                                                X_train, 

                                                                y_train, 

                                                                10)



print("Accuracy: %s" % acc_linear_svc)

print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)
# Stochastic Gradient Descent



train_pred_sgd, acc_sgd, acc_cv_sgd = fit_ml_algo(SGDClassifier(), 

                                                  X_train, 

                                                  y_train,

                                                  10)



print("Accuracy: %s" % acc_sgd)

print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)
# Decision Tree Classifier



train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(), 

                                                                X_train, 

                                                                y_train,

                                                                10)

print("Accuracy: %s" % acc_dt)

print("Accuracy CV 10-Fold: %s" % acc_cv_dt)
# Gradient Boost Trees

train_pred_gbt, acc_gbt, acc_cv_gbt = fit_ml_algo(GradientBoostingClassifier(), 

                                                                       X_train, 

                                                                       y_train,

                                                                       10)



print("Accuracy: %s" % acc_gbt)

print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)
# Define the categorical features for the CatBoost model

cat_features = np.where(X_train.dtypes != np.float)[0]

cat_features
# Use the CatBoost Pool() function to pool together the training data and categorical feature labels

train_pool = Pool(X_train, 

                  y_train,

                  cat_features)
# CatBoost model definition

catboost_model = CatBoostClassifier(iterations=1000,

                                    custom_loss=['Accuracy'],

                                    loss_function='Logloss')



# Fit CatBoost model

catboost_model.fit(train_pool,

                   plot=True)



# CatBoost accuracy

acc_catboost = round(catboost_model.score(X_train, y_train) * 100, 2)
# Set params for cross-validation as same as initial model

cv_params = catboost_model.get_params()



# Run the cross-validation for 10-folds (same as the other models)

cv_data = cv(train_pool,

             cv_params,

             fold_count=10,

             plot=True)





# CatBoost CV results save into a dataframe (cv_data), let's withdraw the maximum accuracy score

acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)
# Print out the CatBoost model metrics

print("---CatBoost Metrics---")

print("Accuracy: {}".format(acc_catboost))

print("Accuracy cross-validation 10-Fold: {}".format(acc_cv_catboost))

models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree', 'Gradient Boosting Trees',

              'CatBoost'],

    'Score': [

        acc_knn, 

        acc_log,  

        acc_gaussian, 

        acc_sgd, 

        acc_linear_svc, 

        acc_dt,

        acc_gbt,

        acc_catboost

    ]})

print("---Reuglar Accuracy Scores---")

models.sort_values(by='Score', ascending=False)


cv_models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree', 'Gradient Boosting Trees',

              'CatBoost'],

    'Score': [

        acc_cv_knn, 

        acc_cv_log,      

        acc_cv_gaussian, 

        acc_cv_sgd, 

        acc_cv_linear_svc, 

        acc_cv_dt,

        acc_cv_gbt,

        acc_cv_catboost

    ]})

print('---Cross-validation Accuracy Scores---')

cv_models.sort_values(by='Score', ascending=False)
# Feature Importance

def feature_importance(model, data):

    """

    Function to show which features are most important in the model.

    ::param_model:: Which model to use?

    ::param_data:: What data to use?

    """

    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': data.columns})

    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]

    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))

    return fea_imp

    #plt.savefig('catboost_feature_importance.png')
# Plot the feature importance scores

feature_importance(catboost_model, X_train)
metrics = ['Precision', 'Recall', 'F1', 'AUC']



eval_metrics = catboost_model.eval_metrics(train_pool,

                                           metrics=metrics,

                                           plot=True)



for metric in metrics:

    print(str(metric)+": {}".format(np.mean(eval_metrics[metric])))
X_train.head()
test.head()
# One hot encode the columns in the test data frame (like X_train)

test_embarked_one_hot = pd.get_dummies(test['Embarked'], 

                                       prefix='embarked')



test_sex_one_hot = pd.get_dummies(test['Sex'], 

                                prefix='sex')



test_plcass_one_hot = pd.get_dummies(test['Pclass'], 

                                   prefix='pclass')
# Combine the test one hot encoded columns with test

test = pd.concat([test, 

                  test_embarked_one_hot, 

                  test_sex_one_hot, 

                  test_plcass_one_hot], axis=1)
# Let's look at test, it should have one hot encoded columns now

test.head()
# Create a list of columns to be used for the predictions

wanted_test_columns = X_train.columns

wanted_test_columns
# Make a prediction using the CatBoost model on the wanted columns

predictions = catboost_model.predict(test[wanted_test_columns])
# Create a submisison dataframe and append the relevant columns

submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = predictions # our model predictions on the test dataset

submission.head()
# What does our submission have to look like?

gender_submission.head()


# Let's convert our submission dataframe 'Survived' column to ints

submission['Survived'] = submission['Survived'].astype(int)

print('Converted Survived column to integers.')
submission.head()
# Are our test and submission dataframes the same length?

if len(submission) == len(test):

    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))

else:

    print("Dataframes mismatched, won't be able to submit to Kaggle.")
# Convert submisison dataframe to csv for submission to csv 

# for Kaggle submisison

submission.to_csv('catboost_submission.csv', index=False)

print('Submission CSV is ready!')