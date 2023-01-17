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

# Import Dependencies

%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import missingno

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



# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')
# Import train & test data 

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv') # example of what a submission should look like
# View the training data

train.head(15)
train.Age.plot.hist()
# View the test data (same columns as the training data)

test.head() # head = view first 5 lines
# View the example submisison dataframe

gender_submission.head()
train.describe()
# Plot graphic of missing values

missingno.matrix(train, figsize = (30,5))
#No of null values

train.isnull().sum()
df_bin = pd.DataFrame() # for discretised continuous variables

df_con = pd.DataFrame() # for continuous variables
train.dtypes
# How many people survived?

fig = plt.figure(figsize=(20,1))

sns.countplot(y='Survived', data=train)

print(train.Survived.value_counts())
df_bin['Survived'] = train['Survived']

df_con['Survived'] = train['Survived']
df_bin.head()
df_con.head()
sns.distplot(train.Pclass)
# missing variables does Pclass have?

train.Pclass.isnull().sum()
df_bin['Pclass'] = train['Pclass']

df_con['Pclass'] = train['Pclass']



df_bin.head()
train.Name.value_counts()

# distribution of Sex

plt.figure(figsize=(20, 5))

sns.countplot(y="Sex", data=train);
train.Sex.isnull().sum()
#Add sex to the dataset 

df_bin['Sex'] = train['Sex']

df_bin['Sex'] = np.where(df_bin['Sex']=='female',1,0)

df_con['Sex'] = train['Sex']
df_bin.head()

#df_con.head()
# How does the Sex variable look compared to Survival?

# We can see this because they're both binarys.

fig = plt.figure(figsize=(10, 10))

sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'label': 'Survived'});

sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'label': 'Did not survive'});
#Missing values in Age variable

train.Age.isnull().sum()
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

                     kde_kws={"label": "Survived"});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={"label": "Did not survive"});

    else:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=data);

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], 

                     kde_kws={"label": "Survived"});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={"label": "Did not survive"});
train.SibSp.isnull().sum()
train.SibSp.value_counts()
df_bin['SibSp'] = train['SibSp']

df_con['SibSp'] = train['SibSp']
# Visualise the counts of SibSp and the distribution of the values

# against Survived

plot_count_dist(train, 

                bin_df=df_bin, 

                label_column='Survived', 

                target_column='SibSp', 

                figsize=(20, 10))
train.Parch.isnull().sum()
train.Parch.value_counts()
df_bin['Parch'] = train['Parch']

df_con['Parch'] = train['Parch']
# Visualise the counts of Parch and the distribution of the values

# against Survived

plot_count_dist(train, 

                bin_df=df_bin,

                label_column='Survived', 

                target_column='Parch', 

                figsize=(20, 10))
df_con.head()
train.Ticket.isnull().sum()

sns.countplot(y="Ticket", data=train);
# How many kinds of ticket are there?

train.Ticket.value_counts()

print("There are {} unique Ticket values.".format(len(train.Ticket.unique())))
train.Fare.isnull().sum()

sns.countplot(y='Fare',data = train)
train.Fare.dtype
print("There are {} unique Fare values.".format(len(train.Fare.unique())))

# Add Fare to sub dataframes

df_con['Fare'] = train['Fare'] 

df_bin['Fare'] = pd.cut(train['Fare'], bins=5) # discretised
df_bin.Fare.value_counts()
# Visualise the Fare bin counts as well as the Fare distribution versus Survived.

plot_count_dist(data=train,

                bin_df=df_bin,

                label_column='Survived', 

                target_column='Fare', 

                figsize=(20,10), 

                use_bin_df=True)
train.Cabin.isnull().sum()
train.Embarked.isnull().sum()
train.Embarked.value_counts()
sns.countplot(y='Embarked', data = train)
# Add Embarked to sub dataframes

df_bin['Embarked'] = train['Embarked']

df_con['Embarked'] = train['Embarked']
# Remove Embarked rows which are missing values

print(len(df_con))

df_con = df_con.dropna(subset=['Embarked'])

df_bin = df_bin.dropna(subset=['Embarked'])

print(len(df_con))
df_bin.head()
one_hot_cols = df_bin.columns.tolist()

one_hot_cols.remove('Survived')

df_bin_enc = pd.get_dummies(df_bin,columns = one_hot_cols)



df_bin_enc.head()
df_con.head(10)
# One hot encode the categorical columns : Embarked, Sex, Pclass

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
df_con_enc.head(10)
selected_df = df_con_enc
selected_df.head()
selected_df.shape
X_train = selected_df.drop('Survived',axis = 1) #data

Y_train = selected_df.Survived #Labels
X_train.shape
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

start_time = time.time()

train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(), 

                                                               X_train, 

                                                               Y_train, 

                                                                    10)

log_time = (time.time() - start_time)

print("Accuracy: %s" % acc_log)

print("Accuracy CV 10-Fold: %s" % acc_cv_log)

print("Running Time: %s" % datetime.timedelta(seconds=log_time))
# k-Nearest Neighbours

start_time = time.time()

train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(), 

                                                  X_train, 

                                                  Y_train, 

                                                  10)

knn_time = (time.time() - start_time)

print("Accuracy: %s" % acc_knn)

print("Accuracy CV 10-Fold: %s" % acc_cv_knn)

print("Running Time: %s" % datetime.timedelta(seconds=knn_time))
# Gaussian Naive Bayes

start_time = time.time()

train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(GaussianNB(), 

                                                                      X_train, 

                                                                      Y_train, 

                                                                           10)

gaussian_time = (time.time() - start_time)

print("Accuracy: %s" % acc_gaussian)

print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)

print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time))
# Linear SVC

start_time = time.time()

train_pred_svc, acc_linear_svc, acc_cv_linear_svc = fit_ml_algo(LinearSVC(),

                                                                X_train, 

                                                                Y_train, 

                                                                10)

linear_svc_time = (time.time() - start_time)

print("Accuracy: %s" % acc_linear_svc)

print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)

print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))
# Stochastic Gradient Descent

start_time = time.time()

train_pred_sgd, acc_sgd, acc_cv_sgd = fit_ml_algo(SGDClassifier(), 

                                                  X_train, 

                                                  Y_train,

                                                  10)

sgd_time = (time.time() - start_time)

print("Accuracy: %s" % acc_sgd)

print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)

print("Running Time: %s" % datetime.timedelta(seconds=sgd_time))
# Decision Tree Classifier

start_time = time.time()

train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(), 

                                                                X_train, 

                                                                Y_train,

                                                                10)

dt_time = (time.time() - start_time)

print("Accuracy: %s" % acc_dt)

print("Accuracy CV 10-Fold: %s" % acc_cv_dt)

print("Running Time: %s" % datetime.timedelta(seconds=dt_time))
# Gradient Boosting Trees

start_time = time.time()

train_pred_gbt, acc_gbt, acc_cv_gbt = fit_ml_algo(GradientBoostingClassifier(), 

                                                                       X_train, 

                                                                       Y_train,

                                                                       10)

gbt_time = (time.time() - start_time)

print("Accuracy: %s" % acc_gbt)

print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)

print("Running Time: %s" % datetime.timedelta(seconds=gbt_time))
# View the data for the CatBoost model

X_train.head()


# Define the categorical features for the CatBoost model

cat_features = np.where(X_train.dtypes != np.float)[0]

cat_features
# Use the CatBoost Pool() function to pool together the training data and categorical feature labels

train_pool = Pool(X_train, 

                  Y_train,

                  cat_features)


# CatBoost model definition

catboost_model = CatBoostClassifier(iterations=1000,

                                    custom_loss=['Accuracy'],

                                    loss_function='Logloss')



# Fit CatBoost model

catboost_model.fit(train_pool,

                   plot=True)



# CatBoost accuracy

acc_catboost = round(catboost_model.score(X_train, Y_train) * 100, 2)
# How long will this take?

start_time = time.time()



# Set params for cross-validation as same as initial model

cv_params = catboost_model.get_params()



# Run the cross-validation for 10-folds (same as the other models)

cv_data = cv(train_pool,

             cv_params,

             fold_count=10,

             plot=True)



# How long did it take?

catboost_time = (time.time() - start_time)



# CatBoost CV results save into a dataframe (cv_data), let's withdraw the maximum accuracy score

acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)
# Print out the CatBoost model metrics

print("---CatBoost Metrics---")

print("Accuracy: {}".format(acc_catboost))

print("Accuracy cross-validation 10-Fold: {}".format(acc_cv_catboost))

print("Running Time: {}".format(datetime.timedelta(seconds=catboost_time)))
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
test.head()
wanted_test_columns = X_train.columns

wanted_test_columns
predictions = catboost_model.predict(test[wanted_test_columns])
predictions[:20]

submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = predictions # our model predictions on the test dataset

submission.head()
submission['Survived'] = submission['Survived'].astype(int)

submission.head()
if len(submission) == len(test):

    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))

else:

    print("Dataframes mismatched, won't be able to submit to Kaggle.")
submission.to_csv('catboost_submission.csv', index=False)

print('Submission CSV is ready!')


# Check the submission csv to make sure it's in the right format

submissions_check = pd.read_csv("catboost_submission.csv")

submissions_check.head()