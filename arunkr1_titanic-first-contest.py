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

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv') # example of what a submission should look like
train.head()
train.Age.plot.hist()
test.head() 
gender_submission.head()
train.describe()
missingno.matrix(train, figsize = (30,10))
train.isnull().sum()
df_bin = pd.DataFrame() 

df_con = pd.DataFrame() 
train.dtypes
train.head()
fig = plt.figure(figsize=(20,1))

sns.countplot(y='Survived', data=train);

print(train.Survived.value_counts())
df_bin['Survived'] = train['Survived']

df_con['Survived'] = train['Survived']
df_bin.head()
df_con.head()
sns.distplot(train.Pclass)
train.Pclass.isnull().sum()
df_bin['Pclass'] = train['Pclass']

df_con['Pclass'] = train['Pclass']
train.Name.value_counts()
plt.figure(figsize=(20, 5))

sns.countplot(y="Sex", data=train);
train.Sex.isnull().sum()
train.Sex.head()
df_bin['Sex'] = train['Sex']

df_bin['Sex'] = np.where(df_bin['Sex'] == 'female', 1, 0) 



df_con['Sex'] = train['Sex']
df_bin.head()
df_con.head()
# How does the Sex variable look compared to Survival?

# We can see this because they're both binarys.

fig = plt.figure(figsize=(10, 10))

try:

    sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'])

except RuntimeError as re:

    if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):

        sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'bw': 0.1, 'label': 'survived'})

    else:

        raise re

# sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'label': 'Survived'});



try:

    sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'])

except RuntimeError as re:

    if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):

        sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'bw': 0.1, 'label': 'Did not survive'})

    else:

        raise re

# sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'label': 'Did not survive'});
sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'bw': 0.1, 'label': 'survived'})

sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'bw': 0.1, 'label': 'Did not survive'});
# How many missing values does age have?

train.Age.isnull().sum()
def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):

    

    if use_bin_df: 

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=bin_df);

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], 

                     kde_kws={'bw' : 0.1, "label": "Survived"});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={'bw' : 0.1,"label": "Did not survive"});

    else:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=data);

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], 

                     kde_kws={'bw' : 0.1,"label": "Survived"});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={'bw' : 0.1,"label": "Did not survive"});
# How many missing values does SibSp have?

train.SibSp.isnull().sum()
# What values are there?

train.SibSp.value_counts()
df_bin['SibSp'] = train['SibSp']

df_con['SibSp'] = train['SibSp']
plot_count_dist(train, 

                bin_df=df_bin, 

                label_column='Survived', 

                target_column='SibSp', 

                figsize=(20, 10))
# How many missing values does Parch have?

train.Parch.isnull().sum()
train.Parch.value_counts()
df_bin['Parch'] = train['Parch']

df_con['Parch'] = train['Parch']
plot_count_dist(train, 

                bin_df=df_bin,

                label_column='Survived', 

                target_column='Parch', 

                figsize=(20, 10))
train.head()
df_con.head()
train.Ticket.isnull().sum()


sns.countplot(y="Ticket", data=train);
train.Ticket.value_counts()
print("There are {} unique Ticket values.".format(len(train.Ticket.unique())))
df_bin.head()
df_con.head()
train.Fare.isnull().sum()
sns.countplot(y="Fare", data=train);
train.Fare.dtype
print("There are {} unique Fare values.".format(len(train.Fare.unique())))
# Add Fare to sub dataframes

df_con['Fare'] = train['Fare'] 

df_bin['Fare'] = pd.cut(train['Fare'], bins=5) # discretised 
df_bin.Fare.value_counts()
plot_count_dist(data=train,

                bin_df=df_bin,

                label_column='Survived', 

                target_column='Fare', 

                figsize=(20,10), 

                use_bin_df=True)
# How many missing values does Cabin have?

train.Cabin.isnull().sum()
train.head()
train.Cabin.value_counts()
train.Embarked.isnull().sum()
train.Embarked.value_counts()
sns.countplot(y='Embarked', data=train);
df_bin['Embarked'] = train['Embarked']

df_con['Embarked'] = train['Embarked']
print(len(df_con))

df_con = df_con.dropna(subset=['Embarked'])

df_bin = df_bin.dropna(subset=['Embarked'])

print(len(df_con))
df_bin.head()
# One-hot encode binned variables

one_hot_cols = df_bin.columns.tolist()

one_hot_cols.remove('Survived')

df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)



df_bin_enc.head()
df_con.head(10)


df_embarked_one_hot = pd.get_dummies(df_con['Embarked'], 

                                     prefix='embarked')



df_sex_one_hot = pd.get_dummies(df_con['Sex'], 

                                prefix='sex')



df_plcass_one_hot = pd.get_dummies(df_con['Pclass'], 

                                   prefix='pclass')
df_con_enc = pd.concat([df_con, 

                        df_embarked_one_hot, 

                        df_sex_one_hot, 

                        df_plcass_one_hot], axis=1)





df_con_enc = df_con_enc.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
df_con_enc.head(20)


selected_df = df_con_enc
selected_df.head()
# Split the dataframe into data and labels

X_train = selected_df.drop('Survived', axis=1) # data

y_train = selected_df.Survived # labels
# Shape of the data (without labels)

X_train.shape
X_train.head()
# Shape of the labels

y_train.shape
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

                                                               y_train, 

                                                                    10)

log_time = (time.time() - start_time)

print("Accuracy: %s" % acc_log)

print("Accuracy CV 10-Fold: %s" % acc_cv_log)

print("Running Time: %s" % datetime.timedelta(seconds=log_time))
# k-Nearest Neighbours

start_time = time.time()

train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(), 

                                                  X_train, 

                                                  y_train, 

                                                  10)

knn_time = (time.time() - start_time)

print("Accuracy: %s" % acc_knn)

print("Accuracy CV 10-Fold: %s" % acc_cv_knn)

print("Running Time: %s" % datetime.timedelta(seconds=knn_time))
# Gaussian Naive Bayes

start_time = time.time()

train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(GaussianNB(), 

                                                                      X_train, 

                                                                      y_train, 

                                                                           10)

gaussian_time = (time.time() - start_time)

print("Accuracy: %s" % acc_gaussian)

print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)

print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time))
# Linear SVC

start_time = time.time()

train_pred_svc, acc_linear_svc, acc_cv_linear_svc = fit_ml_algo(LinearSVC(),

                                                                X_train, 

                                                                y_train, 

                                                                10)

linear_svc_time = (time.time() - start_time)

print("Accuracy: %s" % acc_linear_svc)

print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)

print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))
# Stochastic Gradient Descent

start_time = time.time()

train_pred_sgd, acc_sgd, acc_cv_sgd = fit_ml_algo(SGDClassifier(), 

                                                  X_train, 

                                                  y_train,

                                                  10)

sgd_time = (time.time() - start_time)

print("Accuracy: %s" % acc_sgd)

print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)

print("Running Time: %s" % datetime.timedelta(seconds=sgd_time))
# Decision Tree Classifier

start_time = time.time()

train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(), 

                                                                X_train, 

                                                                y_train,

                                                                10)

dt_time = (time.time() - start_time)

print("Accuracy: %s" % acc_dt)

print("Accuracy CV 10-Fold: %s" % acc_cv_dt)

print("Running Time: %s" % datetime.timedelta(seconds=dt_time))
# Gradient Boosting Trees

start_time = time.time()

train_pred_gbt, acc_gbt, acc_cv_gbt = fit_ml_algo(GradientBoostingClassifier(), 

                                                                       X_train, 

                                                                       y_train,

                                                                       10)

gbt_time = (time.time() - start_time)

print("Accuracy: %s" % acc_gbt)

print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)

print("Running Time: %s" % datetime.timedelta(seconds=gbt_time))
X_train.head()
y_train.head()
# Define the categorical features for the CatBoost model

cat_features = np.where(X_train.dtypes != np.float)[0]

cat_features
train_pool = Pool(X_train, 

                  y_train,

                  cat_features)
y_train.head()
# CatBoost model definition

catboost_model = CatBoostClassifier(iterations=1000,

                                    custom_loss=['Accuracy'],

                                    loss_function='Logloss')



# Fit CatBoost model

catboost_model.fit(train_pool,

                   plot=True)



# CatBoost accuracy

acc_catboost = round(catboost_model.score(X_train, y_train) * 100, 2)
start_time = time.time()



cv_params = catboost_model.get_params()



cv_data = cv(train_pool,

             cv_params,

             fold_count=10,

             plot=True)



catboost_time = (time.time() - start_time)





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
feature_importance(catboost_model, X_train)
metrics = ['Precision', 'Recall', 'F1', 'AUC']



eval_metrics = catboost_model.eval_metrics(train_pool,

                                           metrics=metrics,

                                           plot=True)



for metric in metrics:

    print(str(metric)+": {}".format(np.mean(eval_metrics[metric])))
X_train.head()
test.head()
test_embarked_one_hot = pd.get_dummies(test['Embarked'], 

                                       prefix='embarked')



test_sex_one_hot = pd.get_dummies(test['Sex'], 

                                prefix='sex')



test_plcass_one_hot = pd.get_dummies(test['Pclass'], 

                                   prefix='pclass')
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

submission['Survived'] = predictions 

submission.head()
gender_submission.head()
submission['Survived'] = submission['Survived'].astype(int)

print('Converted Survived column to integers.')
submission.head()
submission.to_csv('../catboost_submission.csv', index=False)

print('Submission CSV is ready!')
submissions_check = pd.read_csv("../catboost_submission.csv")

submissions_check.head()