# import dependencies

import math, time, random, datetime



# data manipulation

import numpy as np

import pandas as pd



# visualization

%matplotlib inline

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



# machine learning

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



# ignore warnings

import warnings

warnings.filterwarnings('ignore')
!ls '../input'
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

gender_submission = pd.read_csv('../input/gender_submission.csv')
train.head()
test.head()
gender_submission.head()
print(train.shape)

print(test.shape)

print(gender_submission.shape)
print(len(train))

print(len(test))

print(len(gender_submission))
train['Survived'].sum()
train.describe()
train.head()
missingno.matrix(train, figsize=(30,10))
train.columns
def find_missing_values(df, columns):

    missing_vals = {}

    print("Number of missing or NaN values for each column:")

    df_length = len(df)

    for column in columns:

        total_column_values = df[column].value_counts().sum()

        missing_vals[column] = df_length - total_column_values

    return missing_vals



missing_values = find_missing_values(train, columns=train.columns)

missing_values
df_bin = pd.DataFrame()

df_con = pd.DataFrame()
train.dtypes
train.head()
#fig = plt.figure(figsize=(1, 5))

sns.countplot(x='Survived', data=train)

print(train.Survived.value_counts())
df_bin['Survived'] = train['Survived']

df_con['Survived'] = train['Survived']
df_bin.head()
df_con.head()
train.Pclass.value_counts()
sns.distplot(train.Pclass)
sns.countplot(x='Pclass', data=train)
missing_values['Pclass']
df_bin['Pclass'] = train['Pclass']

df_con['Pclass'] = train['Pclass']
df_con.head()
train.Name.value_counts()
len(train)
train.Sex.value_counts()
sns.countplot(x='Sex', data=train)
missing_values['Sex']
df_bin['Sex'] = train['Sex']

df_bin['Sex'] = np.where(df_bin['Sex'] == 'female', 1, 0)



df_con['Sex'] = train['Sex']
df_bin.head()
train.head()
fig = plt.figure(figsize=(10, 10))

sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'label': 'Survived'})          # female

sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'label': 'Did Not Survive'})    # male
train.Survived.value_counts()
train.groupby(['Survived', 'Sex']).size()
df_bin.groupby(['Survived', 'Sex']).size()
missing_values['Age']
def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):

    if use_bin_df:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=bin_df)

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column],

                    kde_kws={'label': 'Survived'})

        sns.distplot(data.loc[data[label_column] == 0][target_column],

                    kde_kws={'label': 'Did Not Survived'})

    else:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=bin_df)

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column],

                    kde_kws={'label': 'Survived'})

        sns.distplot(data.loc[data[label_column] == 0][target_column],

                    kde_kws={'label': 'Did Not Survived'})
missing_values['SibSp']
train.SibSp.value_counts()
df_bin['SibSp'] = train['SibSp']

df_con['SibSp'] = train['SibSp']
plot_count_dist(train,

                bin_df=df_bin,

                label_column='Survived',

                target_column='SibSp',

                figsize=(20,5))
missing_values['Parch']
df_bin['Parch'] = train['Parch']

df_con['Parch'] = train['Parch']
plot_count_dist(train,

                bin_df=df_bin,

                label_column='Survived',

                target_column='Parch',

                figsize=(20,5))
train.head()
df_con.head()
missing_values['Ticket']
sns.countplot(y='Ticket', data=train)
train.Ticket.value_counts()
len(train.Ticket.unique())
missing_values['Fare']
sns.countplot(y='Fare', data=train)
train.Fare.dtype
len(train.Fare.unique())
df_con['Fare'] = train['Fare']

df_bin['Fare'] = pd.cut(train['Fare'], bins=5)
df_con.head()
df_bin.head()
df_bin.Fare.value_counts()
plot_count_dist(train,

               bin_df=df_bin,

               label_column='Survived',

               target_column='Fare',

               figsize=(20,10),

               use_bin_df=True)
missing_values['Cabin']
train.Cabin.value_counts()
len(train.Cabin.unique())
missing_values['Embarked']
train.Embarked.value_counts()
train.Embarked.value_counts().sum()
sns.countplot(y='Embarked', data=train)
df_bin['Embarked'] = train['Embarked']

df_con['Embarked'] = train['Embarked']
print(len(df_con))

df_con = df_con.dropna(subset=['Embarked'])

df_bin = df_bin.dropna(subset=['Embarked'])

print(len(df_con))
train.head()
df_con.head()
df_bin.head()
one_hot_cols = df_bin.columns.tolist()

one_hot_cols.remove('Survived')

df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)



df_bin_enc.head()
df_con.head()
df_con_enc = df_con.apply(LabelEncoder().fit_transform)

df_con_enc.head()
selected_df = df_con_enc
X_train = selected_df.drop('Survived', axis=1)

y_train = selected_df.Survived
X_train.head()
y_train.head()
def fit_ml_algo(algo, X_train, y_train, cv):

    

    model = algo.fit(X_train, y_train)

    acc = round(model.score(X_train, y_train) * 100, 2)

    

    train_pred = model_selection.cross_val_predict(algo,

                                                  X_train,

                                                  y_train,

                                                  cv=cv,

                                                  n_jobs = -1)

    

    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    

    return train_pred, acc, acc_cv
start_time = time.time()

train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(),

                                                 X_train,

                                                 y_train,

                                                 10)



log_time = (time.time() - start_time)

print('Accuracy: %s' % acc_log)

print('Accuracy CV 10-Fold %s' % acc_cv_log)

print('Running Time: %s' % datetime.timedelta(seconds=log_time))
start_time = time.time()

train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(),

                                                 X_train,

                                                 y_train,

                                                 10)



log_time = (time.time() - start_time)

print('Accuracy: %s' % acc_knn)

print('Accuracy CV 10-Fold %s' % acc_cv_knn)

print('Running Time: %s' % datetime.timedelta(seconds=log_time))
start_time = time.time()

train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(GaussianNB(),

                                                 X_train,

                                                 y_train,

                                                 10)



log_time = (time.time() - start_time)

print('Accuracy: %s' % acc_gaussian)

print('Accuracy CV 10-Fold %s' % acc_cv_gaussian)

print('Running Time: %s' % datetime.timedelta(seconds=log_time))
start_time = time.time()

train_pred_linear_svc, acc_linear_svc, acc_cv_linear_svc = fit_ml_algo(LinearSVC(),

                                                 X_train,

                                                 y_train,

                                                 10)



log_time = (time.time() - start_time)

print('Accuracy: %s' % acc_linear_svc)

print('Accuracy CV 10-Fold %s' % acc_cv_linear_svc)

print('Running Time: %s' % datetime.timedelta(seconds=log_time))
start_time = time.time()

train_pred_sgd, acc_sgd, acc_cv_sgd = fit_ml_algo(SGDClassifier(),

                                                 X_train,

                                                 y_train,

                                                 10)



log_time = (time.time() - start_time)

print('Accuracy: %s' % acc_sgd)

print('Accuracy CV 10-Fold %s' % acc_cv_sgd)

print('Running Time: %s' % datetime.timedelta(seconds=log_time))
start_time = time.time()

train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(),

                                                 X_train,

                                                 y_train,

                                                 10)



log_time = (time.time() - start_time)

print('Accuracy: %s' % acc_dt)

print('Accuracy CV 10-Fold %s' % acc_cv_dt)

print('Running Time: %s' % datetime.timedelta(seconds=log_time))
start_time = time.time()

train_pred_gbt, acc_gbt, acc_cv_gbt = fit_ml_algo(GradientBoostingClassifier(),

                                                 X_train,

                                                 y_train,

                                                 10)



log_time = (time.time() - start_time)

print('Accuracy: %s' % acc_gbt)

print('Accuracy CV 10-Fold %s' % acc_cv_gbt)

print('Running Time: %s' % datetime.timedelta(seconds=log_time))
X_train.head()
y_train.head()
cat_features = np.where(X_train.dtypes != np.float)[0]

cat_features
train_pool = Pool(X_train,

                 y_train,

                 cat_features)
catboost_model = CatBoostClassifier(iterations=1000,

                                   custom_loss=['Accuracy'],

                                   loss_function='Logloss')



catboost_model.fit(train_pool,

                  plot=True)



acc_catboost = round(catboost_model.score(X_train, y_train) * 100, 2)
start_time = time.time()



cv_params = catboost_model.get_params()



cv_data = cv(train_pool,

            cv_params,

            fold_count=10,

            plot=True)



catboost_time = (time.time() - start_time)



acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)
print("---CatBoost Metrics---")

print("Accuracy: {}".format(acc_catboost))

print("Accuracy 10-Fold: {}".format(acc_cv_catboost))

print("Running Time: {}".format(datetime.timedelta(seconds=catboost_time)))
models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes',  

              'Stochastic Gradient Descent', 'Linear SVC',  

              'Decision Tree', 'Gradient Boosting Trees', 

              'CatBoost'],

    'Score' : [acc_knn,

               acc_log,

               acc_gaussian,

               acc_sgd,

               acc_linear_svc,

               acc_dt,

               acc_gbt,

               acc_catboost]

})



print("---Regular Accuracy Scores---")

models.sort_values(by='Score', ascending=False)
models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes',  

              'Stochastic Gradient Descent', 'Linear SVC',  

              'Decision Tree', 'Gradient Boosting Trees', 

              'CatBoost'],

    'Score' : [acc_cv_knn,

               acc_cv_log,

               acc_cv_gaussian,

               acc_cv_sgd,

               acc_cv_linear_svc,

               acc_cv_dt,

               acc_cv_gbt,

               acc_cv_catboost]

})



print("---Cross-validation Accuracy Scores---")

models.sort_values(by='Score', ascending=False)
models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes',  

              'Stochastic Gradient Descent', 'Linear SVC',  

              'Decision Tree', 'Gradient Boosting Trees', 

              'CatBoost'],

    'Reg Score' : [acc_knn,

               acc_log,

               acc_gaussian,

               acc_sgd,

               acc_linear_svc,

               acc_dt,

               acc_gbt,

               acc_catboost],

    'CV Score' : [acc_cv_knn,

               acc_cv_log,

               acc_cv_gaussian,

               acc_cv_sgd,

               acc_cv_linear_svc,

               acc_cv_dt,

               acc_cv_gbt,

               acc_cv_catboost]

})



print("---Cross-validation Accuracy Scores---")

models.sort_values(by='CV Score', ascending=False)
def feature_importance(model, data):

    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': data.columns})

    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[False, False]).iloc[-30:]

    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=[20, 5])

    _.invert_yaxis()

    return fea_imp
feature_importance(catboost_model, X_train)
metrics = ['Precision', 'Recall', 'F1', 'AUC']



eval_metrics = catboost_model.eval_metrics(train_pool,

                                          metrics=metrics,

                                          plot=True)



for metric in metrics:

    print(str(metric)+": {}".format(np.mean(eval_metrics[metric])))
wanted_test_columns = X_train.columns

wanted_test_columns
predictions = catboost_model.predict(test[wanted_test_columns]

                                    .apply(LabelEncoder().fit_transform))
predictions[:20]
y_train.head()
submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = predictions

submission.head()
gender_submission.head()
submission['Survived'] = submission['Survived'].astype(int)
submission.head()
if len(submission) == len(test):

    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))

else:

    print("DataFrame mismatched, won't be able to submit to Kaggle.")
submission.to_csv('catboost_submission.csv', index=False)
submissions_check = pd.read_csv('catboost_submission.csv')

submissions_check.head()