import pandas as pd

import numpy as np

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation





import math, time, random, datetime

import seaborn as sns

import missingno



from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize





from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier



import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-whitegrid')

%matplotlib inline 

sns.set(color_codes=True)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_gender_sub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
df_train.head(10)
df_test.head(10)
df_gender_sub.head(10)
print("\n Train Data  Shape\n", end = " ")

print(df_train.shape)

print("\n Train Data  Columns\n", end = " ")

print(df_train.columns)



print("\n Train Data Description\n", end = " ")

df_train.describe()
print("\n Data Frame Info\n", end = " ")

df_train.info()
print("\n Get the number of missing values from data frame\n")

df_train.isna().sum()

#df_train.isnull().sum()
missingno.matrix(df_train, figsize=(30,10))
df_train.dtypes
df_train.head(10)
## To Perform Data analysis, let's create two new dataframes



df_bin = pd.DataFrame()

df_con = pd.DataFrame()

df_train["Survived"].value_counts().plot.barh()

print(df_train["Survived"].value_counts())
# Let's add this to our subset dataframes

df_bin['Survived'] = df_train['Survived']

df_con['Survived'] = df_train['Survived']
print(df_train["Pclass"].value_counts())

df_train["Pclass"].value_counts().plot.bar()

sns.distplot(df_train.Pclass)
df_train.Pclass.isna().sum()
df_train.groupby("Pclass")["Survived"].value_counts().plot.bar()



print(df_train.groupby("Pclass")["Survived"].value_counts())
df_bin['Pclass'] = df_train['Pclass']

df_con['Pclass'] = df_train['Pclass']
df_train["Name"].value_counts()
df_train["Name"].isna().sum()
df_train["Sex"].value_counts().plot.barh()

print(df_train["Sex"].value_counts())
df_bin['Sex'] = np.where(df_train['Sex'] == 'female', 1, 0)

df_con['Sex'] = df_train['Sex']
sns.distplot(df_bin.Sex)
df_train.groupby("Sex")["Survived"].value_counts().plot.bar()

#df_train.groupby("Sex")["Survived"].value_counts().plot.pie()



print(df_train.groupby("Sex")["Survived"].value_counts())
df_train["Age"].value_counts().plot.bar()

print(df_train["Age"].value_counts())
df_train["Age"].isna().sum()
df_train["SibSp"].value_counts().plot.bar()

print(df_train["SibSp"].value_counts())
df_train.groupby("SibSp")["Survived"].value_counts().plot.bar()



print(df_train.groupby("SibSp")["Survived"].value_counts())
df_train["Parch"].value_counts().plot.bar()

print(df_train["Parch"].value_counts())
df_train.groupby("Parch")["Survived"].value_counts().plot.bar()



print(df_train.groupby("Parch")["Survived"].value_counts())
df_train["Parch"].isna().sum()
df_train.head()
df_train["Ticket"].value_counts().plot.bar()

print(df_train["Ticket"].value_counts())
df_train.groupby("Ticket")["Survived"].value_counts().plot.bar()



print(df_train.groupby("Ticket")["Survived"].value_counts())
df_train["Ticket"].isna().sum()
df_train["Fare"].value_counts().plot.bar()

print(df_train["Fare"].value_counts())
df_train.groupby("Fare")["Survived"].value_counts().plot.bar()



print(df_train.groupby("Fare")["Survived"].value_counts())
df_train["Fare"].isna().sum()
df_bin['Fare'] = pd.cut(df_train['Fare'], bins=5) 
df_train["Cabin"].value_counts().plot.bar()

print(df_train["Cabin"].value_counts())
df_train.groupby("Cabin")["Survived"].value_counts().plot.bar()



print(df_train.groupby("Cabin")["Survived"].value_counts())
df_train["Cabin"].isna().sum()
df_train["Embarked"].value_counts().plot.bar()

print(df_train["Embarked"].value_counts())
df_train.groupby("Embarked")["Survived"].value_counts().plot.bar()



print(df_train.groupby("Embarked")["Survived"].value_counts())
df_train["Embarked"].isna().sum()
df_bin["SibSp"] = df_train["SibSp"]

df_bin["Parch"] = df_train["Parch"]

df_bin["Embarked"] = df_train["Embarked"]
df_bin.head()
df_con["SibSp"] = df_train["SibSp"]

df_con["Parch"] = df_train["Parch"]

df_con["Embarked"] = df_train["Embarked"]

df_con["Fare"] = df_train["Fare"]
df_con.head()
df_bin = df_bin.dropna(subset=["Embarked"])

df_con = df_con.dropna(subset=["Embarked"])
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



# Drop the original categorical columns (because now they've been one hot encoded)

df_con_enc = df_con_enc.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
df_con_enc.head()
X_train = df_con_enc.drop("Survived", axis=1)

y_train = df_con_enc["Survived"]
X_train.head()
y_train.head()
def ml_algorithm(algo, X_train, y_train, cv):



  model = algo.fit(X_train, y_train)

  acc = round(model.score(X_train, y_train)* 100, 2)



  train_pred = model_selection.cross_val_predict(algo, 

                                                  X_train, 

                                                  y_train, 

                                                  cv=cv, 

                                                  n_jobs = -1)

  

  acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    

  return train_pred, acc, acc_cv

start_time = time.time()

train_pred_log, acc_log, acc_cv_log = ml_algorithm(LogisticRegression(), 

                                                               X_train, 

                                                               y_train, 

                                                                    10)

log_time = (time.time() - start_time)

print("Accuracy: %s" % acc_log)

print("Accuracy CV 10-Fold: %s" % acc_cv_log)

print("Running Time: %s" % datetime.timedelta(seconds=log_time))
# k-Nearest Neighbours

start_time = time.time()

train_pred_knn, acc_knn, acc_cv_knn = ml_algorithm(KNeighborsClassifier(), 

                                                  X_train, 

                                                  y_train, 

                                                  10)

knn_time = (time.time() - start_time)

print("Accuracy: %s" % acc_knn)

print("Accuracy CV 10-Fold: %s" % acc_cv_knn)

print("Running Time: %s" % datetime.timedelta(seconds=knn_time))
# Gaussian Naive Bayes

start_time = time.time()

train_pred_gaussian, acc_gaussian, acc_cv_gaussian = ml_algorithm(GaussianNB(), 

                                                                      X_train, 

                                                                      y_train, 

                                                                           10)

gaussian_time = (time.time() - start_time)

print("Accuracy: %s" % acc_gaussian)

print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)

print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time))
# Linear SVC

start_time = time.time()

train_pred_svc, acc_linear_svc, acc_cv_linear_svc = ml_algorithm(LinearSVC(),

                                                                X_train, 

                                                                y_train, 

                                                                10)

linear_svc_time = (time.time() - start_time)

print("Accuracy: %s" % acc_linear_svc)

print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)

print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))
# Stochastic Gradient Descent

start_time = time.time()

train_pred_sgd, acc_sgd, acc_cv_sgd = ml_algorithm(SGDClassifier(), 

                                                  X_train, 

                                                  y_train,

                                                  10)

sgd_time = (time.time() - start_time)

print("Accuracy: %s" % acc_sgd)

print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)

print("Running Time: %s" % datetime.timedelta(seconds=sgd_time))
# Decision Tree Classifier

start_time = time.time()

train_pred_dt, acc_dt, acc_cv_dt = ml_algorithm(DecisionTreeClassifier(), 

                                                                X_train, 

                                                                y_train,

                                                                10)

dt_time = (time.time() - start_time)

print("Accuracy: %s" % acc_dt)

print("Accuracy CV 10-Fold: %s" % acc_cv_dt)

print("Running Time: %s" % datetime.timedelta(seconds=dt_time))
start_time = time.time()

train_pred_gbt, acc_gbt, acc_cv_gbt = ml_algorithm(GradientBoostingClassifier(), 

                                                                       X_train, 

                                                                       y_train,

                                                                       10)

gbt_time = (time.time() - start_time)

print("Accuracy: %s" % acc_gbt)

print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)

print("Running Time: %s" % datetime.timedelta(seconds=gbt_time))
models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree', 'Gradient Boosting Trees'],

    'Score': [

        acc_knn, 

        acc_log,  

        acc_gaussian, 

        acc_sgd, 

        acc_linear_svc, 

        acc_dt,

        acc_gbt

    ]})

print("---Regular Accuracy Scores---")

models.sort_values(by='Score', ascending=False)
cv_models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree', 'Gradient Boosting Trees'],

    'Score': [

        acc_cv_knn, 

        acc_cv_log,      

        acc_cv_gaussian, 

        acc_cv_sgd, 

        acc_cv_linear_svc, 

        acc_cv_dt,

        acc_cv_gbt

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
grad_boost_model = GradientBoostingClassifier()

grad_boost_model.fit(X_train, y_train)



print("Accuracy for GradBoost Algorithm is {}".format(round(grad_boost_model.score(X_train, y_train)* 100, 2)))



train_pred = model_selection.cross_val_predict(GradientBoostingClassifier(), 

                                                  X_train, 

                                                  y_train, 

                                                  cv=10, 

                                                  n_jobs = -1)

  

print("Accuracy for GradBoost Algorithm using cross validation is {} ".format(round(metrics.accuracy_score(y_train, train_pred) * 100, 2)))

# Plot the feature importance scores

feature_importance(grad_boost_model, X_train)
df_test.head()
df_embarked_one_hot = pd.get_dummies(df_test['Embarked'], 

                                     prefix='embarked')



df_sex_one_hot = pd.get_dummies(df_test['Sex'], 

                                prefix='sex')



df_plcass_one_hot = pd.get_dummies(df_test['Pclass'], 

                                   prefix='pclass')
df_test_enc = pd.concat([df_test, 

                        df_embarked_one_hot, 

                        df_sex_one_hot, 

                        df_plcass_one_hot], axis=1)



# Drop the original categorical columns (because now they've been one hot encoded)

df_test_enc = df_test_enc.drop(['Pclass', 'Sex', 'Embarked', 'Age', 'Name', 'Cabin', 'Ticket', 'PassengerId' ], axis=1)
df_test_enc.head()
df_test_enc.isna().sum()
df_test_enc["Fare"]=df_test_enc["Fare"].fillna(df_test_enc["Fare"].mean())
df_test_enc.isna().sum()
predictions = grad_boost_model.predict(df_test_enc)
submission = pd.DataFrame()

submission['PassengerId'] = df_test['PassengerId']

submission['Survived'] = predictions # our model predictions on the test dataset

submission.head(10)
print(len(df_gender_sub))

print(len(df_test))
df_gender_sub.head(10)