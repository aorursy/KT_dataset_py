import math,time,random, datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

from sklearn.preprocessing import OneHotEncoder,LabelEncoder,label_binarize


import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection,tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression,LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool,cv

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
# looking at the training data
train.head()
# looking at the test data
test.head()
# length of training and test data
len(train) , len(test)
# Submission FIle
gender_submission.head()
train.describe()
# Plotting graphic of missing values
missingno.matrix(train,figsize=(30,10))
train.columns
# let's write a function to see how many missing values are there
def finding_missing_values(df,column):
    missing_values = {}
    df_length=len(df)
    for col in column:
        total_column_values = df[col].value_counts().sum()
        missing_values[col] = df_length-total_column_values
    return missing_values

missing_values = finding_missing_values(train,column=train.columns)
missing_values
df_bin = pd.DataFrame() # for Discritisized continuous variables
df_con = pd.DataFrame() # for continuous variables
train.dtypes
# How many Survived
fig = plt.figure(figsize=(20,1))
sns.countplot(y='Survived',data = train)

train.Survived.value_counts()
# let's add this to our subset dataframe
df_bin['Survived'] = train['Survived']
df_con['Survived'] = train['Survived']
df_bin.head()
df_con.head()
# lets see the distribution
sns.distplot(train.Pclass)
# adding to out sub dataframes
df_bin['Pclass'] = train['Pclass']
df_con['Pclass'] = train['Pclass']
df_con.head()
train.Name.value_counts()
# count of each gender
fig = plt.figure(figsize=(20,5))
sns.countplot(y='Sex',data=train)
# adding sex to subset dataframes
df_bin['Sex'] = train['Sex']

# female-1, male -0
df_bin['Sex'] = np.where(df_bin['Sex']=='female' ,1,0)

df_con['Sex'] = train['Sex']
df_bin.head()
missing_values['Age']
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
train.SibSp.value_counts()
df_bin['SibSp'] = train['SibSp']
df_con['SibSp'] = train['SibSp']
plot_count_dist(train,
                bin_df =df_bin,
               label_column = 'Survived',
               target_column = 'SibSp',
               figsize=(20,10))
train.Parch.value_counts()
df_bin['Parch'] = train['Parch']
df_con['Parch'] = train['Parch']
sns.countplot(y='Ticket',data=train)
train.Ticket.value_counts()
sns.countplot(y='Fare',data=train)
train.Fare.value_counts()
train.Fare.dtype
df_con['Fare'] = train['Fare']
df_bin['Fare'] = pd.cut(train['Fare'],bins=5)
df_bin.Fare.value_counts()
plot_count_dist(train,
                bin_df =df_bin,
               label_column = 'Survived',
               target_column = 'Fare',
               figsize=(20,10),
               use_bin_df=True)
# How many missing values does Cabin have?
train.Cabin.isnull().sum()
# How many missing values does Embarked have?
train.Embarked.isnull().sum()
# What kind of values are in Embarked?
train.Embarked.value_counts()
# What do the counts look like?
sns.countplot(y='Embarked', data=train);
# Add Embarked to sub dataframes
df_bin['Embarked'] = train['Embarked']
df_con['Embarked'] = train['Embarked']
# Remove Embarked rows which are missing values
print(len(df_con))
df_con = df_con.dropna(subset=['Embarked'])
df_bin = df_bin.dropna(subset=['Embarked'])
print(len(df_con))
df_bin.head()
# One hot encodding binned variables
one_hot_cols = df_bin.columns.tolist()
one_hot_cols.remove('Survived')
df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)

df_bin_enc.head()
df_con.head()
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
df_con_enc.head()
selected_df= df_con_enc
selected_df.head()
# Split the dataframe into data and labels
X_train = selected_df.drop('Survived', axis=1) # data
y_train = selected_df.Survived # labels
# Shape of the data 
X_train.shape,y_train.shape
def fit_ml_algo(algo,X_train,y_train,cv):
    
    model = algo.fit(X_train,y_train)
    acc= round(model.score(X_train,y_train)*100,2)
    
    # Cross validation
    train_pred = model_selection.cross_val_predict(algo,
                                                  X_train,
                                                  y_train,
                                                  cv=cv,
                                                  n_jobs=-1)
    
    # Cross validation accuracy metrics
    acc_cv= round(metrics.accuracy_score(y_train,train_pred)*100,2)
    
    return train_pred,acc,acc_cv
train_pred, acc_log1,acc_cv_log1 = fit_ml_algo(LogisticRegression(),
                                            X_train,
                                            y_train,
                                            10)

acc_log1,acc_cv_log1
train_pred, acc_log2,acc_cv_log2 = fit_ml_algo(KNeighborsClassifier(),
                                            X_train,
                                            y_train,
                                            10)

acc_log2,acc_cv_log2
train_pred, acc_log3,acc_cv_log3 = fit_ml_algo(GaussianNB(),
                                            X_train,
                                            y_train,
                                            10)

acc_log3,acc_cv_log3
train_pred, acc_log4,acc_cv_log4 = fit_ml_algo(LinearSVC(),
                                            X_train,
                                            y_train,
                                            10)

acc_log4,acc_cv_log4
train_pred, acc_log5,acc_cv_log5 = fit_ml_algo(SGDClassifier(),
                                            X_train,
                                            y_train,
                                            10)

acc_log5,acc_cv_log5
train_pred, acc_log6,acc_cv_log6 = fit_ml_algo(DecisionTreeClassifier(),
                                            X_train,
                                            y_train,
                                            10)

acc_log6,acc_cv_log6
train_pred, acc_log7,acc_cv_log7 = fit_ml_algo(GradientBoostingClassifier(),
                                            X_train,
                                            y_train,
                                            10)

acc_log7,acc_cv_log7
# View the data for the CatBoost model
X_train.head()
y_train.head()
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
    'Model': [ 'Logistic Regression','KNN', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees',
              'CatBoost'],
    'Score': [
        acc_log1,
        acc_log2,
        acc_log3,
        acc_log5,
        acc_log4,
        acc_log6,
        acc_log7,
        acc_catboost,
    ]})
print("---Reuglar Accuracy Scores---")
models.sort_values(by='Score', ascending=False)

cv_models = pd.DataFrame({
    'Model': [ 'Logistic Regression','KNN', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees',
              'CatBoost'],
    'Score': [
        acc_cv_log1,
        acc_cv_log2,      
        acc_cv_log3,
        acc_cv_log5, 
        acc_cv_log4,
        acc_cv_log6,
        acc_cv_log7,
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
# Plot the feature importance scores
feature_importance(catboost_model, X_train)
metrics = ['Precision', 'Recall', 'F1', 'AUC']

eval_metrics = catboost_model.eval_metrics(train_pool,
                                           metrics=metrics,
                                           plot=True)

for metric in metrics:
    print(str(metric)+": {}".format(np.mean(eval_metrics[metric])))
# We need our test dataframe to look like this one
X_train.head()
# Our test dataframe has some columns our model hasn't been trained on
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
# Our predictions array is comprised of 0's and 1's (Survived or Did Not Survive)
predictions[:20]
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
# How does our submission dataframe look?
submission.head()
# Are our test and submission dataframes the same length?
if len(submission) == len(test):
    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))
else:
    print("Dataframes mismatched, won't be able to submit to Kaggle.")
# Convert submisison dataframe to csv for submission to csv 
# for Kaggle submisison
submission.to_csv('../catboost_submission.csv', index=False)
print('Submission CSV is ready!')
# Check the submission csv to make sure it's in the right format
submissions_check = pd.read_csv("../catboost_submission.csv")
submissions_check.head()
