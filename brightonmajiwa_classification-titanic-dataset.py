

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Data Manipulation

import  pandas as pd

import numpy as np

#Visualization

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')

# Processing

!pip install sklearn

from sklearn.preprocessing import OneHotEncoder,LabelEncoder,LabelBinarizer
# Machine Learning

!pip install catboost

import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection,tree , preprocessing,metrics,linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression,LogisticRegression,SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier,Pool,cv

from sklearn.neighbors import KNeighborsClassifier

import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

train_df.head()
test_df.head()
# Columns in the Dataset

train_df.columns
# Brief Description of the Dataset

train_df.describe(

)
train_df.shape

# 418 rows , 11 columns
# Spotting Missing Values in the Training Dataset

missingno.matrix(train_df,figsize=(30,5))
# Finding The exact number of missing values in the dataframe

train_columns = train_df.columns

def find_missing_values(df,columns):

    missing_vals = {}

    print("number of missing values for each column")

    df_length =len(df)

    

    for column in columns:

        total_column_values =df[column].value_counts().sum()

        missing_vals[column] =  df_length - total_column_values

        

    return missing_vals

    

# function call passing in train dataset and it'slist of columns

missing_values = find_missing_values(train_df,train_columns)

missing_values
df_bin = pd.DataFrame()

df_con = pd.DataFrame()
fig = plt.figure(figsize=(20,2))

sns.countplot(y='Survived',data=train_df)

print(train_df.Survived.value_counts())
# added the survived column to the two subdataframes

df_bin['Survived'] = train_df['Survived']

df_con['Survived'] = train_df['Survived']

# Test

df_bin.head()

sns.distplot(train_df.Pclass)
missing_values['Pclass']
df_bin['Pclass'] = train_df['Pclass']

df_con['Pclass'] = train_df['Pclass']

df_con.head()
train_df.Name.value_counts()
plt.figure(figsize = (20,5))

sns.countplot(y='Sex',data=train_df)
missing_values['Sex']
df_bin['Sex'] = train_df['Sex']

# For the binned dataset we change the Male to 0 and Female to 1

df_bin['Sex']= np.where(df_bin['Sex'] == 'female',1,0)

#df_bin['Sex']

df_bin.head()
# for continious dataset

df_con['Sex'] = train_df['Sex']

df_con.head()
fig = plt.figure(figsize=(8,5))

sns.distplot(df_bin.loc[df_bin['Survived']==1]['Sex'],kde_kws={'label':'Survived'})

sns.distplot(df_bin.loc[df_bin['Survived']==0]['Sex'],kde_kws={'label':'Did not Survive'})
missing_values['Age']
missing_values['SibSp']
train_df.SibSp.value_counts()
df_bin['SibSp'] = train_df['SibSp']

df_con['SibSp'] = train_df['SibSp']

df_bin.head()
def plot_count_dist(data,bin_df,label_column,target_column,figsize=(20,5),use_bin_df=False):

    if use_bin_df:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1,2,1)

        sns.countplot(y=target_column,data=bin_df)

        plt.subplot(1,2,2)

        sns.distplot(data.loc[data[label_column]==1][target_column],kde_kws={'label':'Survived'})

        sns.distplot(data.loc[data[label_column]==0][target_column],kde_kws={'label':'Did Not Survive'})

    else:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1,2,1)

        sns.countplot(y=target_column,data=data)

        plt.subplot(1,2,2)

        sns.distplot(data.loc[data[label_column]==1][target_column],kde_kws={'label':'Survived'})

        sns.distplot(data.loc[data[label_column]==0][target_column],kde_kws={'label':'Did Not Survive'})

        

        
plot_count_dist(train_df,bin_df=df_bin,label_column='Survived',target_column='SibSp',figsize=(20,10))
train_df.Parch.value_counts()
missing_values['Parch']
df_bin['Parch'] = train_df['Parch']

df_bin.head()
plot_count_dist(train_df,

                bin_df=df_bin,

                label_column='Survived',

               target_column='Parch',

               figsize=(20,10))
missing_values['Ticket']
train_df.head()
train_df.Ticket.value_counts()
sns.countplot(y='Ticket',data=train_df)
print('There are {} unique fare values'.format(len(train_df.Fare.unique())))
train_df.Fare
df_con['Fare'] =train_df['Fare']

df_bin['Fare'] = pd.cut(train_df['Fare'],bins=5)

df_bin.Fare.sample(20)
plot_count_dist(data=train_df,

               bin_df=df_bin,

               label_column='Survived',

               target_column='Fare',

               figsize=(20,10),

               use_bin_df=True)
missing_values['Cabin']
train_df.Cabin.value_counts()
train_df.Embarked
train_df.Embarked.value_counts()
train_df.Embarked.value_counts().sum()
sns.countplot(y='Embarked',data=train_df)
missing_values['Embarked']
df_con['Embarked'] = train_df['Embarked']

df_bin['Embarked'] = train_df['Embarked']

df_bin.head()

print(len(df_con))

df_con.head()
df_con = df_con.dropna(subset=['Embarked'])

df_bin = df_bin.dropna(subset=['Embarked'])

print(len(df_con))
# select a list of columns to encode

one_hot_cols =df_bin.columns.tolist()

one_hot_cols.remove('Survived')

print(one_hot_cols)

#Ecoding

df_bin_enc = pd.get_dummies(df_bin,columns=one_hot_cols)

df_bin_enc.head()
df_con_enc = df_con.apply(LabelEncoder().fit_transform)

df_con_enc.head()
selected_df = df_con_enc

x_train = selected_df.drop('Survived',axis=1)# axis = 1 implies vertically

print(x_train.shape)

x_train.head()
y_train = selected_df.Survived

print(y_train.shape)

y_train.head()
import time

import datetime
def fit_ml_algo(algo,x_train,y_train,cv):

    #One Pass(trains once)

    model = algo.fit(x_train,y_train)

    acc = round(model.score(x_train,y_train)*100,2)# accuracy in percentage rounded of to 2dp

    

    #cross validation

    train_pred = model_selection.cross_val_predict(

        algo,

        x_train,

        y_train,

        cv=cv,

        n_jobs=-1)

    # cross validation accuracy metrics

    acc_cv = round(metrics.accuracy_score(y_train,train_pred)*100,2)

    

    return train_pred, acc, acc_cv
start_time = time.time()

# unpacking what will be returned by the (fit_ml_algo)function

train_ped_logreg,acc_logreg,acc_cv_logreg = fit_ml_algo(LogisticRegression(),x_train,y_train,10)

logreg_time =(time.time()-start_time)

print('Accuracy: %s' %acc_logreg)

print('Accuracy of cv 10-fold: %s' %acc_cv_logreg)

print('Running Time: %s' %datetime.timedelta(seconds=logreg_time))
start_time = time.time()

# unpacking what will be returned by the (fit_ml_algo)function

train_ped_knn,acc_knn,acc_cv_knn = fit_ml_algo(KNeighborsClassifier(),x_train,y_train,10)

logreg_time =(time.time()-start_time)

print('Accuracy: %s' %acc_knn)

print('Accuracy of cv 10-fold: %s' %acc_cv_knn)

print('Running Time: %s' %datetime.timedelta(seconds=logreg_time))
start_time = time.time()

# unpacking what will be returned by the (fit_ml_algo)function

train_ped_gaussian,acc_gaussian,acc_cv_gaussian = fit_ml_algo(GaussianNB(),x_train,y_train,10)

logreg_time =(time.time()-start_time)

print('Accuracy: %s' %acc_gaussian)

print('Accuracy of cv 10-fold: %s' %acc_cv_gaussian)

print('Running Time: %s' %datetime.timedelta(seconds=logreg_time))
start_time = time.time()

# unpacking what will be returned by the (fit_ml_algo)function

train_ped_linSVC,acc_linSVC,acc_cv_linSVC = fit_ml_algo(LinearSVC(),x_train,y_train,10)

logreg_time =(time.time()-start_time)

print('Accuracy: %s' %acc_linSVC)

print('Accuracy of cv 10-fold: %s' %acc_cv_linSVC)

print('Running Time: %s' %datetime.timedelta(seconds=logreg_time))
start_time = time.time()

# unpacking what will be returned by the (fit_ml_algo)function

train_ped_sgd,acc_sgd,acc_cv_sgd = fit_ml_algo(SGDClassifier(),x_train,y_train,10)

logreg_time =(time.time()-start_time)

print('Accuracy: %s' %acc_sgd)

print('Accuracy of cv 10-fold: %s' %acc_cv_sgd)

print('Running Time: %s' %datetime.timedelta(seconds=logreg_time))
start_time = time.time()

# unpacking what will be returned by the (fit_ml_algo)function

train_ped_dtree,acc_dtree,acc_cv_dtree = fit_ml_algo(DecisionTreeClassifier(),x_train,y_train,10)

logreg_time =(time.time()-start_time)

print('Accuracy: %s' %acc_dtree)

print('Accuracy of cv 10-fold: %s' %acc_cv_dtree)

print('Running Time: %s' %datetime.timedelta(seconds=logreg_time))
start_time = time.time()

# unpacking what will be returned by the (fit_ml_algo)function

train_ped_gbc,acc_gbc,acc_cv_gbc = fit_ml_algo(GradientBoostingClassifier(),x_train,y_train,10)

logreg_time =(time.time()-start_time)

print('Accuracy: %s' %acc_gbc)

print('Accuracy of cv 10-fold: %s' %acc_cv_gbc)

print('Running Time: %s' %datetime.timedelta(seconds=logreg_time))
"""

 can tell catboost what categorical features you have in your dataset

 so it knows how to handle them

"""

# defining Categorical features

cat_features = np.where(x_train.dtypes != np.float)[0]

cat_features

train_pool = Pool(x_train,y_train,cat_features)
# Model Definition

catboost_model = CatBoostClassifier(iterations=1000,

                                   custom_loss=['Accuracy'],

                                   loss_function='Logloss')

# Fit(Train the model)

catboost_model.fit(train_pool,plot=True)

# Training Accuracy

acc_catboost = round(catboost_model.score(x_train,y_train)*100,2)
start_time = time.time()

#set parameters

cv_params = catboost_model.get_params()

# run a cv of 10-Folds

cv_data = cv(train_pool,cv_params,fold_count=10,plot=True)

# How long did it take it?

catboost_time = (time.time()-start_time)

# GET Maximun accuracy score

acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean'])*100,2)
print('---(catboost Metrics)---')

print('Accuracy :{}'.format(acc_catboost))

print('Accuracy cv 10-fold :{}'.format(acc_cv_catboost))

print('Running Time  :{}'.format(datetime.timedelta(seconds=catboost_time)))
cv_models = pd.DataFrame({

    'Model':['Knn','Logistic Regression','Naive Bayes','Stochastic Gradient Descent','Linear SVC','Decision Tree',

             'Gradient Boosting Tree','catboost'],

    'scores':[acc_cv_knn,acc_cv_logreg,acc_cv_gaussian,acc_cv_sgd,acc_cv_linSVC,acc_cv_dtree,acc_cv_gbc,

              acc_cv_catboost]

})
# sort the datframe out

cv_models.sort_values(by='scores',ascending=False)
def feature_importance(model,data):

    fea_imp = pd.DataFrame({'imp':model.feature_importances_,

                           'col':data.columns})

    fea_imp = fea_imp.sort_values(['imp','col'],ascending=[True,False]).iloc[-30:]

    _ = fea_imp.plot(kind='barh',x='col',y='imp',figsize=(20,10))

    

    return fea_imp
feature_importance(catboost_model,x_train)
metrics = ['Precision','Recall','F1','AUC']

eval_metrics = catboost_model.eval_metrics(train_pool,metrics=metrics,plot=True)



for metric in metrics:

    print(str(metric) + ':{}'.format(np.mean(eval_metrics[metric])))

x_train.head()
# Columns we used in train are also going to be used in Testing 

wanted_test_columns = x_train.columns

print(wanted_test_columns)
"""

conduct prediction

first select the columns 

then encode then

predict

"""
predictions = catboost_model.predict(test_df[wanted_test_columns].apply(LabelEncoder().fit_transform))
# pick the first 19

predictions[:20]
# creating a submission dataframe

test_df.head()
submission =pd.DataFrame()

submission['PassengerId'] = test_df['PassengerId']

submission.head()
submission['Survived'] = predictions

submission.head()
# converting survived column to integer

submission['Survived']= submission['Survived'].astype(int)
print(len(submission))

print(len(test_df))
submission.to_csv('submission.csv',index=False)