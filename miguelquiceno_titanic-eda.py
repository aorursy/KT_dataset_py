import numpy as np # linear algebra

import pandas as pd # data processing



#python imports

import time, math, random, datetime



#Visualization

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')

%matplotlib inline



#Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



#Machine learning

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



# For ignoring warnings

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import train & test Data

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

# example of submission file
# view training data

train.head(10)
len(train) #Training set size
test.head() #View test Data
len(test) # test set size
gender_submission.head()
# Data description

train.describe()
# Missing values visualization(Data)

missingno.matrix(train, figsize = (30,10))
# Let's see how many missing values there are on each feature/column on the data

def find_missing_values(df, columns):

    missing_values = {}

    length = len(df)

    for feature in columns:

        total = df[feature].value_counts().sum()

        missing_values[feature] = length - total

        

    return missing_values



mv = find_missing_values(train, columns = train.columns)

mv  
# Another way to see the missing values of each column is this:

train.isnull().sum()
# We create two dataFrames for separate the discrete variables from the continuous variables

df_con = pd.DataFrame() # For continuous variables

df_dis = pd.DataFrame() # For discrete variables
# We check each dataType of the columns for explore them individually

train.dtypes
# Let's explore each feature individually, but first let's take another look at the data

train.head()
# We're gonna plot the feature 'Survived':

# And see how many people survived

fig = plt.figure(figsize = (20, 2))

sns.countplot(y = 'Survived', data = train)

print(train.Survived.value_counts())
# We add this feature to our subsets

df_con['Survived'] = train['Survived']

df_dis['Survived'] = train['Survived']
df_con.head()
df_dis.head()
# Now we'll look at the feature 'Pclass' by looking at the distribution for check outliers

sns.distplot(train['Pclass'])
mv['Pclass']

# train.Pclass.isnull().sum()
# add the feature to our subsets data

df_con['Pclass'] = train['Pclass']

df_dis['Pclass'] = train['Pclass']

df_con.head()
# The feature Name, we'll analize it with the function value_counts

train.Name.value_counts()
# Now lets pass to the feature sex.

# lets view the distribution of sex.

plt.figure(figsize = (25, 4))

sns.countplot(y = 'Sex', data = train)
# missing values on the column

train.Sex.head()
# lets add it to our subsets, but in the discret variable subset instead of male,female, we'll add it as 0, 1.

df_dis['Sex'] = train['Sex']

df_dis['Sex'] = np.where(df_dis['Sex'] == 'female', 1, 0)# 0 for male, 1 for female

df_con['Sex'] = train['Sex']
df_dis.head()
# Now, how does the Sex variable look compared to Survival?

# they're both binarys, we can see his

fig = plt.figure(figsize = (10, 10))

sns.distplot(df_dis.loc[df_dis['Survived'] == 1]['Sex'], kde_kws={'label': 'Survived'})

sns.distplot(df_dis.loc[df_dis['Survived'] == 0]['Sex'], kde_kws={'bw': 0.1, 'label': 'Did not survive'})
# How many missing values are in the feature Age?

train.Age.isnull().sum()
# For the moment we won't add this feature in our subsets, but we can also fill the missing values with Linear Regression
# Function that creates two diferent types of graphs

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

                     kde_kws={"label": "Did not survive", 'bw': 0.1});

    else:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=data);

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], 

                     kde_kws={"label": "Survived"});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={"label": "Did not survive", 'bw': 0.1});
# We will start with the feature SibSp by counting the missing values

train.SibSp.isnull().sum()
train.SibSp.value_counts()
# Lets add it to our subsets

df_con['SibSp'] = train['SibSp']

df_dis['SibSp'] = train['SibSp']
#We'll graph our info about the number of persons that had siblings on titanic

# the second graph is the info of the first compared to the survival ratio

plot_count_dist(train, bin_df = df_dis,

               label_column = 'Survived',

               target_column = 'SibSp',

               figsize = (20, 10))
# The Parch feature is very similar to SibSp, so the process to analize it is the same

train.Parch.isnull().sum()
train.Parch.value_counts()
df_con['Parch'] = train['Parch']

df_dis['Parch'] = train['Parch']
plot_count_dist(train, bin_df = df_dis,

               label_column = 'Survived',

               target_column = 'Parch',

               figsize = (20, 10))
# lets visualize one of our dataframes and check if everything is all right by checking also the original training data

df_con.head()
train.head()
# Ticket feature

train.Ticket.isnull().sum()
# How many kinds of ticket are there?

sns.countplot(y = "Ticket", data = train)
# Another way to visualize that is:

train.Ticket.value_counts()
# there are 681 unique ticket values.

# there may be some way to reduce this down. But We're gonna eliminate it
# For the feature Fare is a similar process

train.Fare.isnull().sum()
sns.countplot(y = "Fare", data = train)
train.Fare.dtypes
# Because Fare its a continuous value we're gonna add it to our continuous variables dataframe

# and in the categorical variables dataframe we're gonna add it as a discrete value by cut it in 5.

df_con['Fare'] = train['Fare']

df_dis['Fare'] = pd.cut(train['Fare'], bins = 5)
df_dis.Fare.value_counts()
# Visualise the Fare bin counts as well as the Fare distribution versus Survived.

plot_count_dist(data = train,

                bin_df = df_dis,

                label_column = 'Survived', 

                target_column = 'Fare', 

                figsize = (20,10), 

                use_bin_df = True)
# Feature cabin

train.Cabin.isnull().sum()
# Wow, there's a lot of missing values, lets see how do the cabin values looks like

train.Cabin.value_counts()
# We wont be using Cabin, cause of the lots of missing values
# Feature: Embarked

train.Embarked.isnull().sum()
# As we see, there are two missing values

train.Embarked.value_counts()
# Visualize the distribution

sns.countplot(y = "Embarked", data = train)
# We're gonna drop the missing values:

df_dis['Embarked'] = train['Embarked']

df_con['Embarked'] = train['Embarked']
print("Length of the subDataFrame before removing the rows: ", len(df_con))

df_con = df_con.dropna(subset = ['Embarked'])

df_dis = df_dis.dropna(subset = ['Embarked'])

print("Length of the subDataFrame after removing the rows: ", len(df_con))
# Now lets see our two sub dataframes

df_con.head()
df_dis.head()
# Now, before to apply machine learning models lets do feature encoding and transorm the data to a more understandable data for a ML model
# One hot encode binary variables

one_hot_columns = df_dis.columns.tolist()

one_hot_columns.remove('Survived')

df_dis_enc = pd.get_dummies(df_dis, columns = one_hot_columns)

df_dis_enc.head()
df_con.head()
#One hot encode the categorical columns

df_embarked_one_hot = pd.get_dummies(df_con['Embarked'], prefix = 'embarked')



df_sex_one_hot = pd.get_dummies(df_con['Sex'], prefix = 'sex')



df_pclass_one_hot = pd.get_dummies(df_con['Pclass'], prefix = 'pclass')
#combine the columns

df_con_enc = pd.concat([df_con, df_embarked_one_hot, df_sex_one_hot, df_pclass_one_hot], axis = 1)



#Drop the original columns cause they have been encoded already

df_con_enc = df_con_enc.drop(['Pclass', 'Sex', 'Embarked'], axis = 1)
df_con_enc.head()
#Lets start building machine learning models and separating the data

#The data we will use to train the models is our continous data:

selected_df = df_con_enc
#Split the dataframes into data and labels

X_train = selected_df.drop('Survived', axis = 1)

y_train = selected_df['Survived']
X_train.shape
X_train.head()
y_train.shape
y_train.head()
#First lets create a function that runs all machine learning models

def fit_ml_algo(algo, X_train, y_train, cv):

    #Fit

    model = algo.fit(X_train, y_train)

    acc = round(model.score(X_train, y_train) * 100, 2)

    

    # K-fold Cross Validation

    train_pred = model_selection.cross_val_predict(algo, X_train, y_train,

                                                  cv = cv, n_jobs = -1)

    

    #Cross Validation accuracy

    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    

    return train_pred, acc, acc_cv
# Logistic Regression

start_time = time.time()

train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(),

                                                 X_train, y_train, 10)



log_time = (time.time() - start_time)

print("Train Accuracy: ", acc_log)

print("10 - fold Cross validation: ", acc_cv_log)

print("Runing time: ", datetime.timedelta(seconds = log_time))
#K-Nearest Neighbors

start_time = time.time()

train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(),

                                                 X_train, y_train, 10)



knn_time = (time.time() - start_time)

print("Train Accuracy: ", acc_knn)

print("10 - fold Cross validation: ", acc_cv_knn)

print("Runing time: ", datetime.timedelta(seconds = knn_time))
#Gaussian Naive Bayes

start_time = time.time()

train_pred_gnv, acc_gnv, acc_cv_gnv = fit_ml_algo(GaussianNB(),

                                                 X_train, y_train, 10)



gnv_time = (time.time() - start_time)

print("Train Accuracy: ", acc_gnv)

print("10 - fold Cross validation: ", acc_cv_gnv)

print("Runing time: ", datetime.timedelta(seconds = gnv_time))
# Linear Support Vector Machines

start_time = time.time()

train_pred_svm, acc_svm, acc_cv_svm = fit_ml_algo(LinearSVC(),

                                                 X_train, y_train, 10)



svm_time = (time.time() - start_time)

print("Train Accuracy: ", acc_svm)

print("10 - fold Cross validation: ", acc_cv_svm)

print("Runing time: ", datetime.timedelta(seconds = svm_time))
# Stochastic Gradient Descent

start_time = time.time()

train_pred_sgd, acc_sgd, acc_cv_sgd = fit_ml_algo(SGDClassifier(),

                                                 X_train, y_train, 10)



sgd_time = (time.time() - start_time)

print("Train Accuracy: ", acc_sgd)

print("10 - fold Cross validation: ", acc_cv_sgd)

print("Runing time: ", datetime.timedelta(seconds = sgd_time))
# Decision Tree Classifier

start_time = time.time()

train_pred_dtc, acc_dtc, acc_cv_dtc = fit_ml_algo(DecisionTreeClassifier(),

                                                 X_train, y_train, 10)



dtc_time = (time.time() - start_time)

print("Train Accuracy: ", acc_dtc)

print("10 - fold Cross validation: ", acc_cv_dtc)

print("Runing time: ", datetime.timedelta(seconds = dtc_time))
#Gradient Boost Trees

start_time = time.time()

train_pred_gbt, acc_gbt, acc_cv_gbt = fit_ml_algo(GradientBoostingClassifier(),

                                                 X_train, y_train, 10)



gbt_time = (time.time() - start_time)

print("Train Accuracy: ", acc_gbt)

print("10 - fold Cross validation: ", acc_cv_gbt)

print("Runing time: ", datetime.timedelta(seconds = gbt_time))
# CatBoost Algorithm

"""

This is an State-of-the-art algorithm that mixes a lot about secision trees,

gradient boosting trees, etc.

"""
#lets look at our training data

X_train.head()
y_train.head()
# Lets define the categorical features for the catboost model

cat_features = np.where(X_train.dtypes != np.float)[0]

cat_features
# CatBoost has picked up that all variables except Fare can be treated as categorical.

# Lets use the pool function for pool together all the data

train_pool = Pool(X_train, y_train,

                 cat_features)
# catboost model:

catboost_model = CatBoostClassifier(iterations = 1000,

                                   custom_loss = ['Accuracy'],

                                   loss_function = 'Logloss')



# Fit catboost model

catboost_model.fit(train_pool, plot = True)



#Catboost Accuracy

acc_catboost = round(catboost_model.score(X_train, y_train) * 100, 2)
# time

start_time = time.time()



# set params for cross validation

cv_params = catboost_model.get_params()



# run the cross validation with 10 folds

cv_data = cv(train_pool, cv_params,

            fold_count = 10, plot = True)



#Time

catboost_time = (time.time() - start_time)



# CatBoost CV results save into a dataframe (cv_data), let's withdraw the maximum accuracy score

acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)
#Accuracy of catboost

print("--CatBoost Metrics--")

print("Accuracy: ", acc_catboost)

print("10-fold Cross Validation: ", acc_cv_catboost)

print("Runing time: ", datetime.timedelta(seconds = catboost_time))
#Model results

models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree', 'Gradient Boosting Trees',

              'CatBoost'],

    'Score': [

        acc_knn, 

        acc_log,  

        acc_gnv, 

        acc_sgd, 

        acc_svm, 

        acc_dtc,

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

        acc_cv_gnv, 

        acc_cv_sgd, 

        acc_cv_svm, 

        acc_cv_dtc,

        acc_cv_gbt,

        acc_cv_catboost

    ]})

print('---Cross-validation Accuracy Scores---')

cv_models.sort_values(by='Score', ascending=False)
# Because CatBoost got the best accuracy we will use it to predict on the test set





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
feature_importance(catboost_model, X_train) #Relevance of each feature
# Another metrics for evaluate our model

metrics = ['Precision', 'Recall', 'F1', 'AUC']



eval_metrics = catboost_model.eval_metrics(train_pool,

                                           metrics=metrics,

                                           plot=True)



for metric in metrics:

    print(str(metric)+": {}".format(np.mean(eval_metrics[metric])))
# now, for the submission we have to apply the same one hot encoding that we applied to the train test

test.head()
# One hot encode the columns in the test data frame (like X_train)

test_embarked_one_hot = pd.get_dummies(test['Embarked'], 

                                       prefix='embarked')



test_sex_one_hot = pd.get_dummies(test['Sex'], 

                                prefix='sex')



test_pclass_one_hot = pd.get_dummies(test['Pclass'], 

                                   prefix='pclass')
test = pd.concat([test,

                test_embarked_one_hot,

                test_sex_one_hot,

                test_pclass_one_hot],

                axis = 1)

test.head()
# create a list of columns that will be used in the predictions

wanted_test_cols = X_train.columns

wanted_test_cols
# We create our predictions vector

predictions = catboost_model.predict(test[wanted_test_cols])

predictions[:10]
submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = predictions

submission.head()
gender_submission.head()
# check if the dataframe is the same length as the example

if len(submission) == len(test):

    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))

else:

    print("Dataframes mismatched, won't be able to submit to Kaggle.")
submission.to_csv('my_submission.csv', index = False)

print('Submission CSV is ready!')