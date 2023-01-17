import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# data viualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# machine learning 

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



# Let's ignore warnings for now

import warnings

warnings.filterwarnings("ignore")
# Import the train & test data

train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

test = df_test.copy() # making a copy of test data to make predictions

gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv") # example of submission file
# Let's check shape (number of rows, number of columns) of the train & test data

print("Train data - rows:",train.shape[0],", columns:", train.shape[1])

print()

print("Test data - rows:",test.shape[0],", columns:", test.shape[1])
# training data

train.head(10) # first 10 rows
# test data 

df_test.head() # first 5 rows
# Let's have a look at the submisison dataframe

gender_submission.head()
gender_submission.shape
train.describe()
## Let's visualize the missing values using plot

sns.heatmap(train.isnull(), yticklabels = False, cbar = False)

plt.show()
# Missing values in the train data

train.isnull().sum().sort_values(ascending = False)  ## using sort_values we can sort values in descending order
# Missing values in test data

test.isnull().sum().sort_values(ascending = False) 
# Let's check for different data types in the train dataset

train.dtypes
# Let's check for different data types in the test dataset

test.dtypes
train.head()
# How many people survived

print(train.Survived.value_counts())

print()

plt.figure(figsize=(20,1))

sns.countplot(y= "Survived", data = train)

plt.show()
sns.countplot(train['Pclass'])

plt.show()
# Let's check missing values

train.Pclass.isnull().sum()
train.Name.value_counts()
# Let's drop this Name & PassengerId from data

train.drop(columns = ["Name","PassengerId"], axis = 1, inplace = True)

test.drop(columns = ["Name","PassengerId"], axis = 1, inplace = True)
# Let's view the distribution of Sex

plt.figure(figsize=(15, 2))

sns.countplot(y="Sex", data=train);
sns.countplot(x = 'Survived', hue = 'Sex', data = train)

plt.show()
# Let's check for missing values in train data

train.Sex.isnull().sum()
train['Age'].hist(bins = 50, color = 'blue')

plt.show()
# Let's check for missing values in Age feature

train.Age.isnull().sum()
sns.countplot(train['SibSp'])

plt.show()
sns.countplot(train['Parch'])

plt.show()
# Let's see how many kind's of ticket's are there using plot

sns.countplot(y="Ticket", data=train);
# Let's see how many kind's of ticket's are there

train.Ticket.value_counts()
# How many kinds of Ticket are there?

print("There are {} unique Ticket values.".format(len(train.Ticket.unique())))
# Let's drop this feature from our dataset

train.drop("Ticket", axis = 1, inplace = True)

test.drop("Ticket", axis = 1, inplace = True)
train['Fare'].hist(bins = 50, color = 'red')

plt.show()
# Let's drop this feature because we already have class

train.drop("Fare", axis = 1, inplace = True)

test.drop("Fare", axis = 1, inplace = True)
train.Cabin.value_counts()
# Let's drop Cabin feature

train.drop("Cabin", axis = 1, inplace = True)

test.drop("Cabin", axis =1, inplace = True)
# Let's check what kind of values are in Embarked

train.Embarked.value_counts()
sns.countplot(train['Embarked'])

plt.show()
# Let's check for missing values in Embarked

train["Embarked"].isnull().sum()
# Let's remove Embarked rows which are missing values

print(len(train))

train = train.dropna(subset=['Embarked'])

print(len(train))
# Let's see features correlation matrix using heatmap

plt.figure(figsize=(8,7))

sns.heatmap(train.corr(), annot = True, cmap = "coolwarm")

plt.show()
## Let's impute missing values of Age feature using Pclass since they have the highest correlation in absolute numbers

train['Age'] = train.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))

test['Age'] = test.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))
print(train["Age"].isnull().sum())

print(test["Age"].isnull().sum())
train.head()
test.head()
# Let's convert the categorical variables into dummy/indicator variables using get_dummies() 

train = pd.get_dummies(data = train, columns = ["Sex", "Embarked", "Pclass"])

test = pd.get_dummies(data = test, columns = ["Sex", "Embarked", "Pclass"])
# Let's split the dataset into data and labels

X_train = train.drop("Survived", axis = 1)  # data

y_train = train["Survived"] # labels
# Let's check the shape of the data without labels

X_train.shape
# Let's check the shape of the labels

y_train.shape
# Let's write a function that runs the requested algorithm and returns the accuracy metrics

def fit_model(algo, X_train, y_train, cv):

    

    model = algo.fit(X_train, y_train)    

    y_pred = algo.predict(X_train)    

    accuracy = round(accuracy_score(y_train, y_pred) * 100 , 2)

    

    #cross validation

    y_pred_cv = cross_val_predict(algo, X_train, y_train, cv = cv)

    # cross validation accuracy

    accuracy_cv = round(accuracy_score(y_train, y_pred_cv) * 100 , 2)

    

    return y_pred_cv, accuracy, accuracy_cv
# Logistic Regression



y_pred_cv_lr, accuracy_lr, accuracy_cv_lr = fit_model(LogisticRegression(random_state = 3),

                                                     X_train, y_train, 10)



print("Accuracy : ",accuracy_lr)

print("Accuracy CV :",accuracy_cv_lr)
# Random Forest

rf = RandomForestClassifier(n_estimators = 100, random_state = 3)

rf.fit(X_train, y_train)



y_train_pred = rf.predict(X_train)



print('Confusion Matrix : ','\n', confusion_matrix(y_train, y_train_pred))

print()

print("Accuracy : ", round(accuracy_score(y_train, y_train_pred) * 100, 2))
# let's optimize hyperparameters in random forest classifier

from scipy.stats import randint as sp_randint

rfc = RandomForestClassifier(random_state=3)



params = {'n_estimators' : sp_randint(50,200),

         'max_depth' : sp_randint(2,100),

          'max_depth' : sp_randint(2,100),

         'min_samples_split' : sp_randint(2,100),

         'min_samples_leaf' : sp_randint(1,200),

         'criterion' : ['gini', 'entropy']}



# RandomizedSearchCV

rsearch_rfc = RandomizedSearchCV(rfc, param_distributions = params, n_iter = 100, cv = 3, scoring = 'roc_auc', n_jobs = -1,\

                             return_train_score = True, random_state = 3)



rsearch_rfc.fit(X_train,y_train)
# Print best hyperparameters

rsearch_rfc.best_params_
pd.DataFrame(rsearch_rfc.cv_results_).head(2)
# let's fit our model to the training set with best hyperparameters

rfc = RandomForestClassifier(**rsearch_rfc.best_params_, random_state = 3)



rfc.fit(X_train, y_train)



y_train_pred = rfc.predict(X_train)



print('Confusion Matrix : ','\n', confusion_matrix(y_train, y_train_pred))

print()

print("Accuracy : ", round(accuracy_score(y_train, y_train_pred) * 100, 2))
# Feature Importance

imp = pd.DataFrame(rfc.feature_importances_, index = X_train.columns, columns = ['imp'])

imp = imp.sort_values(by ='imp', ascending = False)

imp
# Let's make a prediction on the test dataset using our random forest



y_test_pred = rfc.predict(test)
y_test_pred[:20]
# Let's create a submisison dataframe and append the relevant columns

submission = pd.DataFrame()

submission['PassengerId'] = df_test['PassengerId']

submission['Survived'] = y_test_pred # our model predictions on the test dataset

submission.head()
# Let's check shape of test and submission DataFrame

print(submission.shape)

print(df_test.shape)
# Let's convert submisison dataframe to csv



submission.to_csv('rf_submission.csv', index=False)

print('Submission CSV is ready!')
# let's check the submission csv

submissions_check = pd.read_csv("rf_submission.csv")

submissions_check.head()