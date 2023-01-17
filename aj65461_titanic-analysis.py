import numpy as np
import pandas as pd
import seaborn as sns
from plotnine import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load in the training data
df_train = pd.read_csv('../input/train.csv')
print('Training data shape:', df_train.shape)
df_train.head()
# Load in the test data
df_test = pd.read_csv('../input/test.csv')
print('Testing data shape:', df_test.shape)
df_test.head()
df_train['Survived'].value_counts()
df_train['Survived'].astype(int).plot.hist()
# Function to calculate missing values by column 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
# Missing values statistics
missing_values = missing_values_table(df_train)
missing_values
df_train.dropna(subset = ['Embarked'], inplace=True)
missing_values_table(df_test)
df_train.dtypes.value_counts()
# Number of unique classes in each object column
df_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# Drop Name, Ticket, and Cabin columns for now
del df_train['Name']
del df_train['Ticket']
del df_train['Cabin']
del df_test['Name']
del df_test['Ticket']
del df_test['Cabin']
print('Training data shape:', df_train.shape)
print('Testing data shape:', df_test.shape)
# Use describe method to check for anomalies.
df_train.describe()
len(df_train[df_train['Fare'] > 200])
df_train['Fare'].value_counts().sort_index().plot.line()
df_train['Pclass'].value_counts().plot.bar()
print(df_train.Pclass.value_counts())
fare_class = df_train.groupby(['Pclass']).mean()[['Fare']]
fare_class.plot.bar()

fare_first = df_train[df_train['Pclass'] == 1]['Fare'].mean()
fare_second = df_train[df_train['Pclass'] == 2]['Fare'].mean()
fare_third = df_train[df_train['Pclass'] == 3]['Fare'].mean()

print("Average 1st class fare: " + str(fare_first) +"\n"
     "Average 2nd class fare: " + str(fare_second) + "\n"
     "Average 3rd class fare: " + str(fare_third) + "\n")
(df_train['Sex'].value_counts() / len(df_train)).plot.bar()
# Create dataset with only the survivors
survived = df_train[df_train['Survived'] == 1]

survived['Sex'].value_counts().plot.bar()
print(survived['Sex'].value_counts())
survived['Age'].value_counts().sort_index().plot.line()
survived['SibSp'].value_counts().sort_index().plot.line()
survived['Parch'].value_counts().sort_index().plot.line()
survived['Embarked'].value_counts().plot.bar()
print(survived['Embarked'].value_counts())
(survived['Embarked'].value_counts() / len(survived)).plot.bar()
print(survived['Embarked'].value_counts()/len(survived))
ax = sns.countplot(x = 'Embarked', hue = 'Pclass', data = df_train)
# Impute the Age column in training and test sets
df_train['Age'].fillna((df_train['Age'].mean()), inplace=True)
df_test['Age'].fillna((df_test['Age'].mean()), inplace=True)
# Impute the Fare column in the test set
df_test['Fare'].fillna((df_test['Fare'].mean()), inplace=True)
# Use label encoder on the 'Sex' variable
labelencoder_X = LabelEncoder()

df_train['Sex'] = labelencoder_X.fit_transform(df_train['Sex'])
df_test['Sex'] = labelencoder_X.fit_transform(df_test['Sex'])
# Convert non-numeric data using one-hot encoding
df_train = pd.get_dummies(df_train, columns = ['Embarked'])
df_test = pd.get_dummies(df_test, columns = ['Embarked'])

print('Training Features shape: ', df_train.shape)
print('Testing Features shape: ', df_test.shape)
df_train.columns

# Create X and y arrays for the dataset
X = df_train[['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                       'Embarked_C', 'Embarked_Q', 'Embarked_S']].copy()
y = df_train['Survived'].values
# Split the dataset and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting SVM to the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Evaluate the Results
# Training Set
mse = mean_absolute_error(y_train, classifier.predict(X_train))
print('Training Set Mean Absolute Error: %.2f' % mse)

# Test Set
mse = mean_absolute_error(y_test, classifier.predict(X_test))
print('Test Set Mean Absolute Error: %.2f' % mse)
# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
# Applying Grid Search to find the best model and the best parameters
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print('Best Accuracy: {}\n'.format(best_accuracy))
print('Best Parameters: {}'.format(best_parameters))
# Create X and y arrays for the dataset
X = df_train[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                       'Embarked_C', 'Embarked_Q', 'Embarked_S']].copy()
y = df_train['Survived'].values
# Split the dataset and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)
# Fitting XGBoost to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('XGBoost Confusion Matrix:\n {}'.format(cm))
# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('Accuracy Mean: {}\n'.format(accuracies.mean()))
print('Accuracy Standard Deviation: {}'.format(accuracies.std()))
# Create X and y arrays for the dataset
X_train = df_train[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                       'Embarked_C', 'Embarked_Q', 'Embarked_S']].copy()
y_train = df_train['Survived'].values
# Create X_test from our 'test.csv' file
X_test = df_test[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                       'Embarked_C', 'Embarked_Q', 'Embarked_S']].copy()
# Fitting XGBoost to the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('Accuracy Mean: {}\n'.format(accuracies.mean()))
print('Accuracy Standard Deviation: {}'.format(accuracies.std()))
# Saving data for competition score
output = pd.DataFrame({'PassengerId': df_test.PassengerId,
                       'Survived': y_pred})

output.to_csv('submission.csv', index=False)