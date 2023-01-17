import pandas as pd                  # A fundamental package for linear algebra and multidimensional arrays

import numpy as np                   # Data analysis and data manipulating tool

import random                        # Library to generate random numbers

from collections import Counter      # Collection is a Python module that implements specialized container datatypes providing 

                                     # alternatives to Python’s general purpose built-in containers, dict, list, set, and tuple.

                                     # Counter is a dict subclass for counting hashable objects

# Visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



# To ignore warnings in the notebook

import warnings

warnings.filterwarnings("ignore")
fraud_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Imbalanced_classes/master/fraud_data.csv")
fraud_data.head()
fraud_data.info()
fraud_data.describe()
# Taking a look at the target variable

fraud_data.isFraud.value_counts()
fraud_data.isFraud.value_counts(normalize = True)      # Normalize = True will find the proportion of fraud transaction and not fraud transaction 
# we can also use countplot form seaborn to plot the above information graphically.

sns.countplot(fraud_data.isFraud)
def miss_val_info(df):

  """

  This function will take a dataframe and calculates the frequency and percentage of missing values in each column.

  """

  missing_count = df.isnull().sum().sort_values(ascending = False)

  missing_percent = round(missing_count / len(df) * 100, 2)

  missing_info = pd.concat([missing_count, missing_percent], axis = 1, keys=['Missing Value Count','Percent of missing values'])

  return missing_info[missing_info['Missing Value Count'] != 0]

miss_val_info(fraud_data)      # Display the frequency and percentage of data missing in each column
fraud_data = fraud_data[fraud_data.columns[fraud_data.isnull().mean() < 0.2]]
# filling missing values of numerical columns with mean value.

num_cols = fraud_data.select_dtypes(include=np.number).columns      # getting all the numerical columns



fraud_data[num_cols] = fraud_data[num_cols].fillna(fraud_data[num_cols].mean())   # fills the missing values with mean
cat_cols = fraud_data.select_dtypes(include = 'object').columns    # getting all the categorical columns



fraud_data[cat_cols] = fraud_data[cat_cols].fillna(fraud_data[cat_cols].mode().iloc[0])  # fills the missing values with maximum occuring element in the column
# Let's have a look if there still exist any missing values

miss_val_info(fraud_data)
fraud_data = pd.get_dummies(fraud_data, columns=cat_cols)    # earlier we have collected all the categorical columns in cat_cols
fraud_data.head()
# Separate input features and output feature

X = fraud_data.drop(columns = ['isFraud'])       # input features

Y = fraud_data.isFraud      # output feature



from sklearn.model_selection import train_test_split



# Split randomly into 70% train data and 30% test data

X_train, X_Test, Y_train, Y_Test = train_test_split(X, Y, test_size = 0.3, random_state = 123)
# import SMOTE 

from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state = 25, ratio = 1.0)   # again we are eqalizing both the classes
# fit the sampling

X_train, Y_train = sm.fit_sample(X_train, Y_train)
np.unique(Y_train, return_counts=True)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(criterion='entropy')
rfc.fit(X_train, Y_train)
rfc.score(X_Test, Y_Test)
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)     # Let's say we select 10 best features
X_new = selector.fit_transform(X, Y)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size = 0.2, random_state = 42)
rfc.fit(X_train, Y_train)
rfc.score(X_test, Y_test)
# We will use here k - fold cross validation technique

from sklearn.model_selection import cross_validate
cv_results = cross_validate(rfc, X_new, Y, cv=10, scoring=["accuracy", "precision", "recall"])

cv_results
print("Accuracy: ", cv_results["test_accuracy"].mean())
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
# Different parameters in random forest



criterion = ['gini', 'entropy']        # what criteria to consider



n_estimators = [100, 200, 300]       # Number of trees in random forest



max_features = ['auto', 'sqrt']       # Number of features to consider at every split



max_depth = [10, 20]      # Maximum number of levels in tree. Hope you remember linspace function from numpy session



max_depth.append(None)     # also appendin 'None' in max_depth i.e. no maximum depth to be considered.



params = {'criterion': criterion,

          'n_estimators': n_estimators,

          'max_features': max_features,

          'max_depth': max_depth}

params
gs = GridSearchCV(rfc, param_grid=params, n_jobs=2)
gs.fit(X_train, Y_train)    # this will take a lot of time to execute; have some patience
gs.best_params_
gs.best_score_
gs.score(X_test, Y_test)