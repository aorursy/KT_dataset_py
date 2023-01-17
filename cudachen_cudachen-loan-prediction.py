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
import pandas as pd

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns
loan_train = pd.read_csv("/kaggle/input/loanprediction/train_ctrUa4K.csv")

print("Print the first 5 records in the dataset:")

print(loan_train.head())

print()

print("Basic descriptive statistics of all the variables:")

print(loan_train.describe())

print()

print("Present the feature attributes:")

print(loan_train.info())
# count the null columns

null_columns = loan_train.columns[loan_train.isnull().any()]

loan_train[null_columns].isnull().sum()
loan_train.hist(bins=50, figsize=(15, 10))

plt.show()
df_obj = loan_train.select_dtypes(include='object')



for col in df_obj.iloc[:, 1:].columns:

    print(sns.countplot(x=col, data=df_obj))

    plt.show()
df_approved = loan_train[loan_train['Loan_Status'] == 'Y']

df_rejected = loan_train[loan_train['Loan_Status'] == 'N']



non_object_variables = ['ApplicantIncome', 'CoapplicantIncome', 

                        'Credit_History', 'LoanAmount', 'Loan_Amount_Term']



for obj in non_object_variables:

    sns.distplot(df_approved[obj][df_approved[obj].isnull() == False], label = 'Loan_Status == Y')

    sns.distplot(df_rejected[obj][df_rejected[obj].isnull() == False], label = 'Loan_Status == N')

    plt.legend()

    plt.show()
object_variables = ['Gender', 'Married', 'Dependents', 

                   'Self_Employed', 'Property_Area']



for obj in object_variables:

    print(sns.countplot(x=obj, data=df_obj, hue='Loan_Status'))

    plt.show()
def fill_with_mode(df, x):

    df[x].fillna(df[x].mode()[0], inplace=True)

    

has_null_objects = ['Gender', 'Married', 'Dependents',

                    'Self_Employed', 'LoanAmount', 'Loan_Amount_Term',

                    'Credit_History']



for obj in has_null_objects:

    fill_with_mode(loan_train, obj)
# make sure all null value has bee filled

for col in loan_train.columns:

    print('Variable', col, 'has missing entry:', sum(loan_train[col].isnull()))
X = loan_train.iloc[:, 1:-1]

y = loan_train.iloc[:, -1]
print(X.shape)

print(y.shape)
# these four columns need to be onehotted

onehot_targets = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 

                  'Property_Area']
for target in onehot_targets:

    onehot_temp = pd.get_dummies(X[target])

    X = X.drop(target, axis=1)

    X = pd.concat([onehot_temp, X], axis=1)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(y)
y = le.transform(y)
# prepare train and validation data

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

scores = cross_val_score(clf, X, y, cv=5)

print("Random forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.model_selection import RandomizedSearchCV



# number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start=10, stop=700, num=10)]

# number of features to consider in every split

max_features = ['auto', 'sqrt']

# maximum level in tree

max_depth = [int(x) for x in np.linspace(10, 110, num=11)]

max_depth.append(None)

# minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# method of selecting samples of training each tree

bootstrap = [True, False]



# create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
# start hyperparameter searching

rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100,

                              cv=5, verbose=2, random_state=42, n_jobs=-1)

rf_random.fit(X, y)
# print the best hyperparameters

rf_random.best_params_
clrf_grid = RandomForestClassifier(n_estimators=546, min_samples_split=2, min_samples_leaf=4,

                                  max_features='sqrt', max_depth=10, bootstrap=True)

scores = cross_val_score(clrf_grid, X, y, cv=5)
print("Grid Random Forst Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
loan_test = pd.read_csv('/kaggle/input/loanprediction/test_lAUu6dG.csv')

    

X_test = loan_test.iloc[:, 1:]

    

for feature in has_null_objects:

    fill_with_mode(X_test, feature)

    

# make sure all null value has bee filled

for col in X_test.columns:

    print('Variable', col, 'has missing entry:', sum(X_test[col].isnull()))
# onehot encoding to test data specific features

for target in onehot_targets:

    onehot_temp = pd.get_dummies(X_test[target])

    X_test = X_test.drop(target, axis=1)

    X_test = pd.concat([onehot_temp, X_test], axis=1)
clrf_grid.fit(X, y)



results = clrf_grid.predict(X_test)

SJ_submit = pd.DataFrame({"Loan_ID": loan_test['Loan_ID'], "Loan_Status": results})

print(SJ_submit.head())
pd.DataFrame(SJ_submit).to_csv("submit_CudaChen.csv", index=False)