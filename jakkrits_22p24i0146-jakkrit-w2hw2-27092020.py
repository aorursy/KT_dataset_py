# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

train_df.head()
# Check for null

train_df.isnull().sum()
import pandas_profiling
report = pandas_profiling.ProfileReport(train_df)

display(report)
%pip install autoviz # installing and importing autoviz, another library for automatic data visualization

from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()



report_2 = AV.AutoViz("/kaggle/input/titanic/train.csv")

display(report_2)
from scipy import stats # statistical library



from statsmodels.stats.weightstats import ztest # statistical library for hypothesis testing



# Creating different datasets for survivors and non-survivors

df_survivors = train_df[train_df['Survived'] == 1]

df_nonsurvivors = train_df[train_df['Survived'] == 0]



# First distribution for the hypothesis test: Ages of survivors

dist_a = df_survivors['Age'].dropna()



# Second distribution for the hypothesis test: Ages of non-survivors

dist_b = df_nonsurvivors['Age'].dropna()



# Z-test: Checking if the distribution means (ages of survivors vs ages of non-survivors) are statistically different

t_stat, p_value = ztest(dist_a, dist_b)

print("----- Z Test Results -----")

print("T stat. = " + str(t_stat))

print("P value = " + str(p_value)) # P-value is less than 0.05



print("")



# T-test: Checking if the distribution means (ages of survivors vs ages of non-survivors) are statistically different

t_stat_2, p_value_2 = stats.ttest_ind(dist_a, dist_b)

print("----- T Test Results -----")

print("T stat. = " + str(t_stat_2))

print("P value = " + str(p_value_2)) # P-value is less than 0.05
train_df.shape
train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train_df = train_df.dropna()
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

test_df = test_df.dropna()

display(test_df.head())

print(test_df.shape)
# Need PassengerId for survivor info

survived_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv', index_col='PassengerId')

survived_df.head()
test_df = test_df.join(survived_df, on='PassengerId')

test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

test_df.head()
test_df.info()
def generateFeaturesAndTargets(df):

    y = df['Survived'].values

    X = pd.get_dummies(df).drop('Survived', axis=1).values

    return X, y
X_train, y_train = generateFeaturesAndTargets(train_df)

display(X_train.shape)

print(X_train[:5])

display(y_train.shape)

print(y_train[:5])
X_test, y_test = generateFeaturesAndTargets(test_df)

display(X_test.shape)

print(X_test[:5])

display(y_test.shape)

print(y_test[:5])
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def get_model_accuracy(cls, X, y, cv=5):

    kf = KFold(n_splits=cv)

    kf.get_n_splits(X)

    

    accuracies = []

    

    for train_idx, test_idx in kf.split(X):

        X_train = X[train_idx]

        X_test = X[test_idx]

        y_train = y[train_idx]

        y_test = y[test_idx]

        

        cls.fit(X_train, y_train)

        

        y_pred = np.round(cls.predict(X_test))

        

        acc = accuracy_score(y_test, y_pred)

        

        accuracies.append(acc)

        

    return np.mean(accuracies)
models = [

    DecisionTreeClassifier(random_state=42),

    GaussianNB(),

    MLPClassifier(random_state=42)

]



print('K-fold = 5')

for model in models:

    acc = get_model_accuracy(model, X_train, y_train, 5)

    print(f'{model}:\naccuracy mean score = {acc}\n---------\n')
def evaluate(cls, X, y):

    y_pred = np.round(cls.predict(X))

    

    acc = accuracy_score(y, y_pred)

    precision = precision_score(y, y_pred)

    recall = recall_score(y, y_pred)

    f1 = f1_score(y, y_pred)

    

    return {

        'accuracy': acc,

        'precision': precision,

        'recall': recall,

        'f1': f1,

    }
for model in models:

    res = evaluate(model, X_test, y_test)

    print(f'{model}:\nAccuracy = {res["accuracy"]}\nPrecision = {res["precision"]}\nRecall = {res["recall"]}\nF1 = {res["f1"]}')

    print('===='*20)