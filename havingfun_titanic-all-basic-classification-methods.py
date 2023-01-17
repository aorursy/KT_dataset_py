# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # For data visualizations

import os

import seaborn as sns

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
y_train = train['Survived']

X_train, X_test = train, test

X_train.drop('Survived', axis=1, inplace=True)

X_train['train'] = 1

X_test['train'] = 0

combined_train = pd.concat([X_train, test])
def standardize_sibsp(x):

    if x == 0:

        return 0

    elif x == 1:

        return 1

    else:

        return 2
def standardize_parch(x):

    if x == 0:

        return 0

    elif x == 1:

        return 1

    else:

        return 2
def imputer(df, strategy='mean'):

    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)

    imputer = imputer.fit(df)

    df = imputer.transform(df)

    return df
def label_encode(df):

    from sklearn.preprocessing import LabelEncoder

    X_LabelEncoder = LabelEncoder()

    df = X_LabelEncoder.fit_transform(df)

    return df
def onehot_encode(df, indices):

    from sklearn.preprocessing import OneHotEncoder

    from sklearn.compose import ColumnTransformer

    ct = ColumnTransformer([('onehotencoder', OneHotEncoder(), indices)], remainder='passthrough')

    df = np.array(ct.fit_transform(df), dtype=np.float)

    return df
def train_scalar(X):

    from sklearn.preprocessing import StandardScaler

    scalar = StandardScaler()

    scalar.fit(X)

    return scalar
# Final input for training generated here

combined_train['StandardSibSp'] = combined_train['SibSp'].apply(standardize_sibsp)

combined_train['StandardParch'] = combined_train['Parch'].apply(standardize_parch)

X = combined_train.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'StandardSibSp', 'StandardParch', 'train']].values

X[:, 2:4] = imputer(X[:, 2:4])

X[:, 4:5] = imputer(X[:,4:5], strategy='most_frequent')

X[:, 1] = label_encode(X[:, 1])

X[:, 4] = label_encode(X[:, 4])

X = onehot_encode(X, [0,1,4,5,6])
# Splitting train and test after the encoding and imputer process

X_train = X[X[:, -1] == 1][:, :-1]

X_test = X[X[:, -1] == 0][:, :-1]
# Feature scaling after the train test split

X_scalar = train_scalar(X_train)

X_train = X_scalar.transform(X_train)

X_test = X_scalar.transform(X_test)

(X_train.shape, X_test.shape, y_train.shape)
# Add all classifiers and generate the output

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

logisticRegressionClassifier = LogisticRegression()

naiveBayesClassifier = GaussianNB()

decisionTreeClassifier = DecisionTreeClassifier(criterion='entropy')

randomForestClassifier = RandomForestClassifier(n_estimators=10, criterion='entropy')

linearSVMClassifier = SVC(kernel='linear')

kernelSVMClassifier = SVC(kernel='rbf')

knnClassifier = KNeighborsClassifier(n_neighbors = 50, metric = 'minkowski', p = 2)

classifiers = [

    (logisticRegressionClassifier, 'logisticRegression'),

    (naiveBayesClassifier, 'naiveBayes'),

    (decisionTreeClassifier, 'decisionTree'),

    (randomForestClassifier, 'randomForest'),

    (linearSVMClassifier, 'linearSVM'),

    (kernelSVMClassifier, 'kernelSVM'),

    (knnClassifier, 'knn')

]
for classifier, classifier_label in classifiers:

    classifier.fit(X_train, y_train)

    y_preds = classifier.predict(X_test)

    output_dataframe = pd.DataFrame({'PassengerId': test.iloc[:, 0].values, 'Survived': y_preds})

    output_dataframe.to_csv('/kaggle/working/' + classifier_label + '_' + str(datetime.datetime.now())+ '.csv', index=False)