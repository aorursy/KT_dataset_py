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
train_data = pd.read_csv("../input/loanprediction/train_ctrUa4K.csv")

train_data.info()

train_data.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

category = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

for cat in category:

    plt.figure()

    sns.countplot(x=cat, hue='Loan_Status',data=train_data)
category = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', ]

for cat in category:

    plt.figure()

    sns.catplot(x='Loan_Status', y=cat, data=train_data)
from sklearn.linear_model import LogisticRegression

from sklearn.experimental import enable_iterative_imputer

#from sklearn.impute import IterativeImputer

#from sklearn.impute import KNNImputer

from fancyimpute import KNN

from sklearn.preprocessing import RobustScaler



mapping_X = {'Gender': {'Male': 0, 'Female': 1},

    'Married': {'No': 0, 'Yes': 1}, 

    'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},

    'Education': {'Graduate': 0, 'Not Graduate': 1},

    'Self_Employed': {'No': 0, 'Yes': 1}, 

    'Property_Area': {'Urban': 0, 'Rural': 1, 'Semiurban': 2}}

mapping_y = {'N': 0, 'Y': 1}

train_data = pd.read_csv("../input/loanprediction/train_ctrUa4K.csv")

X = train_data.iloc[:, 1:-1].replace(mapping_X).to_numpy()

y = train_data.iloc[:, -1].replace(mapping_y).to_numpy()

y_train = y



# impute missing value

# for those categorical value that impute to non-integer, do rounding

#imputer = IterativeImputer(max_iter=10, random_state=0)

#imputer = KNNImputer(n_neighbors=2, weights="uniform")

#imputer.fit(X)

imputer = KNN(k=5)

X = imputer.fit_transform(X)



def categorical_rounding(X, columns):

    for col in columns.keys():

        X[:, columns[col]] = np.round(X[:, columns[col]])



columns = {'Gender': 0, 'Married': 1, 'Dependents': 2, 'Education': 3, 'Self_Employed': 4, 'Credit_History': 9}

categorical_rounding(X, columns)



# standardize

X_train = RobustScaler().fit_transform(X)



# logistic regression

logreg = LogisticRegression(C=1e5)

logreg.fit(X_train, y_train)



test_data = pd.read_csv("../input/loanprediction/test_lAUu6dG.csv")

loan_id = test_data.iloc[:, 0].to_numpy()

X = test_data.iloc[:, 1:].replace(mapping_X).to_numpy()

X = imputer.fit_transform(X)

categorical_rounding(X, columns)

X_test = RobustScaler().fit_transform(X)



# predicted test sample

y_predicted = logreg.predict(X_test)



# output result to csv file]

df = pd.DataFrame({'Loan_ID': loan_id, 'Loan_Status': y_predicted})

df.to_csv('submission_CC.csv', index=False)