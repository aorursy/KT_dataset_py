# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



prefix = '/kaggle/input/santander-customer-transaction-prediction/'
df = pd.read_csv(prefix + "train.csv", index_col='ID_code')

trues = df.loc[df['target'] == 1]

falses = df.loc[df['target'] != 1].sample(frac=1)[:len(trues)]

data = pd.concat([trues, falses], ignore_index=True).sample(frac=1)

data.head()
data.isna().sum().sum()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



y = data['target']

X = data.drop('target', axis=1)



scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression



models = []

models.append(('Random Forest', RandomForestClassifier(n_estimators=1000)))

models.append(('GNB', GaussianNB()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('Logistic Regression', LogisticRegression()))



DECISION_FUNCTIONS = {}
from sklearn.metrics import roc_auc_score, roc_curve

%matplotlib inline



best_model = None

best_model_name = ""

best_valid = 0

for name, model in models:

    model.fit(X_train, y_train)

    if name in DECISION_FUNCTIONS:

        proba = model.decision_function(X_test)

    else:

        proba = model.predict_proba(X_test)[:, 1]

    score = roc_auc_score(y_test, proba)

    fpr, tpr, _  = roc_curve(y_test, proba)

    plt.figure()

    plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (auc = {score})")

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.title(f"{name} Results")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right")

    plt.show()

    if score > best_valid:

        best_valid = score

        best_model = model

        best_model_name = name



print(f"Best model is {best_model_name}")
test = pd.read_csv(prefix + "test.csv", index_col='ID_code')

submission = pd.read_csv(prefix + "sample_submission.csv", index_col='ID_code')



test_X = scaler.transform(test)

if best_model_name in DECISION_FUNCTIONS:

    submission['target'] = best_model.decision_function(test_X)

else:

    submission['target'] = best_model.predict_proba(test_X)[:, 1]

submission.to_csv(f"submission.csv")