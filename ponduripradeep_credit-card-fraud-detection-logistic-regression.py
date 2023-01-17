# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (15.0,8.0)

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
credit_card_df = pd.read_csv('../input/creditcard.csv')

print(credit_card_df.shape)

credit_card_df.head(10)
print(credit_card_df.isnull().sum())
features = ['Amount'] + ['V%d'%num for num in range(1,29)]

target = 'Class'

X = credit_card_df[features]

y = credit_card_df[target]
def normalize(X):

    for feature in X.columns:

        X[feature] -= X[feature].mean()

        X[feature] /= X[feature].std()

    return X
model = LogisticRegression()



splitter = StratifiedShuffleSplit(n_splits=1,test_size=0.5,random_state=0)



for train_indices, test_indices in splitter.split(X, y):

    # Select the train and test data

    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]

    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    

    # Normalize the data

    X_train = normalize(X_train)

    X_test = normalize(X_test)

    

    # Fit and predict!

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    # And finally: show the results

    print(classification_report(y_test, y_pred))


