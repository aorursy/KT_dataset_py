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
test_set = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv', index_col='ID_code')

train_set = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv', index_col='ID_code')
test_set.head()
test_set.tail(2)
train_set.head()
train_set.tail(2)
test_set.describe()
train_set.describe()
y = train_set.iloc[:,0]

X = train_set.drop(columns=['target'])

X
#Calculate of new features, aggreated of existing

idx = X.columns.values

for df in [X, test_set]:

    df['sum'] = df[idx].sum(axis=1)

    df['min'] = df[idx].min(axis=1)

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurtosis'] = df[idx].kurtosis(axis=1)

    df['med'] = df[idx].median(axis=1)
X.head()
test_set.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

print("Accuracy: {:.2f} %".format(accuracies.mean()))

print("Standar Deviation: {:.2f} %".format(accuracies.std()*100))
sc = StandardScaler()

X = sc.fit_transform(X)

test_set_predict = sc.transform(test_set)

classifier.fit(X, y)

y_submit = classifier.predict(test_set_predict)

y_submit
submission = pd.DataFrame({

        "ID_code": test_set.index,

        "target": y_submit

    })

submission.to_csv('sja-c2-log-reg.csv', index=False)
# Credits:

# Kirill Eremenko, Hadelin Ponteves, SuperDataScience

# https://www.kaggle.com/gpreda/santander-eda-and-prediction#Model