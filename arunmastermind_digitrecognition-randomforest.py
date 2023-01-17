# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_data.head()
test_data.head()
train_data.shape
test_data.shape
train_data.isnull().sum()
test_data.isnull().any().sum()
train_data['label'].value_counts().sort_values(ascending=True)
labeled_data = train_data['label']
train_data.drop('label', inplace=True, axis=1)

train_data.head()
X_train, X_test, y_train, y_test = train_test_split(train_data, labeled_data, test_size=0.2, random_state=2)
X_train.head()
y_train.head()
X_test.head()
y_test.head()
rfc  = RandomForestClassifier(n_estimators = 300, n_jobs=-1)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
confusion_matrix = confusion_matrix(y_test, rfc_pred)

confusion_matrix
accuracy_score(y_test, rfc_pred)
classification_report = classification_report(y_test, rfc_pred)

print(classification_report)
submissions=pd.DataFrame({"ImageId": list(range(1,len(rfc_pred)+1)),

                         "Label": rfc_pred})

submissions.to_csv("submission.csv", index=False, header=True)