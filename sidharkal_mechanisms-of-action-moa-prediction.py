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
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import hamming_loss

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report
train = pd.read_csv('../input/lish-moa/train_features.csv')

test = pd.read_csv('../input/lish-moa/test_features.csv')

train_target = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

sample = pd.read_csv('../input/lish-moa/sample_submission.csv')

train.head()
def preprocess(df):

    

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    del df['sig_id']

    return df



train = preprocess(train)

test = preprocess(test)
train.head()
test.head()
y_train = train_target.drop(["sig_id"], axis=1)

y_train.head()
print("training data size is ",train.shape)

print("target data size is   ",y_train.shape)

print("testing data size is ",test.shape)
X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size=0.33, random_state=42)
classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.00001, penalty='l1'), n_jobs=-1)

%time classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print("accuracy :",accuracy_score(y_test,predictions))

print("macro f1 score :",f1_score(y_test, predictions, average = 'macro'))

print("micro f1 scoore :",f1_score(y_test, predictions, average = 'micro'))

print("hamming loss :",hamming_loss(y_test,predictions))

print("Precision recall report :\n",classification_report(y_test, predictions))
pred = classifier.predict(test)

pred.shape
sample.iloc[:,1:] = pred

sample.head()
sample.to_csv('submission.csv', index=False)