# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

from sklearn import svm
train = pd.read_csv('../input/train.csv')

labels = LabelEncoder().fit(train['species'])

y = labels.transform(train['species'])

train = train.drop(['id', 'species'], axis=1)
model = svm.SVC(probability=True)

model.fit(train, y)
test = pd.read_csv('../input/test.csv')

test_ids = test['id']

test = test.drop('id', axis=1)

predictions = model.predict_proba(test)
submission = pd.DataFrame(predictions, columns=labels.classes_, index=test_ids)
submission.to_csv('svm_submission.csv')
type(predictions)
from sklearn.metrics import log_loss
validate = model.predict_proba(train)

log_loss(y, validate)