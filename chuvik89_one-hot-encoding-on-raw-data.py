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
import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, roc_auc_score

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder
sample = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')
y = train['target']

train = train.drop(['target', 'id'], axis=1)

test = test.drop(['id'], axis=1)
# оставляем только числовые признаки в таблице 



drop_object_train = train.select_dtypes(exclude=['object'])

drop_object_test = test.select_dtypes(exclude=['object'])
# выберем колонки с низким числом вариаций возможных значений. Допустим меньше 10



object_train = train[[name for name in train.columns if train[name].nunique() < 100 and train[name].dtype == 'object']]

object_test = test[[name for name in test.columns if test[name].nunique() < 100 and test[name].dtype == 'object']]
# применим метод LabelEncoder к нашим столбцам



label_encoder = LabelEncoder()

for col in object_train.columns:

    object_train[col] = label_encoder.fit_transform(object_train[col])

    object_test[col] = label_encoder.transform(object_test[col])
train_digit = pd.concat([drop_object_train, object_train], axis=1)

test_digit = pd.concat([drop_object_test, object_test], axis=1)

train_digit.shape, test_digit.shape
model = LogisticRegression()

model.fit(train_digit, y)

preds = model.predict_proba(train_digit)

mean_absolute_error(y, preds[:, 1]), roc_auc_score(y, preds[:, 1])
# попробуем закодировать наши данные без каких-либо преобразований



OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)



OH_train = OH_encoder.fit_transform(train)

OH_test = OH_encoder.transform(test)

OH_train.shape, OH_test.shape
model = LogisticRegression(solver='lbfgs', tol=0.0003, max_iter=5000)

model.fit(OH_train, y)

preds = model.predict_proba(OH_train)

mean_absolute_error(y, preds[:, 1]), roc_auc_score(y, preds[:, 1])
preds = model.predict_proba(OH_test)

pd.DataFrame({'id': sample['id'], 'target': preds[:, 1]}).to_csv('/kaggle/working/submission.csv', index=False)