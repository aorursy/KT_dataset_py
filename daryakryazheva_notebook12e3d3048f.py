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
train_dt = pd.read_csv('/kaggle/input/ffe/train.csv')

test_dt = pd.read_csv('/kaggle/input/ffe/test.csv')
rows = []

counter = 0

for idx, row in train_dt.iterrows():

    if row['target'] == 0:

        counter += 1

        if counter < 31000:

            rows.append(row)

    else:

        rows.append(row)

        

train_dt = pd.DataFrame(rows)

from sklearn.linear_model import LogisticRegression



model = LogisticRegression()



y = train_dt['target']

del train_dt['target']

del train_dt['id']

X = train_dt.values
%%time

model.fit(X, y)
test_idxs = test_dt['id']

del test_dt['id']

X_test = test_dt.values
pred = model.predict(X_test)
submission_dt = pd.DataFrame()



submission_dt['id'] = test_idxs

submission_dt['target'] = pred.astype('int')
submission_dt
submission_dt.to_csv('submission.csv', index=False)