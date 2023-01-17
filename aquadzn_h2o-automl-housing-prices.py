import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import h2o

from h2o.automl import H2OAutoML

h2o.init(max_mem_size='16G')
train = h2o.import_file('../input/home-data-for-ml-course/train.csv')

test = h2o.import_file('../input/home-data-for-ml-course/test.csv')
train.tail(2)
test.head(2)
print("Train shape:", train.shape, '\nTest shape:', test.shape)
y = "SalePrice"

x = train.columns

x.remove('Id')

x.remove(y)

test = test.drop(['Id'])
aml = H2OAutoML()

aml.train(x=x, y=y, training_frame=train)
lb = aml.leaderboard

lb.head(rows=lb.nrows)
aml.leader
preds = aml.predict(test)
preds.head()
submission = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')

submission.head()
submission[y] = preds.as_data_frame().values

submission
submission.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv')