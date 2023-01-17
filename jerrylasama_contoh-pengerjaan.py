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
train = pd.read_csv('/kaggle/input/sircleai-orientation-2020/train.csv')

train = train.set_index('id')

test = pd.read_csv('/kaggle/input/sircleai-orientation-2020/test.csv')

test = test.set_index('id')
sample_submission = pd.read_csv('/kaggle/input/sircleai-orientation-2020/sample_submission.csv')

sample_submission = sample_submission.set_index('id')

sample_submission.head()
train.head()
test.head()
x = train.drop(columns='gender')

y = train['gender']
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(x,y)
pred = clf.predict(test)
submission = sample_submission.copy()

submission['gender'] = pred
submission.head()
sample_submission.head()
submission.to_csv('submission.csv')