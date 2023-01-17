# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/train.csv')

test = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/test.csv')
X = train.drop('Class',axis=1).values

y = train['Class'].values
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(max_depth=8,learning_rate=0.1,n_estimators=80)

model.fit(X, y)
model.score(X,y)
X = test.values
p = model.predict_proba(X)
df_submit = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv',index_col=0)

df_submit['Class'] = p[:,1]

df_submit.to_csv('submission.csv')