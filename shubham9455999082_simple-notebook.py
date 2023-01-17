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
train=pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')

test=pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')
train.head()
test.head()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['premise']=le.fit_transform(train['premise'])

train['hypothesis']=le.fit_transform(train['hypothesis'])

train['language']=le.fit_transform(train['language'])
train
test['premise']=le.fit_transform(test['premise'])

test['hypothesis']=le.fit_transform(test['hypothesis'])

test['language']=le.fit_transform(test['language'])
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
x=train[['premise','hypothesis','language']]
x
y=train['label']
clf.fit(x,y)
pp=test[['premise','hypothesis','language']]
d=clf.predict(pp)
d
sample=pd.read_csv('/kaggle/input/contradictory-my-dear-watson/sample_submission.csv')
sample
test1=test['id']
c=pd.DataFrame({'id':test1,'prediction':d})
c.to_csv('submission.csv',index=False)