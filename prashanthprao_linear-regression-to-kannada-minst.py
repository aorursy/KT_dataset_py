# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

train=pd.read_csv('../input/Kannada-MNIST/train.csv')

test=pd.read_csv('../input/Kannada-MNIST/test.csv')

sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')



test = test.drop('id', axis=1)



X_train = train.drop(columns='label')

y_train = train.label



X_train = X_train / 255

test = test / 255



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0,test_size=0.15)



clf = LogisticRegression(random_state=0).fit(X_train, y_train)

clf.predict(X_test)



clf.score(X_test, y_test)
sample_sub['label']=clf.predict(test)

sample_sub.to_csv('submission.csv',index=False)
sample_sub.head()