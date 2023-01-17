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
#reading the train data 

train = pd.read_csv("../input/train.csv")
#reading the test data

test = pd.read_csv("../input/test.csv")
#top 5 rows of the data. 

train.head()
#shape of the data. 

train.shape
#test data shape 

test.shape
train.target.value_counts()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

std = StandardScaler()

y = train['target']

del train['target']



X= train
std.fit_transform(X)

std.transform(test)
pca = PCA(0.90)

principle_components = pca.fit_transform(X)

pc_test = pca.transform(test)
principle_components
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X, y)
xx = logisticRegr.predict(X)
yy = logisticRegr.predict(test)
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.columns
sample_submission['target'] = yy
sample_submission.head()
sample_submission.to_csv("final_submission", index=False, sep=",")