# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd 
import numpy as np 

from sklearn.tree import DecisionTreeClassifier
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
sample_sub = pd.read_csv('../input/sampleSubmission.csv')
X = df_train.profession.values # this line is awful 
y = df_train.target.values 

X_test = df_test.profession.values
model = DecisionTreeClassifier(max_depth=4)
model.fit(X.reshape(-1,1),y)
y_hat = model.predict_proba(X_test.reshape(-1,1))[:,1]
sample_sub['target'] = y_hat 

sample_sub.head()

sample_sub.to_csv('v_01.csv', index=False)