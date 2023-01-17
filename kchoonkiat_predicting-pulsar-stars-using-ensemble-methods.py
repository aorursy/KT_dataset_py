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
df = pd.read_csv('../input/pulsar_stars.csv')

df.head()

        
df.shape
df.info()
import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt



fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches

corr = df.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,annot=True)



X = df.drop('target_class',axis='columns')

y= df['target_class']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1')
lr.fit(X_train,y_train)

lr.score(X_test,y_test)