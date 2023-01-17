# As a ING developer who wants to grow as datascientist,
# I submitted this challenge, to refresh my knowledge on Datascience, win awsum prizes and be of the street.
# Allthough I think I'm close, my computer takes too long to run and I should/want/need to go to bed.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.read_csv("../input/train.csv")
#df['revenue2014'].fillna(df['revenue2015'], inplace=True)
#df['revenue2016'].fillna(df['revenue2015'], inplace=True)
df['trend'] = df.apply(lambda row: np.polyfit([1,2,3],[row.revenue2014, row.revenue2015, row.revenue2016],1)[0], axis=1) 

# Any results you write to the current directory are saved as output.
df['down'] =  df['trend'].apply(lambda x: 1 if x<0 else 0)
df.groupby('down')['bankrupt'].agg(['count','mean']).sort_values(by='mean', ascending=False)
from sklearn.ensemble import GradientBoostingClassifier
y = df['bankrupt']
df.drop(columns='bankrupt',inplace=True)
df.fillna(0,inplace=True)
X= pd.get_dummies(df, drop_first=True) 

clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,learning_rate=1)
clf.fit(X,y)