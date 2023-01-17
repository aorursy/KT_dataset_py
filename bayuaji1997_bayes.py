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
data_train=pd.read_csv("../input/train.csv")

data_test=pd.read_csv("../input/test.csv")
data_train.head()
data_test.head()
data_train.columns
data_train.isnull().sum()
from sklearn.naive_bayes import GaussianNB

X=data_train.drop(["ID_code","target"],axis=1)

y=data_train["target"]

model=GaussianNB()

model.fit(X,y)
predictions=model.predict(X)

print(predictions)
output=pd.DataFrame({'ID_code':data_test['ID_code'],'target':predictions})

output.to_csv('sample_submission.csv',index=False)