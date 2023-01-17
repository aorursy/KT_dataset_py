# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#LEAVE IT ALONE

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df1=pd.read_csv('../input/X_train.csv')  #This happens when comptetion maker try to be smart give x_train &Y_train to help

df2=pd.read_csv('../input/y_train.csv')  #but it ends up confusing you. 

test=pd.read_csv('../input/X_test.csv')

sample=pd.read_csv('../input/sample_submission.csv')
print((df1.shape),(df2.shape),(test.shape),(sample.shape))
df1.head()
df1.info()
df2.head()
df2.info()
sample.head()
test.head()
df2.surface.nunique()