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
from sklearn.model_selection import train_test_split
train=pd.read_csv('/kaggle/input/loandata/Loan payments data.csv')

print('Shape=>',train.shape)

train.head()
train['Gender'].value_counts()*100/train['Gender'].count()
X=train.drop(columns='Gender').values

Y=train['Gender'].values
print(X.shape,Y.shape)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print(x_train.shape,y_train.shape)

print(x_test.shape,y_test.shape)
pd.Series(y_train).value_counts()*100/len(y_train)
pd.Series(y_test).value_counts()*100/len(y_test)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)
print(x_train.shape,y_train.shape)

print(x_test.shape,y_test.shape)
pd.Series(y_train).value_counts()*100/len(y_train)
pd.Series(y_test).value_counts()*100/len(y_test)