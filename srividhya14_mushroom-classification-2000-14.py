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
import pandas as pd

filename = "/kaggle/input/mushroom-classification/mushrooms.csv"

df = pd.read_csv(filename)

df.head()
df1=pd.get_dummies(df,drop_first=True)
df1.tail()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.countplot(x='class',data=df)
sns.countplot(x='odor',data=df)
sns.countplot(x='bruises',data=df)
from sklearn.model_selection import train_test_split
X=df1.drop('class_p',axis=1)

y=df1['class_p']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
predictions=lr.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))