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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv('../input/advertising/advertising.csv')

data.head()
data.info()
data.describe()
data['Age'].plot.hist(bins=30)
sns.jointplot(x='Age',y='Area Income',data=data)
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=data, color='red',kind='kde')
sns.jointplot(x='Daily Internet Usage',y='Daily Time Spent on Site',data=data)
sns.pairplot(data, hue='Clicked on Ad',palette='bwr')
from sklearn.model_selection import train_test_split
y=data['Clicked on Ad']

X=data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,y_train)
prediction=lr.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(prediction,y_test))