# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ad_data = pd.read_csv('/kaggle/input/advertising/advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()
ad_data.columns
ad_data['Age'].plot.hist(bins=35)
import seaborn as sns

sns.jointplot(x='Age',y='Area Income',data=ad_data)
sns.jointplot(x='Age',y='Daily Time Spent on Site',kind='kde',data=ad_data)
sns.jointplot(y='Daily Time Spent on Site',x='Age',data=ad_data,kind='scatter')
ad_data.columns
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,kind='scatter')
sns.pairplot(ad_data,hue='Clicked on Ad')
from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site','Age', 'Area Income',

       'Daily Internet Usage','Male']]

y= ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=101)
from sklearn.linear_model import LogisticRegression
Logmodel = LogisticRegression()

Logmodel.fit(X_train,y_train)
predictions = Logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))