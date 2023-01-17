# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/diabetes-dataset/diabetes2.csv')
df.head()
df.info()
df.describe()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='Outcome',data=df)
sns.distplot(df['Age'].dropna(),kde=True)
df.corr()
sns.heatmap(df.corr())
sns.pairplot(df)
plt.subplots(figsize=(20,15))

sns.boxplot(x='Age', y='BMI', data=df)
x = df.drop('Outcome',axis=1)

y = df['Outcome']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
predictions = logmodel.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)