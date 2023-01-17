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
import  seaborn as sns
df=pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
df.head()
#sns.pairplot(df)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.distplot(df['age'])
import matplotlib.pyplot as plt 
plt.figure(figsize=(20,10))
dia=sns.countplot(df['age'])
sns.countplot(df['DEATH_EVENT'])
plt.figure(figsize=(5,5))
sns.boxplot(y="age", x="DEATH_EVENT", data=df,palette='rainbow')
plt.figure(figsize=(5,5))
sns.boxplot(y="age", x="DEATH_EVENT",hue='sex', data=df,palette='rainbow')
sns.stripplot(x="DEATH_EVENT", y="time", data=df)
plt.figure(figsize=(8,9))
sns.boxplot(y="age", x="DEATH_EVENT",hue='diabetes', data=df,palette='rainbow')
plt.figure(figsize=(8,9))
sns.boxplot(y="age", x="DEATH_EVENT",hue='high_blood_pressure', data=df,palette='rainbow')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('DEATH_EVENT',axis=1), 
                                                    df['DEATH_EVENT'], test_size=0.30, 
                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
