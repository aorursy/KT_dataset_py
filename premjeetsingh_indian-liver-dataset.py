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
liver_df = pd.read_csv("../input/indian_liver_patient.csv")
liver_df.head()
liver_df.describe()
liver_df.info()
liver_df.isnull().sum()
liver_df[liver_df['Albumin_and_Globulin_Ratio'].isnull()]
import seaborn as sns

sns.countplot(data = liver_df, x = 'Dataset')

Diagnosed, Not_Diagnosed = liver_df['Dataset'].value_counts()

print('No. of patients diagnosed with liver cancer', Diagnosed)

print('No. of patients not diagnosed with liver cancer', Not_Diagnosed)
sns.countplot(data = liver_df, x = 'Gender')

M, F = liver_df['Gender'].value_counts()

print('No. of patients that are Male:', M)

print('No. of patients that are Female:', F)
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

sns.heatmap(data = liver_df.corr(), annot = True)
#liver_df['Albumin_and_Globulin_Ratio'].mean()
#liver_df['Albumin_and_Globulin_Ratio'].fillna(0.947063,inplace=True)
liver_df.dropna(axis=0,inplace=True)
liver_df.info()
liver_df.isnull().sum()
liver_df['Dataset'].value_counts()
sex = ['Gender']

final_df = pd.get_dummies(liver_df,columns = sex,drop_first=True)
final_df.head()
from sklearn.model_selection import train_test_split

X = final_df.drop('Dataset',axis=1)

y = final_df['Dataset']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=700)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))