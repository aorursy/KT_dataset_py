import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns 

data=pd.read_csv('../input/StudentsPerformance.csv')

data.head()
from sklearn.preprocessing import LabelEncoder

label_data=[]

for i in range(5):

    label=LabelEncoder()

    label.fit(data.iloc[:,i])

    new_data=label.transform(data.iloc[:,i])

    print(label.classes_)

    label_data.append(new_data)
data1=data[['math score', 'reading score',

       'writing score']]

label_data=pd.DataFrame(label_data).T

frames=[label_data,data1]

final_data=pd.concat(frames,axis=1)

final_data.columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch',

       'test preparation course', 'math score', 'reading score',

       'writing score']
plt.figure(figsize=(10,10))

sns.heatmap(final_data.corr(),annot=True,vmax=1,vmin=-1)
final_data.head()
plt.figure(figsize=(10,4))

sns.boxplot(data['race/ethnicity'],data['math score'])
plt.figure(figsize=(10,4))

sns.boxplot(data['parental level of education'],data['math score'])
plt.figure(figsize=(6,4))

sns.boxplot(data['test preparation course'],data['math score'])
plt.figure(figsize=(6,4))

sns.boxplot(data['lunch'],data['math score'])