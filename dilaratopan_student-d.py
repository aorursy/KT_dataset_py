# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import warnings 

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data= pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
data.columns

data.head()
data.tail()
data.describe()
data.info()
data["parental level of education"].unique()
data["race/ethnicity"].unique()
data["parental level of education"].value_counts()
 #Gender show bar plot

sns.set(style='whitegrid')

ax=sns.barplot(x=data['gender'].value_counts().index,y=data['gender'].value_counts().values,palette="Blues_d",hue=['female','male'])

plt.legend(loc=8)

plt.xlabel('gender')

plt.ylabel('Frequency')

plt.title('Show of Gender Bar Plot')

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(x=data["gender"], y=data["math score"], data=data)

plt.xticks(rotation=90)
plt.figure(figsize=(5,5))

sns.barplot(x=data["gender"], y=data["reading score"], data=data)

plt.xticks(rotation=90)
plt.figure(figsize=(5,5))

sns.barplot(x=data["gender"], y=data["writing score"], data=data)

plt.xticks(rotation=90)
data.head()
f,ax = plt.subplots=(10,7)

sns.barplot(x=data['math score'], y=data['race/ethnicity'] , hue= data['race/ethnicity'])
