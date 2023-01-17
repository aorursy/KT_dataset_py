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
data = pd.read_csv('/kaggle/input/student-performance-data-set/student-por.csv')
data
data.describe(include = 'all')
data.mean()
data.median()
data.mode()
numeric = list(data._get_numeric_data().columns)
numeric
categorical = list(set(data.columns) - set(data._get_numeric_data().columns))
categorical
data["average_marks"]=(data["G1"]+data["G2"]+data["G3"])/3
data
def marks(average_marks):
    if(average_marks<=6):
        return("low")
    elif(average_marks>=7 and average_marks<=14):
        return("average")
    elif(average_marks>=15):
        return("high")
data["grades"]=data["average_marks"].apply(marks)
data
#visualizing the grades
plt.figure(figsize=(8,6))
sns.countplot(data["grades"], order=["low","average","high"], palette='Set1')
plt.title('Final Grade - Number of Students',fontsize=20)
plt.xlabel('Final Grade', fontsize=16)
plt.ylabel('Number of Student',fontsize = 16)
# Visual on basis of gender and age of students
res = data.groupby(['sex', 'age']).size().unstack()
res.plot(kind='bar', figsize=(15,10))
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
#describing correlation
correlation=data.corr()

plt.figure(figsize=(20,20))
sns.heatmap(correlation, annot=True, cmap="Reds")
plt.title('Correlation Heatmap', fontsize=20)
