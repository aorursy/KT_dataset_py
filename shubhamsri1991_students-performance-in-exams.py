# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
students_performance_dataset = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
students_performance_dataset.shape
students_performance_dataset.head()
students_performance_dataset.info()
students_performance_dataset.duplicated().sum()
students_performance_dataset.describe()
students_performance_dataset['test preparation course'].unique()
students_performance_dataset['test preparation course'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(10,8))

plt.title('Percentage Of Students Completed The Course')
plt.figure(figsize=(10,5))

sns.distplot(students_performance_dataset['math score'][students_performance_dataset['test preparation course']=='none'], color='red')

sns.distplot(students_performance_dataset['math score'][students_performance_dataset['test preparation course']=='completed'], color='green')

plt.title('Test Preparation Course Vs Math Score')

plt.show()
plt.figure(figsize=(10,5))

sns.distplot(students_performance_dataset['reading score'][students_performance_dataset['test preparation course']=='none'], color='red')

sns.distplot(students_performance_dataset['reading score'][students_performance_dataset['test preparation course']=='completed'], color='green')

plt.title('Test Preparation Course Vs Reading Score')

plt.show()
plt.figure(figsize=(10,5))

sns.distplot(students_performance_dataset['writing score'][students_performance_dataset['test preparation course']=='none'], color='red')

sns.distplot(students_performance_dataset['writing score'][students_performance_dataset['test preparation course']=='completed'], color='green')

plt.title('Test Preparation Course Vs Writing Score')

plt.show()
students_performance_dataset['parental level of education'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(10,8))

plt.title('Parental Level Of Education')
plt.figure(figsize=(10,5))

sns.countplot(students_performance_dataset['race/ethnicity'][students_performance_dataset['test preparation course']=='completed'])

plt.title('Race/Ethnicity Vs Test Preparation Course')
plt.figure(figsize=(10,5))

sns.countplot(students_performance_dataset['lunch'][students_performance_dataset['test preparation course']=='completed'])

plt.title('Lunch Vs Test Preparation Course')
plt.figure(figsize=(12,5))

sns.countplot(students_performance_dataset['parental level of education'][students_performance_dataset['test preparation course']=='completed'])

plt.title('Parental Level Of Education Vs Test Preparation Course')
d1 = students_performance_dataset[(students_performance_dataset['gender']=='male') & (students_performance_dataset['test preparation course']=='completed')]

d2 = students_performance_dataset[(students_performance_dataset['gender']=='male') & (students_performance_dataset['test preparation course']=='none')]

d3 = students_performance_dataset[(students_performance_dataset['gender']=='female') & (students_performance_dataset['test preparation course']=='completed')]

d4 = students_performance_dataset[(students_performance_dataset['gender']=='female') & (students_performance_dataset['test preparation course']=='none')]



values = [len(d1),len(d2),len(d3),len(d4)]

labels = ['Male Completed Course','Male Not Completed Course', "Female Completed Course", "Female Not Completed Course"]

colors = ['green', 'orange', 'blue', 'grey']



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(title_text="ANALYSIS OF Gender Vs Test Preparation Course")

fig.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=1)))

fig.show()