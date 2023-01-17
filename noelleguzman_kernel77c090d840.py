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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

df.head()
#check for any null data

df.isnull().sum()
df['gender'].value_counts()
df['race/ethnicity'].value_counts()
df['parental level of education'].value_counts()
df['test preparation course'].value_counts()
df.describe()
df.groupby('gender').describe()
df.corr()
sns.pairplot(df)

plt.show()

#strong correlation with reading and writing
math_plot = sns.distplot(df['math score'])
reading_plot = sns.distplot(df['reading score'])
writing_plot = sns.distplot(df['writing score'])
df.groupby(['parental level of education']).mean().plot.bar()

plt.show()
df.groupby(['gender']).mean().plot.bar()

plt.show()
sns.heatmap(df.corr(), annot = True)

plt.xticks(rotation = 50)

plt.yticks (rotation = 0)

plt.title('Correlation Analysis')