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
# Import utility libraries

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import chi2_contingency



%matplotlib inline
# Loading Dataset

file_location = "/kaggle/input/students-performance-in-exams/StudentsPerformance.csv"

df = pd.read_csv(file_location)

df.head()
# Spliting data by gender

female_scores = df[ df['gender'] == 'female']

male_scores = df [ df['gender'] == 'male']



# Plotting distributin

sns.distplot( male_scores['math score'], color = 'green', label='male')

sns.distplot( female_scores['math score'], color='blue', label='female')

plt.title('Gender-wise marks distribution')

plt.legend()

plt.show()
# Creating 4 Bins to categorizes math score

bin_labels = ['Low', 'Medium', 'High', 'Excellent']

df['math score grads'] = pd.cut(df['math score'],

                               bins=len(bin_labels),

                               labels=bin_labels

                               )
# Cropping out paired list of (gender, math grad)

ZippedList = list(zip(df['gender'], df['math score grads']))



# Converting to DataFrame

df_genderMathScore = pd.DataFrame(ZippedList,

                                  columns=['gender', 'math score grade'],

                                 )

df_genderMathScore['freq']=1 #dummy column just for later counting computation, can have any value

df_genderMathScore.head()
# Creating Frequency DataFrame of above

df_genderMathScoreContingencyTable = df_genderMathScore.pivot_table(index='gender',

           columns='math score grade',

           values='freq',

           aggfunc='count').fillna(0)

df_genderMathScoreContingencyTable
chiState, pValue, dof, expected = chi2_contingency(df_genderMathScoreContingencyTable)

chiState, pValue, dof, expected