# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.head()
df.shape
df.isnull().sum()
df.info()
## object type features are categorical variables here.

col_list = ['sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s',

       'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p',

       'status', 'salary']



cat = []

for col in col_list:

    if df[col].dtypes == 'object':

        cat.append(col)

        

cat
## Visualisation using countplot

for col in cat:

    print(df[col].value_counts())

    sns.countplot(df[col])

    plt.show()
male_pct = round((100/139) * 100, 2)

female_pct = round((48/76) * 100, 2)

print(f"Male placement percentage: {male_pct}%")

print(f"Female placement percentage: {female_pct}%")
y = round((64/74) * 100, 2)

n = round((84/141) * 100, 2)

print(f"Percentage of students with workex who got placed: {y}%")

print(f"Percentage of students without workex who got placed: {n}%")
hr = round((53/95) * 100, 2)

fin = round((95/120) * 100, 2)

print(f"Percentage of students in Mkt&HR who got placed: {hr}%")

print(f"Percentage of students in Mkt&Fin who got placed: {fin}%")
pct = df[['ssc_p', 'hsc_p', 'degree_p', 'mba_p']]

sns.heatmap(pct.corr(), annot=True)
sns.scatterplot(x='mba_p', y='etest_p', data=df, hue='status')