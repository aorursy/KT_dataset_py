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
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.shape
data.info()
data.head()
data.describe()
data['salary'].median()
data.describe(include = 'O')
data.isnull().sum()
df = data.copy(deep = True)
df['salary'].fillna(0, inplace = True)
df.isna().sum()
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x="gender", data=df, hue="status")
sns.countplot(x="workex", data=df, hue="status")
sns.countplot(x="degree_t", data=df, hue="status")
sns.boxplot(x= df['gender'],y= df['salary'])
sns.boxplot(x= df['degree_t'],y= df['salary'])
sns.boxplot(x=df['hsc_s'],y=df['salary'],hue=df['gender'])
sns.boxplot(x=df['hsc_b'],y=df['salary'],hue=df['gender'])
sns.scatterplot(x='degree_p', y='salary', hue='status', data=df)
sns.set(style="darkgrid")
sns.regplot(x=df['degree_p'],y=df['hsc_p'], fit_reg=False, color='r')
sns.distplot(df['salary'], kde=False, bins=7, color='g')
sns.distplot(df['degree_p'], kde=False, bins=8, color='y')
sns.catplot(x="degree_t", y="etest_p", data=df);
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='GnBu')