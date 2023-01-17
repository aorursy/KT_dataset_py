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
#activity 6
#subodh kumar 18scse1010125   https://www.kaggle.com/subodhkumar4957/cat-3-ds/edit
#Shubhi Saxena 18SCSE1050040  https://www.kaggle.com/shubhisaxena/notebookede319203e/edit/run/44312229
#Siddhant Chaudhary 18SCSE1140054



#https://www.kaggle.com/rhuebner/human-resources-data-set
#1. Read the dataset.
#2. Describe the dataset.
#3. Find mean,median and mode of columns.
#4. Find the distribution of columns with numerical data. Evaluate if they are normal or
#not.
#5. Draw different types of graphs to represent information hidden in the dataset.
#6. Find columns which are correlated.
#7. Find columns which are not correlated.
#8. Compare different columns of dataset
#
#9. Is Any supervised machine learning possible ? if yes explain.
# 1. Read the dataset.
import pandas as pd
df=pd.read_csv("/kaggle/input/human-resources-data-set/HRDataset_v13.csv")
df
# 2. Describe the dataset. 
df.describe(include = 'all')

# 3. Find mean,median and mode of columns.
print("\n----------- Calculate Mean -----------\n")
print(df.mean())
 
print("\n----------- Calculate Median -----------\n")
print(df.median())
 
print("\n----------- Calculate Mode -----------\n")
print(df.mode())
# 4. Find the distribution of columns with numerical data. Evaluate if they are normal or not.
# Numeric Dataset and these features are normal


numeric = list(df._get_numeric_data().columns)
numeric
# 4. Find the distribution of columns with numerical data. Evaluate if they are normal or not.
# Ordinal columns
# columns which are not normal are ordinal
categorical = list(set(df.columns) - set(df._get_numeric_data().columns))
categorical
# 5. Draw different types of graphs to represent information hidden in the dataset.
import matplotlib.pyplot as plt
print(df['Sex'].value_counts())
df['Sex'].value_counts().plot(kind='bar')
# 5. Draw different types of graphs to represent information hidden in the dataset.
df['PayRate'].plot.hist()
# 5. Draw different types of graphs to represent information hidden in the dataset.
df.plot.hist()

# 5. Draw different types of graphs to represent information hidden in the dataset.
plt.hist(df.PerfScoreID)
plt.xlabel('PerfScoreID')
plt.ylabel('Total Employee')
# 5. Draw different types of graphs to represent information hidden in the dataset.
df1=df.loc[0:300,:]
df1
plt.plot(df1.PayRate,label='PayRate')
plt.plot(df1.Sex,label='Sex')
plt.plot(df1.PerfScoreID,label='PerfScoreID')
plt.legend()
df.plot.bar()

# 6. Find columns which are correlated.
# 7. Find columns which are not correlated.
df.corr()
# 8. Compare different columns of dataset
comparison_column = np.where(df["PayRate"] == df["SpecialProjectsCount"], True, False)
df["equal"] = comparison_column
df
#9. Is Any supervised machine learning possible ? if yes
#explain
#YES SUPERVISED LEARNING IS POSSIBLE. BECAUSE IN THIS DATASET WE HAVE LOTS OF LABELS AND SUPERVISED
#MACHINE LEARNING IS USED WHEN WE HAVE LABELS