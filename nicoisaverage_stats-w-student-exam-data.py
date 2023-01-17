import seaborn as sns 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from pandas import plotting

from scipy import stats 

plt.style.use("ggplot")

import warnings 

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))

df = pd.read_csv('../input/StudentsPerformance.csv')
# Take a quick look at the data 

df.shape

df.columns
df.head()
print(df['gender'].unique())

print(df['race/ethnicity'].unique())

print(df['lunch'].unique())

print(df['test preparation course'].unique())
free_reduced = df[df['lunch'] == "free/reduced"]
standard = df[df['lunch'] == 'standard']
free_reduced.describe()

standard.describe()
math_s = plt.hist(standard['math score'],bins=30,label='Standard')

math_f = plt.hist(free_reduced['math score'],bins=30,alpha=0.6,color='blue',label='Free/Reduced')

plt.legend()

plt.xlabel('Math Score')

plt.ylabel('Students')

plt.show()



                
math_s = plt.hist(standard['reading score'],bins=30,label='Standard')

math_f = plt.hist(free_reduced['reading score'],bins=30,alpha=0.6,color='blue',label='Free/Reduced')

plt.legend()

plt.xlabel('Reading Score')

plt.ylabel('Students')

plt.show()
math_s = plt.hist(standard['writing score'],bins=30,label='Standard')

math_f = plt.hist(free_reduced['writing score'],bins=30,alpha=0.6,color='blue',label='Free/Reduced')

plt.legend()

plt.xlabel('Writing Score')

plt.ylabel('Students')

plt.show()