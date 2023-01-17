import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()
df.info()
sns.heatmap(df.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')
df['salary'] = df['salary'].fillna(value = 0)
df.head()
df.info()
sns.heatmap(df.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')
print(df['gender'].unique())

print(df['status'].unique())

print(df['workex'].unique())

print(df['hsc_b'].unique())

print(df['ssc_b'].unique())
df.set_index('sl_no',inplace = True)
df.head()
sns.set_style('whitegrid')

sns.countplot(x = 'gender', data = df)

plt.title('Gender Comparison')

df['gender'].value_counts()
sns.set_style('whitegrid')

sns.countplot(x = 'ssc_b', data = df)

plt.title('College Comparison')

df['ssc_b'].value_counts()
sns.set_style('whitegrid')

sns.countplot(x = 'degree_t', data = df)

plt.title('Degree Comparison')

df['degree_t'].value_counts()
sns.set_style('whitegrid')

sns.countplot(x = 'hsc_s', data = df)

plt.title('Stream Comparison')

df['hsc_s'].value_counts()
plt.pie(df.gender.value_counts(),labels = ['Male','Female'], autopct = '%1.2f%%', shadow = True, startangle = 90)
sns.set_style('whitegrid')

sns.countplot(x = 'status', data = df)

plt.title('No of students placed')

df['status'].value_counts()
sns.set_style('darkgrid')

sns.countplot(x = 'gender', hue = 'status', data = df)

plt.title('No of students placed amongst gender')

print('Out of 139 males, 100 got placed which is {}'.format((100/139)*100))

print('Out of 76 females, 48 got placed which is {}'.format((48/76)*100))
sns.set_style('darkgrid')

sns.countplot(x = 'ssc_b', hue = 'status', data = df)

plt.title('No of students placed amongst board in SSC')
sns.set_style('darkgrid')

sns.countplot(x = 'hsc_b', hue = 'status', data = df)

plt.title('No of students placed amongst board in HSC')
sns.set_style('darkgrid')

sns.countplot(x = 'degree_t', hue = 'status', data = df)

plt.title('No of students placed amongst degree')
sns.set_style('darkgrid')

sns.countplot(x = 'workex', hue = 'status', data = df)

plt.title('No of students placed amongst workex')
sns.scatterplot(x = 'ssc_p', y = 'status', data = df)
sns.scatterplot(x = 'hsc_p', y = 'status', data = df)
sns.scatterplot(x = 'degree_p', y = 'status', data = df)
sns.scatterplot(x = 'mba_p', y = 'status', data = df)
sns.set_style('darkgrid')

sns.countplot(x = 'specialisation', hue = 'status', data = df)

plt.title('No of students placed amongst specialisation')
data = df[['salary','ssc_p','hsc_p','degree_p','etest_p','mba_p']]

corr = data.corr()

corr
plt.figure(figsize = (8,6))

sns.heatmap(corr, annot = True)