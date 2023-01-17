import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

%matplotlib inline
campus_DF = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
campus_DF.shape
campus_DF.head(10)
campus_DF = campus_DF.drop(columns = ['sl_no'])
campus_DF['salary'] = campus_DF['salary'].fillna(0)
campus_DF_placed = campus_DF[campus_DF['status'] == 'Placed']

campus_DF_not_placed = campus_DF[campus_DF['status'] == 'Not Placed']
fig, a1 = plt.subplots(2, 2,figsize=(16,12))

sns.countplot(x = "gender", data = campus_DF, ax = a1[0,0])

sns.countplot(x = "status", data = campus_DF, ax = a1[0,1])

sns.countplot(x = "ssc_b", data = campus_DF, ax = a1[1,0])

sns.countplot(x = "hsc_b", data = campus_DF, ax = a1[1,1])

plt.show()
fig, a2 = plt.subplots(2, 2,figsize=(16,12))

sns.countplot(x = "hsc_s", data = campus_DF, ax = a2[0,0])

sns.countplot(x = "degree_t", data = campus_DF, ax = a2[0,1])

sns.countplot(x = "specialisation", data = campus_DF, ax = a2[1,0])

sns.countplot(x = "workex", data = campus_DF, ax = a2[1,1])

plt.show()
f, axes = plt.subplots(4, 2,figsize=(12,20))

sns.countplot(x = "gender", hue = "status", data = campus_DF, ax = axes[0,0])

sns.countplot(x = "ssc_b", hue = "status", data = campus_DF, ax = axes[0,1])

sns.countplot(x = "hsc_b", hue = "status", data = campus_DF, ax = axes[1,0])

sns.countplot(x = "hsc_s", hue = "status", data = campus_DF, ax = axes[1,1])

sns.countplot(x = "degree_t", hue = "status", data = campus_DF, ax = axes[2,0])

sns.countplot(x = "workex", hue = "status", data = campus_DF, ax = axes[2,1])

sns.countplot(x = "specialisation", hue = "status", data = campus_DF, ax = axes[3,0])

f.delaxes(axes[3,1])

plt.show()
plt.figure(figsize=(10,5))

sns.boxplot(campus_DF_placed['salary'], color = '#b8adf0', linewidth = 2.2)

plt.show()



campus_DF_placed['salary'].describe()
plt.figure(figsize=(10,5))

sns.kdeplot(campus_DF_placed['salary'], color = '#b8adf0', shade = True)

plt.show()
plt.figure(figsize=(10, 5))

sns.boxplot(x = 'salary', y = 'gender', data = campus_DF_placed, linewidth = 2.2)

plt.show()
campus_DF_placed.groupby('gender').describe()['salary']
plt.figure(figsize=(10, 5))

sns.boxplot(x = 'salary', y = 'ssc_b', data = campus_DF_placed, linewidth = 2.2)

plt.show()
campus_DF_placed.groupby('ssc_b').describe()['salary']
plt.figure(figsize=(10, 5))

sns.boxplot(x = 'salary', y = 'hsc_b', data = campus_DF_placed, linewidth = 2.2)

plt.show()
campus_DF_placed.groupby('hsc_b').describe()['salary']
plt.figure(figsize=(10, 5))

sns.boxplot(x = 'salary', y = 'hsc_s', data = campus_DF_placed, linewidth = 2.2)

plt.show()
campus_DF_placed.groupby('hsc_s').describe()['salary']
plt.figure(figsize=(10,5))

sns.boxplot(x = 'salary', y = 'degree_t', data = campus_DF_placed, linewidth = 2.2)

plt.show()
campus_DF_placed.groupby('degree_t').describe()['salary']
plt.figure(figsize=(10,5))

sns.boxplot(x = 'salary', y = 'workex', data = campus_DF_placed, linewidth = 2.2)

plt.show()
campus_DF_placed.groupby('workex').describe()['salary']
plt.figure(figsize=(10,5))

sns.boxplot(x = 'salary', y = 'specialisation', data = campus_DF_placed, linewidth = 2.2)

plt.show()
campus_DF_placed.groupby('specialisation').describe()['salary']
f, ax2 = plt.subplots(3,2,figsize = (15,14))

sns.distplot(campus_DF['ssc_p'], bins = 10, color = '#b8adf0', kde = False, ax = ax2[0,0])

sns.distplot(campus_DF['hsc_p'], bins = 12, color = '#da19e0', kde = False, ax = ax2[0,1])

sns.distplot(campus_DF['degree_p'], bins = 8, color = '#ab0eb0', kde = False, ax = ax2[1,0])

sns.distplot(campus_DF['etest_p'], bins = 10, color = '#660769', kde = False, ax = ax2[1,1])

sns.distplot(campus_DF['mba_p'], bins = 6, color = '#450247', kde = False, ax = ax2[2,0])

f.delaxes(ax2[2,1])

plt.show()
plt.figure(figsize=(9,4))

sns.boxplot(x = 'etest_p', y = 'status', data = campus_DF, linewidth = 2.2)

plt.show()
plt.figure(figsize=(8,6))

sns.kdeplot(campus_DF['etest_p'].loc[campus_DF['status']=='Placed'])

sns.kdeplot(campus_DF['etest_p'].loc[campus_DF['status']=='Not Placed'])

plt.legend(['Placed', 'Not Placed'])

plt.show()
campus_DF.groupby('status').describe()['etest_p']
plt.figure(figsize=(9,4))

sns.boxplot(x = 'mba_p', y = 'status', data = campus_DF, linewidth = 2.2)

plt.show()
plt.figure(figsize=(8,6))

sns.kdeplot(campus_DF['mba_p'].loc[campus_DF['status']=='Placed'])

sns.kdeplot(campus_DF['mba_p'].loc[campus_DF['status']=='Not Placed'])

plt.legend(['Placed', 'Not Placed'])

plt.show()
campus_DF.groupby('status').describe()['mba_p']
plt.figure(figsize=(9,4))

sns.boxplot(x = 'degree_p', y = 'status', data = campus_DF, linewidth = 2.2)

plt.show()
plt.figure(figsize=(8,6))

sns.kdeplot(campus_DF['degree_p'].loc[campus_DF['status']=='Placed'])

sns.kdeplot(campus_DF['degree_p'].loc[campus_DF['status']=='Not Placed'])

plt.legend(['Placed', 'Not Placed'])

plt.show()
campus_DF.groupby('status').describe()['degree_p']
plt.figure(figsize=(9,4))

sns.boxplot(x = 'hsc_p', y = 'status', data = campus_DF, linewidth = 2.2)

plt.show()
plt.figure(figsize=(8,6))

sns.kdeplot(campus_DF['hsc_p'].loc[campus_DF['status']=='Placed'])

sns.kdeplot(campus_DF['hsc_p'].loc[campus_DF['status']=='Not Placed'])

plt.legend(['Placed', 'Not Placed'])

plt.show()
campus_DF.groupby('status').describe()['hsc_p']
plt.figure(figsize=(9,4))

sns.boxplot(x = 'ssc_p', y = 'status', data = campus_DF, linewidth = 2.2)

plt.show()
plt.figure(figsize=(8,6))

sns.kdeplot(campus_DF['ssc_p'].loc[campus_DF['status']=='Placed'])

sns.kdeplot(campus_DF['ssc_p'].loc[campus_DF['status']=='Not Placed'])

plt.legend(['Placed', 'Not Placed'])

plt.show()
campus_DF.groupby('status').describe()['ssc_p']