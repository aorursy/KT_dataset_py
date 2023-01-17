# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'

data_campus = pd.read_csv(path, index_col = 'sl_no')

data_campus.head()
data_campus.info()
data_campus['salary']=data_campus['salary'].fillna(0)
print(data_campus['gender'].unique())

print(data_campus['ssc_b'].unique())

print(data_campus['hsc_b'].unique())

print(data_campus['degree_t'].unique())

print(data_campus['workex'].unique())

print(data_campus['specialisation'].unique())

print(data_campus['status'].unique())
#plt.figure(figsize=(15,10))

sns.catplot('gender' , kind='count', hue = 'status',data = data_campus)

plt.title('Gender vs Status', fontsize=16)
gender_status = data_campus.groupby(['gender', 'status'])

group_gen_sts = gender_status.size()

group_gen_sts.name = 'total'

group_gen_sts = group_gen_sts.reset_index()

group_gen_sts

def normal_total(group):

    group['normal_data'] = group.total/group.total.sum()

    return group



gender_normal = group_gen_sts.groupby('gender').apply(normal_total)

gender_normal
sns.barplot(x = 'gender', y = 'normal_data', hue = 'status',data = gender_normal, order = ['M', 'F'], hue_order = ['Placed', 'Not Placed'])

plt.title('Gender vs Status (normalised data)', fontsize=16)
sns.set(style="ticks")

sns.catplot('workex', kind = 'count', hue = 'status',data = data_campus)

plt.title('Work experience vs Status', fontsize=16)
workex_status = data_campus.groupby(['workex', 'status'])

group_work_sts = workex_status.size()

group_work_sts.name = 'total'

group_work_sts = group_work_sts.reset_index()

group_work_sts
workex_normal = group_work_sts.groupby('workex').apply(normal_total)

workex_normal
sns.barplot(x = 'workex', y = 'normal_data', hue = 'status',data = workex_normal, hue_order = ['Placed', 'Not Placed'])

plt.title('Work experience vs Status (normalised data)', fontsize=16)
sns.pairplot(data_campus, vars=['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p'], hue='status')

plt.title('Relationship between the different percentage', fontsize=16)
sns.catplot('hsc_s', kind = 'count', hue = 'status',data = data_campus)

plt.title('Higher secondary stream vs Status', fontsize=16)
hsc_status = data_campus.groupby(['hsc_s', 'status'])

group_hsc_sts = hsc_status.size()

group_hsc_sts.name = 'total'

group_hsc_sts = group_hsc_sts.reset_index()

group_hsc_sts
hsc_normal = group_hsc_sts.groupby('hsc_s').apply(normal_total)

hsc_normal
sns.barplot(x = 'hsc_s', y = 'normal_data', hue = 'status',data = hsc_normal, order = ['Commerce', 'Science', 'Arts'], hue_order = ['Placed', 'Not Placed'])

plt.title('Higher secondary stream vs Status (normalised data)', fontsize=16)
sns.catplot('degree_t', kind = 'count', hue = 'status',data = data_campus, order = ['Comm&Mgmt', 'Sci&Tech', 'Others'])

plt.title('Degree type vs Status', fontsize=16)
degree_status = data_campus.groupby(['degree_t', 'status'])

group_degree_sts = degree_status.size()

group_degree_sts.name = 'total'

group_degree_sts = group_degree_sts.reset_index()

group_degree_sts
degree_normal = group_degree_sts.groupby('degree_t').apply(normal_total)

degree_normal
sns.barplot(x = 'degree_t', y = 'normal_data', hue = 'status',data = degree_normal, order = ['Comm&Mgmt', 'Sci&Tech', 'Others'], hue_order = ['Placed', 'Not Placed'])

plt.title('Degree type vs Status (normalised data)', fontsize=16)
sns.catplot('specialisation', kind = 'count', hue = 'status',data = data_campus)

plt.title('MBA specialisation vs Status', fontsize=16)
mba_status = data_campus.groupby(['specialisation', 'status'])

group_mba_sts = mba_status.size()

group_mba_sts.name = 'total'

group_mba_sts = group_mba_sts.reset_index()

group_mba_sts
mba_normal = group_mba_sts.groupby('specialisation').apply(normal_total)

mba_normal
sns.barplot(x = 'specialisation', y = 'normal_data', hue = 'status',data = mba_normal, order = ['Mkt&HR', 'Mkt&Fin'], hue_order = ['Placed', 'Not Placed'])

plt.title('MBA specialisation vs Status (normalised data)', fontsize=16)
sns.distplot(data_campus.salary[data_campus.salary > 0])

plt.title('Salary distribution',size=15)

mean_salary = data_campus.salary[data_campus.salary > 0].mean()

median_salary = data_campus.salary[data_campus.salary > 0].median()

plt.axvline(mean_salary,color='red')

plt.axvline(median_salary,color='green')

plt.title('Salary distribution \n Mean={0:.2f}   Median={1:.2f}'.format(mean_salary,median_salary))

sns.boxplot(data_campus.salary[data_campus.salary > 0],orient='v')

plt.title('Boxplot of salary', fontsize=16)
mba_salary = data_campus[data_campus.salary > 0].groupby('specialisation')[['salary']].mean()



mba_salary