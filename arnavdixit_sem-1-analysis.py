#Importing the libraries

import numpy as np 

import pandas as pd 

import seaborn as sns 

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
#Importing the dataset

dataset = pd.read_csv("/kaggle/input/manit-201923-batch-sem1-result/first_year.csv")

dataset
#Removing Students with 0 GPA who didn't appear for exam

students_absent = dataset[dataset.GPA == 0]

dataset = dataset[dataset.GPA != 0]

students_absent.head()
# Plotting Section wise students who didn't appear for exams

sns.catplot(x="Section", data=students_absent, kind='count', order='ABCDEFGHIJ')
#Students who did not clear the semester

not_clear = dataset[dataset.Result == 'Not Clear']

sns.catplot(x = "Section", data=not_clear, kind='count', aspect=2, order="ABCDEFGHIJ")
#Total student strength sectionwise

sec_plt = sns.catplot(x='Section', data=dataset, aspect=2, kind='count', order="ABCDEFGHIJ")

sec_plt.set(ylim=(80, 105))

sns.catplot(x='Section', data=dataset, aspect=2, kind='count', hue='Gender', order="ABCDEFGHIJ")
#Branchwise Strength in different sections

sns.catplot(x='Section', data=dataset, kind='count', col='Branch', col_wrap=4, order="ABCDEFGHIJ")
#Kernel Density Extimate for GPA

sns.kdeplot(data=dataset['GPA'], bw=0.1, shade=True)
#Kernel Density Extimate Gender wise

m_gpa = dataset[dataset.Gender=='M']

f_gpa = dataset[dataset.Gender=='F']

sns.kdeplot(m_gpa['GPA'], bw=0.12, shade = True, label='Males')

sns.kdeplot(f_gpa['GPA'], bw=0.12, shade = True, label='Females')
#Section wise GPA

sns.catplot(x='Section', y='GPA', data=dataset, kind='box')
#Sectionwise GPA by Gender

sns.lineplot(x='Section', y='GPA', hue='Gender', data=dataset)
#Branchwise GPA

sns.catplot(x='Branch', y='GPA', data=dataset, kind='boxen')
#Branchwise GPA for Males and Females

sns.catplot(x='Branch', y='GPA', data=dataset, kind='point', hue='Gender')
#Branchwise GPA across all sections

sns.catplot(x='Section', y='GPA', data=dataset, hue='Branch', kind='box', aspect=3)
#Adding new feature: second year section

pd.set_option('mode.chained_assignment', None)

x = dataset['Scholar No'].copy() % 1000

sec=[]

for temp in x:

    if temp // 200 == 0:

        sec.append(1)

    elif temp // 200 == 1:

        sec.append(2)

    else:

        sec.append(3)

dataset['Section-Sem3'] = sec
#Plotting Second year section wise GPA

sns.catplot(x='Section-Sem3', y='GPA', col='Branch', data=dataset, kind='point', aspect=0.6, col_wrap=7)
#Extracting initials

x = dataset.Name

d=[]

for i in dataset.Name:

    d.append(i[0])

dataset['Initials']=d
#Plotting count of students with same initials

sns.catplot(x='Initials', kind='count', data=dataset)