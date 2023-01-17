import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



print('All libraries have been imported.')
filepath = '../input/students-performance-in-exams/StudentsPerformance.csv'

student = pd.read_csv(filepath)



student.head()
student.shape
student.dtypes
student.columns = ['gender', 'race', 'parent_education', 'lunch', 'test_prep', 'math_score', 'reading_score', 'writing_score']



student.head()
student.isnull().sum()
student.describe()
# Framework for subplots and subplot titles.

fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 5))

chart_titles = ['Math Score Distribution', 'Reading Score Distribution', 'Writing Score Distribution']



# Plot charts.

for col, ax, chart_title in zip(student.columns[-3:], axes.flatten(), chart_titles):

    sns.distplot(student[col], norm_hist = True, ax = ax, kde = False).set_title(chart_title, fontsize = 14)

    ax.set(xlabel = 'Score')

    

# Add gridlines to all plots from this point forward.

plt.rcParams['axes.grid'] = True
for col in student.columns[:5]:

    print('\n' + '-'*50)       # Serves as a divider between each column summary.

    print(col.upper() + ' COLUMN SUMMARY:\n') # Indicates which column is being summarized in each section.

    print(student[col].value_counts())
# Framework and titles for subplots.

fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (15, 15))

chart_titles = ['Gender (Total: 1,000)', 'Race (Total: 1,000)', 'Parental Education (Total: 1,000)', 'Lunch (Total: 1,000)', 'Test Preparation (Total: 1,000)']



# Plot charts.

for col, ax, chart_title in zip(student.columns[:5], axes.flatten(), chart_titles):

    sns.countplot(y = str(col), ax = ax, data = student).set_title(chart_title, fontsize = 14)

    ax.set(xlabel = 'Count', ylabel = col.replace('_', ' ').title())

fig.delaxes(axes[2,1]) # Delete extra plot, only needed 5.



# Adjust spacing.

plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
# New col for composite test score averages.

student['composite'] = student.mean(axis = 1).round(2)



student.head()
sns.distplot(student.composite, norm_hist = True, kde = False)

plt.title('Composite Score Distribution')

plt.xlabel('Score')
student = student.drop(columns = ['math_score', 'reading_score', 'writing_score'])

student.head()
# Grouped-by summary statistics

for col in student.columns[:5]:

    print('-'*50)

    print(col.upper() + ' COLUMN STATISTICAL SUMMARY:\n')

    print(student.groupby(col).describe())
# Framework and titles for subplots.

fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (15, 20))

chart_titles = ['Composite Test Score Avg.\nby Gender', 'Composite Test Score Avg.\nby Race','Composite Test Score Avg.\nby Level of Parental Education',

                'Composite Test Score Avg.\nby Lunch Type', 'Composite Test Score Avg.\nby Status of Test Prep Course']



# Plot charts.

for col, ax, chart_title in zip(student.columns[:5], axes.flatten(), chart_titles):

    sns.boxplot(x = str(col), y = 'composite', ax = ax, data = student).set_title(chart_title, fontsize = 14)

    ax.tick_params(axis = 'x', labelrotation = 45)

    ax.set(xlabel = 'Count', ylabel = col.replace('_', ' ').title())

fig.delaxes(axes[2,1]) # Delete extra plot, only need 5.



# Adjust spacing.

plt.subplots_adjust(wspace = 0.2, hspace = 0.55)