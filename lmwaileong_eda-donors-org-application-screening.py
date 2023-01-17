import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
import squarify
from pandas import set_option
import os
import re

plt.rcParams['patch.force_edgecolor']=True
plt.rcParams['patch.facecolor']='b'

# get colors
rb = []
colors = plt.cm.rainbow_r(np.linspace(0,1,51))
for c in colors:
    rb.append(c)
rb = list(rb)
data = pd.read_csv('../input/train.csv')
data.head(3)
data['teacher_gender'] = data['teacher_prefix'].map({'Mrs.':'Female','Ms.':'Female','Mr.':'Male',
                                                     'Teacher':'Unknown','Dr.':'Unknown'})
# dealing with datetime
data['project_submitted_datetime'] = pd.to_datetime(data['project_submitted_datetime'], format='%Y-%m-%d %H:%M:%S', utc=True)

data['year'] = data['project_submitted_datetime'].dt.year
data['month'] = data['project_submitted_datetime'].dt.month
data['year'] = data['project_submitted_datetime'].dt.year
data['day'] = data['project_submitted_datetime'].map(lambda x: x.isoweekday())
fig, ax = plt.subplots(1,2, figsize=(18,6))

# ax[0] Number of submissions by respective teachers
ax[0].boxplot(data['teacher_id'].value_counts(), showfliers=False)
ax[0].set_title('# project submission by individual teachers')
ax[0].set_xticklabels(' ')
ax[0].set_ylabel('# of submission')

# ax[1] Number of submissions by month
month = data.groupby(['month','year'], as_index=False)['id'].count()
sns.barplot(x='month', y='id', data=month, hue='year', ax=ax[1], palette='viridis')
ax[1].legend(loc='upper left')
ax[1].set_ylabel('Number of projects submmited')
ax[1].set_title('# project submission by month')

plt.tight_layout()
plt.show()

# Number of submissions by state
plt.figure(figsize=(18,9))
school = data['school_state'].value_counts()/data['school_state'].count()*100
squarify.plot(sizes=school.values, label=school.index, color=rb, alpha=0.6,value=school.values.round(1))
plt.axis('off')
plt.title('# project submission and approval rate by state')

plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1,3, figsize=(20,6))

# ax[0] Overall approval rate
approval_rate = data['project_is_approved'].value_counts()/data['id'].count()*100
ax[0].pie(approval_rate, labels=['Approved', 'Rejected'], autopct='%1.1f%%', startangle=90, 
          colors=['powderblue', 'lightcoral'], wedgeprops={'linewidth' :4,'edgecolor':'white'}, pctdistance=0.8)
ax[0].set_title('Percentage of approved projects')
ax[0].axis('equal')

white_circle=plt.Circle( (0,0), 0.6, color='white') # adding white space to create donut chart
ax[0].add_artist(white_circle)

# ax[1] Submissions & approval rate by gender
gender = data.groupby('teacher_gender', as_index=False).agg({'teacher_id':'size','project_is_approved':'mean'})
ax[1].bar(gender['teacher_gender'].values, gender['teacher_id'], color='cadetblue')
ax2 = ax[1].twinx() # instantiate a second axes that shares the same x-axis
ax2.plot(gender['project_is_approved'], color='lightcoral', linewidth = 3, marker='o', markersize=8)
ax[1].set_title('Submissions and approval rate by gender')
ax2.set_ylabel('approval rate',rotation=-90)
ax[1].set_ylabel('# of submissions')

# Approval by school grade
grade_approved = data[data['project_is_approved']==1][['project_grade_category','project_is_approved']].\
    groupby('project_grade_category')['project_is_approved'].count().tolist()
grade_not_approved = data[data['project_is_approved']==0][['project_grade_category','project_is_approved']].\
    groupby('project_grade_category')['project_is_approved'].count().tolist()

n = data.groupby('project_grade_category')['id'].count().index
ax[2].bar(n, grade_approved, label='Approved', color='gold', alpha=0.5)
ax[2].bar(n, grade_not_approved, label='Not approved', bottom=grade_approved, color='lightcoral', alpha=0.7)
ax[2].legend(loc='best')
ax[2].set_title('Submissions and approval rate by grade')

plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1,2, figsize=(18,7))

# ax[0] Subject categories
subjects = pd.DataFrame(data['project_subject_categories'].str.split(', ',expand=True)).stack().value_counts()
    # Splitting subject categories; e.g. 'Health & Sports, Applied Learning' to 'Health & Sports','Applied Learning'
subjects = subjects.reset_index()
subjects = subjects.rename(columns={'index':'subjects',0:'frequency'}) # rename columns
sns.barplot(x='frequency', y='subjects', data=subjects, palette='viridis', ax=ax[0])
ax[0].set_title('Number of projects involving unique categories')

# ax[1] Subject categories
subcategories = pd.DataFrame(data['project_subject_subcategories'].str.split(', ',expand=True)).stack().value_counts()
subcategories = subcategories.reset_index()
subcategories = subcategories.rename(columns={'index':'subcategories',0:'frequency'})
sns.barplot(x='frequency', y='subcategories', data=subcategories, palette='viridis', ax=ax[1])
ax[1].set_title('Number of projects involving unique sub-categories')

plt.tight_layout()
plt.show()
fig, ax1 = plt.subplots(figsize=(18,8))

subject_count = data.groupby('project_subject_categories', as_index=False)['project_is_approved'].count()
subject_approval = data.groupby('project_subject_categories', as_index=False)['project_is_approved'].mean()

sns.barplot(x='project_subject_categories',y='project_is_approved',data=subject_count, ax=ax1)
ax1.set_xticklabels(subject_count['project_subject_categories'].tolist(),rotation=90)
ax1.set_ylabel('Number of submission')
ax1.set_title('Submissions categories (incl.combo) and approval rate')
ax2 = ax1.twinx()
ax2.plot('project_subject_categories','project_is_approved',data=subject_approval, color='lightcoral', linewidth=3,
        marker='o', markersize=9)
ax2.set_ylabel('approval rate')


plt.tight_layout()
plt.show()
from nltk.corpus import stopwords
s=list(stopwords.words('english'))

essay1 = pd.DataFrame(data['project_essay_1'].str.split(expand=True)).stack().value_counts()
essay1 = essay1.reset_index()
essay1 = essay1.rename(columns={'index':'words',0:'count'})
essay_cleaned = essay1[~essay1['words'].isin(s)]

essay2 = pd.DataFrame(data['project_essay_2'].str.split(expand=True)).stack().value_counts()
essay2 = essay2.reset_index()
essay2 = essay2.rename(columns={'index':'words',0:'count'})
essay_cleaned2 = essay2[~essay2['words'].isin(s)]
fig, ax = plt.subplots(1,2, figsize=(18,6))
es1_top10 = essay_cleaned.head(10)
es2_top10 = essay_cleaned2.head(10)

sns.barplot(x='words',y='count', data=es1_top10, palette='magma', ax=ax[0])
sns.barplot(x='words',y='count', data=es2_top10, palette='magma', ax=ax[1])
ax[0].set_title('Top 10 words used in essay 1')
ax[1].set_title('Top 10 words used in essay 2')

plt.tight_layout()
plt.show()