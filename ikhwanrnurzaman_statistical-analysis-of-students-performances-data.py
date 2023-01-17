import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



data = pd.read_csv('../input/StudentsPerformance.csv')

print(data.info())
data = data.rename(columns={'race/ethnicity':'group', 'parental level of education':'p_edu', 'test preparation course':'t_prep', 'math score':'math', 'reading score':'read', 'writing score':'write'})

data['final'] = data.mean(numeric_only=True, axis=1).round(2)

data['gender'] = data.gender.astype('category')

data['group'] = data.group.astype('category')

data['p_edu'] = data.p_edu.astype('category')

data['t_prep'] = data.t_prep.astype('category')

data['lunch'] = data.lunch.astype('category')

print(data.info())

print(data.head())

print(data.describe())

print('Group : ', data.group.unique())

print("Parents' education level : ", data.p_edu.unique())

print('Test preparation : ', data.t_prep.unique())

print('lunch : ', data.lunch.unique())
sns.set_style('darkgrid')

fig = plt.figure(figsize=(8,8))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



plt.subplot(2,2,1)

ax1 = sns.distplot(data.math, axlabel='score')

plt.axvline(data.math.mean(), color='r')

ax1.annotate(s=data.math.mean(), xy=(data.math.mean(), 0.026))

plt.title('Math Score')



plt.subplot(2,2,2)

ax2 = sns.distplot(data.read, axlabel='score')

plt.axvline(data.read.mean(), color='r')

ax2.annotate(s=data.read.mean(), xy=(data.read.mean(), 0.026))

plt.title('Reading Score')



plt.subplot(2,2,3)

ax3 = sns.distplot(data.write, axlabel='score')

plt.axvline(data.write.mean(), color='r')

ax3.annotate(s=data.write.mean(), xy=(data.write.mean(), 0.026))

plt.title('Writing Score')



plt.subplot(2,2,4)

ax4 = sns.distplot(data.final, axlabel='score')

plt.axvline(data.final.mean(), color='r')

ax4.annotate(s=round(data.final.mean(),3), xy=(data.final.mean(), 0.026))

plt.title('Final Score')
plt.figure(figsize=(8,8))

sns.boxenplot(data=data, x='group', y='final')

plt.title('Final score distribution based on group')
data['lunch_count'] = data['lunch']

data_1 = data.groupby(['group', 'lunch']).lunch_count.count()

data_1_ = data_1.groupby(level=0).apply(lambda x : 100*x/float(x.sum())).reset_index()

p = sns.catplot(data=data_1_, x='lunch', y='lunch_count', col='group', col_order=['group A', 'group E'], kind='bar', sharex=False)

p.set_axis_labels('Lunch Type', 'Percentage')
data['t_prep_count'] = data['t_prep']

data_2 = data.groupby(['group', 't_prep']).t_prep_count.count()

data_2_ = data_2.groupby(level=0).apply(lambda x : 100*x/float(x.sum())).reset_index()

p = sns.catplot(data=data_2_, x='t_prep', y='t_prep_count', col='group', col_order=['group A', 'group E'], col_wrap=3, kind='bar', sharex=False)

p.set_axis_labels('Test Preparation Completion', 'Percentage')
data['p_edu_count'] = data['p_edu']

data_3 = data.groupby(['group', 'p_edu']).p_edu_count.count()

data_3_ = data_3.groupby(level=0).apply(lambda x : 100*x/float(x.sum())).reset_index()

p = sns.catplot(data=data_3_, x='p_edu', order=["master's degree", "bachelor's degree", "associate's degree", 'some college', 'high school', 'some high school'], y='p_edu_count', col='group', col_order=['group A', 'group E'], col_wrap=3, kind='bar', sharex=False)

p.set_xticklabels(rotation=30)

p.set_axis_labels("Parents' Level Education", 'Percentage')
free = data[data['lunch'] == 'free/reduced']

standard = data[data['lunch'] == 'standard']



ax1 = sns.distplot(free.final, axlabel='final score', color='b', hist_kws=dict(alpha=0.5), label='Free Lunch', bins=15)

plt.axvline(round(free.final.mean(),2), color='b')

ax1.annotate(s=round(free.final.mean(),3), xy=(free.final.mean(), 0.025))

ax2 = sns.distplot(standard.final, axlabel='final score', color='r', hist_kws=dict(alpha=0.5), label='Standard', bins=15)

plt.axvline(round(standard.final.mean(),2), color='r')

ax1.annotate(s=round(standard.final.mean(),3), xy=(standard.final.mean(), 0.03))

plt.legend()
final_mean = data.final.mean()

free_mean = free.final.mean()

standard_mean = standard.final.mean()

lunch_diff = free_mean - standard_mean



free_shifted = free.final - free_mean + final_mean

standard_shifted = standard.final - standard_mean + final_mean



lunch_diff_shifted = []

for i in range(3000):

    free_samp = free_shifted.sample(frac=1, replace=True)

    standard_samp = standard_shifted.sample(frac=1, replace=True)

    lunch_diff_shifted.append(free_samp.mean() - standard_samp.mean())

    

p_val = np.sum(np.array(lunch_diff_shifted) <= lunch_diff)/len(lunch_diff_shifted)

print('Lunch final score different', round(lunch_diff,2))

print('Lunch P-Value : ', p_val)
completed = data[data['t_prep'] == 'completed']

none = data[data['t_prep'] == 'none']



ax1 = sns.distplot(completed.final, axlabel='final score', color='b', hist_kws=dict(alpha=0.5), label='Completed', bins=15)

plt.axvline(round(completed.final.mean(),2), color='b')

ax1.annotate(s=round(completed.final.mean(),3), xy=(completed.final.mean(), 0.03))

ax2 = sns.distplot(none.final, axlabel='final score', color='r', hist_kws=dict(alpha=0.5), label='None', bins=15)

plt.axvline(round(none.final.mean(),2), color='r')

ax1.annotate(s=round(none.final.mean(),3), xy=(none.final.mean(), 0.025))

plt.legend()
completed_mean = completed.final.mean()

none_mean = none.final.mean()

t_prep_diff = completed_mean - none_mean



completed_shifted = completed.final - completed_mean + final_mean

none_shifted = none.final - none_mean + final_mean



t_prep_diff_shifted = []

for i in range(3000):

    completed_samp = completed_shifted.sample(frac=1, replace=True)

    none_samp = none_shifted.sample(frac=1, replace=True)

    t_prep_diff_shifted.append(completed_samp.mean() - none_samp.mean())

    

p_val = np.sum(np.array(t_prep_diff_shifted) >= t_prep_diff)/len(t_prep_diff_shifted)

conf_int = np.percentile(t_prep_diff_shifted, [2.5, 97.5])

print('t_prep final score different', round(t_prep_diff,2))

print('t_prep P-Value : ', p_val)
high_school = data.loc[(data.p_edu == 'high school') | (data.p_edu == 'some high school')]

college = data.loc[(data.p_edu != 'high school') & (data.p_edu != 'some high school')]



ax1 = sns.distplot(college.final, axlabel='final score', color='b', hist_kws=dict(alpha=0.5), label='college', bins=15)

plt.axvline(round(college.final.mean(),2), color='b')

ax1.annotate(s=round(college.final.mean(),3), xy=(college.final.mean(), 0.03))

ax2 = sns.distplot(high_school.final, axlabel='final score', color='r', hist_kws=dict(alpha=0.5), label='high school', bins=15)

plt.axvline(round(high_school.final.mean(),2), color='r')

ax2.annotate(s=round(high_school.final.mean(),3), xy=(high_school.final.mean(), 0.025))

plt.legend()
high_school_mean = high_school.final.mean()

college_mean = college.final.mean()

p_edu_diff = high_school_mean - college_mean



high_school_shifted = high_school.final - high_school_mean + final_mean

college_shifted = college.final - college_mean + final_mean



p_edu_diff_shifted = []

for i in range(3000):

    high_school_samp = high_school_shifted.sample(frac=1, replace=True)

    college_samp = college.sample(frac=1, replace=True)

    p_edu_diff_shifted.append(high_school_samp.mean() - college_samp.mean())

    

p_val = np.sum(np.array(p_edu_diff_shifted) <= p_edu_diff)/len(p_edu_diff_shifted)

conf_int = np.percentile(p_edu_diff_shifted, [2.5, 97.5])

print('p_edu final score different : ', round(p_edu_diff,2))

print('p_edu P-Value : ', round(p_val,4))