import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 8)

plt.rcParams['figure.titlesize'] = 16

plt.rcParams['figure.titleweight'] = 'bold'

plt.rcParams['axes.titlesize'] = 16

plt.rcParams['axes.titleweight'] = 'bold'

plt.rcParams['axes.labelsize'] = 13

plt.rcParams['axes.labelweight'] = 'bold'

plt.rcParams['legend.fontsize'] = 12

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12

sns.set_style('whitegrid')
df = pd.read_csv('../input/2016-FCC-New-Coders-Survey-Data.csv', low_memory=False)

df.head()
df['Genders'] = df.Gender.copy()

df.loc[df.Genders.isin(['genderqueer', 'trans', 'agender']) | df.Genders.isnull(), 

       'Genders'] = 'other/NA'
ax = df.Genders.value_counts(ascending=True).plot(kind='barh', width=0.7, figsize=(10,5))

_ = (ax.set_title('Gender Distribution'),

     ax.set_xlabel('Number of Coders'),

     ax.set_ylabel('Gender'))
df['AgeGroups'] = pd.cut(df.Age, [0, 21, 25, 30, 40, 99], 

                         labels=['10-20', '21-24', '25-29', '30-39', '40-86'],

                         right=False)
ax = df.AgeGroups.value_counts(sort=False).plot(kind='bar', rot=0, figsize=(10,5))



_ = (ax.set_title('Age Distribution'),

     ax.set_xlabel('Age Groups'),

     ax.set_ylabel('Number of Coders'))
df['MonthsProgGroups'] = pd.cut(df.MonthsProgramming, [0, 6, 24, 999],

                                right=False,

                                labels=['< 6', '6-24', '24+'])
ax = df.MonthsProgGroups.value_counts(sort=False).plot(kind='bar', rot=0, figsize=(10,5))



_ = (ax.set_title('Programming Experience'),

     ax.set_xlabel('Months of Programming'),

     ax.set_ylabel('Number of Coders'))
df['MoneyBins'] = pd.cut(df.MoneyForLearning, [-1, 0, 100, 1000, 200000], 

                         labels=['0', '1-100', '100-1000', '1000+'])
ax = df.MoneyBins.value_counts(sort=False).plot(kind='bar', rot=0, figsize=(10,5))



_ = (ax.set_title('Money Spent on Learning'),

     ax.set_xlabel('Money in US$'),

     ax.set_ylabel('Number of Coders'))
ax = sns.boxplot(data=df, x='AgeGroups', y='Income', hue='IsSoftwareDev')



_ = (ax.set_title('Current Income vs Age Group and Developer Job'),

     ax.set_xlabel('Age Groups'),

     ax.set_ylabel('Current Income in US$'),

     ax.set_ylim(0, 200000),

     ax.set_yticks(np.linspace(0, 200000, 11)),

    )
ax = sns.boxplot(data=df, hue='Genders', y='Income', x='IsUnderEmployed')



_ = (ax.set_title('Current Income vs Under-Employment and Gender'),

     ax.set_xlabel('Is Under Employed?'),

     ax.set_xticklabels(['no', 'yes']),

     ax.set_ylabel('Current Income in US$'),

     ax.set_ylim(0, 160000),

     ax.set_yticks(np.linspace(0, 160000, 9)),

    )
ax = sns.boxplot(data=df, x='AgeGroups', y='Income', hue='IsUnderEmployed')



_ = (ax.set_title('Current Income vs Age Group and Under-Employment'),

     ax.set_xlabel('Age Groups'),

     ax.set_ylabel('Current Income in US$'),

     ax.set_ylim(0, 200000),

     ax.set_yticks(np.linspace(0, 200000, 11)),

    )
fig, ax = plt.subplots(figsize=(7, 7))



sns.boxplot(data=df, x='Income', y='SchoolDegree', ax=ax, orient='h')



_ = (ax.set_title('Current Income vs School Degree'),

     ax.set_xlabel('Current Income in US$'),

     ax.set_ylabel('School Degree'),

     ax.set_xlim(0, 160000),

     ax.set_xticks(np.linspace(0, 160000, 9)),

    )
no_hi = df.Income[df.SchoolDegree=='no high school (secondary school)'].count()



print('Only {} people with no high school.'.format(no_hi))
fig, ax = plt.subplots(figsize=(7, 9))



sns.boxplot(data=df, x='Income', y='EmploymentField', ax=ax, orient='h')



_ = (ax.set_title('Current Income vs Employment Field'),

     ax.set_xlabel('Current Income in US$'),

     ax.set_ylabel('Employment Field'),

     ax.set_xlim(0, 160000),

     ax.set_xticks(np.linspace(0, 160000, 9)),

    )
df.Income.groupby(df.EmploymentField).count()
ax = sns.boxplot(data=df, x='CityPopulation', y='Income', hue='Genders',

                 order=['less than 100,000', 

                        'between 100,000 and 1 million', 

                        'more than 1 million'])



_ = (ax.set_title('Current Income vs City Size and Gender'),

     ax.set_xlabel('City Size'),

     ax.set_ylabel('Current Income in US$'),

     ax.set_ylim(0, 200000),

     ax.set_yticks(np.linspace(0, 200000, 11)),

    )
fig = sns.lmplot(data=df, x='Income', y='ExpectedEarning', hue='Genders',

                 fit_reg=True, ci=None, size=8, 

                 x_jitter=500, y_jitter=500,

                 scatter_kws={'alpha':0.25})



_ = (fig.ax.set_title('Expected Earning vs Current Income and Gender'),

     fig.ax.set_xlabel('Current Income in US$'),

     fig.ax.set_ylabel('Expected Earning in US$'),

     fig.ax.set_xlim(0, 120000),

     fig.ax.set_ylim(0, 120000),

     fig.ax.add_line(plt.Line2D((0, 120000), (0, 120000), 

                                linewidth=2, color='black', alpha=0.2)),

    )
third_group = df[(df.Genders == 'other/NA') & df.Income.notnull() & df.ExpectedEarning.notnull()].shape[0]



print('Only {} people in the other/NA group in the above chart.'.format(third_group))
fig = sns.lmplot(data=df, x='Age', y='ExpectedEarning',

                 fit_reg=False, size=9, x_jitter=1, y_jitter=1000,

                 scatter_kws={'alpha':0.2})



_ = (fig.ax.set_title('Expected Earning vs Age'),

     fig.ax.set_xlabel('Age'),

     fig.ax.set_ylabel('Expected Earning in US$'),

     fig.ax.set_xlim(10, 60.05),

     fig.ax.set_ylim(0, 120000),

    )
ax = sns.boxplot(data=df, x='AgeGroups', y='ExpectedEarning')



_ = (ax.set_title('Expected Earning vs Age Group'),

     ax.set_xlabel('Age Groups'),

     ax.set_ylabel('Expected Earning in US$'),

     ax.set_ylim(0, 160000),

     ax.set_yticks(np.linspace(0, 160000, 9)),

    )
ax = sns.boxplot(data=df, x='AgeGroups', y='ExpectedEarning', hue='Genders')



_ = (ax.set_title('Expected Earning vs Age Group and Gender'),

     ax.set_xlabel('Age Groups'),

     ax.set_ylabel('Expected Earning'),

     ax.set_ylim(0, 200000),

     ax.set_yticks(np.linspace(0, 200000, 11)),

    )
ax = sns.boxplot(data=df, x='AgeGroups', y='ExpectedEarning', hue='IsEthnicMinority')



_ = (ax.set_title('Expected Earning vs Age Group and Minority Status'),

     ax.set_xlabel('Age Groups'),

     ax.set_ylabel('Expected Earning'),

     ax.set_ylim(0, 200000),

     ax.set_yticks(np.linspace(0, 200000, 11)),

    )
fig = sns.lmplot(data=df, x='HoursLearning', y='ExpectedEarning', 

                 fit_reg=False, size=9, 

                 x_jitter=2, y_jitter=1000,

                 scatter_kws={'alpha':0.2})



_ = (fig.ax.set_title('Expected Earning vs Hours Learning'),

     fig.ax.set_xlabel('Hours Learning per Week'),

     fig.ax.set_ylabel('Expected Earning in US$'),

     fig.ax.set_xlim(0, 70),

     fig.ax.set_ylim(0, 140000),

    )
ax = sns.boxplot(data=df, x='MoneyBins', y='ExpectedEarning')



_ = (ax.set_title('Expected Earning vs Money Spent on Learning'),

     ax.set_xlabel('Money Spent on Learning in US$'),

     ax.set_ylabel('Expected Earning in US$'),

     ax.set_ylim(0, 140000)

    )
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))



sns.boxplot(data=df, x='MoneyBins', y='Age', ax=ax1)

sns.boxplot(data=df, x='MoneyBins', y='MonthsProgramming', ax=ax2)



_ = (ax1.set_title('Age vs Money Spent on Learning'),

     ax1.set_xlabel('Money Spent on Learning in US$'),

     ax1.set_ylabel('Age'),

     ax1.set_ylim(10, 60),

     

     ax2.set_title('Months Programming vs Money Spent on Learning'),

     ax2.set_xlabel('Money Spent on Learning in US$'),

     ax2.set_ylabel('Months Programming'),

     ax2.set_ylim(0, 80)

    )
cols = ['ExpectedEarning', 'Age', 'Income', 

        'HoursLearning', 'MonthsProgramming', 'MoneyForLearning']

corr_mat = np.corrcoef(df[cols].dropna(subset=cols).values.T)



ax = sns.heatmap(corr_mat, annot=True, fmt='.2f',

                 xticklabels=cols, yticklabels=cols,

                )



_ = (ax.set_title('Correlation Matrix'))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,14))



sns.boxplot(data=df, x='AgeGroups', y='MonthsProgramming', hue='IsSoftwareDev', ax=ax1)

sns.boxplot(data=df, x='AgeGroups', y='HoursLearning', hue='IsSoftwareDev', ax=ax2)



_ = (ax1.set_title('Months Programming vs Age Group and Developer Job'),

     ax1.set_xlabel('Age Groups'),

     ax1.set_ylabel('Months of Programming'),

     ax1.set_ylim(0, 360),

     ax1.set_yticks(np.linspace(0, 360, 10)),

     

     ax2.set_title('Hours Learning vs Age Group and Developer Job'),

     ax2.set_xlabel('Age Groups'),

     ax2.set_ylabel('Weekly Hours Spent on Learning'),

     ax2.set_ylim(0, 60),

    )
df.MonthsProgramming.dropna().groupby(df.IsSoftwareDev).describe().unstack().round()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,14))



sns.boxplot(data=df, x='AgeGroups', y='MonthsProgramming', hue='Genders', ax=ax1)

sns.boxplot(data=df, x='AgeGroups', y='HoursLearning', hue='Genders', ax=ax2)



_ = (ax1.set_title('Months Programming vs Age Group and Gender'),

     ax1.set_xlabel('Age Groups'),

     ax1.set_ylabel('Months of Programming'),

     ax1.set_ylim(0, 120),

     

     ax2.set_title('Hours Learning vs Age Group and Gender'),

     ax2.set_xlabel('Age Groups'),

     ax2.set_ylabel('Weekly Hours Spent on Learning'),

     ax2.set_ylim(0, 60),

    )
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,12))



sns.boxplot(data=df, x='MonthsProgramming', y='SchoolDegree', ax=ax1, orient='h')

sns.boxplot(data=df, x='HoursLearning', y='SchoolDegree', ax=ax2, orient='h')



_ = (ax1.set_title('Programming Experience vs School Degree'),

     ax1.set_xlabel('Months of Programming'),

     ax1.set_ylabel('School Degree'),

     ax1.set_xlim(0, 120),

     

     ax2.set_title('Weekly Hours Spent on Learning vs School Degree'),

     ax2.set_xlabel('Weekly Hours Spent on Learning'),

     ax2.set_ylabel('School Degree'),

     ax2.set_xlim(0, 60),

    )
ax = sns.countplot(data=df, x='Genders', hue='JobPref')



_ = (ax.set_title('Gender vs Job Preference'),

     ax.set_xlabel('Number of People'),

     ax.set_ylabel('Gender'),

    )  
ax = sns.countplot(data=df, x='Genders', hue='JobRoleInterest')



_ = (ax.set_title('Job Role Interest vs Gender'),

     ax.set_xlabel('Number of People'),

     ax.set_ylabel('Job Role'),

    )
ax = sns.countplot(data=df, x='JobWherePref', hue='AgeGroups')



_ = (ax.set_title('Job Location vs Age'),

     ax.set_xlabel('Job Location'),

     ax.set_ylabel('Number of People'),

    )
ax = sns.countplot(data=df, x='Genders', hue='JobRelocateYesNo')



_ = (ax.set_title('Relocation vs Gender'),

     ax.set_xlabel('Gender'),

     ax.set_ylabel('Number of Coders'),

    )
ax = sns.countplot(data=df, x='AgeGroups', hue='JobRelocateYesNo')



_ = (ax.set_title('Relocation vs Age'),

     ax.set_xlabel('Age Groups'),

     ax.set_ylabel('Number of Coders'),

    )
ax = sns.countplot(data=df, x='AgeGroups', hue='HasChildren')



_ = (ax.set_title('Having Children vs Age'),

     ax.set_xlabel('Age Groups'),

     ax.set_ylabel('Number of Coders'),

    )