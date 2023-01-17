import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
survey_raw_data = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')



survey_raw_data_index = survey_raw_data.drop(survey_raw_data.index[0])
survey_raw_data_index = survey_raw_data_index[(survey_raw_data_index.Q5 != 'Student') & (survey_raw_data_index.Q5 != 'Not employed')]
survey_raw_data_index['Group'] = np.where(survey_raw_data_index.iloc[:,44]=='University Courses (resulting in a university degree)', 'University-trained Data Worker', 'Self-trained Data Worker')

survey_raw_data_group = survey_raw_data_index
survey_raw_data_group_a1 = survey_raw_data_group[survey_raw_data_group.iloc[:,1] == '70+']

survey_raw_data_group_a1['age_modified'] = 70



survey_raw_data_group_a2 = survey_raw_data_group[survey_raw_data_group.iloc[:,1] != '70+']

survey_raw_data_group_a2['age_modified_1'] = survey_raw_data_group_a2.iloc[:,1].str.extract('(\d\d)', expand=False)

survey_raw_data_group_a2['age_modified_2'] = survey_raw_data_group_a2.iloc[:,1].str.extract(r'-(\d\d)', expand=False)

survey_raw_data_group_a2['age_modified_1'] = pd.to_numeric(survey_raw_data_group_a2['age_modified_1'])

survey_raw_data_group_a2['age_modified_2'] = pd.to_numeric(survey_raw_data_group_a2['age_modified_2'])

survey_raw_data_group_a2['age_modified'] = (survey_raw_data_group_a2['age_modified_1'] + survey_raw_data_group_a2['age_modified_2']) / 2

survey_raw_data_group_a2 = survey_raw_data_group_a2.drop(['age_modified_1','age_modified_2'], axis = 1)



survey_raw_data_age_frame = [survey_raw_data_group_a1,survey_raw_data_group_a2]

survey_raw_data_age = pd.concat(survey_raw_data_age_frame)
survey_raw_data_age_y1 = survey_raw_data_age[survey_raw_data_age.iloc[:,55] == '< 1 years']

survey_raw_data_age_y1['yr_coding'] = survey_raw_data_age_y1.iloc[:,55].str.extract('(\d)', expand=False)



survey_raw_data_age_y2 = survey_raw_data_age[survey_raw_data_age.iloc[:,55] == '20+ years']

survey_raw_data_age_y2['yr_coding'] = survey_raw_data_age_y2.iloc[:,55].str.extract(r'(\d\d)+', expand=False)



survey_raw_data_age_y3 = survey_raw_data_age[survey_raw_data_age.iloc[:,55] == 'I have never written code']

survey_raw_data_age_y3['yr_coding'] = 0



survey_raw_data_age_y4 = survey_raw_data_age[(survey_raw_data_age.iloc[:,55] != '20+ years') & (survey_raw_data_age.iloc[:,55] != '< 1 years') & (survey_raw_data_age.iloc[:,55] != 'I have never written code')]

survey_raw_data_age_y4['yr_coding_1'] = survey_raw_data_age_y4.iloc[:,55].str.extract(r'(\d+)-', expand=False)

survey_raw_data_age_y4['yr_coding_2'] = survey_raw_data_age_y4.iloc[:,55].str.extract(r'-(\d+)', expand=False)

survey_raw_data_age_y4['yr_coding_1'] = pd.to_numeric(survey_raw_data_age_y4['yr_coding_1'])

survey_raw_data_age_y4['yr_coding_2'] = pd.to_numeric(survey_raw_data_age_y4['yr_coding_2'])

survey_raw_data_age_y4['yr_coding'] = (survey_raw_data_age_y4['yr_coding_1'] + survey_raw_data_age_y4['yr_coding_2']) / 2

survey_raw_data_age_y4 = survey_raw_data_age_y4.drop(['yr_coding_1','yr_coding_2'], axis = 1)

survey_raw_data_age_y4.head()



survey_raw_data_year_frame = [survey_raw_data_age_y1,survey_raw_data_age_y2,survey_raw_data_age_y3,survey_raw_data_age_y4]

survey_raw_data_year = pd.concat(survey_raw_data_year_frame)
survey_raw_data_year_nonull = survey_raw_data_year[survey_raw_data_year.loc[:,'Q15'].notna()]



survey_raw_data_year_nonull['age_modified'] = pd.to_numeric(survey_raw_data_year_nonull['age_modified'])

survey_raw_data_year_nonull['yr_coding'] = pd.to_numeric(survey_raw_data_year_nonull['yr_coding'])

survey_raw_data_year_nonull['age_coding'] = survey_raw_data_year_nonull['age_modified'] - survey_raw_data_year_nonull['yr_coding']



survey_raw_data_year_nonull['age_coding'].value_counts().sort_index()
conditions = [

    (survey_raw_data_year_nonull['age_coding'] <= 17),

    (survey_raw_data_year_nonull['age_coding'] > 17) & (survey_raw_data_year_nonull['age_coding'] <= 25),

    (survey_raw_data_year_nonull['age_coding'] > 25) & (survey_raw_data_year_nonull['age_coding'] <= 35),

    (survey_raw_data_year_nonull['age_coding'] > 35) & (survey_raw_data_year_nonull['age_coding'] <= 50),

    (survey_raw_data_year_nonull['age_coding'] > 50) & (survey_raw_data_year_nonull['age_coding'] <= 70)

    ]

choices = ['Before college', 'Higher and graduate education', 'Early career','Mid career','Late career']



survey_raw_data_year_nonull['age_group_coding'] = np.select(conditions, choices)
df_age_coding = survey_raw_data_year_nonull.loc[:,['Group','age_group_coding']]

df_age_coding.sort_index().head()
self_trained = survey_raw_data_year_nonull[survey_raw_data_year_nonull.loc[:,'Group'] == 'Self-trained Data Worker'].loc[:,'age_group_coding'].value_counts()

university_trained = survey_raw_data_year_nonull[survey_raw_data_year_nonull.loc[:,'Group'] == 'University-trained Data Worker'].loc[:,'age_group_coding'].value_counts()

df = pd.DataFrame({'Self-trained Data Worker': self_trained,

                   'University-trained Data Worker': university_trained})

df = df.rename_axis("fields", axis='columns').rename_axis("age_group", axis = 'rows').reset_index()

df
age_data = df_age_coding.groupby('Group').age_group_coding.value_counts(normalize = True).to_frame('percent').reset_index()



plt.figure(figsize=(21,10))

sns.barplot(x = age_data['age_group_coding'], y = age_data['percent'], hue = age_data['Group'], order = ['Before college','Higher and graduate education','Early career','Mid career','Late career'])

plt.xlabel('Age Start Coding to Analyze Data')

plt.ylabel('Percent in Group')
df_gender = survey_raw_data_year_nonull.loc[:,['Group','Q2']]



gender_data = df_gender.groupby('Group').Q2.value_counts(normalize = True).to_frame('percent').reset_index()



plt.figure(figsize = (21,10))

sns.barplot(x = gender_data['Q2'], y = gender_data['percent'], hue = gender_data['Group'], order = ['Male','Female','Prefer to self-describe','Prefer not to say'])

plt.xlabel('Gender')

plt.ylabel('Percent in Group')
df_edu = survey_raw_data_year_nonull.loc[:,['Group','Q4']]



edu_data = df_edu.groupby('Group').Q4.value_counts(normalize = True).to_frame('percent').reset_index()



plt.figure(figsize = (21,10))

chart2 = sns.barplot(x = edu_data['Q4'], y = edu_data['percent'], hue = edu_data['Group'], order = ['Doctoral degree',"Master’s degree",'Professional degree',"Bachelor’s degree","Some college/university study without earning a bachelor’s degree",'No formal education past high school','I prefer not to answer'])

chart2.set_xticklabels(labels = ['Doctoral degree',"Master’s degree",'Professional degree',"Bachelor’s degree","Some college/university study without earning a bachelor’s degree",'No formal education past high school','I prefer not to answer'], rotation=60)

plt.xlabel('Educational Attainment')

plt.ylabel('Percent in Group')
df_title = survey_raw_data_year_nonull.loc[:,['Group','Q5']]



title_data = df_title.groupby('Group').Q5.value_counts(normalize = True).to_frame('percent').reset_index()



plt.figure(figsize = (21,10))

chart3 = sns.barplot(x = title_data['Q5'], y = title_data['percent'], hue = title_data['Group'])

chart3.set_xticklabels(labels = title_data['Q5'], rotation=60)

plt.xlabel('Professional Title')

plt.ylabel('Percent in Group')
df_salary = survey_raw_data_year_nonull.loc[:,['Group','Q10']]

salary_data = df_salary.groupby('Group').Q10.value_counts(normalize = True).to_frame('percent').reset_index()

plt.figure(figsize = (21,10))

chart4 = sns.barplot(x = salary_data['Q10'], y = salary_data['percent'], hue = salary_data['Group'], order = ['$0-999','1,000-1,999','2,000-2,999','3,000-3,999','4,000-4,999','5,000-7,499','7,500-9,999','10,000-14,999','15,000-19,999','20,000-24,999','25,000-29,999','30,000-39,999','40,000-49,999','50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999','90,000-99,999','100,000-124,999','125,000-149,999','150,000-199,999','200,000-249,999','250,000-299,999','300,000-500,000','> $500,000'])

chart4.set_xticklabels(labels = ['$0-999','1,000-1,999','2,000-2,999','3,000-3,999','4,000-4,999','5,000-7,499','7,500-9,999','10,000-14,999','15,000-19,999','20,000-24,999','25,000-29,999','30,000-39,999','40,000-49,999','50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999','90,000-99,999','100,000-124,999','125,000-149,999','150,000-199,999','200,000-249,999','250,000-299,999','300,000-500,000','> $500,000'], rotation=60)

plt.xlabel('Salary')

plt.ylabel('Percent in Group')
conditions = [

    (df_salary['Q10'] == '$0-999') | (df_salary['Q10'] == '10,000-14,999') | (df_salary['Q10'] == '1,000-1,999') | (df_salary['Q10'] == '15,000-19,999') | (df_salary['Q10'] == '2,000-2,999') | (df_salary['Q10'] == '3,000-3,999') | (df_salary['Q10'] == '4,000-4,999') | (df_salary['Q10'] == '5,000-7,499') | (df_salary['Q10'] == '7,500-9,999'),

    (df_salary['Q10'] == '25,000-29,999') | (df_salary['Q10'] == '20,000-24,999') | (df_salary['Q10'] == '30,000-39,999'),

    (df_salary['Q10'] == '40,000-49,999') | (df_salary['Q10'] == '50,000-59,999'),

    (df_salary['Q10'] == '60,000-69,999') | (df_salary['Q10'] == '70,000-79,999'),

    (df_salary['Q10'] == '80,000-89,999') | (df_salary['Q10'] == '90,000-99,999') | (df_salary['Q10'] == '100,000-124,999') | (df_salary['Q10'] == '125,000-149,999') | (df_salary['Q10'] == '150,000-199,999'),

    (df_salary['Q10'] == '> $500,000') | (df_salary['Q10'] == '200,000-249,999') | (df_salary['Q10'] == '250,000-299,999') | (df_salary['Q10'] == '300,000-500,000')

    ]

choices = ['Low outlier', 'Below regular', 'Bottom Half regular','Top Half regular','Beyond regular','High outlier']



df_salary['salary_range'] = np.select(conditions, choices)

salary_data_2 = df_salary.groupby('Group').salary_range.value_counts(normalize = True).to_frame('percent').reset_index()
plt.figure(figsize = (21,10))

sns.barplot(x = salary_data_2['salary_range'], y = salary_data_2['percent'], hue = salary_data_2['Group'], order = ['Low outlier', 'Below regular', 'Bottom Half regular','Top Half regular','Beyond regular','High outlier'])

plt.xlabel('Salary Range')

plt.ylabel('Percent in Group')
df_course = survey_raw_data_year_nonull.iloc[:,np.r_[35:47,246]]



self = df_course[df_course['Group'] == 'Self-trained Data Worker']

univ = df_course[df_course['Group'] == 'University-trained Data Worker']



n_self_1 = self[self['Q13_Part_1'].notnull()].Q13_Part_1.count()

n_self_2 = self[self['Q13_Part_2'].notnull()].Q13_Part_2.count()

n_self_3 = self[self['Q13_Part_3'].notnull()].Q13_Part_3.count()

n_self_4 = self[self['Q13_Part_4'].notnull()].Q13_Part_4.count()

n_self_5 = self[self['Q13_Part_5'].notnull()].Q13_Part_5.count()

n_self_6 = self[self['Q13_Part_6'].notnull()].Q13_Part_6.count()

n_self_7 = self[self['Q13_Part_7'].notnull()].Q13_Part_7.count()

n_self_8 = self[self['Q13_Part_8'].notnull()].Q13_Part_8.count()

n_self_9 = self[self['Q13_Part_9'].notnull()].Q13_Part_9.count()

n_self_10 = self[self['Q13_Part_10'].notnull()].Q13_Part_10.count()

n_self_11 = self[self['Q13_Part_11'].notnull()].Q13_Part_11.count()

n_self_12 = self[self['Q13_Part_12'].notnull()].Q13_Part_12.count()



n_univ_1 = univ[univ['Q13_Part_1'].notnull()].Q13_Part_1.count()

n_univ_2 = univ[univ['Q13_Part_2'].notnull()].Q13_Part_2.count()

n_univ_3 = univ[univ['Q13_Part_3'].notnull()].Q13_Part_3.count()

n_univ_4 = univ[univ['Q13_Part_4'].notnull()].Q13_Part_4.count()

n_univ_5 = univ[univ['Q13_Part_5'].notnull()].Q13_Part_5.count()

n_univ_6 = univ[univ['Q13_Part_6'].notnull()].Q13_Part_6.count()

n_univ_7 = univ[univ['Q13_Part_7'].notnull()].Q13_Part_7.count()

n_univ_8 = univ[univ['Q13_Part_8'].notnull()].Q13_Part_8.count()

n_univ_9 = univ[univ['Q13_Part_9'].notnull()].Q13_Part_9.count()

n_univ_10 = univ[univ['Q13_Part_10'].notnull()].Q13_Part_10.count()

n_univ_11 = univ[univ['Q13_Part_11'].notnull()].Q13_Part_11.count()

n_univ_12 = univ[univ['Q13_Part_12'].notnull()].Q13_Part_12.count()



course_counts = pd.DataFrame({'Self-trained Data Worker': [n_self_1,

                           n_self_2,

                           n_self_3,

                           n_self_4,

                           n_self_5,

                           n_self_6,

                           n_self_7,

                           n_self_8,

                           n_self_9,

                           n_self_10,

                           n_self_11,

                           n_self_12

                           ],

                           'University-trained Data Worker':[

                           n_univ_1,

                           n_univ_2,

                           n_univ_3,

                           n_univ_4,

                           n_univ_5,

                           n_univ_6,

                           n_univ_7,

                           n_univ_8,

                           n_univ_9,

                           n_univ_10,

                           n_univ_11,

                           n_univ_12

                           ]},

                          index=[

                           'Udacity',

                           'Coursera',

                           'edX',

                           'DataCamp',

                           'DataQuest',

                           'Kaggle Courses (i.e. Kaggle Learn)',

                           'Fast.ai',

                           'Udemy',

                           'LinkedIn Learning',

                           'University Courses (resulting in a university degree)',

                           'None',

                           'Other'

                           ])



course_counts
course_counts['Total'] = course_counts['Self-trained Data Worker'] + course_counts['University-trained Data Worker']

plt.figure(figsize=(21,10))

sns.barplot(x = course_counts.index, y = course_counts.Total, color = "C1").set_xticklabels(labels = course_counts.index, rotation = 60)

sns.barplot(x = course_counts.index, y = course_counts['Self-trained Data Worker'], color = "C0").set_xticklabels(labels = course_counts.index, rotation = 60)



topbar = plt.Rectangle((0,0),1,1,fc="C1", edgecolor = 'none')

bottombar = plt.Rectangle((0,0),1,1,fc='C0',  edgecolor = 'none')

plt.legend([bottombar, topbar], ['Self-trained Data Worker', 'University-trained Data Worker'], loc=1, ncol = 2, prop={'size':16})

plt.xlabel('Course Platforms')
df_media = survey_raw_data_year_nonull.iloc[:,np.r_[22:34,246]]



self = df_media[df_media['Group'] == 'Self-trained Data Worker']

univ = df_media[df_media['Group'] == 'University-trained Data Worker']



n_self_1 = self[self['Q12_Part_1'].notnull()].Q12_Part_1.count()

n_self_2 = self[self['Q12_Part_2'].notnull()].Q12_Part_2.count()

n_self_3 = self[self['Q12_Part_3'].notnull()].Q12_Part_3.count()

n_self_4 = self[self['Q12_Part_4'].notnull()].Q12_Part_4.count()

n_self_5 = self[self['Q12_Part_5'].notnull()].Q12_Part_5.count()

n_self_6 = self[self['Q12_Part_6'].notnull()].Q12_Part_6.count()

n_self_7 = self[self['Q12_Part_7'].notnull()].Q12_Part_7.count()

n_self_8 = self[self['Q12_Part_8'].notnull()].Q12_Part_8.count()

n_self_9 = self[self['Q12_Part_9'].notnull()].Q12_Part_9.count()

n_self_10 = self[self['Q12_Part_10'].notnull()].Q12_Part_10.count()

n_self_11 = self[self['Q12_Part_11'].notnull()].Q12_Part_11.count()

n_self_12 = self[self['Q12_Part_12'].notnull()].Q12_Part_12.count()



n_univ_1 = univ[univ['Q12_Part_1'].notnull()].Q12_Part_1.count()

n_univ_2 = univ[univ['Q12_Part_2'].notnull()].Q12_Part_2.count()

n_univ_3 = univ[univ['Q12_Part_3'].notnull()].Q12_Part_3.count()

n_univ_4 = univ[univ['Q12_Part_4'].notnull()].Q12_Part_4.count()

n_univ_5 = univ[univ['Q12_Part_5'].notnull()].Q12_Part_5.count()

n_univ_6 = univ[univ['Q12_Part_6'].notnull()].Q12_Part_6.count()

n_univ_7 = univ[univ['Q12_Part_7'].notnull()].Q12_Part_7.count()

n_univ_8 = univ[univ['Q12_Part_8'].notnull()].Q12_Part_8.count()

n_univ_9 = univ[univ['Q12_Part_9'].notnull()].Q12_Part_9.count()

n_univ_10 = univ[univ['Q12_Part_10'].notnull()].Q12_Part_10.count()

n_univ_11 = univ[univ['Q12_Part_11'].notnull()].Q12_Part_11.count()

n_univ_12 = univ[univ['Q12_Part_12'].notnull()].Q12_Part_12.count()



media_counts = pd.DataFrame({'Self-trained Data Worker': [n_self_1,

                           n_self_2,

                           n_self_3,

                           n_self_4,

                           n_self_5,

                           n_self_6,

                           n_self_7,

                           n_self_8,

                           n_self_9,

                           n_self_10,

                           n_self_11,

                           n_self_12

                           ],

                           'University-trained Data Worker':[

                           n_univ_1,

                           n_univ_2,

                           n_univ_3,

                           n_univ_4,

                           n_univ_5,

                           n_univ_6,

                           n_univ_7,

                           n_univ_8,

                           n_univ_9,

                           n_univ_10,

                           n_univ_11,

                           n_univ_12

                           ]},

                          index=[

                           'Twitter (data science influencers)',

                           'Hacker News (https://news.ycombinator.com/)',

                           'Reddit (r/machinelearning, r/datascience, etc)',

                           'Kaggle (forums, blog, social media, etc)',

                           'Course Forums (forums.fast.ai, etc)',

                           'YouTube (Cloud AI Adventures, Siraj Raval, etc)',

                           'Podcasts (Chai Time Data Science, Linear Digressions, etc)',

                           'Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)',

                           'Journal Publications (traditional publications, preprint journals, etc)',

                           'Slack Communities (ods.ai, kagglenoobs, etc)',

                           'None',

                           'Other'

                           ])



media_counts
media_counts['Total'] = media_counts['Self-trained Data Worker'] + media_counts['University-trained Data Worker']

plt.figure(figsize=(21,10))

sns.barplot(x = course_counts.index, y = media_counts.Total, color = "C1").set_xticklabels(labels = media_counts.index, rotation = 60)

sns.barplot(x = course_counts.index, y = media_counts['Self-trained Data Worker'], color = "C0").set_xticklabels(labels = media_counts.index, rotation = 60)



topbar = plt.Rectangle((0,0),1,1,fc="C1", edgecolor = 'none')

bottombar = plt.Rectangle((0,0),1,1,fc='C0',  edgecolor = 'none')

plt.legend([bottombar, topbar], ['Self-trained Data Worker', 'University-trained Data Worker'], loc=1, ncol = 2, prop={'size':16})

plt.xlabel('Media Platforms')