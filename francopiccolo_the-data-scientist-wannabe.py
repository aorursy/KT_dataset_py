import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import warnings 

warnings.simplefilter('ignore')
# data engineers vs data scientists

mc_responses = pd.read_csv('../input/multipleChoiceResponses.csv', encoding = "ISO-8859-1")

mc_responses_subset = mc_responses[(mc_responses['CurrentJobTitleSelect'].isin(['DBA/Database Engineer', 'Data Scientist','Software Developer/Software Engineer','Data Analyst','Business Analyst','Programmer','Computer Scientist','Statistician']))

                                    & (mc_responses['EmploymentStatus'] == 'Employed full-time')]
jobfunction = mc_responses_subset[['CurrentJobTitleSelect','JobFunctionSelect']].groupby(['CurrentJobTitleSelect','JobFunctionSelect'])['CurrentJobTitleSelect'].count()

jobfunction_ds = jobfunction.loc['Data Scientist'].sort_values(ascending = False)[:5]

jobfunction_de = jobfunction.loc['DBA/Database Engineer'].sort_values(ascending = False)[:5]



fig, axs = plt.subplots(1,2, figsize=(25,10))

axs[0].barh(range(len(jobfunction_de)),jobfunction_de.values,tick_label = jobfunction_de.index)

axs[0].invert_yaxis()

axs[0].set_title('Data Engineer Function')

axs[1].barh(range(len(jobfunction_ds)),jobfunction_ds.values,tick_label = jobfunction_ds.index)

axs[1].invert_yaxis()

axs[1].set_title('Data Scientist Function')

plt.show()
worktools = pd.DataFrame(mc_responses_subset['WorkToolsSelect'].str.split(',', expand = True).stack()\

                          .reset_index(level = 1, drop = True)).rename(columns = {0:'Tools'})

worktools = pd.DataFrame(mc_responses['CurrentJobTitleSelect']).join(worktools, how = 'inner')

worktools = worktools.groupby(['CurrentJobTitleSelect', 'Tools'])['CurrentJobTitleSelect'].count()

worktools_ds = worktools.loc['Data Scientist'].sort_values(ascending = False)[:5]

worktools_de = worktools.loc['DBA/Database Engineer'].sort_values(ascending = False)[:5]



fig, axs = plt.subplots(1,2, figsize=(25,10))

axs[1].barh(range(len(worktools_ds)),worktools_ds.values,tick_label = worktools_ds.index)

axs[1].invert_yaxis()

axs[1].set_title('Data Scientist tools')

axs[0].barh(range(len(worktools_de)),worktools_de.values,tick_label = worktools_de.index)

axs[0].invert_yaxis()

axs[0].set_title('Data Engineer tools')

plt.show()
time_spent = mc_responses_subset[['CurrentJobTitleSelect','TimeGatheringData','TimeModelBuilding', 'TimeProduction', 'TimeVisualizing', 'TimeFindingInsights', 'TimeOtherSelect']]

time_spent = time_spent.groupby(['CurrentJobTitleSelect']).mean()

timespent_de = time_spent.loc['DBA/Database Engineer']

timespent_ds = time_spent.loc['Data Scientist']



plt.figure(figsize=(12, 5))

plt.bar(range(len(timespent_ds)), timespent_ds, tick_label = timespent_ds.index, alpha = 0.5, label = 'Data Scientist')

plt.bar(range(len(timespent_de)), timespent_de, tick_label = timespent_de.index, alpha = 0.5, label = 'Data Engineer')

plt.title('Percentage of time spent on each stage of the data-pipeline')

plt.legend()

plt.show()
challenges = pd.DataFrame(mc_responses_subset['WorkChallengesSelect'].str.split(',', expand = True).stack()\

                          .reset_index(level = 1, drop = True)).rename(columns = {0:'Challenge'})

challenges = pd.DataFrame(mc_responses['CurrentJobTitleSelect']).join(challenges, how = 'inner')

challenges = challenges.groupby(['CurrentJobTitleSelect', 'Challenge'])['CurrentJobTitleSelect'].count()

challenges_ds = challenges.loc['Data Scientist'].sort_values(ascending = False)[:5]

challenges_de = challenges.loc['DBA/Database Engineer'].sort_values(ascending = False)[:5]



fig, axs = plt.subplots(1,2, figsize=(25,10))

axs[1].barh(range(len(challenges_ds)),challenges_ds.values,tick_label = challenges_ds.index)

axs[1].invert_yaxis()

axs[1].set_title('Data Scientists Challenges')

axs[0].barh(range(len(challenges_de)),challenges_de.values,tick_label = challenges_de.index)

axs[0].invert_yaxis()

axs[0].set_title('Data Engineers Challenges')

plt.show()
mc_responses_subset['JobSatisfaction'][mc_responses_subset['JobSatisfaction'] == '1 - Highly Dissatisfied'] = 1

mc_responses_subset['JobSatisfaction'][mc_responses_subset['JobSatisfaction'] == '10 - Highly Satisfied'] = 10

mc_responses_subset['JobSatisfaction'][mc_responses_subset['JobSatisfaction'] == 'I prefer not to share'] = np.nan

mc_responses_subset['JobSatisfaction'] = mc_responses_subset['JobSatisfaction'].astype(float)



job_satisfaction = mc_responses_subset.groupby(['CurrentJobTitleSelect'])['JobSatisfaction'].mean().sort_values()



ax = plt.subplot()

ax.barh(range(len(job_satisfaction)), job_satisfaction.values,tick_label = job_satisfaction.index)

ax.set_title('Job Satisfaction by profession')

plt.show()
n = mc_responses_subset['CurrentJobTitleSelect'].value_counts()

career_switch = mc_responses_subset['CurrentJobTitleSelect'][(mc_responses_subset['CareerSwitcher'] == 'Yes')].value_counts()/n*100

career_switch = career_switch.sort_values()



ax = plt.subplot()

ax.barh(range(len(career_switch)), career_switch.values,tick_label = career_switch.index)

ax.set_title('Career switch into data science intention')

plt.show()
n_datascience_belief = mc_responses_subset['CurrentJobTitleSelect'][mc_responses_subset['DataScienceIdentitySelect'].isin(['Yes','No'])].value_counts()

datascience_belief = mc_responses_subset['CurrentJobTitleSelect'][mc_responses_subset['DataScienceIdentitySelect'] == 'Yes'].value_counts()/n_datascience_belief*100

datascience_belief = datascience_belief.sort_values()

ax = plt.subplot()

ax.barh(range(len(datascience_belief)), datascience_belief.values,tick_label = datascience_belief.index)

ax.set_title('How much each profession believes they are data scientists')

plt.show()
n_stats_importance = mc_responses_subset['CurrentJobTitleSelect'][~mc_responses_subset['JobSkillImportanceStats'].isnull()].value_counts()

stats_importance = mc_responses_subset['CurrentJobTitleSelect'][mc_responses_subset['JobSkillImportanceStats'] == 'Necessary'].value_counts()/n_stats_importance*100

stats_importance = stats_importance.sort_values()

ax = plt.subplot()

ax.barh(range(len(stats_importance)), stats_importance.values,tick_label = stats_importance.index)

ax.set_title('How important is Advanced Statistics skill to land a job in data science')

plt.show()
n_sql_importance = mc_responses_subset['CurrentJobTitleSelect'][~mc_responses_subset['JobSkillImportanceSQL'].isnull()].value_counts()

sql_importance = mc_responses_subset['CurrentJobTitleSelect'][mc_responses_subset['JobSkillImportanceSQL'] == 'Necessary'].value_counts()/n_sql_importance*100

sql_importance = sql_importance.sort_values()

ax = plt.subplot()

ax.barh(range(len(sql_importance)), sql_importance.values,tick_label = sql_importance.index)

ax.set_title('How important is SQL skill to land a job in data science')

plt.show()
lang = mc_responses_subset[['CurrentJobTitleSelect','LanguageRecommendationSelect']].groupby(['CurrentJobTitleSelect','LanguageRecommendationSelect'])['CurrentJobTitleSelect'].count()

lang_ds = lang.loc['Data Scientist'].sort_values(ascending = False)[:5]

lang_de = lang.loc['DBA/Database Engineer'].sort_values(ascending = False)[:5]



fig, axs = plt.subplots(1,2, figsize=(25,10))

axs[0].barh(range(len(lang_ds)),lang_ds.values,tick_label = lang_ds.index)

axs[0].invert_yaxis()

axs[0].set_title('Recommended language for data science by Data Scientists')

axs[1].barh(range(len(lang_de)),lang_de.values,tick_label = lang_de.index)

axs[1].invert_yaxis()

axs[1].set_title('Recommended language for data science by Data Engineers')

plt.show()
ml_tool_next = mc_responses_subset[['CurrentJobTitleSelect','MLToolNextYearSelect']].groupby(['CurrentJobTitleSelect','MLToolNextYearSelect'])['CurrentJobTitleSelect'].count()

ml_tool_next_ds = ml_tool_next.loc['Data Scientist'].sort_values(ascending = False)[:5]

ml_tool_next_de = ml_tool_next.loc['DBA/Database Engineer'].sort_values(ascending = False)[:5]



fig, axs = plt.subplots(1,2, figsize=(25,10))

axs[0].barh(range(len(ml_tool_next_ds)),ml_tool_next_ds.values,tick_label = ml_tool_next_ds.index)

axs[0].invert_yaxis()

axs[0].set_title('Tool data scientists are planning to learn next year')

axs[1].barh(range(len(ml_tool_next_de)),ml_tool_next_de.values,tick_label = ml_tool_next_de.index)

axs[1].invert_yaxis()

axs[1].set_title('Tool data engineers are planning to learn next year')

plt.show()