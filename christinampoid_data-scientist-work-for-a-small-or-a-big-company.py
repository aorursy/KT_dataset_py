import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_colwidth', 200)

pd.set_option('display.max_columns', None)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px
schema = pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')

schema.head(2)
responses = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')

responses
m_choice_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

m_choice_df.head()
def custom_format(ax):

    ax.spines["top"].set_visible(False)

    ax.spines["bottom"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_visible(False)

    ax.tick_params(axis='both', labelsize=12, colors='#565656', bottom=False, left=False)

    ax.set_ylabel('');

    ax.set_xlabel('');

    return ax
fig, ax = plt.subplots(figsize=(12, 6))

sns.countplot(y='Q5', 

              data=m_choice_df.iloc[1:], 

              order=m_choice_df['Q5'].value_counts().index[:-1],

              color='#66C2A5');

ax = custom_format(ax)

sns.set_style("whitegrid")
print('{0:1f}% of participants are Data Scientists.'.format(

    m_choice_df.loc[m_choice_df['Q5']=='Data Scientist'].shape[0]/m_choice_df.shape[0] * 100

))
data_scientists = m_choice_df.loc[m_choice_df['Q5']=='Data Scientist']

data_scientists.head()
fig, ax = plt.subplots(figsize=(8, 4))

sns.countplot(y=data_scientists['Q6'], 

              color='#66C2A5',

              order=['0-49 employees', '50-249 employees', '250-999 employees', '1000-9,999 employees', '> 10,000 employees']);

ax = custom_format(ax)

ax.set_title('Number of Data Scientists respondents per company size');
fig, ax = plt.subplots(figsize=(6, 4))

sns.countplot(y='Q2', 

              data=m_choice_df.iloc[1:], 

              order=m_choice_df['Q2'].value_counts().index[:-1],

              color='#66C2A5');

ax = custom_format(ax)

ax.set_title('Number of Data Scientists respondents per gender');
gender_counts = (data_scientists.groupby(['Q6'])['Q2']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

gender_counts.head()
gender_pivoted_q6 = gender_counts.pivot('Q6', 'Q2', 'percentage')

gender_pivoted_q6 = gender_pivoted_q6.reindex(index=["0-49 employees", 

                                                        "50-249 employees", 

                                                        "250-999 employees", 

                                                        "1000-9,999 employees", 

                                                        "> 10,000 employees"])



gender_pivoted_q6.head()
fig, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(gender_pivoted_q6.T, 

            cmap="Greens",

            annot=True);

ax.set_title('Percentage of Data Scientists per gender on each company size');

ax.set_xlabel('');

ax.set_ylabel('');
fig, ax = plt.subplots(figsize=(10, 4))

sns.countplot(y='Q1', 

              data=m_choice_df.iloc[1:], 

              order=['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70+'],

              color='#66C2A5')

ax = custom_format(ax)
age_counts = (data_scientists.groupby(['Q6'])['Q1']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

age_counts.head()
age_pivoted_q6 = age_counts.pivot('Q6', 'Q1', 'percentage')

age_pivoted_q6 = age_pivoted_q6.reindex(index=["0-49 employees", 

                                                        "50-249 employees", 

                                                        "250-999 employees", 

                                                        "1000-9,999 employees", 

                                                        "> 10,000 employees"])



age_pivoted_q6.head()
fig, ax = plt.subplots(figsize=(9, 9))

sns.heatmap(age_pivoted_q6.T, annot=True, cmap = "Greens");

ax.set_title('Percentage of Data Scientists per age on each company size');

ax.set_xlabel('');

ax.set_ylabel('');
fig, ax = plt.subplots(figsize=(10, 4))

sns.countplot(y='Q4', 

              data=m_choice_df.iloc[1:], 

              order=['No formal education past high school',

                                                      'Some college/university study without earning a bachelor’s degree',

                                                      'Professional degree',

                                                      'Bachelor’s degree',

                                                      'Master’s degree',

                                                      'Doctoral degree',

                                                      'I prefer not to answer'

                                                     ],

              color='#66C2A5')

ax = custom_format(ax)
degree_counts = (data_scientists.groupby(['Q6'])['Q4']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

degree_counts.head(5)
degree_pivoted_q6 = degree_counts.pivot('Q6', 'Q4', 'percentage')

degree_pivoted_q6 = degree_pivoted_q6.reindex(index=["0-49 employees", 

                                                        "50-249 employees", 

                                                        "250-999 employees", 

                                                        "1000-9,999 employees", 

                                                        "> 10,000 employees"],

                                             columns=['No formal education past high school',

                                                      'Some college/university study without earning a bachelor’s degree',

                                                      'Professional degree',

                                                      'Bachelor’s degree',

                                                      'Master’s degree',

                                                      'Doctoral degree',

                                                      'I prefer not to answer'

                                                     ])



degree_pivoted_q6.head()
fig, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(degree_pivoted_q6.T, annot=True, cmap = "Greens");

ax.set_title('Percentage of Data Scientists per education type on each company size');

ax.set_xlabel('');

ax.set_ylabel('');
fig, ax = plt.subplots(figsize=(10, 12))

sns.countplot(y='Q3', 

              data=m_choice_df.iloc[1:], 

              order=m_choice_df['Q3'].value_counts().index[:-1],

              color='#66C2A5')

ax = custom_format(ax)
country_counts = (data_scientists.groupby(['Q6'])['Q3']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

country_counts.head()
country_pivoted_q6 = country_counts.pivot('Q6', 'Q3', 'percentage')

country_pivoted_q6 = country_pivoted_q6.reindex(index=["0-49 employees", 

                                                        "50-249 employees", 

                                                        "250-999 employees", 

                                                        "1000-9,999 employees", 

                                                        "> 10,000 employees"])



country_pivoted_q6.head()
fig, ax = plt.subplots(figsize=(20, 16))

sns.heatmap(country_pivoted_q6.T.sort_values('0-49 employees', ascending=False), 

            cmap="Greens",

            annot=True);

ax.set_title('Percentage of Data Scientists per country on each company size');

ax.set_xlabel('');

ax.set_ylabel('');
salary_counts = (data_scientists.groupby(['Q6'])['Q10']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

salary_counts.head()
salary_pivoted_q6 = salary_counts.pivot('Q6', 'Q10', 'percentage')

salary_pivoted_q6 = salary_pivoted_q6.reindex(index=["0-49 employees", 

                                                        "50-249 employees", 

                                                        "250-999 employees", 

                                                        "1000-9,999 employees", 

                                                        "> 10,000 employees"], 

                                             columns=['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999',

                                                      '4,000-4,999', '5,000-7,499', '7,500-9,999', '10,000-14,999',

                                                      '15,000-19,999', '20,000-24,999', '25,000-29,999', '30,000-39,999',

                                                      '40,000-49,999', '50,000-59,999', '60,000-69,999', '70,000-79,999',

                                                      '80,000-89,999', '90,000-99,999', '100,000-124,999', '125,000-149,999',

                                                      '150,000-199,999', '200,000-249,999', '250,000-299,999', '300,000-500,000',

                                                      '> $500,000'])



salary_pivoted_q6.head()
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(salary_pivoted_q6.T,

            cmap="Greens",

            annot=True);

ax.set_title('Worldwide: Percentage of Data Scientists per salary band on each company size');

ax.set_xlabel('');

ax.set_ylabel('');
salary_counts_india = (data_scientists.loc[data_scientists['Q3']=='India'].groupby(['Q6'])['Q10']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

salary_counts_india.head()
salary_pivoted_q6_india = salary_counts_india.pivot('Q6', 'Q10', 'percentage')

salary_pivoted_q6_india = salary_pivoted_q6_india.reindex(index=["0-49 employees", 

                                                        "50-249 employees", 

                                                        "250-999 employees", 

                                                        "1000-9,999 employees", 

                                                        "> 10,000 employees"], 

                                             columns=['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999',

                                                      '4,000-4,999', '5,000-7,499', '7,500-9,999', '10,000-14,999',

                                                      '15,000-19,999', '20,000-24,999', '25,000-29,999', '30,000-39,999',

                                                      '40,000-49,999', '50,000-59,999', '60,000-69,999', '70,000-79,999',

                                                      '80,000-89,999', '90,000-99,999', '100,000-124,999', '125,000-149,999',

                                                      '150,000-199,999', '200,000-249,999', '250,000-299,999', '300,000-500,000',

                                                      '> $500,000'])



salary_pivoted_q6_india.head()
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(salary_pivoted_q6_india.T,

            cmap="Greens",

            annot=True);

ax.set_title('India: Percentage of Data Scientists per salary band on each company size');

ax.set_xlabel('');

ax.set_ylabel('');
salary_counts_us = (data_scientists.loc[data_scientists['Q3']=='United States of America'].groupby(['Q6'])['Q10']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

salary_counts_us.head()
salary_pivoted_q6_us = salary_counts_us.pivot('Q6', 'Q10', 'percentage')

salary_pivoted_q6_us = salary_pivoted_q6_us.reindex(index=["0-49 employees", 

                                                        "50-249 employees", 

                                                        "250-999 employees", 

                                                        "1000-9,999 employees", 

                                                        "> 10,000 employees"], 

                                             columns=['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999',

                                                      '4,000-4,999', '5,000-7,499', '7,500-9,999', '10,000-14,999',

                                                      '15,000-19,999', '20,000-24,999', '25,000-29,999', '30,000-39,999',

                                                      '40,000-49,999', '50,000-59,999', '60,000-69,999', '70,000-79,999',

                                                      '80,000-89,999', '90,000-99,999', '100,000-124,999', '125,000-149,999',

                                                      '150,000-199,999', '200,000-249,999', '250,000-299,999', '300,000-500,000',

                                                      '> $500,000'])



salary_pivoted_q6_us.head()
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(salary_pivoted_q6_us.T,

            cmap="Greens",

            annot=True);

ax.set_title('USA: Percentage of Data Scientists per salary band on each company size');

ax.set_xlabel('');

ax.set_ylabel('');
salary_counts_uk = (data_scientists.loc[data_scientists['Q3']=='United Kingdom of Great Britain and Northern Ireland'].groupby(['Q6'])['Q10']

                 .value_counts(normalize=True)

                 .rename('percentage')

                 .mul(100)

                 .reset_index()

                 .sort_values('Q6'))

salary_counts_uk.head()
salary_pivoted_q6_uk = salary_counts_uk.pivot('Q6', 'Q10', 'percentage')

salary_pivoted_q6_uk = salary_pivoted_q6_uk.reindex(index=["0-49 employees", 

                                                        "50-249 employees", 

                                                        "250-999 employees", 

                                                        "1000-9,999 employees", 

                                                        "> 10,000 employees"], 

                                             columns=['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999',

                                                      '4,000-4,999', '5,000-7,499', '7,500-9,999', '10,000-14,999',

                                                      '15,000-19,999', '20,000-24,999', '25,000-29,999', '30,000-39,999',

                                                      '40,000-49,999', '50,000-59,999', '60,000-69,999', '70,000-79,999',

                                                      '80,000-89,999', '90,000-99,999', '100,000-124,999', '125,000-149,999',

                                                      '150,000-199,999', '200,000-249,999', '250,000-299,999', '300,000-500,000',

                                                      '> $500,000'])



salary_pivoted_q6_uk.head()
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(salary_pivoted_q6_uk.T,

            cmap="Greens",

            annot=True);

ax.set_title('UK: Percentage of Data Scientists per salary band on each company size');

ax.set_xlabel('');

ax.set_ylabel('');
fig, ax = plt.subplots(figsize=(8,4))

sns.countplot(y='Q15', color='#66C2A5', data=data_scientists, order=['I have never written code', '< 1 years', '1-2 years',

                                                         '3-5 years', '5-10 years', '10-20 years', '20+ years',

                                                        ])

custom_format(ax);
code_counts_uk = (data_scientists.groupby(['Q6'])['Q15']

                 .value_counts(normalize=True)

                 .rename('percentage')

                 .mul(100)

                 .reset_index()

                 .sort_values('Q6'))

code_counts_uk.head()
code_pivoted_q6_uk = code_counts_uk.pivot('Q6', 'Q15', 'percentage')

code_pivoted_q6_uk = code_pivoted_q6_uk.reindex(index=["0-49 employees", 

                                                        "50-249 employees", 

                                                        "250-999 employees", 

                                                        "1000-9,999 employees", 

                                                        "> 10,000 employees"], 

                                                columns=['I have never written code', '< 1 years', '1-2 years',

                                                         '3-5 years', '5-10 years', '10-20 years', '20+ years',

                                                        ]

                                             )



code_pivoted_q6_uk.head()
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(code_pivoted_q6_uk.T,

            cmap="Greens",

            annot=True);

ax.set_title('Percentage of Data Scientists per code experience on each company size');

ax.set_xlabel('');

ax.set_ylabel('');
fig, ax = plt.subplots(figsize=(8, 4))

sns.countplot(y=data_scientists['Q8'], 

              color='#66C2A5')

ax = custom_format(ax)
ml_counts = (data_scientists.groupby(['Q6'])['Q8']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

ml_counts.head()
ml_pivoted_q6 = ml_counts.pivot('Q6', 'Q8', 'percentage')

ml_pivoted_q6 = ml_pivoted_q6.reindex(index=["0-49 employees", 

                                                        "50-249 employees", 

                                                        "250-999 employees", 

                                                        "1000-9,999 employees", 

                                                        "> 10,000 employees"],

                                     columns=['We have well established ML methods (i.e., models in production for more than 2 years)',

                                              'We recently started using ML methods (i.e., models in production for less than 2 years)',

                                              'We are exploring ML methods (and may one day put a model into production)',

                                              'We use ML methods for generating insights (but do not put working models into production)',

                                              'No (we do not use ML methods)',

                                              'I do not know',

                                             ])



ml_pivoted_q6.head()
fig, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(ml_pivoted_q6.T, annot=True, cmap = "Greens");

ax.set_title('Percentage of Data Scientists per ML presence on each company size');

ax.set_xlabel('');

ax.set_ylabel('');
fields = ['Q9_Part_1', 'Q9_Part_2', 'Q9_Part_3', 'Q9_Part_4', 

          'Q9_Part_5', 'Q9_Part_6', 'Q9_Part_7', 'Q9_Part_8'

         ]
for field in fields:

    data_scientists[field+'_new'] = data_scientists[field].apply(

        lambda x: 1 if type(x) == str else 0

    )

    data_scientists[field+'_new'] = data_scientists[field+'_new'].fillna(0);
q9_part1_counts = (data_scientists.groupby(['Q6'])['Q9_Part_1_new']

                     .value_counts(normalize=True)

                     .rename('Analyze and understand data to influence product or business decisions')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q9_part1_counts = q9_part1_counts.loc[q9_part1_counts['Q9_Part_1_new']==1].drop(columns=['Q9_Part_1_new'])
q9_part2_counts = (data_scientists.groupby(['Q6'])['Q9_Part_2_new']

                     .value_counts(normalize=True)

                     .rename('Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q9_part2_counts = q9_part2_counts.loc[q9_part2_counts['Q9_Part_2_new']==1].drop(columns=['Q9_Part_2_new'])

q9_part3_counts = (data_scientists.groupby(['Q6'])['Q9_Part_3_new']

                     .value_counts(normalize=True)

                     .rename('Build prototypes to explore applying machine learning to new areas')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q9_part3_counts = q9_part3_counts.loc[q9_part3_counts['Q9_Part_3_new']==1].drop(columns=['Q9_Part_3_new'])

q9_part4_counts = (data_scientists.groupby(['Q6'])['Q9_Part_4_new']

                     .value_counts(normalize=True)

                     .rename('Build and/or run a machine learning service that operationally improves my product or workflows')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q9_part4_counts = q9_part4_counts.loc[q9_part4_counts['Q9_Part_4_new']==1].drop(columns=['Q9_Part_4_new'])

q9_part5_counts = (data_scientists.groupby(['Q6'])['Q9_Part_5_new']

                     .value_counts(normalize=True)

                     .rename('Experimentation and iteration to improve existing ML models')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q9_part5_counts = q9_part5_counts.loc[q9_part5_counts['Q9_Part_5_new']==1].drop(columns=['Q9_Part_5_new'])

q9_part6_counts = (data_scientists.groupby(['Q6'])['Q9_Part_6_new']

                     .value_counts(normalize=True)

                     .rename('Do research that advances the state of the art of machine learning')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q9_part6_counts = q9_part6_counts.loc[q9_part6_counts['Q9_Part_6_new']==1].drop(columns=['Q9_Part_6_new'])

q9_part7_counts = (data_scientists.groupby(['Q6'])['Q9_Part_7_new']

                     .value_counts(normalize=True)

                     .rename('None of these activities are an important part of my role at work')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q9_part7_counts = q9_part7_counts.loc[q9_part7_counts['Q9_Part_7_new']==1].drop(columns=['Q9_Part_7_new'])

q9_part8_counts = (data_scientists.groupby(['Q6'])['Q9_Part_8_new']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q9_part8_counts = q9_part8_counts.loc[q9_part8_counts['Q9_Part_8_new']==1]

subtables = [q9_part3_counts,

            q9_part4_counts, q9_part5_counts, q9_part6_counts,

            q9_part7_counts]
merged = q9_part1_counts.merge(q9_part2_counts, on='Q6')

for subtable in subtables:

    merged = merged.merge(subtable, on='Q6')

merged.head()
merged = merged.set_index('Q6').unstack().to_frame()

merged.head()
merged = merged.reset_index()

merged = merged.rename(columns={'level_0': 'Task', 'Q6': 'Company size', 0: 'percentage'})
import warnings

warnings.filterwarnings("ignore")



grid = sns.FacetGrid(merged, col="Company size", hue="Company size",

                     palette="Set2", col_wrap=1, height=3, sharey=True, sharex=True,

                    col_order=['0-49 employees', '50-249 employees', '250-999 employees',

                               '1000-9,999 employees', '> 10,000 employees']);

grid.map(sns.barplot, 'percentage', 'Task');

sns.despine();

plt.subplots_adjust(hspace=0.2, wspace=1.2);
# Approximately how many individuals are responsible for data science workloads at your place of business?

fig, ax = plt.subplots(figsize=(9, 5))

sns.countplot(y=data_scientists['Q7'],

             order=['0', '1-2', '3-4', '5-9', '10-14', '15-19', '20+'], 

             color='#66C2A5');

custom_format(ax);
workload_counts = (data_scientists.groupby(['Q6'])['Q7']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

workload_counts.head()
workload_pivoted_q6 = workload_counts.pivot('Q6', 'Q7', 'percentage')

workload_pivoted_q6 = workload_pivoted_q6.reindex(index=["0-49 employees", 

                                                        "50-249 employees", 

                                                        "250-999 employees", 

                                                        "1000-9,999 employees", 

                                                        "> 10,000 employees"],

                                                 columns=['0', '1-2', '3-4', '5-9', '10-14', '15-19', '20+'])



workload_pivoted_q6.head()
fig, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(workload_pivoted_q6.T, 

            cmap="Greens",

            annot=True);

ax.set_title('Percentage of cases per people responsible for DS workload on each company size')

ax.set_xlabel('');

ax.set_ylabel('');
# create an auxiliary column that shows zero or more people in the DS workload

data_scientists['people_workload'] = data_scientists['Q7'].apply(lambda x: '0' if x=='0' else '1+')

data_scientists['people_workload'].value_counts()
people_workload_counts = (data_scientists.groupby(['people_workload'])['Q8']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('people_workload'))
fig, ax = plt.subplots(figsize=(8, 6));

sns.barplot(y="Q8", x="percentage", hue="people_workload", data=people_workload_counts, palette="Set2");

custom_format(ax);
fields = ['Q18_Part_1', 'Q18_Part_2', 'Q18_Part_3', 'Q18_Part_4', 

          'Q18_Part_5', 'Q18_Part_6', 'Q18_Part_7', 'Q18_Part_8', 'Q18_Part_9', 'Q18_Part_10'

         ]
for field in fields:

    data_scientists[field+'_new'] = data_scientists[field].apply(

        lambda x: 1 if type(x) == str else 0

    )

    data_scientists[field+'_new'] = data_scientists[field+'_new'].fillna(0);
q18_part1_counts = (data_scientists.groupby(['Q6'])['Q18_Part_1_new']

                     .value_counts(normalize=True)

                     .rename('Python')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q18_part1_counts = q18_part1_counts.loc[q18_part1_counts['Q18_Part_1_new']==1].drop(columns=['Q18_Part_1_new'])
q18_part2_counts = (data_scientists.groupby(['Q6'])['Q18_Part_2_new']

                     .value_counts(normalize=True)

                     .rename('R')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q18_part2_counts = q18_part2_counts.loc[q18_part2_counts['Q18_Part_2_new']==1].drop(columns=['Q18_Part_2_new'])
q18_part3_counts = (data_scientists.groupby(['Q6'])['Q18_Part_3_new']

                     .value_counts(normalize=True)

                     .rename('SQL')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q18_part3_counts = q18_part3_counts.loc[q18_part3_counts['Q18_Part_3_new']==1].drop(columns=['Q18_Part_3_new'])
q18_part4_counts = (data_scientists.groupby(['Q6'])['Q18_Part_4_new']

                     .value_counts(normalize=True)

                     .rename('C')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q18_part4_counts = q18_part4_counts.loc[q18_part4_counts['Q18_Part_4_new']==1].drop(columns=['Q18_Part_4_new'])
q18_part5_counts = (data_scientists.groupby(['Q6'])['Q18_Part_5_new']

                     .value_counts(normalize=True)

                     .rename('C++')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q18_part5_counts = q18_part5_counts.loc[q18_part5_counts['Q18_Part_5_new']==1].drop(columns=['Q18_Part_5_new'])
q18_part6_counts = (data_scientists.groupby(['Q6'])['Q18_Part_6_new']

                     .value_counts(normalize=True)

                     .rename('Java')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q18_part6_counts = q18_part6_counts.loc[q18_part6_counts['Q18_Part_6_new']==1].drop(columns=['Q18_Part_6_new'])
q18_part7_counts = (data_scientists.groupby(['Q6'])['Q18_Part_7_new']

                     .value_counts(normalize=True)

                     .rename('Javascript')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q18_part7_counts = q18_part7_counts.loc[q18_part7_counts['Q18_Part_7_new']==1].drop(columns=['Q18_Part_7_new'])
q18_part8_counts = (data_scientists.groupby(['Q6'])['Q18_Part_8_new']

                     .value_counts(normalize=True)

                     .rename('Typescript')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q18_part8_counts = q18_part8_counts.loc[q18_part8_counts['Q18_Part_8_new']==1].drop(columns=['Q18_Part_8_new'])
q18_part9_counts = (data_scientists.groupby(['Q6'])['Q18_Part_9_new']

                     .value_counts(normalize=True)

                     .rename('Bash')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q18_part9_counts = q18_part9_counts.loc[q18_part9_counts['Q18_Part_9_new']==1].drop(columns=['Q18_Part_9_new'])
q18_part10_counts = (data_scientists.groupby(['Q6'])['Q18_Part_10_new']

                     .value_counts(normalize=True)

                     .rename('Matlab')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q18_part10_counts = q18_part10_counts.loc[q18_part10_counts['Q18_Part_10_new']==1].drop(columns=['Q18_Part_10_new'])
subtables = [q18_part3_counts,

            q18_part4_counts, q18_part5_counts, q18_part6_counts,

             q18_part7_counts, q18_part8_counts, q18_part9_counts, q18_part10_counts

            ]
merged = q18_part1_counts.merge(q18_part2_counts, on='Q6')

for subtable in subtables:

    merged = merged.merge(subtable, on='Q6')

merged.head()
merged = merged.set_index('Q6').unstack().to_frame()

merged.head()
merged = merged.reset_index()

merged = merged.rename(columns={'level_0': 'Language', 'Q6': 'Company size', 0: 'percentage'})
grid = sns.FacetGrid(merged, col="Company size", hue="Company size",

                     palette="Set2", col_wrap=3, height=3, sharey=True, sharex=True,

                    col_order=['0-49 employees', '50-249 employees', '250-999 employees',

                               '1000-9,999 employees', '> 10,000 employees']);

grid.map(sns.barplot, 'percentage', 'Language');

sns.despine();

plt.subplots_adjust(hspace=0.2, wspace=1.2);
fields = ['Q28_Part_1', 'Q28_Part_2', 'Q28_Part_3', 'Q28_Part_4', 

          'Q28_Part_5', 'Q28_Part_6', 'Q28_Part_7', 'Q28_Part_8', 'Q28_Part_9', 'Q28_Part_10', 'Q28_Part_11'

         ]
for field in fields:

    data_scientists[field+'_new'] = data_scientists[field].apply(

        lambda x: 1 if type(x) == str else 0

    )

    data_scientists[field+'_new'] = data_scientists[field+'_new'].fillna(0);
q28_part1_counts = (data_scientists.groupby(['Q6'])['Q28_Part_1_new']

                     .value_counts(normalize=True)

                     .rename('Scikit-learn')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q28_part1_counts = q28_part1_counts.loc[q28_part1_counts['Q28_Part_1_new']==1].drop(columns=['Q28_Part_1_new'])
q28_part2_counts = (data_scientists.groupby(['Q6'])['Q28_Part_2_new']

                     .value_counts(normalize=True)

                     .rename('TensorFlow')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q28_part2_counts = q28_part2_counts.loc[q28_part2_counts['Q28_Part_2_new']==1].drop(columns=['Q28_Part_2_new'])
q28_part3_counts = (data_scientists.groupby(['Q6'])['Q28_Part_3_new']

                     .value_counts(normalize=True)

                     .rename('Keras')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q28_part3_counts = q28_part3_counts.loc[q28_part3_counts['Q28_Part_3_new']==1].drop(columns=['Q28_Part_3_new'])
q28_part4_counts = (data_scientists.groupby(['Q6'])['Q28_Part_4_new']

                     .value_counts(normalize=True)

                     .rename('RandomForest')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q28_part4_counts = q28_part4_counts.loc[q28_part4_counts['Q28_Part_4_new']==1].drop(columns=['Q28_Part_4_new'])
q28_part5_counts = (data_scientists.groupby(['Q6'])['Q28_Part_5_new']

                     .value_counts(normalize=True)

                     .rename('Xgboost')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q28_part5_counts = q28_part5_counts.loc[q28_part5_counts['Q28_Part_5_new']==1].drop(columns=['Q28_Part_5_new'])
q28_part6_counts = (data_scientists.groupby(['Q6'])['Q28_Part_6_new']

                     .value_counts(normalize=True)

                     .rename('Pytorch')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q28_part6_counts = q28_part6_counts.loc[q28_part6_counts['Q28_Part_6_new']==1].drop(columns=['Q28_Part_6_new'])
q28_part7_counts = (data_scientists.groupby(['Q6'])['Q28_Part_7_new']

                     .value_counts(normalize=True)

                     .rename('Caret')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q28_part7_counts = q28_part7_counts.loc[q28_part7_counts['Q28_Part_7_new']==1].drop(columns=['Q28_Part_7_new'])
q28_part8_counts = (data_scientists.groupby(['Q6'])['Q28_Part_8_new']

                     .value_counts(normalize=True)

                     .rename('LightGBM')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q28_part8_counts = q28_part8_counts.loc[q28_part8_counts['Q28_Part_8_new']==1].drop(columns=['Q28_Part_8_new'])
q28_part9_counts = (data_scientists.groupby(['Q6'])['Q28_Part_9_new']

                     .value_counts(normalize=True)

                     .rename('Spark MLib')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q28_part9_counts = q28_part9_counts.loc[q28_part9_counts['Q28_Part_9_new']==1].drop(columns=['Q28_Part_9_new'])
q28_part10_counts = (data_scientists.groupby(['Q6'])['Q28_Part_10_new']

                     .value_counts(normalize=True)

                     .rename('Fast.ai')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q28_part10_counts = q28_part10_counts.loc[q28_part10_counts['Q28_Part_10_new']==1].drop(columns=['Q28_Part_10_new'])
q28_part11_counts = (data_scientists.groupby(['Q6'])['Q28_Part_11_new']

                     .value_counts(normalize=True)

                     .rename('None')

                     .mul(100)

                     .reset_index()

                     .sort_values('Q6'))

q28_part11_counts = q28_part11_counts.loc[q28_part11_counts['Q28_Part_11_new']==1].drop(columns=['Q28_Part_11_new'])
subtables = [q28_part3_counts, q28_part4_counts, q28_part5_counts, q28_part6_counts, q28_part7_counts, q28_part8_counts,

            q28_part9_counts, q28_part10_counts, q28_part11_counts]

merged = q28_part1_counts.merge(q28_part2_counts, on='Q6')

for subtable in subtables:

    merged = merged.merge(subtable, on='Q6')

merged.head()
merged = merged.set_index('Q6').unstack().to_frame()

merged.head()
merged = merged.reset_index()

merged = merged.rename(columns={'level_0': 'ML Framework', 'Q6': 'Company size', 0: 'percentage'})

merged.head()
grid = sns.FacetGrid(merged, col="Company size", hue="Company size",

                     palette="Set2", col_wrap=3, height=3, sharey=True, sharex=True,

                    col_order=['0-49 employees', '50-249 employees', '250-999 employees',

                               '1000-9,999 employees', '> 10,000 employees']);

grid.map(sns.barplot, 'percentage', 'ML Framework');

sns.despine();

plt.subplots_adjust(hspace=0.2, wspace=1.2);