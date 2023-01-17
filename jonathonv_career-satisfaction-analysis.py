import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
% matplotlib inline
df = pd.read_csv('../input/survey_results_public.csv')

schema_df = pd.read_csv('../input/survey_results_schema.csv')
# Cherry pick a few columns for now

chosen_columns = ['Professional', 'ProgramHobby', 'University', 'EmploymentStatus',

                  'FormalEducation', 'MajorUndergrad', 'YearsProgram', 'YearsCodedJob',

                  'DeveloperType', 'CareerSatisfaction']



column_definitions = schema_df[schema_df['Column'].isin(chosen_columns)]



for ix, row in column_definitions.iterrows():

    print(row['Column'])

    print(row['Question'], '\n')
data = df[chosen_columns]



# Rename columns to be more 'Pythonic' ;)

data.columns = ['professional', 'program_hobby', 'university' ,'employment_status',

                'formal_education', 'major_undergrad', 'years_program', 'years_coded_job',

                'developer_type', 'career_satisfaction']
# Drop responses with no answers in any of the columns

data = data.dropna()

# Inspect data

data.head(5)
for column in data.columns:

    print(column)

    print(data[column].unique(), '\n')
# Convert code length columns to integers to make it easier to plot

data['years_program_int'] = data['years_program'].map(lambda resp: int(resp.split(" ")[0]) if not resp == 'Less than a year' else 0)

data['years_coded_int'] = data['years_coded_job'].map(lambda resp: int(resp.split(" ")[0]) if not resp == 'Less than a year' else 0)

# Create a simple yes/no column for hobby response

data['hobby'] = data['program_hobby'].map(lambda resp: resp.split(",")[0])

# Count number of developer types respondent's picked

data['type_count'] = data['developer_type'].map(lambda resp: len(resp.split(';')))
# People who have coded as part of their job longer than they have known how to code :)

len(data[data['years_coded_int'] > data['years_program_int']])
# People who have been coding as part of their job for longer must really like their job :)

# Looks like there is a slump before 15 years

sns.regplot(x='years_coded_int', y='career_satisfaction', data=data, x_estimator=pd.np.mean)
# What kind of jobs do these developers have in years 13 & 14

years1314 = data[data['years_coded_int'].isin([13,14])]



# Need to flatten and count the responses

# Copy code from stackoverflow of course! :) 

# https://stackoverflow.com/questions/11264684/flatten-list-of-lists

years1314['developer_type_list'] = years1314['developer_type'].map(lambda resp: resp.split('; '))

nested_lists = list(years1314['developer_type_list'])

flattened = [val for sublist in nested_lists for val in sublist]

# Now to count the types

from collections import Counter

c = Counter()

c.update(flattened)

c
# Which of the types in these troubling years offer the least satisfaction?

for col in c:

    years1314[col] = False

    years1314.ix[years1314['developer_type'].str.contains(col), col] = True

    

type_sat = pd.melt(years1314, id_vars=['career_satisfaction'], value_vars=list(c), var_name='developer_type')

type_sat = type_sat[type_sat['value']]

type_sat.groupby('developer_type')['career_satisfaction'].mean().sort_values()
# Looks like overwhelming majority treat coding as a hobby

data.groupby(['hobby'])['hobby'].count()
# Looks like the more they code as a hobby the more they're satisfied

data.groupby(['program_hobby'])['career_satisfaction'].mean()
# Career satisfaction by developer type, bit messy

data.groupby(['developer_type'])['career_satisfaction'].mean()
# Looks like the more diverse your job is, the more you are satisfied

data.groupby(['type_count'])['career_satisfaction'].agg(['count', 'mean'])
sns.regplot(x='type_count', y='career_satisfaction', data=data, x_estimator=pd.np.mean)
# For those who picked one type, which are more satisfied

(

    data[data['type_count'] == 1]

    .groupby('developer_type')['career_satisfaction']

    .mean()

    .sort_values(ascending=False)

)
# Satisfaction according to if respondent is attending university

(

    data

    .groupby('university')['career_satisfaction']

    .mean()

    .sort_values(ascending=False)

)
# Satisfaction according to respondent's employment status

(

    data

    .groupby('employment_status')['career_satisfaction']

    .mean()

    .sort_values(ascending=False)

)
# Satisfaction according to respondent's highest level of formal education

(

    data

    .groupby('formal_education')['career_satisfaction']

    .mean()

    .sort_values(ascending=False)

)
# Satisfaction according to respondent's major

# Interesting how people who didn't major in Info Tech disciplines ranked highest

(

    data

    .groupby('major_undergrad')['career_satisfaction']

    .mean()

    .sort_values(ascending=False)

)