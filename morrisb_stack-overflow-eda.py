# To store the data
import pandas as pd

# To do linear algebra
import numpy as np

# To map Country-Names to ISO Country-Codes
import pycountry

# To create interactive maps
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

# To create plots
import matplotlib.pyplot as plt

# To create nicer plots
import seaborn as sns

# To search strings
import re
# Load the survey-data
df = pd.read_csv('../input/survey_results_public.csv', low_memory=False)
print('The survey-data has {} entries with {} features.'.format(df.shape[0], df.shape[1]))

# Load the questions
questions_df = pd.read_csv('../input/survey_results_schema.csv').set_index('Column')
print('The survey has {} questions.'.format(questions_df.shape[0]))


# Create a mapping from country-names to country-codes
country_codes = {country.name:country.alpha_3 for country in pycountry.countries}
country_codes['Democratic Republic of the Congo'] = country_codes.pop('Congo, The Democratic Republic of the')
country_codes['Venezuela, Bolivarian Republic of...'] = country_codes.pop('Venezuela, Bolivarian Republic of')
country_codes['Micronesia, Federated States of...'] = country_codes.pop('Micronesia, Federated States of')
country_codes['The former Yugoslav Republic of Macedonia'] = country_codes.pop('Macedonia, Republic of')
country_codes['United Republic of Tanzania'] = country_codes.pop('Tanzania, United Republic of')
country_codes['Iran, Islamic Republic of...'] = country_codes.pop('Iran, Islamic Republic of')
country_codes['North Korea'] = country_codes.pop("Korea, Democratic People's Republic of")
country_codes['Republic of Moldova'] = country_codes.pop('Moldova, Republic of')
country_codes['Bolivia'] = country_codes.pop('Bolivia, Plurinational State of')
country_codes['Taiwan'] = country_codes.pop('Taiwan, Province of China')
country_codes['South Korea'] = country_codes.pop('Korea, Republic of')
country_codes['Libyan Arab Jamahiriya'] = country_codes.pop('Libya')
country_codes['Hong Kong (S.A.R.)'] = country_codes.pop('Hong Kong')
country_codes['Czech Republic'] = country_codes.pop('Czechia')
country_codes['Cape Verde'] = country_codes.pop('Cabo Verde')

# Handle duplicates for the same country
country_codes['Congo, Republic of the...'] = country_codes['Democratic Republic of the Congo']
country_codes["Democratic People's Republic of Korea"] = country_codes['North Korea']
country_codes['Republic of Korea'] = country_codes['South Korea']

# Create a country-code column
df['Code'] = df.Country.map(country_codes)
# Create the DataFrame for plotting
plot = df.groupby(['Country', 'Code']).Respondent.count().reset_index().rename(columns={'Respondent':'Respondents'})

# Data for the map
data = [dict(type = 'choropleth',
             locations = plot['Code'],
             z = plot['Respondents'],
             text = plot['Country'],
             colorscale = [[0,"rgb(0, 0, 0)"],[0.5,"rgb(255, 0, 0)"],[1,"rgb(240, 240, 240)"]],
             autocolorscale = False,
             reversescale = True,
             marker = dict(line = dict (color = '#000000',
                                        width = 0.5)),
            colorbar = dict(title = 'Respondents'))]

# Layout for the map
layout = dict(title = 'Respondents of the Stack Overflow 2018 survey around the world',
              geo = dict(showframe = False,
                         showcoastlines = False,
                         projection = dict(type = 'Mercator'),
                         countrycolor = '#000000',
                         showcountries = True))

# Create the map
fig = dict(data=data, layout=layout)
iplot(fig)
order_age = ['Under 18 years old', '18 - 24 years old', '25 - 34 years old', '35 - 44 years old', '45 - 54 years old', '55 - 64 years old', '65 years or older']

plt.figure(figsize=(14,4))
plt.grid()
sns.countplot(data=df, x='Age', order=order_age, palette='viridis')
plt.title(questions_df.loc['Age'].QuestionText)
plt.ylabel('Count')
plt.xticks(rotation=60)
plt.show()
fig, axarr = plt.subplots(2,1, figsize=(18,6))

order_yearscoding = ['0-2 years', '3-5 years', '6-8 years', '9-11 years', '12-14 years', '15-17 years', '18-20 years', '21-23 years', '24-26 years', '27-29 years', '30 or more years']

axarr[0].grid()
sns.countplot(data=df, x='YearsCoding', order=order_yearscoding, palette='viridis', ax=axarr[0])
axarr[0].set_title(questions_df.loc['YearsCoding'].QuestionText)
axarr[0].set_ylabel('Count')

order_yearscodingprof = ['0-2 years', '3-5 years', '6-8 years', '9-11 years', '12-14 years', '15-17 years', '18-20 years', '21-23 years', '24-26 years', '27-29 years', '30 or more years']

axarr[1].grid()
sns.countplot(data=df, x='YearsCodingProf', order=order_yearscodingprof, palette='viridis', ax=axarr[1])
axarr[1].set_title(questions_df.loc['YearsCodingProf'].QuestionText)
axarr[1].set_ylabel('Count')

plt.tight_layout()
plt.show()
gender = pd.DataFrame.from_records(df.Gender.dropna().apply(lambda x: x.split(';')).tolist()).stack().to_frame().rename(columns={0:'Gender'}).reset_index(drop=True)

plt.figure(figsize=(14,6))
plt.grid()
sns.countplot(data=gender, y='Gender', order=gender.Gender.value_counts().index, palette='viridis')
plt.title(questions_df.loc['Gender'].QuestionText)
plt.xlabel('Count')
plt.show()
raceethnicity = pd.DataFrame.from_records(df.RaceEthnicity.dropna().apply(lambda x: x.split(';')).tolist()).stack().to_frame().rename(columns={0:'RaceEthnicity'}).reset_index(drop=True)

plt.figure(figsize=(14,6))
plt.grid()
sns.countplot(data=raceethnicity, y='RaceEthnicity', order=raceethnicity.RaceEthnicity.value_counts().index, palette='viridis')
plt.title(questions_df.loc['RaceEthnicity'].QuestionText)
plt.xlabel('Count')
plt.show()
sexualorientation = pd.DataFrame.from_records(df.SexualOrientation.dropna().apply(lambda x: x.split(';')).tolist()).stack().to_frame().rename(columns={0:'SexualOrientation'}).reset_index(drop=True)

plt.figure(figsize=(14,6))
plt.grid()
sns.countplot(data=sexualorientation, y='SexualOrientation', order=sexualorientation.SexualOrientation.value_counts().index, palette='viridis')
plt.title(questions_df.loc['SexualOrientation'].QuestionText)
plt.xlabel('Count')
plt.show()
order_formaleducation = ['I never completed any formal education', 
                         'Primary/elementary school',
                         'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
                         'Some college/university study without earning a degree', 
                         'Associate degree',
                         'Bachelor’s degree (BA, BS, B.Eng., etc.)',
                         'Master’s degree (MA, MS, M.Eng., MBA, etc.)',
                         'Professional degree (JD, MD, etc.)',
                         'Other doctoral degree (Ph.D, Ed.D., etc.)']

plt.figure(figsize=(14,4))
plt.grid()
sns.countplot(data=df, y='FormalEducation', order=order_formaleducation, palette='viridis')
plt.title(questions_df.loc['FormalEducation'].QuestionText)
plt.xlabel('Count')
plt.show()
plt.figure(figsize=(14,4))
plt.grid()
sns.countplot(data=df, y='UndergradMajor', order=df.UndergradMajor.value_counts().index, palette='viridis')
plt.title(questions_df.loc['UndergradMajor'].QuestionText)
plt.xlabel('Count')
plt.show()
order_companysize = ['Fewer than 10 employees','10 to 19 employees','20 to 99 employees','100 to 499 employees','500 to 999 employees','1,000 to 4,999 employees', '5,000 to 9,999 employees', '10,000 or more employees']

plt.figure(figsize=(14,4))
plt.grid()
sns.countplot(data=df, x='CompanySize', order=order_companysize, palette='viridis')
plt.title(questions_df.loc['CompanySize'].QuestionText)
plt.ylabel('Count')
plt.xticks(rotation=60)
plt.show()
fig, axarr = plt.subplots(2,1, figsize=(18,6))

order_jobsatisfaction = ['Extremely satisfied', 'Moderately satisfied', 'Slightly satisfied', 'Neither satisfied nor dissatisfied', 'Slightly dissatisfied', 'Moderately dissatisfied', 'Extremely dissatisfied']

axarr[0].grid()
sns.countplot(data=df, x='JobSatisfaction', order=order_jobsatisfaction, palette='viridis', ax=axarr[0])
axarr[0].set_title(questions_df.loc['JobSatisfaction'].QuestionText)
axarr[0].set_ylabel('Count')


order_careersatisfaction = ['Extremely satisfied', 'Moderately satisfied', 'Slightly satisfied', 'Neither satisfied nor dissatisfied', 'Slightly dissatisfied', 'Moderately dissatisfied', 'Extremely dissatisfied']

axarr[1].grid()
sns.countplot(data=df, x='CareerSatisfaction', order=order_careersatisfaction, palette='viridis', ax=axarr[1])
axarr[1].set_title(questions_df.loc['CareerSatisfaction'].QuestionText)
axarr[1].set_ylabel('Count')
plt.tight_layout()
plt.show()
order_lastnewjob = ["I've never had a job", 'Less than a year ago', 'Between 1 and 2 years ago', 'Between 2 and 4 years ago', 'More than 4 years ago']

plt.figure(figsize=(14,4))
plt.grid()
sns.countplot(data=df, x='LastNewJob', order=order_lastnewjob, palette='viridis')
plt.title(questions_df.loc['LastNewJob'].QuestionText)
plt.ylabel('Count')
plt.xticks(rotation=60)
plt.show()
plt.figure(figsize=(14,4))
plt.grid()
sns.countplot(data=df, y='HopeFiveYears', order=df.HopeFiveYears.value_counts().index, palette='viridis')
plt.title(questions_df.loc['HopeFiveYears'].QuestionText)
plt.xlabel('Count')
plt.show()
plt.figure(figsize=(14,4))
plt.grid()
sns.countplot(data=df, y='JobSearchStatus', order=df.JobSearchStatus.value_counts().index, palette='viridis')
plt.title(questions_df.loc['JobSearchStatus'].QuestionText)
plt.xlabel('Count')
plt.show()
order_stackoverflowvisit = ['Multiple times per day', 'Daily or almost daily', 'A few times per week', 'A few times per month or weekly', 'Less than once per month or monthly', 'I have never visited Stack Overflow (before today)']

plt.figure(figsize=(14,4))
plt.grid()
sns.countplot(data=df, y='StackOverflowVisit', order=order_stackoverflowvisit, palette='viridis')
plt.title(questions_df.loc['StackOverflowVisit'].QuestionText)
plt.xlabel('Count')
plt.show()
fig, axarr = plt.subplots(2,1, figsize=(18,6))

order_hourscomputer = ['Less than 1 hour', '1 - 4 hours', '5 - 8 hours', '9 - 12 hours', 'Over 12 hours']

axarr[0].grid()
sns.countplot(data=df, x='HoursComputer', order=order_hourscomputer, palette='viridis', ax=axarr[0])
axarr[0].set_title(questions_df.loc['HoursComputer'].QuestionText)
axarr[0].set_ylabel('Count')

order_hoursoutside = ['Less than 30 minutes', '30 - 59 minutes', '1 - 2 hours', '3 - 4 hours', 'Over 4 hours']

axarr[1].grid()
sns.countplot(data=df, x='HoursOutside', palette='viridis', ax=axarr[1])
axarr[1].set_title(questions_df.loc['HoursOutside'].QuestionText)
axarr[1].set_ylabel('Count')

plt.tight_layout()
plt.show()
oder_skipmeals = ['Daily or almost every day', '3 - 4 times per week', '1 - 2 times per week', 'Never']

plt.figure(figsize=(14,4))
plt.grid()
sns.countplot(data=df, x='SkipMeals', order=oder_skipmeals, palette='viridis')
plt.title(questions_df.loc['SkipMeals'].QuestionText)
plt.ylabel('Count')
plt.show()
ergonomicdevies = pd.DataFrame.from_records(df.ErgonomicDevices.dropna().apply(lambda x: x.split(';')).tolist()).stack().to_frame().rename(columns={0:'ErgonomicDevices'}).reset_index(drop=True)

plt.figure(figsize=(14,6))
plt.grid()
sns.countplot(data=ergonomicdevies, y='ErgonomicDevices', order=ergonomicdevies.ErgonomicDevices.value_counts().index, palette='viridis')
plt.title(questions_df.loc['ErgonomicDevices'].QuestionText)
plt.xlabel('Count')
plt.show()
order_waketime = ['I work night shifts', 'Before 5:00 AM', 'Between 5:00 - 6:00 AM', 'Between 6:01 - 7:00 AM', 'Between 7:01 - 8:00 AM','Between 8:01 - 9:00 AM', 'Between 9:01 - 10:00 AM', 'Between 10:01 - 11:00 AM', 'Between 11:01 AM - 12:00 PM', 'After 12:01 PM', 'I do not have a set schedule']

plt.figure(figsize=(14,4))
plt.grid()
sns.countplot(data=df, x='WakeTime', order=order_waketime, palette='viridis')
plt.title(questions_df.loc['WakeTime'].QuestionText)
plt.ylabel('Count')
plt.xticks(rotation=60)
plt.show()
# Create salary distribution with cleaning cut
plt.figure(figsize=(15,3))
sns.distplot(df.ConvertedSalary.dropna(), bins=200)
plt.annotate('Cleaning cut', xy=(500000, 0.0000005), xytext=(500000, 0.000005), ha='center', arrowprops=dict(facecolor='black'))
plt.title('Distribution of the ConvertedSalary')
plt.xlabel('Yearly salary in $')
plt.ylabel('Frequency')
plt.show()

# Cleaned ConvertedSalary DataFrame
clean_salary_df = df[df.ConvertedSalary<500000]
# Create the DataFrame for plotting
plot = df.groupby(['Country', 'Code']).agg({'ConvertedSalary':'median', 'Respondent':'count'}).reset_index()
plot = plot[plot.Respondent>=100]

# Data for the map
data = [dict(type = 'choropleth',
             locations = plot['Code'],
             z = plot['ConvertedSalary'].apply(lambda x: round(x, -2)),
             text = plot['Country'],
             colorscale = [[0,"rgb(0, 0, 0)"],[0.5,"rgb(255, 0, 0)"],[1,"rgb(240, 240, 240)"]],
             autocolorscale = False,
             reversescale = True,
             marker = dict(line = dict (color = '#000000',
                                        width = 0.5)),
            colorbar = dict(title = 'ConvertedSalary',
                            tickprefix = '$'))]

# Layout for the map
layout = dict(title = 'Median ConvertedSalary with more than 100 entries around the world in US-Dollar',
              geo = dict(showframe = False,
                         showcoastlines = False,
                         projection = dict(type = 'Mercator'),
                         countrycolor = '#000000',
                         showcountries = True))

# Create the map
fig = dict( data=data, layout=layout )
iplot(fig)
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.Age.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('Age')).dropna()
order = ['Under 18 years old', '18 - 24 years old', '25 - 34 years old', '35 - 44 years old', '45 - 54 years old', '55 - 64 years old', '65 years or older']

# Create the plot
plt.figure(figsize=(14,5))
sns.barplot(data=plot_df, x='ConvertedSalary', y='Age', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by Age')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('Age')
plt.grid()
plt.show()
# Create subplots
fig, axarr = plt.subplots(2, 1, figsize=(14,5))

# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.YearsCoding.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('YearsCoding')).dropna()
order = plot_df.groupby('YearsCoding').ConvertedSalary.mean().sort_values().index

# Create the plot
axarr[0].grid()
sns.barplot(data=plot_df, x='ConvertedSalary', y='YearsCoding', order=order, palette='viridis', ax=axarr[0])
axarr[0].set_title('ConvertedSalary grouped by YearsCoding')
axarr[0].set_xlabel('ConvertedSalary in US-Dollar')
axarr[0].set_ylabel('YearsCoding')
axarr[0].set_xlim(0, 125000)


# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.YearsCodingProf.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('YearsCodingProf')).dropna()
order = plot_df.groupby('YearsCodingProf').ConvertedSalary.mean().sort_values().index

# Create the plot
axarr[1].grid()
sns.barplot(data=plot_df, x='ConvertedSalary', y='YearsCodingProf', order=order, palette='viridis', ax=axarr[1])
axarr[1].set_title('ConvertedSalary grouped by YearsCodingProf')
axarr[1].set_xlabel('ConvertedSalary in US-Dollar')
axarr[1].set_ylabel('YearsCodingProf')
axarr[1].set_xlim(0, 125000)

# Display plot
plt.tight_layout()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.Gender.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('Gender')).dropna()
order = plot_df.groupby('Gender').ConvertedSalary.mean().sort_values().index

# Create the plot
plt.figure(figsize=(14,5))
sns.barplot(data=plot_df, x='ConvertedSalary', y='Gender', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by Gender')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('Gender')
plt.grid()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.RaceEthnicity.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('RaceEthnicity')).dropna()
order = plot_df.groupby('RaceEthnicity').ConvertedSalary.mean().sort_values().index

# Create the plot
plt.figure(figsize=(14,5))
sns.barplot(data=plot_df, x='ConvertedSalary', y='RaceEthnicity', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by RaceEthnicity')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('RaceEthnicity')
plt.grid()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.FormalEducation.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('FormalEducation')).dropna()
order = plot_df.groupby('FormalEducation').ConvertedSalary.mean().sort_values().index

# Create the plot
plt.figure(figsize=(14,5))
sns.barplot(data=plot_df, x='ConvertedSalary', y='FormalEducation', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by FormalEducation')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('FormalEducation')
plt.grid()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.UndergradMajor.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('UndergradMajor')).dropna()
order = plot_df.groupby('UndergradMajor').ConvertedSalary.mean().sort_values().index

# Create the plot
plt.figure(figsize=(14,5))
sns.barplot(data=plot_df, x='ConvertedSalary', y='UndergradMajor', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by UndergradMajor')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('UndergradMajor')
plt.grid()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.CompanySize.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('CompanySize')).dropna()
order = plot_df.groupby('CompanySize').ConvertedSalary.mean().sort_values().index

# Create the plot
plt.figure(figsize=(14,5))
sns.barplot(data=plot_df, x='ConvertedSalary', y='CompanySize', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by CompanySize')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('CompanySize')
plt.grid()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.JobSatisfaction.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('JobSatisfaction')).dropna()
order = ['Extremely satisfied', 'Moderately satisfied', 'Slightly satisfied', 'Neither satisfied nor dissatisfied', 'Slightly dissatisfied', 'Moderately dissatisfied', 'Extremely dissatisfied']

# Create the plot
plt.figure(figsize=(14,5))
sns.barplot(data=plot_df, x='ConvertedSalary', y='JobSatisfaction', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by JobSatisfaction')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('JobSatisfaction')
plt.grid()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.LastNewJob.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('LastNewJob')).dropna()
order = plot_df.groupby('LastNewJob').ConvertedSalary.mean().sort_values().index

# Create the plot
plt.figure(figsize=(14,5))
sns.barplot(data=plot_df, x='ConvertedSalary', y='LastNewJob', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by LastNewJob')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('LastNewJob')
plt.grid()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.StackOverflowVisit.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('StackOverflowVisit')).dropna()
order = ['Multiple times per day', 'Daily or almost daily', 'A few times per week', 'A few times per month or weekly', 'Less than once per month or monthly', 'I have never visited Stack Overflow (before today)']

# Create the plot
plt.figure(figsize=(14,5))
sns.barplot(data=plot_df, x='ConvertedSalary', y='StackOverflowVisit', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by StackOverflowVisit')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('StackOverflowVisit')
plt.grid()
plt.show()
# Create subplots
fig, axarr = plt.subplots(3, 1, figsize=(14,8))

# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.WakeTime.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('WakeTime')).dropna()
order = ['I work night shifts', 'Before 5:00 AM', 'Between 5:00 - 6:00 AM', 'Between 6:01 - 7:00 AM', 'Between 7:01 - 8:00 AM','Between 8:01 - 9:00 AM', 'Between 9:01 - 10:00 AM', 'Between 10:01 - 11:00 AM', 'Between 11:01 AM - 12:00 PM', 'After 12:01 PM', 'I do not have a set schedule']

# Create the plot
axarr[0].grid()
sns.barplot(data=plot_df, x='ConvertedSalary', y='WakeTime', order=order, palette='viridis', ax=axarr[0])
axarr[0].set_title('ConvertedSalary grouped by WakeTime')
axarr[0].set_xlabel('ConvertedSalary in US-Dollar')
axarr[0].set_ylabel('WakeTime')
axarr[0].set_xlim(0, 80000)


# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.HoursComputer.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('HoursComputer')).dropna()
order = ['Less than 1 hour', '1 - 4 hours', '5 - 8 hours', '9 - 12 hours', 'Over 12 hours']

# Create the plot
axarr[1].grid()
sns.barplot(data=plot_df, x='ConvertedSalary', y='HoursComputer', order=order, palette='viridis', ax=axarr[1])
axarr[1].set_title('ConvertedSalary grouped by HoursComputer')
axarr[1].set_xlabel('ConvertedSalary in US-Dollar')
axarr[1].set_ylabel('HoursComputer')
axarr[1].set_xlim(0, 80000)


# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.HoursOutside.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('HoursOutside')).dropna()
order = ['Less than 30 minutes', '30 - 59 minutes', '1 - 2 hours', '3 - 4 hours', 'Over 4 hours']

# Create the plot
axarr[2].grid()
sns.barplot(data=plot_df, x='ConvertedSalary', y='HoursOutside', order=order, palette='viridis', ax=axarr[2])
axarr[2].set_title('ConvertedSalary grouped by HoursOutside')
axarr[2].set_xlabel('ConvertedSalary in US-Dollar')
axarr[2].set_ylabel('HoursOutside')
axarr[2].set_xlim(0, 80000)

# Display plot
plt.tight_layout()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.SkipMeals.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('SkipMeals')).dropna()
order = ['Daily or almost every day', '3 - 4 times per week', '1 - 2 times per week', 'Never']

# Create the plot
plt.figure(figsize=(14,5))
sns.barplot(data=plot_df, x='ConvertedSalary', y='SkipMeals', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by SkipMeals')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('SkipMeals')
plt.grid()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.LanguageWorkedWith.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('LanguageWorkedWith')).dropna()
order = plot_df.groupby('LanguageWorkedWith').ConvertedSalary.mean().sort_values().index

# Create the plot
plt.figure(figsize=(14,8))
sns.barplot(data=plot_df, x='ConvertedSalary', y='LanguageWorkedWith', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by LanguageWorkedWith')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('LanguageWorkedWith')
plt.grid()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.AIDangerous.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('AIDangerous')).dropna()
order = plot_df.groupby('AIDangerous').ConvertedSalary.mean().sort_values().index

# Create the plot
plt.figure(figsize=(14,8))
sns.barplot(data=plot_df, x='ConvertedSalary', y='AIDangerous', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by AIDangerous')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('AIDangerous')
plt.grid()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.AIInteresting.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('AIInteresting')).dropna()
order = plot_df.groupby('AIInteresting').ConvertedSalary.mean().sort_values().index

# Create the plot
plt.figure(figsize=(14,8))
sns.barplot(data=plot_df, x='ConvertedSalary', y='AIInteresting', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by AIInteresting')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('AIInteresting')
plt.grid()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.AIResponsible.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('AIResponsible')).dropna()
order = plot_df.groupby('AIResponsible').ConvertedSalary.mean().sort_values().index

# Create the plot
plt.figure(figsize=(14,8))
sns.barplot(data=plot_df, x='ConvertedSalary', y='AIResponsible', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by AIResponsible')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('AIResponsible')
plt.grid()
plt.show()
# Compute the data
plot_df = clean_salary_df.ConvertedSalary.to_frame().join(pd.DataFrame.from_records(df.AIFuture.apply(lambda x: x.split(';') if type(x)==str else []).tolist()).stack().reset_index(level=1, drop=True).rename('AIFuture')).dropna()
order = plot_df.groupby('AIFuture').ConvertedSalary.mean().sort_values().index

# Create the plot
plt.figure(figsize=(14,8))
sns.barplot(data=plot_df, x='ConvertedSalary', y='AIFuture', order=order, palette='viridis')
plt.title('ConvertedSalary grouped by AIFuture')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('AIFuture')
plt.grid()
plt.show()
data = []
pattern = re.compile('t\. (.*)$')

for col in df[['AssessJob1', 'AssessJob2', 'AssessJob3', 'AssessJob4', 'AssessJob5', 'AssessJob6', 'AssessJob7', 'AssessJob8', 'AssessJob9', 'AssessJob10']].mean(axis=0).sort_values(ascending=False).index:
    plot_df = df[col].dropna()
    data.append(go.Box(y = plot_df,
                       boxmean = True,
                       name = pattern.search(questions_df.loc[col]['QuestionText'])[1]))
    
layout = go.Layout(title = 'Rank the aspects of the job opportunity in order of importance.',
                   xaxis = dict(showticklabels=False),
                   legend=dict(x=0, y=-10))

fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = []
pattern = re.compile('t\. (.*)$')

for col in df[['AssessBenefits1', 'AssessBenefits2', 'AssessBenefits3', 'AssessBenefits4', 'AssessBenefits5', 'AssessBenefits6', 'AssessBenefits7', 'AssessBenefits8', 'AssessBenefits9', 'AssessBenefits10', 'AssessBenefits11']].mean(axis=0).sort_values(ascending=False).index:
    plot_df = df[col].dropna()
    data.append(go.Box(y = plot_df,
                       boxmean = True,
                       name = pattern.search(questions_df.loc[col]['QuestionText'])[1]))
    
layout = go.Layout(title = "Rank the aspects of a job's benefits package.",
                   xaxis = dict(showticklabels=False),
                   legend=dict(x=0, y=-20))

fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = []
pattern = re.compile('d\. (.*)$')

for col in df[['JobContactPriorities1', 'JobContactPriorities2', 'JobContactPriorities3', 'JobContactPriorities4', 'JobContactPriorities5']].mean(axis=0).sort_values(ascending=False).index:
    plot_df = df[col].dropna()
    data.append(go.Box(y = plot_df,
                       boxmean = True,
                       name = pattern.search(questions_df.loc[col]['QuestionText'])[1]))
    
layout = go.Layout(title = 'Rank your preference in how you are contacted.',
                   xaxis = dict(showticklabels=False),
                   legend=dict(x=0, y=-10))

fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = []
pattern = re.compile('t\. (.*)$')

for col in df[['JobEmailPriorities1', 'JobEmailPriorities2', 'JobEmailPriorities3', 'JobEmailPriorities4', 'JobEmailPriorities5', 'JobEmailPriorities6', 'JobEmailPriorities7']].mean(axis=0).sort_values(ascending=False).index:
    plot_df = df[col].dropna()
    data.append(go.Box(y = plot_df,
                       boxmean = True,
                       name = pattern.search(questions_df.loc[col]['QuestionText'])[1]))
    
layout = go.Layout(title = 'Rank the items by how important it is to include them in the message.',
                   xaxis = dict(showticklabels=False),
                   legend=dict(x=0, y=-10))

fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = []
pattern = re.compile('\? (.*)$')
scale = ['Strongly disagree', 'Disagree', 'Neither Agree nor Disagree', 'Agree', 'Strongly agree']
mapper = {name:value for value, name in enumerate(scale, 1)}

for col in df[['AgreeDisagree1', 'AgreeDisagree2', 'AgreeDisagree3']].dropna().applymap(lambda x: mapper[x]).mean(axis=0).sort_values(ascending=False).index:
    plot_df = df[col].dropna().apply(lambda x: mapper[x])
    data.append(go.Box(y = plot_df,
                       boxmean = True,
                       name = pattern.search(questions_df.loc[col]['QuestionText'])[1]))
    
layout = go.Layout(title = 'To what extent do you agree or disagree with the following statements.',
                   xaxis = dict(showticklabels=False),
                   yaxis = dict(tickvals = list(range(1, len(scale)+1)),
                                tickangle = -60,
                                ticktext = scale),
                   legend=dict(x=0, y=-10))

fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = []
pattern = re.compile(': (.*)$')
scale = ['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']
mapper = {name:value for value, name in enumerate(scale, 1)}

for col in df[['AdsAgreeDisagree1', 'AdsAgreeDisagree2', 'AdsAgreeDisagree3']].dropna().applymap(lambda x: mapper[x]).mean(axis=0).sort_values(ascending=False).index:
    plot_df = df[col].dropna().apply(lambda x: mapper[x])
    data.append(go.Box(y = plot_df,
                       boxmean = True,
                       name = pattern.search(questions_df.loc[col]['QuestionText'])[1]))
    
layout = go.Layout(title = 'To what extent do you agree or disagree with the following statements.',
                   xaxis = dict(showticklabels=False),
                   yaxis = dict(tickvals = list(range(1, len(scale)+1)),
                                tickangle = -60,
                                ticktext = scale),
                   legend=dict(x=0, y=-10))

fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = []
pattern = re.compile('t\. (.*)$')

for col in df[['AdsPriorities1', 'AdsPriorities2', 'AdsPriorities3', 'AdsPriorities4', 'AdsPriorities5', 'AdsPriorities6', 'AdsPriorities7']].mean(axis=0).sort_values(ascending=False).index:
    plot_df = df[col].dropna()
    data.append(go.Box(y = plot_df,
                       boxmean = True,
                       name = pattern.search(questions_df.loc[col]['QuestionText'])[1]))
    
layout = go.Layout(title = 'To what extent do you agree or disagree with the following statements.',
                   xaxis = dict(showticklabels=False),
                   legend=dict(x=0, y=-10))

fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = []
pattern = re.compile('d\. (.*)$')
scale = ['Not at all interested', 'Somewhat interested', 'A little bit interested', 'Very interested', 'Extremely interested']
mapper = {name:value for value, name in enumerate(scale, 1)}

for col in df[['HypotheticalTools1', 'HypotheticalTools2', 'HypotheticalTools3', 'HypotheticalTools4', 'HypotheticalTools5']].dropna().applymap(lambda x: mapper[x]).mean(axis=0).sort_values(ascending=False).index:
    plot_df = df[col].dropna().apply(lambda x: mapper[x])
    data.append(go.Box(y = plot_df,
                       boxmean = True,
                       name = pattern.search(questions_df.loc[col]['QuestionText'])[1]))
    
layout = go.Layout(title = 'Rate your interest in participating in the hypothetical tools on Stack Overflow.',
                   xaxis = dict(showticklabels=False),
                   yaxis = dict(tickvals = list(range(1, len(scale)+1)),
                                tickangle = -60,
                                ticktext = scale),
                   legend=dict(x=0, y=-10))

fig = go.Figure(data=data, layout=layout)
iplot(fig)
a = pd.DataFrame.from_records(df['DevType'].dropna().apply(lambda x: x.split(';') if type(x)==str else x).tolist()).stack().rename('DevType').to_frame()
a.index = a.index.get_level_values(0)
b = pd.DataFrame.from_records(df['LanguageWorkedWith'].dropna().apply(lambda x: x.split(';') if type(x)==str else x).tolist()).stack().rename('LanguageWorkedWith')
b.index = b.index.get_level_values(0)
dev_lang = a.join(b).assign(N=1).pivot_table(index='DevType', columns='LanguageWorkedWith', values='N', aggfunc='count')
mat = dev_lang.div(dev_lang.sum().sum()).values

plt.figure(figsize=(14,7))
plt.title('Developers with their languages')
plt.xticks(range(dev_lang.shape[1]), dev_lang.columns, rotation='vertical')
plt.yticks(range(dev_lang.shape[0]), dev_lang.index)
plt.imshow(mat)
plt.show()
a = pd.DataFrame.from_records(df['FormalEducation'].dropna().apply(lambda x: x.split(';') if type(x)==str else x).tolist()).stack().rename('FormalEducation').to_frame()
a.index = a.index.get_level_values(0)
b = pd.DataFrame.from_records(df['LanguageWorkedWith'].dropna().apply(lambda x: x.split(';') if type(x)==str else x).tolist()).stack().rename('LanguageWorkedWith')
b.index = b.index.get_level_values(0)
dev_lang = a.join(b).assign(N=1).pivot_table(index='FormalEducation', columns='LanguageWorkedWith', values='N', aggfunc='count')
mat = dev_lang.div(dev_lang.sum().sum()).values

plt.figure(figsize=(14,7))
plt.title('Developers with their languages')
plt.xticks(range(dev_lang.shape[1]), dev_lang.columns, rotation='vertical')
plt.yticks(range(dev_lang.shape[0]), dev_lang.index)
plt.imshow(mat)
plt.show()
a = pd.DataFrame.from_records(df['RaceEthnicity'].dropna().apply(lambda x: x.split(';') if type(x)==str else x).tolist()).stack().rename('RaceEthnicity').to_frame()
a.index = a.index.get_level_values(0)
b = pd.DataFrame.from_records(df['LanguageWorkedWith'].dropna().apply(lambda x: x.split(';') if type(x)==str else x).tolist()).stack().rename('LanguageWorkedWith')
b.index = b.index.get_level_values(0)
dev_lang = a.join(b).assign(N=1).pivot_table(index='RaceEthnicity', columns='LanguageWorkedWith', values='N', aggfunc='count')
mat = dev_lang.div(dev_lang.sum().sum()).values

plt.figure(figsize=(14,7))
plt.title('Developers with their languages')
plt.xticks(range(dev_lang.shape[1]), dev_lang.columns, rotation='vertical')
plt.yticks(range(dev_lang.shape[0]), dev_lang.index)
plt.imshow(mat)
plt.show()
df.head()
import itertools
edges = {}
for row in df['LanguageWorkedWith'].dropna().apply(lambda x: x.split(';')):
    if len(row)>1:
        for edge in itertools.combinations(row, 2):
            first, second = sorted(edge)
            key = '_'.join([first, second])
            if key in edges.keys():
                edges[key] += 1
            else:
                edges[key] = 1
edges = [[key.split('_')[0], key.split('_')[1], value] for key, value in edges.items()]
import networkx as nx

G = nx.Graph()
G.add_weighted_edges_from(edges)
plt.figure(figsize=(15,5))
pos = nx.spring_layout(G, k=30)
nx.draw_networkx_nodes(G,pos,node_size=700)
nx.draw_networkx_edges(G,pos,width=2, edge_color='grey')
nx.draw_networkx_labels(G,pos,font_size=15,font_family='sans-serif')
plt.axis('off')
plt.show()


sns.pointplot(x="Hobby", y="ConvertedSalary", hue="Gender", data=df[(df.Gender.isin(['Male', 'Female'])) & (df.ConvertedSalary<500000)])
plt.show()
order = ['I never completed any formal education','Primary/elementary school','Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)','Professional degree (JD, MD, etc.)','Some college/university study without earning a degree', 'Associate degree','Bachelor’s degree (BA, BS, B.Eng., etc.)','Master’s degree (MA, MS, M.Eng., MBA, etc.)','Other doctoral degree (Ph.D, Ed.D., etc.)']

plt.figure(figsize=(14,5))
sns.pointplot(y="FormalEducation", x="ConvertedSalary", hue="Gender", data=df[(df.Gender.isin(['Male', 'Female'])) & (df.ConvertedSalary<500000)], order=order)
plt.show()

cols = ['Respondent','Hobby','OpenSource','Country','Student','Employment','FormalEducation','UndergradMajor','CompanySize','DevType','YearsCoding','YearsCodingProf','JobSatisfaction','CareerSatisfaction','HopeFiveYears','JobSearchStatus','LastNewJob','UpdateCV','Currency','Salary','SalaryType','CurrencySymbol','CommunicationTools', 'TimeFullyProductive','EducationTypes','SelfTaughtTypes','TimeAfterBootcamp','HackathonReasons','LanguageWorkedWith','LanguageDesireNextYear','DatabaseWorkedWith','DatabaseDesireNextYear','PlatformWorkedWith','PlatformDesireNextYear','FrameworkWorkedWith','FrameworkDesireNextYear','IDE','OperatingSystem','NumberMonitors','Methodology','VersionControl','CheckInCode','AdBlocker','AdBlockerDisable','AdBlockerReasons','AdsActions','AIDangerous','AIInteresting','AIResponsible','AIFuture','EthicsChoice','EthicsReport','EthicsResponsible','EthicalImplications','StackOverflowRecommend','StackOverflowVisit','StackOverflowHasAccount','StackOverflowParticipate','StackOverflowJobs','StackOverflowDevStory','StackOverflowJobsRecommend','StackOverflowConsiderMember','WakeTime','HoursComputer','HoursOutside','SkipMeals','ErgonomicDevices','Exercise', 'ConvertedSalary','Gender','SexualOrientation','EducationParents','RaceEthnicity','Age','Dependents','MilitaryUS','SurveyTooLong','SurveyEasy','Code']
from xgboost import XGBRegressor

tmp = pd.get_dummies(df[df.ConvertedSalary<500000][cols].dropna())

y = tmp.ConvertedSalary
X = tmp.drop(['ConvertedSalary', 'Respondent'], axis=1)

xgb = XGBRegressor(n_estimators=100, n_jobs=32)
xgb.fit(X, y)
n = 30

feature_importance = xgb.feature_importances_
imp, fea = zip(*sorted(zip(feature_importance, X.columns), reverse=True))
imp_2, fea_2 = imp[:n], fea[:n]

colors = []
for i in range(len(fea_2)):
    n = np.corrcoef(tmp[fea_2[i]], y)[0,1]
    if n>=0:
        colors.append('g')
    else:
        colors.append('r')

plt.figure(figsize=(16, 8))
#plt.xscale('log', nonposy='clip')
plt.barh(range(len(imp_2)), imp_2, align='center', color=colors)
plt.yticks(range(len(imp_2)), [elem[:50] for elem in fea_2])
plt.show()