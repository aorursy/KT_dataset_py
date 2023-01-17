# For notebook plotting
%matplotlib inline

# Standard dependencies
import os
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns
# Read in data
DIR = '../input/'
freeform_df = pd.read_csv(DIR + 'freeFormResponses.csv', low_memory=False, header=[0,1])
multi_df = pd.read_csv(DIR + 'multipleChoiceResponses.csv', low_memory=False, header=[0,1])
SurveySchema = pd.read_csv(DIR + 'SurveySchema.csv', low_memory=False, header=[0,1])
# Format Dataframes
freeform_df.columns = freeform_df.columns.map('_'.join)
multi_df.columns = multi_df.columns.map('_'.join)
SurveySchema.columns = SurveySchema.columns.map('_'.join)

# For getting all columns
pd.set_option('display.max_columns', None)
# Get a feel for the shape and magnitude of the data
print('The free form response csv has {} rows and {} columns.\n\
The multiple choice csv has {} rows and {} columns.\n\
The survey schema has {} rows and {} columns.'.format(freeform_df.shape[0], 
                                                      freeform_df.shape[1], 
                                                      multi_df.shape[0], 
                                                      multi_df.shape[1], 
                                                      SurveySchema.shape[0],
                                                      SurveySchema.shape[1]))
# Free form questions
freeform_df.head(2)
# Multiple Choice Questions
multi_df.head(2)
# Survey Schema (Response data)
SurveySchema.head(3)
# Rename columns from multiple choice data
multi_df = multi_df.rename({'Time from Start to Finish (seconds)_Duration (in seconds)' : 'duration', 
                 'Q1_What is your gender? - Selected Choice' : 'gender', 
                 'Q1_OTHER_TEXT_What is your gender? - Prefer to self-describe - Text' : 'gender_other', 
                 'Q2_What is your age (# years)?' : 'age', 
                 'Q3_In which country do you currently reside?' : 'country', 
                 'Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?' : 'education', 
                 'Q5_Which best describes your undergraduate major? - Selected Choice' : 'major', 
                 'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice' : 'current_title', 
                 'Q6_OTHER_TEXT_Select the title most similar to your current role (or most recent title if retired): - Other - Text' : 'current_title_other', 
                 'Q7_In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice' : 'industry', 
                'Q8_How many years of experience do you have in your current role?' : 'experience', 
                'Q9_What is your current yearly compensation (approximate $USD)?' : 'compensation', 
                'Q10_Does your current employer incorporate machine learning methods into their business?' : 'employerML?', 
                'Q12_MULTIPLE_CHOICE_What is the primary tool that you use at work or school to analyze data? (include text response) - Selected Choice' : 'primary_tool', 
                'Q17_What specific programming language do you use most often? - Selected Choice' : 'language_often', 
                'Q18_What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice' : 'language_recommend', 
                'Q20_Of the choices that you selected in the previous question, which ML library have you used the most? - Selected Choice' : 'library_most', 
                'Q23_Approximately what percent of your time at work or school is spent actively coding?' : 'time_coding'}, axis='columns')

# Rename columns from free form data
freeform_df = freeform_df.rename({'Q1_OTHER_TEXT_What is your gender? - Prefer to self-describe - Text' : 'gender', 
                                  'Q21_OTHER_TEXT_What data visualization libraries or tools have you used in the past 5 years? (Select all that apply) - Other - Text' : 'viz'}, axis = 'columns')
# Check columns
print('Renamed columns in multi_df:')
display(multi_df[['duration', 'gender', 'gender_other', 
          'age', 'country', 'education', 'major', 
          'current_title', 'current_title_other', 
        'industry', 'experience', 'compensation', 
        'employerML?', 'primary_tool', 'language_often', 
        'language_recommend', 'library_most', 'time_coding']].head(0))

print('Renamed columns in freeform_df:')
freeform_df[['gender', 'viz']].head(0)
# Gender distribution
multi_df['gender'].value_counts().plot(kind='bar', 
                                       rot=20, 
                                       title='Gender distribution of Kagglers', 
                                       figsize=(14,5))
male_perc = len(multi_df[multi_df['gender'] == 'Male']) / 23860 * 100
female_perc = len(multi_df[multi_df['gender'] == 'Female']) / 23860 * 100
nospecifics_perc = len(multi_df[multi_df['gender'] == 'Prefer not to say']) / 23860 * 100

print('Of the 23860 people that answered the gender question:\n\
{:.2f} % of the participants are male.\n\
{:.2f} % of the participants are female.\n\
{:.2f} % of the participants did not specify their gender'.format(male_perc, 
                                                                  female_perc, 
                                                                  nospecifics_perc))
# Examples of self-specified genders
freeform_df['gender'].dropna().head(15)
# Plot age ranges
age_plot = multi_df['age'].value_counts().reindex(['18-21','22-24',
                                                   '25-29','30-34',
                                                   '35-39','40-44',
                                                   '45-49','50-54',
                                                   '55-59','60-69',
                                                   '70-79','80+'])

_, ax = plt.subplots()
age_plot.plot(ax=ax, 
              color='r')
age_plot.plot(kind='bar', 
              ax=ax, 
              rot=30, 
              figsize=(14,5), 
              title='Age ranges')
# Age range statement
print('{} of respondents are in the age range of 18-21'.format(len(multi_df[multi_df['age'] == '18-21'])))
# Location of Kagglers
multi_df['country'].value_counts().sort_values(ascending=True).plot(kind='barh', 
                                             title='Where are Kagglers residing?', 
                                             figsize=(10,20))
# 23860 people filled out this question
# Get percentages
American_perc = len(multi_df[multi_df['country'] == 'United States of America']) / 23860 * 100
Indian_perc = len(multi_df[multi_df['country'] == 'India']) / 23860 * 100
Chinese_perc = len(multi_df[multi_df['country'] == 'China']) / 23860 * 100

print('Top 3 countries: \n{:.2f} % of all respondents are American,\n\
{:.2f} % are Indian and \n{:.2f} % are Chinese.'.format(American_perc, 
                                              Indian_perc, 
                                              Chinese_perc))
# Amount of people from the Netherlands
dutch_people = len(multi_df[multi_df['country'] == 'Netherlands'])

print("There are {} people from the Netherlands that filled in Kaggle's survey.".format(dutch_people))
# Highest formal education attained or planning to attain in the next 2 years
multi_df['education'].value_counts().plot(kind='barh', 
                                          title='Highest formal education', figsize=(14,5))
# Undergraduate major
multi_df['major'].value_counts().plot(kind='barh', 
                                      title='Undergraduate majors')
multi_df['industry'].value_counts().plot(kind='barh', 
                                         title='Industries where Kagglers work',
                                         figsize=(10,5))
# Job titles plot
multi_df['current_title'].value_counts().plot(kind='barh', 
                                              title='Job titles of Kagglers', 
                                              figsize=(10,5))
# Compensation plot
multi_df['compensation'].value_counts().plot(kind='barh', 
                                             title='Monetary compensations of Kagglers (in USD)', figsize=(12,7))
# Compensation statement
print('20186 Kagglers responded to the compensation question.')

not_disclosed = multi_df[multi_df['compensation'] == '0-10,000']

print('{:.2f} % of respondents did not disclose their approximate yearly compensation.'. format(len(not_disclosed) / 20186 * 100))

low_comp = multi_df[multi_df['compensation'] == '0-10,000']
students_low_comp = low_comp['current_title'].value_counts()['Student']

print('{:.2f} % of respondents in the 0-10,000 range are students.'.format(students_low_comp / len(low_comp) * 100))
# Plot experience chart
exp = multi_df['experience'].value_counts().reindex(['0-1','1-2',
                                              '2-3','3-4',
                                              '4-5','5-10',
                                              '10-15','15-20',
                                              '20-25','25-30','30 +'])

_, ax = plt.subplots()

exp.plot(ax=ax, color='r')
exp.plot(kind='bar', ax=ax, rot=30, title='Experience years of Kagglers', figsize=(12,5))
exp_01 = multi_df[multi_df['experience'] == '0-1']
student_exp_01 = low_comp['current_title'].value_counts()['Student']

print('{} respondents are in the 0-1 year experience range.\n\
This is {:.2f} % of the total respondents.'.format(len(exp_01), len(exp_01) / 21102 * 100))
print('{:.2f} % of respondents in the 0-1 year experience range are students.'.format(student_exp_01 / len(exp_01) * 100))
multi_df['primary_tool'].value_counts().plot(kind='barh', 
                                             title='Primary tools', figsize=(10,5))
# Getting all columns from Q13 and renaming them
Q13 = multi_df.filter(like=("Q13"))
mapping = dict()
for i in range(16):
    old_index = Q13.columns[i]
    string = 'Q13_part_' + str(i+1)
    mapping.update({old_index : string})
Q13 = Q13.rename(columns=mapping)
# Getting value counts
data=pd.melt(Q13).dropna()
data=data[data['value'].apply(lambda x: type(x)==str)]
data=data['value'].value_counts()
# Plot
display(data.plot(kind='barh', 
                  figsize=(14,5), title="IDE's (Integrated Development Environments)"))
###############
print('{:.2f} % of respondents use Jupyter/iPython\n\
{:.2f} % use RStudio and\n\
{:.2f} % use Notepad++'.format(data[0] / 19117 * 100, 
                               data[1] / 19117 * 100, 
                               data[2] / 19117 * 100))
freeform_df["Q13_OTHER_TEXT_Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Other - Text"].dropna().unique()
# Getting all columns from Q14 and renaming them
Q14 = multi_df.filter(like=("Q14"))
mapping = dict()
for i in range(12):
    old_index = Q14.columns[i]
    string = 'Q14_part_' + str(i+1)
    mapping.update({old_index : string})
Q14 = Q14.rename(columns=mapping)
# Getting value counts
data=pd.melt(Q14).dropna()
data=data[data['value'].apply(lambda x: type(x)==str)]
data=data['value'].value_counts()
data.drop('None', inplace=True)
# Plot
display(data.plot(kind='bar', 
                  figsize=(14,5), 
                  rot=25, title='Hosted notebook environments'))
###############
print('Q14: Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply)\n')
print('{:.2f} % of respondents have used Kaggle Kernels\n\
{:.2f} % have used JupyterHub/Binder and\n\
{:.2f} % have used Google Colab'.format(data[0] / 18971 * 100, 
                               data[1] / 18971 * 100, 
                               data[2] / 18971 * 100))
# Getting all columns from Q15 and renaming them
Q15 = multi_df.filter(like=("Q15"))
mapping = dict()
for i in range(8):
    old_index = Q15.columns[i]
    string = 'Q15_part_' + str(i+1)
    mapping.update({old_index : string})
Q15 = Q15.rename(columns=mapping)
# Getting value counts
data=pd.melt(Q15).dropna()
data=data[data['value'].apply(lambda x: type(x)==str)]
data=data['value'].value_counts()
# Plot
display(data.plot(kind='bar', 
                  figsize=(14,5), 
                  rot=25, title='Cloud providers'))
###############
print('Q15: Which of the following cloud computing services have you used at work or school in the last 5 years? (Select all that apply)\n')
print('{:.2f} % of respondents have used AWS (Amazon Web Services)\n\
{:.2f} % have used no cloud provider before and\n\
{:.2f} % have used GCP (Google Cloud Platform)'.format(data[0] / 18971 * 100, 
                               data[1] / 18971 * 100, 
                               data[2] / 18971 * 100))
# Getting all columns from Q19 and renaming them
Q19 = multi_df.filter(like=("Q19"))
mapping = dict()
for i in range(19):
    old_index = Q19.columns[i]
    string = 'Q19_part_' + str(i+1)
    mapping.update({old_index : string})
Q19 = Q19.rename(columns=mapping)
# Getting value counts
data=pd.melt(Q19).dropna()
data=data[data['value'].apply(lambda x: type(x)==str)]
data=data['value'].value_counts()
# Plot
display(data.plot(kind='bar', 
                  figsize=(14,5), 
                  rot=25, title='Machine Learning Frameworks used'))
###############
print('Q19: What machine learning frameworks have you used in the past 5 years? (Select all that apply)\n')
print('{:.2f} % of respondents have used Scikit-Learn\n\
{:.2f} % have used Tensorflow and\n\
{:.2f} % have used Keras'.format(data[0] / 18697 * 100, 
                               data[1] / 18697 * 100, 
                               data[2] / 18697 * 100))
# ML Libraries used the most
multi_df['library_most'].value_counts().plot(kind='bar', 
                                       rot=25, 
                                       title='ML Libraries used the most', 
                                       figsize=(14,5))

print('Q20: Of the choices that you selected in the previous question, which ML library have you used the most?')
# Getting all columns from Q21 and renaming them
Q21 = multi_df.filter(like=("Q21"))
mapping = dict()
for i in range(14):
    old_index = Q21.columns[i]
    string = 'Q21_part_' + str(i+1)
    mapping.update({old_index : string})
Q21 = Q21.rename(columns=mapping)
# Getting value counts
data=pd.melt(Q21).dropna()
data=data[data['value'].apply(lambda x: type(x)==str)]
data=data['value'].value_counts()
# Plot
display(data.plot(kind='bar', 
                  figsize=(14,5), 
                  rot=25, title='Visualization tools used'))
###############
print('Q21: What data visualization libraries or tools have you used in the past 5 years? (Select all that apply)\n')
print('{:.2f} % of respondents have used Matplotlib\n\
{:.2f} % have used Seaborn and\n\
{:.2f} % have used ggplot2'.format(data[0] / 18697 * 100, 
                               data[1] / 18697 * 100, 
                               data[2] / 18697 * 100))
# Examples of self-specified visualization frameworks
freeform_df['viz'].value_counts().dropna().head(15)
# Getting all columns from Q16 and renaming them
Q16 = multi_df.filter(like=("Q16"))
mapping = dict()
for i in range(19):
    old_index = Q16.columns[i]
    string = 'Q16_part_' + str(i+1)
    mapping.update({old_index : string})
Q16 = Q16.rename(columns=mapping)
# Getting value counts
data=pd.melt(Q16).dropna()
data=data[data['value'].apply(lambda x: type(x)==str)]
data=data['value'].value_counts()
# Plot
display(data.plot(kind='barh', 
                  figsize=(14,5), 
                  rot=0, 
                  title='Programming languages used on a regular basis'))
###############
print('Q16: What programming languages do you use on a regular basis? (Select all that apply)\n')
print('{:.2f} % of respondents have used Python\n\
{:.2f} % have used SQL and\n\
{:.2f} % have used R'.format(data[0] / 18971 * 100, 
                               data[1] / 18971 * 100, 
                               data[2] / 18971 * 100))
# Prgramming languages most often used
multi_df['language_often'].value_counts().plot(kind='barh', 
                                          title='Primary language for Kagglers', figsize=(14,5))
# Programming languages recommended by Kagglers
multi_df['language_recommend'].value_counts().plot(kind='barh', 
                                                   title='What programming language would you recommend an aspiring data scientist to learn first? ', 
                                                   figsize=(14,5))
multi_df['time_coding'].value_counts().reindex(['0% of my time',
                                                '1% to 25% of my time', 
                                                '25% to 49% of my time', 
                                                '50% to 74% of my time', 
                                                '75% to 99% of my time', 
                                                '100% of my time']).plot(kind='barh', 
                                                                         title='How much time do you spend coding?',
                                                                        figsize=(14,5))
# Getting all columns from Q38 and renaming them
Q38 = multi_df.filter(like=("Q38"))
mapping = dict()
for i in range(19):
    old_index = Q16.columns[i]
    string = 'Q38_part_' + str(i+1)
    mapping.update({old_index : string})
Q38 = Q38.rename(columns=mapping)
# Getting value counts
data=pd.melt(Q38).dropna()
data=data[data['value'].apply(lambda x: type(x)==str)]
data=data['value'].value_counts()
# Plot
display(data.plot(kind='barh', 
                  figsize=(14,5), 
                  rot=0, 
                  title='Favorite Media Sources'))
###############
print('Q38: Who/what are your favorite media sources that report on data science topics? (Select all that apply)\n')
print('{:.2f} % of respondents have as their favorite media source Kaggle Forums\n\
{:.2f} % have Medium Blog Posts and\n\
{:.2f} % have ArXiv & Preprints'.format(data[0] / 18971 * 100, 
                               data[1] / 18971 * 100, 
                               data[2] / 18971 * 100))
print('Kagglers on average took {:.2f} seconds (More than 3 hours!) to complete the survey.\n\
This is not really representative, because there are some ridiculous outliers in the data!\n\
The slowest person to complete the survey took {} seconds (10 days and 6 hours)!\n\n'.format(multi_df['duration'].mean(),
                                                                                             multi_df.loc[multi_df['duration'].idxmax()]['duration']))
##### Work in progress #####
##### Ideas #####
# ML models/algorithms
# Learning tools
# Dataset sources
# Reproducability opinions
# Self-confidence of Kagglers
# Engagement with community and website