# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Read all comma-separated values (csv) file of mental health into DataFrames

df_1 = pd.read_csv('../input/mental_health_2014.csv')

df_2 = pd.read_csv('../input/mental_health_2016.csv')

df_3 = pd.read_csv('../input/mental_health_2017.csv')

df_4 = pd.read_csv('../input/mental_health_2018.csv')
#print columns of the desired table 

columns = df_1.rename(str.lower, axis='columns')

df_1.drop(['Timestamp','comments'],axis = 'columns',inplace=True)

df_1.columns = map(str.lower, df_1.columns)
df_1.columns
pd.set_option('display.max_columns', 130)

df_2.head(3)
#Drop unnecessary columns 

df_2.drop(['Is your primary role within your company related to tech/IT?',

                   'Why or why not?',

                   'Have you had a mental health disorder in the past?',

                   'If yes, what condition(s) have you been diagnosed with?',

                   'If maybe, what condition(s) do you believe you have?',

                   'If so, what condition(s) were you diagnosed with?',

                   'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?',

                   'Do you know local or online resources to seek help for a mental health disorder?',

                   'What country do you live in?',

                   'What US state or territory do you live in?',

                   

                   'Which of the following best describes your work position?'],axis = 'columns',inplace=True)
#disorders=df_2['If so, what condition(s) were you diagnosed with?']
df_2.drop(df_2.loc[:,'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?':'Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?'].columns, axis = 1, inplace=True)

df_2.drop(df_2.loc[:,'Why or why not?.1':'Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?'].columns, axis = 1, inplace=True)
#Rename long columns to shorter ones so it's easier to work with them

columns = ['self_employed','no_employees','tech_company','benefits','care_options','wellness_program','seek_help','anonymity',

          'leave','mental_health_consequence','phys_health_consequence','coworkers','supervisor','mental_vs_physical',

           'obs_consequence','medical_coverage','phys_health_interview','mental_health_interview','family_history',

           'treatment1','treatment2','treatment','work_interfere','age','gender','country','state','remote_work']



for i in range(len(columns)):

    df_2.rename(index=str, columns={df_2.columns[i] : columns[i]},inplace=True)



df_2.drop(['treatment1','treatment2'],axis = 'columns',inplace=True)



df_2.head(100)
df_2.isna().sum()
df_2.drop(['medical_coverage'],axis ='columns', inplace=True)
#merge the first dataset with the second one

results = df_1.append(df_2, ignore_index=True, sort=False)

#add disorder to the table

#results['disorders']=disorders

results.head(5)
results.drop(['country', 'state', 'self_employed','work_interfere',

              'remote_work', 'tech_company','phys_health_consequence', 'phys_health_interview',

              'mental_health_consequence'],axis ='columns', inplace=True)
results.head(5)
# Get a quick overview of all of the variables using pandas_profiling

import pandas_profiling

pandas_profiling.ProfileReport(results)
null_columns=results.columns[results.isnull().any()]

results[null_columns].isnull().sum()
#drop rows if more than half of the columns are NaN

results.dropna(thresh=len(results.columns)/2, axis=0, inplace=True)
def clean_value (value):

    if value == "Always" or value == 1:

        return "Yes"

    elif value == "Never" or value == 0:

        return "No"

    elif value == "Yes, I observed" or value == 'Yes, I experienced':

        return "Yes"

    elif value == "I don't know":

        return "Don't know"

    elif value == "I am not sure":

        return "Not sure"

    elif value == "Physical health":

        return "No"

    elif value == "Mental health":

        return "No"

    elif value == "Some of them":

        return "Maybe"

    elif value == "Difficult":

        return "Very difficult"

    elif value == 'Not eligible for coverage / NA':

        return 'Not eligible for coverage / N/A'

    else: return value
results=results.applymap(clean_value)
results.isna().sum()
results.dropna(axis=0, inplace=True)
#deal with the NaN values

#results['care_options']=results['care_options'].fillna('Not sure')

#results['work_interfere']=results['work_interfere'].fillna('Not sure')

#results['self_employed'] = results['self_employed'].fillna('No')
pandas_profiling.ProfileReport(results)
results.to_csv('submission.csv', index=False)
results.shape
np.in1d(df_3.columns,df_4.columns)

np.array_equal(df_3.columns,df_4.columns)

for i in range(len(df_3.columns)): # assuming the lists are of the same length

    if df_3.columns[i]!= df_4.columns[i]:

        df_3 = df_3.rename(index=str, columns={df_3.columns[i] : df_4.columns[i]})

        print()

        print()

        print(df_3.columns[i])

        print()

        print(df_4.columns[i])  
df = pd.concat([df_3, df_4], ignore_index=True, sort=True)

df.shape
df.head(3)
df.rename(index=str,columns={'Have you ever sought treatment for a mental health disorder from a mental health professional?' : 'treatment',

                            'How many employees does your company or organization have?' : 'no_employees',

                            'Does your employer provide mental health benefits as part of healthcare coverage?' : 'benefits',

                            'Do you know the options for mental health care available under your employer-provided health coverage?' : 'care_options',

                            'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?' : 'wellness_program',

                            'Does your employer offer resources to learn more about mental health disorders and options for seeking help?' : 'seek_help',

                            'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?' : 'anonymity',

                            'If a mental health issue prompted you to request a medical leave from work, how easy or difficult would it be to ask for that leave?' : 'leave',

                            'Would you feel comfortable discussing a mental health issue with your coworkers?' : 'coworkers',

                            'Would you feel comfortable discussing a mental health issue with your direct supervisor(s)?' : 'supervisor',

                            'Would you bring up your mental health with a potential employer in an interview?' : 'mental_health_interview',

                            'Would you feel more comfortable talking to your coworkers about your physical health or your mental health?' : 'mental_vs_physical',

                            'What is your age?' : 'age',

                            'What is your gender?' : 'gender',

                            'Do you have a family history of mental illness?' : 'family_history'

                            #'<strong>Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?</strong>' : 'obs_consequence'

                            }, inplace=True)
newDF = pd.DataFrame() #creates a new dataframe that's empty



for column in df.columns:

    if column in results.columns:

        newDF[column] = df[column]
#drop rows if more than half of the columns are NaN

newDF.dropna(thresh=len(newDF.columns)/2, axis=0, inplace=True)
null_columns=newDF.columns[newDF.isnull().any()]

newDF[null_columns].isnull().sum()
submission = pd.concat([results, newDF], ignore_index=True, sort=True)

submission.shape
#submission.dropna(axis=0, inplace=True)
submission=submission.applymap(clean_value)
pandas_profiling.ProfileReport(submission)
submission.to_csv('submission.csv', index=False)