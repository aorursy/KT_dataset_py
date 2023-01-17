# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load both dataset

dete_survey = pd.read_csv('/kaggle/input/employee-exit-survey/dete_survey.csv')

tafe_survey = pd.read_csv('/kaggle/input/employee-exit-survey/tafe_survey.csv')
# display the dataset

pd.options.display.max_columns = 100 #to avoid column truncation

dete_survey.head(5)
dete_survey.info()
tafe_survey.head(5)
tafe_survey.info()
# read 'Not Stated' as NaN

dete_survey = pd.read_csv('/kaggle/input/employee-exit-survey/dete_survey.csv', na_values='Not Stated')

dete_survey
# getting the column index

print(dete_survey.columns.get_loc('Professional Development'))

print(dete_survey.columns.get_loc('Health & Safety'))

print('\n')

print(tafe_survey.columns.get_loc('Main Factor. Which of these was the main factor for leaving?'))

print(tafe_survey.columns.get_loc('Workplace. Topic:Would you recommend the Institute as an employer to others?'))
# remove and update the columns on both dataset

dete_survey_updated = dete_survey.drop(dete_survey.columns[28:49], axis=1)

tafe_survey_updated = tafe_survey.drop(tafe_survey.columns[17:66], axis=1)



# check the resulting dataset

print(dete_survey_updated.columns)

print(tafe_survey_updated.columns)
# rename the column name of dete_survey

dete_survey_updated.columns = dete_survey_updated.columns.str.lower().str.strip().str.replace(' ', '_')



# rename the column name of tefe_survey by referring to the dete_survey column name

tefe_col = {'Record ID': 'id', 'CESSATION YEAR': 'cease_date', 'Reason for ceasing employment': 'separationtype', 'Gender. What is your Gender?': 'gender',

           'CurrentAge. Current Age':'age', 'Employment Type. Employment Type': 'employment_status', 'Classification. Classification': 'position', 

           'LengthofServiceOverall. Overall Length of Service at Institute (in years)': 'institute_service', 'LengthofServiceCurrent. Length of Service at current workplace (in years)': 'role_service'

           }



tafe_survey_updated = tafe_survey_updated.rename(tefe_col, axis=1)
# check the updated dete_survey column name

dete_survey_updated.head()
# check the updated tafe_survey column name

tafe_survey_updated.head()
# check the unique values under 'separationtype' column



dete_survey_updated['separationtype'].value_counts()
# check the unique values under 'separationtype' column

tafe_survey_updated['separationtype'].value_counts()
# update the resignation values in dete_survey into single value: 'Resignation'

dete_survey_updated['separationtype'] = dete_survey_updated['separationtype'].str.split('-').str[0]



# verify the column

dete_survey_updated['separationtype'].value_counts()
# create new dataset with only 'Resignation' value in 'separationtype' column



dete_resignations = dete_survey_updated[dete_survey_updated['separationtype'] == 'Resignation']

tafe_resignations = tafe_survey_updated[tafe_survey_updated['separationtype'] == 'Resignation']
# check the values under 'cease_date' column

dete_resignations['cease_date']
# extract the year string under cease_date column

dete_resignations['cease_date'] = dete_resignations['cease_date'].str.split('/').str[-1]
# creating the copy of the slice

dete_resignations = dete_survey_updated[dete_survey_updated['separationtype'] == 'Resignation'].copy()

tafe_resignations = tafe_survey_updated[tafe_survey_updated['separationtype'] == 'Resignation'].copy()
# extract the year string under cease_date column, on the copy this time. SettingWithCopyWarning will not pop up

# the value also converted into float for further processing

dete_resignations['cease_date'] = dete_resignations['cease_date'].str.split('/').str[-1].astype('float')

dete_resignations['cease_date'].value_counts().sort_index()
# check for outliers in dete_start_date

dete_resignations['dete_start_date'].value_counts().sort_index()
# check for outlier in tafe_survey

tafe_resignations['cease_date'].value_counts().sort_index()
import matplotlib.pyplot as plt

from numpy import arange
plt.hist(dete_resignations['cease_date'], alpha=0.5, bins=np.arange(2006,2016)-0.5, label='dete')

plt.hist(tafe_resignations['cease_date'], alpha=0.5, bins=np.arange(2009,2015)-0.5, label='tafe')

plt.xticks(np.arange(2006, 2015, 1.0))

plt.legend(loc='best')

plt.show()
# create new column in dete by substracting 'dete_start_date' from 'cease_date'

dete_resignations['institute_service'] = dete_resignations['cease_date'] - dete_resignations['dete_start_date']



# check the newly created column

dete_resignations['institute_service'].head()
# check the unique values

tafe_resignations['Contributing Factors. Dissatisfaction'].value_counts(dropna=False)
# check the unique values

tafe_resignations['Contributing Factors. Job Dissatisfaction'].value_counts(dropna=False)
# create a function to update the value to either True, False, or NaN

def update_vals(val):

    if pd.isnull(val):

        return np.nan

    elif val == '-':

        return False

    else:

        return True



# apply the function on both columns, putting the result in the new column

tafe_resignations['dissatisfied'] = tafe_resignations[['Contributing Factors. Dissatisfaction', 'Contributing Factors. Job Dissatisfaction']].applymap(update_vals).any(axis=1, skipna=False)

        

# check the result

tafe_resignations['dissatisfied'].value_counts(dropna=False)
# create similar 'dissatisfied' column in dete_survey data based on the determined column

dis_list = ['job_dissatisfaction','dissatisfaction_with_the_department', 'physical_work_environment', 'lack_of_recognition', 'lack_of_job_security',

           'work_location', 'employment_conditions', 'work_life_balance', 'workload']



dete_resignations['dissatisfied'] = dete_resignations[dis_list].any(axis=1, skipna=False)

dete_resignations['dissatisfied'].value_counts(dropna=False)
dete_resignations1 = dete_resignations.copy()

tafe_resignations1 = tafe_resignations.copy()

# create identifier column for both dataset

dete_resignations['institute'] = 'DETE'

tafe_resignations['institute'] = 'TAFE'
# combine both dataset

combined = pd.concat([dete_resignations, tafe_resignations], ignore_index=True)
# verify the number of non null values in each column

combined.notnull().sum().sort_values()
# drop the column with less than 500 non null values

combined1 = combined.dropna(thresh=500, axis=1) 

combined1.notnull().sum()
combined1['institute_service'].value_counts()
combined1['institute_service']
# extract the year string

combined2 = combined1.copy()

combined2['institute_service'] = combined1['institute_service'].astype('str').str.extract(r'([0-9]+)')



# convert the values to float

combined2['institute_service'] = combined2['institute_service'].astype('float')



# check the result

combined2['institute_service'].value_counts(dropna=False)
# create a function that returns the value according to the duration

def cat(x):

    if pd.isnull(x):

        return np.nan

    elif x<3:

        return 'New'

    elif 3 <= x <= 6:

        return 'Experienced'

    elif 7 <= x <= 10:

        return 'Established'

    else:

        return 'Veteran'

    

# apply the function on the column

combined2['institute_service'] = combined2['institute_service'].apply(cat)
# check the result

combined2['institute_service'].value_counts(dropna=False)
# check the values of 'dissatisfied' column

combined2['dissatisfied'].value_counts(dropna=False)
# fill the null value with the most frequent value, False

combined2['dissatisfied'] = combined2['dissatisfied'].fillna(False)



# aggregate the 'dissatisfied' column with category as the index

table = pd.pivot_table(combined2, values='dissatisfied', index='institute_service')
# plot the aggregate result

%matplotlib inline

table.plot(kind='bar', rot=30)