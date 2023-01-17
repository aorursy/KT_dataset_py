#Import modules and dataset



#Import numpy and pandas

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#Read csv files for DETE and FATE surveys

dete_survey = pd.read_csv('../input/employee-exit-survey/dete_survey.csv')

tafe_survey = pd.read_csv('../input/employee-exit-survey/tafe_survey.csv')
#Explore the Both Dataset



#Create a function that explores the dataset and checks missing for missing values

def explore_dataset(dataset):

    print(dataset.info())

    print(dataset.head(3))

    sns.heatmap(dataset.isnull(),cbar = False)
#Explore DETE survey

explore_dataset(dete_survey)
#Explore TAFE survey

explore_dataset(tafe_survey)
#Read in DETE and FATE datasets

dete_survey = pd.read_csv('../input/employee-exit-survey/dete_survey.csv', na_values = 'Not Stated')

tafe_survey = pd.read_csv('../input/employee-exit-survey/tafe_survey.csv', na_values = 'Not Stated')
#Drop columns that are not needed for project objective



#Drop DETE columns

dete_survey_updated = dete_survey.drop(dete_survey.columns[28:49],axis = 1)



#Drop FATE columns

tafe_survey_updated = tafe_survey.drop(tafe_survey.columns[17:66], axis = 1)

#Adjust both DETE and TAFE column names



#Before DETE column name adjustment

print(dete_survey_updated.columns)



#Use string manipulation to adjust column names in DETE dataset

dete_survey_updated.columns = dete_survey_updated.columns.str.lower().str.strip().str.replace(' ', '_')
#After DETE column name adjustment

dete_survey_updated.columns
#Adjust TAFE column names by using rename



#Before TAFE column name adjustment

print(tafe_survey_updated.columns)



#Renamed column's dictionary

tafe_column_names = {'Record ID': 'id', 'CESSATION YEAR': 'cease_date', 'Reason for ceasing employment': 'separationtype',

                     'Gender. What is your Gender?': 'gender', 'CurrentAge. Current Age': 'age',

                     'Employment Type. Employment Type': 'employment_status', 'Classification. Classification': 'position',

                     'LengthofServiceOverall. Overall Length of Service at Institute (in years)': 'institute_service',

                     'LengthofServiceCurrent. Length of Service at current workplace (in years)': 'role_service'}



#Rename TAFE column names

tafe_survey_updated = tafe_survey_updated.rename(tafe_column_names, axis = 1)
#After TAFE column name adjustment

tafe_survey_updated.columns
#Count Unique values for the Seperationtype column in the DETE and TAFE dataset



#Find unique values for DETE dataset

print(dete_survey_updated['separationtype'].value_counts())



print('\n')



#Find unique values for TAFE dataset

print(tafe_survey_updated['separationtype'].value_counts())
#Update DETE separationtype column



#Use str.split and str to reduce the different resignation values to only one type

dete_survey_updated['separationtype'] = dete_survey_updated['separationtype'].str.split('-').str[0]



#Check the value counts for DETE separationtype column

print(dete_survey_updated['separationtype'].value_counts())
#Filter the DETE and TAFE dataset to contain rows where separationtype is Resignation

dete_resignations = dete_survey_updated[dete_survey_updated['separationtype'] == 'Resignation'].copy()

tafe_resignations = tafe_survey_updated[tafe_survey_updated['separationtype'] == 'Resignation'].copy()



#Check the the Filter

print(dete_resignations['separationtype'].value_counts())

tafe_resignations['separationtype'].value_counts()
#Count unique values within the cease_date column for datasets DETE anda TAFE



#DETE: Find the unique value_counts for the cease_date column

print(dete_resignations['cease_date'].value_counts())

print('\n')



#TAFE: Find the unique value_counts for the cease_date column

tafe_resignations['cease_date'].value_counts()
#Correct the format for the cease_date column within the DETE dataset

dete_resignations['cease_date'] = dete_resignations['cease_date'].str.split('/').str[-1]



#Change the cease_date column values to a float data type

dete_resignations['cease_date'] = dete_resignations['cease_date'].astype('float')



#Check results 

dete_resignations['cease_date'].value_counts()
#Check the unique values within the start_date column for DETE dataset



#DETE: Find the unique value counts for the start_date column

dete_resignations['dete_start_date'].value_counts().sort_index(ascending = True)
#Create a new colum that holds the amount of years worked for each employee



#Create a new column that substracts start_date from cease_date

dete_resignations['institute_service'] = dete_resignations['cease_date'] - dete_resignations['dete_start_date']



#Check the new column results with dete_start_date and cease_date

dete_resignations[['dete_start_date','cease_date','institute_service']].head(5)
#Explore the TAFE dissatisfaction columns and understand their values



#Find the unique values for the dissatisfied columns in the TAFE dataset

print(tafe_resignations['Contributing Factors. Dissatisfaction'].value_counts())

print('\n')

print(tafe_resignations['Contributing Factors. Job Dissatisfaction'].value_counts())

print('\n')



#Check the Datatype of the columns within the TAFE dataset

tafe_resignations.info()



#View the first 5 rows of the TAFE dissastisfaction column

tafe_resignations['Contributing Factors. Dissatisfaction'].head(30)
#Explore the DETE dissatisfaction columns and understand their values

print(dete_resignations['job_dissatisfaction'].value_counts())

print('\n')



#Check the column datatypes within the DETE dataset

dete_resignations.info()
#Create a function that takes an dataframe (x) and returns values (False,True, or NaN) given requirements

def update_values(x):

    if x == '-':

        return False

    elif pd.isnull(x):

        return np.nan

    else:

        return True



#Create a new column that uses the update_values function to determine if emplyees were dissatisfied

tafe_resignations['dissatisfied'] = tafe_resignations[['Contributing Factors. Dissatisfaction',

                                                          'Contributing Factors. Job Dissatisfaction']].applymap(update_values).any(1, skipna = False)

#Copy tafe_resignations dataframe

tafe_resignations_up = tafe_resignations.copy()



#Check new column

print(tafe_resignations_up['dissatisfied'].value_counts(dropna = False))



#Check how the values look

tafe_resignations_up.head(2)
#Create a new column that aggregates many dissatisfiend columns into one new column

dete_resignations['dissatisfied'] = dete_resignations[['job_dissatisfaction',

                                                          'dissatisfaction_with_the_department','physical_work_environment',

                                                          'lack_of_recognition','lack_of_job_security','work_location',

                                                          'employment_conditions','work_life_balance','workload']].any(1, skipna = False)



#Copy dete_resignations dataframe

dete_resignations_up = dete_resignations.copy()



#Check dissatisfaction column in DETE dataframe

dete_resignations_up['dissatisfied'].value_counts(dropna = False)
#Add institue column to the dete dataframe

dete_resignations_up['institute'] = 'DETE'



#Check institute column

print(dete_resignations_up['institute'].head(5))

print('\n')



#Add institute column to the tafe dataframe

tafe_resignations_up['institute'] = 'TAFE'



#Check institute column to the tafe dataframe

tafe_resignations_up['institute'].head(5)
#Combine dataframes and explore the result



#Combine the DETE and TAFE dataframe together

combined = pd.concat([dete_resignations_up, tafe_resignations_up], ignore_index = True)



#Check the results of the combining dataframes

print(combined.head(5))



#Find how many null values there are in the combined dataframe

print(combined.shape)

combined.notnull().sum().sort_values()





#Update the combined dataframe to contain less nulls



#Drop null values with a thresh of 500 or more

combined_updated = combined.dropna(thresh = 500, axis = 1).copy()



#Check updated dataframe

print(combined_updated.notnull().sum())

combined_updated.head(5)
#Check the unique values

combined_updated['institute_service'].value_counts(dropna=False)
#Create a new column that has the extracted values from the institute_service column as strings

combined_updated['institute_service_up'] = combined_updated['institute_service'].astype('str').str.extract(r'(\d+)')



#Change values from new columns into a float type

combined_updated['institute_service_up'] = combined_updated['institute_service_up'].astype('float')



#Check the new column unique values

combined_updated['institute_service_up'].value_counts()
#Create a function that takes float values and categorizes them into strings

def transform_service(val):

    if val >= 11:

        return 'Veteran'

    elif 7 <= val < 11:

        return 'Established'

    elif 3 <= val < 7:

        return 'Experienced'

    elif pd.isnull(val):

        return np.nan

    else:

        return 'New'

    

#Create a new column to hold the results of the function being used on the institute_service_up    

combined_updated['service_cat'] = combined_updated['institute_service_up'].apply(transform_service)



#check to see if values have been categorized

combined_updated['service_cat'].value_counts()
#Find how many unique value are in the dissatisfied column

print(combined_updated['dissatisfied'].value_counts(dropna= False))
#Replace the missing values in the dissatisfied column

combined_updated['dissatisfied'] =  combined_updated['dissatisfied'].fillna(value = False)



#Check the unique values in the dissatisfied column

combined_updated['dissatisfied'].value_counts(dropna = False)



#Calculate the percentage of dissatisfied employees in each service_cat group

dis_pct = combined_updated.pivot_table(index = 'service_cat', values = 'dissatisfied')

print(round(dis_pct * 100, 0))
#Create a barchart of the dis_pct results

dis_pct.plot(kind = 'bar', title = 'The Percentage of Employees that Resigned Due To Dissatisfaction',xlabel = 'Experience Level', ylabel = '% of Employees')