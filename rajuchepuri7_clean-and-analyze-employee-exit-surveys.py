# Read in the data

import pandas

import numpy



dete_survey = pandas.read_csv("../input/dete_survey.csv")

tafe_survey = pandas.read_csv("../input/tafe_survey.csv", encoding = 'Windows-1252')



#quick exploration of the data

pandas.options.display.max_columns = 150 # to avoid truncated output

dete_survey.head()
dete_survey.info()
tafe_survey.head()
tafe_survey.info()
# Read in the data again, but this time read `Not Stated` values as `NaN`

dete_survey = pandas.read_csv("../input/dete_survey.csv", na_values = 'Not Stated')



# quick exploration of the data

dete_survey.head()
# Remove columns we don't need for our analysis

dete_survey_updated = dete_survey.drop(dete_survey.columns[28:49], axis = 1)

tafe_survey_updated = tafe_survey.drop(tafe_survey.columns[17:66], axis = 1)



#verify that the columns were dropped

print(dete_survey_updated.columns)

print(tafe_survey_updated.columns)
# clean the column names

dete_survey_updated.columns = dete_survey_updated.columns.str.lower().str.strip().str.replace(" ", "_")



# check whether the column names were updated correctly

dete_survey_updated.columns
# Update column names to match the names in dete_survey_updated

tafe_survey_updated = tafe_survey_updated.rename(columns={'Record ID': 'id', 

                                   'CESSATION YEAR': 'cease_date', 

                                   'Reason for ceasing employment': 'separationtype', 

                                   'Gender. What is your Gender?': 'gender', 

                                   'CurrentAge. Current Age': 'age',

                                   'Employment Type. Employment Type': 'employment_status', 

                                   'Classification. Classification': 'position', 

                                   'LengthofServiceOverall. Overall Length of Service at Institute (in years)': 'institute_service'})



# check whether the specified column names were updated correctly

tafe_survey_updated.columns
dete_survey_updated['separationtype'].value_counts()
tafe_survey_updated['separationtype'].value_counts()
dete_survey_updated['separationtype'] = dete_survey_updated['separationtype'].str.split("-").str[0]



dete_survey_updated['separationtype'].value_counts()
# Select only the resignation separation types from each dataframe

dete_resignations = dete_survey_updated[dete_survey_updated['separationtype'] == 'Resignation'].copy()



tafe_resignations = tafe_survey_updated[tafe_survey_updated['separationtype'] == 'Resignation'].copy()
# verify the unique values

dete_resignations['cease_date'].value_counts()
# Extract the years and convert them to a float type

dete_resignations['cease_date'] = dete_resignations['cease_date'].str.split("/").str[-1]

dete_resignations['cease_date'] = dete_resignations['cease_date'].astype('float')



# Check the values again and look for outliers

dete_resignations['cease_date'].value_counts()
# Check the unique values and look for outliers

dete_resignations['dete_start_date'].value_counts().sort_index()
# check for the unique values



tafe_resignations['cease_date'].value_counts()
# Calculate the length of time an employee spent in their respective workplace and create a new column

dete_resignations['institute_service'] = dete_resignations['cease_date'] - dete_resignations['dete_start_date']



# quick check of the values



dete_resignations['institute_service'].head()
# verify the unique values

tafe_resignations['Contributing Factors. Dissatisfaction'].value_counts()
# verify the unique values

tafe_resignations['Contributing Factors. Job Dissatisfaction'].value_counts()
# Update the values in the contributing factors columns to be either True, False, or NaN

def update_vals(value):

    if pandas.isnull(value):

        return numpy.nan

    elif value == '-':

        return False

    else:

        return True

        

tafe_resignations['dissatisfied'] = tafe_resignations[['Contributing Factors. Dissatisfaction', 'Contributing Factors. Job Dissatisfaction']].applymap(update_vals).any(1, skipna = False)

tafe_resignations_up = tafe_resignations.copy()



# Check the unique values after the updates

tafe_resignations_up['dissatisfied'].value_counts(dropna = False)
# Update the values in columns related to dissatisfaction to be either True, False, or NaN

dete_resignations['dissatisfied'] = dete_resignations[['job_dissatisfaction', 'dissatisfaction_with_the_department', 'physical_work_environment', 'lack_of_recognition', 'lack_of_job_security', 'work_location', 'employment_conditions', 'work_life_balance', 'workload']].any(1, skipna=False)

dete_resignations_up = dete_resignations.copy()



# Check the unique values after the updates

dete_resignations_up['dissatisfied'].value_counts(dropna = False)
# Add an institute column

dete_resignations_up['institute'] = 'DETE'

tafe_resignations_up['institute'] = 'TAFE'
# combine the dataframes

combined = pandas.concat([dete_resignations_up, tafe_resignations_up], ignore_index = True)



# Verify the number of non null values in each column

combined.notnull().sum().sort_values()
# Drop columns with less than 500 non null values

combined_updated = combined.dropna(axis = 1, thresh = 500).copy()
# Check the unique values

combined_updated['institute_service'].value_counts(dropna = False)
# Extract the years of service and convert the type to float

combined_updated['institute_service_up'] = combined_updated['institute_service'].astype('str').str.extract(r'(\d+)')

combined_updated['institute_service_up'] = combined_updated['institute_service_up'].astype('float')



# Check the years extracted are correct

combined_updated['institute_service_up'].value_counts()
# Convert years of service into categories

def transform_service(value):

    if pandas.isnull(value):

        return numpy.nan

    elif value < 3:

        return "New"

    elif value >=3 and value < 7:

        return "Experienced"

    elif value >= 7 and value < 11:

        return "Established"

    else:

        return "Veteran"

combined_updated['service_cat'] = combined_updated['institute_service_up'].apply(transform_service)



# Quick check of the update

combined_updated['service_cat'].value_counts()
# Verify the unique values

combined_updated['dissatisfied'].value_counts(dropna = False)
# Replace missing values with the most frequent value, i.e. False

combined_updated['dissatisfied'] = combined_updated['dissatisfied'].fillna(False)
# Calculate the percentage of employees who resigned due to dissatisfaction in each category

dissatisfaction_pct = combined_updated.pivot_table(index = 'service_cat', values = 'dissatisfied')



# Plot the results

%matplotlib inline

dissatisfaction_pct.plot(kind = 'bar', rot = 30)