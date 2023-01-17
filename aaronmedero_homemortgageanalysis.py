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
import matplotlib.pyplot as plt

import seaborn as sns
raw_data = pd.read_csv('/kaggle/input/ny-home-mortgage/ny_hmda_2015.csv', sep=',')

raw_data.head()
raw_data.shape
raw_data.columns
# Calculate the approval rate

raw_data['action_taken'].value_counts().sort_index()
raw_data['action_taken_name'].value_counts()
# We count "Loan originated" and "Application approved but not accepted" as approved applications, and "Application denied by financial institution"

# as denied applications. Other categories are dropped.



filter1 = (raw_data['action_taken'] >= 1) & (raw_data['action_taken'] <= 3)

data = raw_data[filter1]

data.shape
freqs = data['action_taken'].value_counts()

freqs

# 1 and 2 - approved

# 3 - denied
# Calculate the overall approval rate

overall_approval_rate = (freqs[1] + freqs[2]) / data.shape[0]

print("Overall approval rate:", overall_approval_rate)
# Calculate the approval rate for each gender

data['applicant_sex'].value_counts()
data['applicant_sex'].hist()
data['applicant_sex_name'].value_counts()
# Let's keep "Male" and "Female" only.

filter2 = (data['applicant_sex_name'] == 'Male') | (data['applicant_sex_name'] == 'Female')

data = data[filter2]

data.shape
# Split the data into male subset and female subset

filter3 = (data['applicant_sex_name'] == 'Male')

data_male = data[filter3]

print("Male:", data_male.shape)



data_female = data[data['applicant_sex_name'] == 'Female']

print("Female:", data_female.shape)
# Calculate the approval rate of male applicants

freqs = data_male['action_taken'].value_counts()

approval_rate_male = (freqs[1] + freqs[2]) / data_male.shape[0]

print("Male approvale rate:", approval_rate_male)



# Calculate the approval rate of female applicants

freqs = data_female['action_taken'].value_counts()

approval_rate_female = (freqs[1] + freqs[2]) / data_female.shape[0]

print("Female approvale rate:", approval_rate_female)
# Calculate the approval rates of different races



data['applicant_race_1'].value_counts()
data['applicant_race_1'].hist()
data['applicant_race_name_1'].value_counts()
# Let's get rid of "N/A" and "Infomation not provided".

filter4 = (data['applicant_race_name_1'] == 'White') | (data['applicant_race_name_1'] == 'Asian') | (data['applicant_race_name_1'] == 'Black or African American') | (data['applicant_race_name_1'] == 'American Indian or Alaska Native') | (data['applicant_race_name_1'] == 'Native Hawaiian or Other Pacific Islander')

data = data[filter4]

data.shape
# Split the data into subsets

filter5 = (data['applicant_race_name_1'] == 'White')

data_White = data[filter5]

print("White:", data_White.shape)



data_Asian = data[data['applicant_race_name_1'] == 'Asian']

print("Asian:", data_Asian.shape)



data_Black_or_African_American = data[data['applicant_race_name_1'] == 'Black or African American']

print("Black or African American:", data_Black_or_African_American.shape)



data_American_Indian_or_Alaska_Native = data[data['applicant_race_name_1'] == 'American Indian or Alaska Native']

print("American Indian or Alaska Native:", data_American_Indian_or_Alaska_Native.shape)



data_Native_Hawaiian_or_Other_Pacific_Islander = data[data['applicant_race_name_1'] == 'Native Hawaiian or Other Pacific Islander']

print("Native Hawaiian or Other Pacific Islander:", data_Native_Hawaiian_or_Other_Pacific_Islander.shape)
# Calculate the approval rate of White applicants

freqs = data_White['action_taken'].value_counts()

approval_rate_White = (freqs[1] + freqs[2]) / data_White.shape[0]

print("White approval rate:", approval_rate_White)



# Calculate the approval rate of Asian applicants

freqs = data_Asian['action_taken'].value_counts()

approval_rate_Asian = (freqs[1] + freqs[2]) / data_Asian.shape[0]

print("Asian approval rate:", approval_rate_Asian)



# Calculate the approval rate of Black_or_African_American applicants

freqs = data_Black_or_African_American['action_taken'].value_counts()

approval_rate_Black_or_African_American = (freqs[1] + freqs[2]) / data_Black_or_African_American.shape[0]

print("Black_or_African_American approval rate:", approval_rate_Black_or_African_American)



# Calculate the approval rate of American_Indian_or_Alaska_Native applicants

freqs = data_American_Indian_or_Alaska_Native['action_taken'].value_counts()

approval_rate_American_Indian_or_Alaska_Native = (freqs[1] + freqs[2]) / data_American_Indian_or_Alaska_Native.shape[0]

print("American_Indian_or_Alaska_Native approval rate:", approval_rate_American_Indian_or_Alaska_Native)



# Calculate the approval rate of Native_Hawaiian_or_Other_Pacific_Islander applicants

freqs = data_Native_Hawaiian_or_Other_Pacific_Islander['action_taken'].value_counts()

approval_rate_Native_Hawaiian_or_Other_Pacific_Islander = (freqs[1] + freqs[2]) / data_Native_Hawaiian_or_Other_Pacific_Islander.shape[0]

print("Native_Hawaiian_or_Other_Pacific_Islander approval rate:", approval_rate_Native_Hawaiian_or_Other_Pacific_Islander)
#Finished Finding Approval Rate based off Race
def get_tax_bracket(income):

    """

    Tax brackets 2020:

    9875, 40125, 85525, 163300, 207350, 518400

    """

    if income < 9.875:

        return 0

    elif income < 40.125:

        return 1

    elif income < 85.525:

        return 2

    elif income < 163.3:

        return 3

    elif income < 207.35:

        return 4

    elif income < 518.4:

        return 5

    else:

        return 6
def tax_Bracket_Name (tax_bracket):

    """

    

    9875, 40125, 85525, 163300, 207350, 518400

    

    """

        

    if tax_bracket == 0:

        return 'Income (0) < 9875'

    elif tax_bracket == 1:

        return 'Income (1) < 40,125'

    elif tax_bracket == 2:

        return 'Income (2) < 85,525'

    elif tax_bracket == 3:

        return 'Income (3) < 163,300'

    elif tax_bracket == 4:

        return 'Income (4) < 207,350'

    elif tax_bracket == 5:

        return 'Income (5) < 518,400'

    elif tax_bracket == 6:

        return 'Income (6) > 518,400'

    else:

        return 'Not a Number'

    
# Create a column in raw_data, showing the tax_bracket of each applicant

data_income = raw_data.copy()

data_income_name = data_income.copy()

data_income['tax_bracket'] = data_income['applicant_income_000s'].apply(get_tax_bracket)

data_income_name['tax_bracket_name'] = data_income['tax_bracket'].apply(tax_Bracket_Name)
data_income['tax_bracket'].value_counts().sort_index()
# Calculate the approval rates of different income levels



data_income['tax_bracket'].hist()
data_income_name['tax_bracket_name'].value_counts().sort_index()
# Split the data into subsets

filter6 = (data_income_name['tax_bracket_name']                     == 'Income (0) < 9875')

data_Income_9875 = data[filter5]

print("Income (0) < 9875:", data_Income_9875.shape)



data_Income_40125 = data[data_income_name['tax_bracket_name']       == 'Income (1) < 40,125']

print("Income (1) < 40,125:", data_Income_40125.shape)



data_Income_85525 = data[data_income_name['tax_bracket_name']       == 'Income (2) < 85,525']

print("Income (2) < 85,525:", data_Income_85525.shape)



data_Income_163300 = data[data_income_name['tax_bracket_name']      == 'Income (3) < 163,300']

print("Income (3) < 163,300:", data_Income_163300.shape)



data_Income_207350 = data[data_income_name['tax_bracket_name']      == 'Income (4) < 207,350']

print("Income (4) < 207,350:", data_Income_207350.shape)



data_Income_518400 = data[data_income_name['tax_bracket_name']      == 'Income (5) < 518,400']

print("Income (5) < 518,400:", data_Income_518400.shape)



data_Income_over_518400 = data[data_income_name['tax_bracket_name'] == 'Income (6) > 518,400']

print("Income (6) > 518,400:", data_Income_over_518400.shape)
# Calculate the approval rate of Income (0) < 9875 applicants

freqs = data_Income_9875['action_taken'].value_counts()

approval_Income_9875 = (freqs[1] + freqs[2]) / data_Income_9875.shape[0]

print("Income (0) < 9875 approval rate:", approval_Income_9875)



# Calculate the approval rate of Income (1) < 40,125 applicants

freqs = data_Income_40125['action_taken'].value_counts()

approval_Income_40125 = (freqs[1] + freqs[2]) / data_Income_40125.shape[0]

print("Income (1) < 40,125 approval rate:", approval_Income_40125)



# Calculate the approval rate of Income (2) < 85,525 applicants

freqs = data_Income_85525['action_taken'].value_counts()

approval_Income_85525 = (freqs[1] + freqs[2]) / data_Income_85525.shape[0]

print("Income (2) < 85,525 approval rate:", approval_Income_85525)



# Calculate the approval rate of Income (3) < 163,300 applicants

freqs = data_Income_163300['action_taken'].value_counts()

approval_Income_163300 = (freqs[1] + freqs[2]) / data_Income_163300.shape[0]

print("Income (3) < 163,300 approval rate:", approval_Income_163300)



# Calculate the approval rate of Income (4) < 207,350 applicants

freqs = data_Income_207350['action_taken'].value_counts()

approval_Income_207350 = (freqs[1] + freqs[2]) / data_Income_207350.shape[0]

print("Income (4) < 207,350 approval rate:", approval_Income_207350)



# Calculate the approval rate of Income (5) < 518,400 applicants

freqs = data_Income_518400['action_taken'].value_counts()

approval_Income_518400 = (freqs[1] + freqs[2]) / data_Income_518400.shape[0]

print("Income (5) < 518,400 approval rate:", approval_Income_518400)



# Calculate the approval rate of Income (6) > 518,400 applicants

freqs = data_Income_over_518400['action_taken'].value_counts()

approval_Income_over_518400 = (freqs[1] + freqs[2]) / data_Income_over_518400.shape[0]

print("Income (6) > 518,400 approval rate:", approval_Income_over_518400)