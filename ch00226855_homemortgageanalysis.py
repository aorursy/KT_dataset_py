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
# Create a column in raw_data, showing the tax_bracket of each applicant

data_income = raw_data.copy()

data_income['tax_bracket'] = data_income['applicant_income_000s'].apply(get_tax_bracket)
data_income['tax_bracket'].value_counts().sort_index()
# Calculate the approval rates of different income levels














