# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2015.csv')
data.describe()
cols = ['MENTHLTH', 'ADPLEASR', 'ADDOWN', 'ADSLEEP', 'ADENERGY', 'ADEAT1', 'ADFAIL', 'ADTHINK', 'ADMOVE']

# Section 25: Anxiety and Depression

# Questions 1 - 8

# 25.1 - 25.8



data = pd.DataFrame(data, columns=cols)



# 99, BLANK = No answer

nulls = [99, 77]

data = data[cols].replace(nulls, np.nan)

# Set values of '99' and '77' to null as it is a "Refused to Answer"/"Don't know" response



data = data[cols].replace(88, 0)

# Replace values of 88 with 0

# 88 in handbook is 'None', equivalent of zero days

# We will be adding the total days later anyway



# Convert DF to True/False values. Determine if each cell is NULL or not.

is_null_data = pd.DataFrame(np.where(data.loc[:, 'ADPLEASR':].isnull(), False, True))

# is_null_data



# Add and apply 'At Least One' column

# If person answered at least one question

data['At Least One'] = is_null_data.any(axis=1)

data.iloc[209810:209835] # Print a sample
# If all values in row is True, this means every question was answered

# Else, not all 8 were answered

data['All Eight'] = is_null_data.all(axis=1, skipna=False)

data[209810:209835] # Print a sample
data['Total Days'] = np.nan # Make new column and set values to NaN

data['Total Days'] = data[cols[1:]].sum(axis = 1, min_count = 8)

# Set values in column to the sum of each row

# df.sum() performs the sum of each row

# axis param: Performs the operation across each row

# min_count param: Requirement to have AT LEAST X amount of valid values

# Null/NaN is not considered a valid value



data[209810:209835]
# testing



data['ADPLEASR'].mean()

# 4.006002164715143



practice_data = pd.DataFrame(data, columns=['ADPLEASR', 'ADDOWN', 'ADSLEEP'])

# practice_data['ADPLEASR'].mean()



practice_data = pd.DataFrame(np.where(practice_data['ADPLEASR'].isnull(), practice_data['ADPLEASR'].mean(), data['ADPLEASR']))

practice_data = practice_data.round(0)



# Sample NaN indexes: 1, 2, 3, 209814, 209816, 209824

sample_nan = [1, 2, 3, 209814, 209816, 209824]



print('Original data without mean imputations: ', data['ADPLEASR'].iloc[sample_nan])

print('Data with mean imputations: ', practice_data.iloc[sample_nan])
# Impute missing data using mean of each column



data_imputed = pd.DataFrame(data)



for label in data_imputed.columns[1:9]: # iterate over cols

    data_imputed[label] = pd.DataFrame(np.where(data_imputed[label].isnull(), data_imputed[label].mean(), data_imputed[label]))

    # set NaN cells of each column to the column's mean

data_imputed = data_imputed.round(0)

# round float value cells



# data_imputed.iloc[209810:209835]

data_imputed.iloc[sample_nan]
thirty_days = len(data_imputed[data_imputed['MENTHLTH'] == 30].index)

# thirty_days

# 22184

# Total number of respondents who answered '30 days' for MENTHLTH

total_responses = len(data_imputed.index)

# total_reponses

# 441456



perc_thirty_menthlth = thirty_days / total_responses

perc_thirty_menthlth

# 0.05025189373346381

# 5%
x = len(data_imputed[(data_imputed['Total Days'] >= 40) & (data_imputed['MENTHLTH'] == 30)].index)

y = len(data_imputed[(data_imputed['Total Days'] < 40) & (data_imputed['MENTHLTH'] == 30)].index)

x/(x+y) * 100