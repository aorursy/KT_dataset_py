import numpy as np

import pandas as pd

from sklearn import preprocessing
raw_data = pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv')

raw_data.head()
raw_data.describe(include='all')
raw_data.isnull().sum()
# Checking why and which countries has null value for alcohol

null_alcohol = raw_data[raw_data["Alcohol"].isnull()]

#alcohol_na = raw_data.query('Alcohol == 0')

#alcohol_na

null_alcohol
null_bmi = raw_data[raw_data[" BMI "].isnull()]

null_bmi
## Turns out only Sudan and South Sudan do not report the BMI. We can use imputation for Monaco and San Marino from

## previous years because only one year missing from those countries. I will drop Sudan and South Sudan from the data
## Question: Does Life Expectancy have positive or negative relationship with drinking alcohol?

## Data is missing for almost every country in 2015, so I will drop the 2015 from the data

is_2015 = raw_data[raw_data["Year"]==2015].index

is_2015

data_wo_2015 = raw_data.drop(is_2015)

data_wo_2015
## South Sudan does not have any Alcohol data, so I will drop South Sudan completely

is_s_sudan = data_wo_2015[data_wo_2015["Country"]=="South Sudan"].index

is_s_sudan

data_alcohol = data_wo_2015.drop(is_s_sudan)

data_alcohol
data_alcohol.isnull().sum()
data_1 = data_alcohol[data_alcohol['Life expectancy '].isnull()].index

data_1
data_2 = data_alcohol.drop(data_1) 

data_2
na_bmi = data_2[data_2[" BMI "].isnull()].index

na_bmi
data_3 = data_2.drop(na_bmi)

data_3
data_3.isnull().sum()
data_4 = data_3[data_3['Alcohol'].isnull()].index

data_4
data_clean = data_3.drop(data_4)

data_clean.isnull().sum()
data_clean['Status'].unique()
# Transform to categorical data to numerical data, 1 stands for "Developed countries, and 0 for "developing countries

data_clean["Status"] = data_clean["Status"].map({'Developed':1,'Developing':0})
data_clean['Status'].unique()
## Dropping multiple columns at the same time. I do not need year info for my DL model, no need for county name either
to_drop = ['Country','Year', "Hepatitis B", "Polio", "Total expenditure", "Diphtheria ", "GDP", "Population", "Income composition of resources","Schooling"]

data_clean.drop(to_drop, inplace=True, axis=1)



#passing in the inplace parameter as True and the axis parameter as 1. This tells Pandas that we want the changes to be made directly in our object and that it should look for the values to be dropped in the columns of the object.
#include='all' shows all the data not only numerical

data_clean.describe(include='all')
data_clean.isnull().sum()
targets_csv = data_clean['Life expectancy ']

inputs_csv = data_clean.drop(['Life expectancy '], axis=1)
targets_csv.head()
inputs_csv.head()
targets_csv.to_csv('target_csv.csv',header=False,index=False)
inputs_csv.to_csv('inputs_csv.csv',header=False,index=False)
unscaled_inputs = np.loadtxt("inputs_csv.csv", delimiter = ',')

targets = np.loadtxt("target_csv.csv", delimiter=',')
print(targets.shape[0])

print(unscaled_inputs.shape[0])
scaled_inputs = preprocessing.scale(unscaled_inputs)
shuffled_indicies = np.arange(scaled_inputs.shape[0])

np.random.shuffle(shuffled_indicies)



shuffled_inputs = scaled_inputs[shuffled_indicies]

shuffled_targets = targets[shuffled_indicies]
samples_count = shuffled_inputs.shape[0]

train_samples_count = int(0.8*samples_count)

validation_samples_count = int(0.1*samples_count)

test_samples_count = samples_count - train_samples_count - validation_samples_count



train_inputs = shuffled_inputs[:train_samples_count]

train_targets = shuffled_targets[:train_samples_count]



validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]

validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]



test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]

test_targets = shuffled_targets[train_samples_count+validation_samples_count:]



## Check if we seperated them correctly



print(samples_count)

print(train_samples_count)

print(validation_samples_count)

print(test_samples_count)

np.savez('life_expectancy_data_train',inputs= train_inputs, targets=train_targets)

np.savez('life_expectancy_data_validation', inputs=validation_inputs, targets=validation_targets)

np.savez('life_expectancy_data_test',inputs=test_inputs,targets=test_targets)