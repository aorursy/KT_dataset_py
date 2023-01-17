# importing are basic libraries.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# %matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
reported_data = pd.read_csv('/kaggle/input/malaria-dataset/reported_numbers.csv')

estimated_data = pd.read_csv('/kaggle/input/malaria-dataset/estimated_numbers.csv')

incidence_data = pd.read_csv('/kaggle/input/malaria-dataset/incidence_per_1000_pop_at_risk.csv')
print(reported_data.head() , reported_data.info())  # checking the info and few data of reported data.
print(estimated_data.head() , estimated_data.info())  # checking the info and few data of estimated data.
print(incidence_data.head() , incidence_data.info())  # checking the info and few data of incidence data.
# using this function to find number of unique values in a column of a dataframe

def getUnique(df):

    for col in df.columns:

        print(col + " : " + str(df[col].nunique()))
getUnique(reported_data)
getUnique(estimated_data)
getUnique(incidence_data)
reported = reported_data.groupby(['WHO Region']).agg({'No. of cases' : 'sum'}).reset_index()

x = reported_data['No. of cases'].sum()

reported['Percentage'] = (reported['No. of cases']/x)*100

reported
fig_dims = (20, 5)



fig, axes = plt.subplots(1, 2, figsize=fig_dims)

sns.barplot(x = 'Year' , y = 'No. of cases' , data = reported_data[reported_data['WHO Region'] == 'Africa'], ax= axes[0]).set_title("In Africa")

sns.barplot(x = 'Year' , y = 'No. of cases' , data = reported_data[reported_data['WHO Region'] != 'Africa'], ax= axes[1]).set_title("In other WHO Regions")
estimated = estimated_data.groupby(['WHO Region']).agg({'No. of cases_median' : 'sum'}).reset_index()

x = estimated_data['No. of cases_median'].sum()

estimated['Percentage'] = (estimated['No. of cases_median']/x)*100

estimated
fig_dims = (20, 5)



fig, axes = plt.subplots(1,2 , figsize=fig_dims)

sns.barplot(x = 'Year' , y = 'No. of cases_median' 

            , data = estimated_data[estimated_data['WHO Region'] == 'Africa'], ax= axes[0]).set_title("In Africa")

sns.barplot(x = 'Year' , y = 'No. of cases_median' 

            , data = estimated_data[estimated_data['WHO Region'] != 'Africa'], ax= axes[1]).set_title("In other WHO Regions")
incidence_data.info()
incidence = incidence_data.groupby(['WHO Region']).agg({'No. of cases' : 'sum'}).reset_index()

x = incidence_data['No. of cases'].sum()

incidence['Percentage'] = (incidence['No. of cases']/x)*100

incidence
fig_dims = (20, 5)



fig, axes = plt.subplots(1,2 , figsize=fig_dims)

sns.barplot(x = 'Year' , y = 'No. of cases' 

            , data = incidence_data[incidence_data['WHO Region'] == 'Africa'], ax= axes[0]).set_title("In Africa")

sns.barplot(x = 'Year' , y = 'No. of cases' 

            , data = incidence_data[incidence_data['WHO Region'] != 'Africa'], ax= axes[1]).set_title("In other WHO Regions")