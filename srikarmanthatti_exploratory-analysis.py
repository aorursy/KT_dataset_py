# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

infec_df = pd.read_csv('../input/infectious-diseases-by-american-counties/infectious-diseases-by-disease-county-year-and-sex-1.csv')
infec_df.shape
infec_df.head(10)
infec_df
print(infec_df.columns)

infec_df.columns = ['disease', 'county', 'year', 'sex', 'cases', 'population','lower_95_ci', 'upper_95_ci','rate']

print("Update column names:", infec_df.columns)
infec_df.dtypes
print("Number of Unique diseases:", len(infec_df.disease.unique()))

print("Number of counties we have in the dataset are:", len(infec_df.county.unique()))
# checking null values count in each column

for i in list(infec_df.columns):

    print("for column",i,"the number of NA values are:", sum(infec_df[i].isnull()))

    #print(infec_df.i.count())
#replacing the NAN with 0 in the cases column

infec_df.cases.fillna(0, inplace=True)
year_sex_df  = pd.DataFrame(infec_df.groupby(['year','sex']).sum()[['cases']]) #group the data w.r.t to year and sex and summing the total cases for those groups

year_sex_df.reset_index(level=['year','sex'], inplace = True) 

year_sex_df['year'] = year_sex_df.year.astype('category') #converting the year to category for better xticks


fig, ax = plt.subplots(figsize=(8,6))

for label, df in year_sex_df.groupby('sex'):

    df.plot(x ='year',y = 'cases',ax=ax, label=label)

plt.legend()
diseas_df = pd.DataFrame(infec_df.groupby('disease').sum()[['cases']])

diseas_df.reset_index(level = 'disease', inplace = True)

diseas_df = diseas_df.sort_values(by = 'cases', ascending=False)
fid, ax = plt.subplots(figsize = (10,8))

disease_list = list(diseas_df['disease'])

cases_list = list(diseas_df['cases'])

ax.bar(disease_list,cases_list)

plt.xticks(disease_list, rotation = 'vertical')
fid, ax = plt.subplots(figsize = (10,8))

disease_list = list(diseas_df['disease'])

cases_list = list(diseas_df['cases'])

ax.bar(disease_list[:10],cases_list[:10])

plt.xticks(disease_list[:10], rotation = 90)
max_diease_df = pd.DataFrame(infec_df.groupby('disease').max()[['cases']]).reset_index(level = 'disease')

def get_county(row):

    """function that returns the county name with the mathcing disease and number of cases"""

    county = infec_df[(infec_df['disease']==row['disease']) & (infec_df['cases']== row['cases'])]['county'].iloc[0]

    return county

max_diease_df['county'] = max_diease_df.apply(lambda x: get_county(x), axis = 1)
max_diease_df = max_diease_df.sort_values('cases', ascending = False)

max_diease_df.county.value_counts()