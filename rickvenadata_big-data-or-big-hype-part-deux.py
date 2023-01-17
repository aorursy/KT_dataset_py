# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

color = sns.color_palette()

%matplotlib inline
# take a slice of the full survey data with just the fields we are interested in

answers = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)

answers = answers[['WorkDatasetSize','CurrentEmployerType','EmployerIndustry','EmployerSize']]

answers = answers.dropna(subset = ['WorkDatasetSize'])

answers.head(10)
# set up groups for plotting data set sizes

ltmb1 = answers.groupby('WorkDatasetSize').get_group('<1MB')

gb100 = answers.groupby('WorkDatasetSize').get_group('100GB')

mb100 = answers.groupby('WorkDatasetSize').get_group('100MB')

tb100 = answers.groupby('WorkDatasetSize').get_group('100TB')

pb100 = answers.groupby('WorkDatasetSize').get_group('100PB')

mb10 = answers.groupby('WorkDatasetSize').get_group('10MB')

gb10 = answers.groupby('WorkDatasetSize').get_group('10GB')

tb10 = answers.groupby('WorkDatasetSize').get_group('10TB')

pb10 = answers.groupby('WorkDatasetSize').get_group('10PB')

mb1 = answers.groupby('WorkDatasetSize').get_group('1MB')

gb1 = answers.groupby('WorkDatasetSize').get_group('1GB')

tb1 = answers.groupby('WorkDatasetSize').get_group('1TB')

pb1 = answers.groupby('WorkDatasetSize').get_group('1PB')

eb1 = answers.groupby('WorkDatasetSize').get_group('1EB')

gteb1 = answers.groupby('WorkDatasetSize').get_group('>1EB')



size_groups = [ltmb1,mb1,mb10,mb100,gb1,gb10,gb100,tb1,tb10,tb100,pb1,pb10,pb100,eb1,gteb1]

dsizes = pd.concat(size_groups).reset_index(drop=True)
plt.figure(figsize=(12,9))

sns.set(font_scale=2)

sns.countplot(y=dsizes['WorkDatasetSize'],orient='h', data=dsizes)

plt.title('Dataset Sizes Used By Data Scientists', fontsize=16)

plt.xlabel('Count', fontsize=16)

plt.ylabel('Size', fontsize=16)

plt.show()
# Rarity of large datasets in use by data scientists surveyed

size_groups = [tb100,pb1,pb10,pb100,eb1,gteb1]

dsizes = pd.concat(size_groups).reset_index(drop=True)

plt.figure(figsize=(12,9))

sns.set(font_scale=2)

sns.countplot(y=dsizes['WorkDatasetSize'],orient='h', data=dsizes)

plt.title('"Big Data" Dataset Sizes Not Typical!', fontsize=16)

plt.xlabel('Count', fontsize=16)

plt.ylabel('Size', fontsize=16)

plt.show()
# check counts for industries of employers

empindustries = answers['EmployerIndustry'].dropna()

empindustries_count = empindustries.value_counts()

empindustries_count
# set up employer industry groups for plotting

empind = answers.dropna(subset = ['EmployerIndustry'])

empind = empind.loc[~empind['EmployerIndustry'].isin(["Mix of fields","Other"])]

empind = empind.loc[empind['WorkDatasetSize'].isin(["100GB","1TB","10TB","100TB","1PB"])]



tech = empind.groupby('EmployerIndustry').get_group('Technology')

edu = empind.groupby('EmployerIndustry').get_group('Academic')

fin = empind.groupby('EmployerIndustry').get_group('Financial')

web = empind.groupby('EmployerIndustry').get_group('Internet-based')

gov = empind.groupby('EmployerIndustry').get_group('Government')

crm = empind.groupby('EmployerIndustry').get_group('CRM/Marketing')

telco = empind.groupby('EmployerIndustry').get_group('Telecommunications')

mfg = empind.groupby('EmployerIndustry').get_group('Manufacturing')

ins = empind.groupby('EmployerIndustry').get_group('Insurance')

ret = empind.groupby('EmployerIndustry').get_group('Retail')

rx = empind.groupby('EmployerIndustry').get_group('Pharmaceutical')

org = empind.groupby('EmployerIndustry').get_group('Non-profit')

fun = empind.groupby('EmployerIndustry').get_group('Hospitality/Entertainment/Sports')

sec = empind.groupby('EmployerIndustry').get_group('Military/Security')



ind_groups = [tech,edu,fin,web,gov]

industries = pd.concat(ind_groups).reset_index(drop=True)
# Plot large datasets in use, by top 5 industries

plt.figure(figsize=(20,15))

sns.set(font_scale=2)

sns.countplot(y=industries['EmployerIndustry'],orient='h', data=industries, hue='WorkDatasetSize')

plt.title('Large Datasets Used By Top 5 Industries Surveyed', fontsize=16)

plt.xlabel('Responses', fontsize=16)

plt.ylabel('Dataset Size', fontsize=16)

plt.legend(loc=(1.04,0))

plt.show()
# Industries grouped by dataset size

empind = answers.dropna(subset = ['EmployerIndustry'])

industries = ["Technology","Academic","Financial","Internet-based","Government"]

empind = empind.loc[empind['EmployerIndustry'].isin(industries)]



ltmb1 = empind.groupby('WorkDatasetSize').get_group('<1MB')

gb100 = empind.groupby('WorkDatasetSize').get_group('100GB')

mb100 = empind.groupby('WorkDatasetSize').get_group('100MB')

tb100 = empind.groupby('WorkDatasetSize').get_group('100TB')

pb100 = empind.groupby('WorkDatasetSize').get_group('100PB')

mb10 = empind.groupby('WorkDatasetSize').get_group('10MB')

gb10 = empind.groupby('WorkDatasetSize').get_group('10GB')

tb10 = empind.groupby('WorkDatasetSize').get_group('10TB')

pb10 = empind.groupby('WorkDatasetSize').get_group('10PB')

mb1 = empind.groupby('WorkDatasetSize').get_group('1MB')

gb1 = empind.groupby('WorkDatasetSize').get_group('1GB')

tb1 = empind.groupby('WorkDatasetSize').get_group('1TB')

pb1 = empind.groupby('WorkDatasetSize').get_group('1PB')

eb1 = empind.groupby('WorkDatasetSize').get_group('1EB')

gteb1 = empind.groupby('WorkDatasetSize').get_group('>1EB')



size_groups = [mb10,mb100,gb1,gb10,gb100,tb1,tb10,tb100]

dsizes = pd.concat(size_groups).reset_index(drop=True)
plt.figure(figsize=(20,15))

sns.set(font_scale=2)

sns.countplot(y=dsizes['WorkDatasetSize'],orient='h', data=dsizes, hue='EmployerIndustry')

plt.title('Moderately Sized Datasets In Use By Top 5 Industries', fontsize=16)

plt.xlabel('Responses', fontsize=16)

plt.ylabel('Dataset Size', fontsize=16)

plt.legend(loc=(1.04,0))

plt.show()
# check employer sizes

empsizes = answers['EmployerSize'].dropna()

empsizes_count = empsizes.value_counts()

empsizes_count
# set up grouping by employer size

orgsize = answers.dropna(subset = ['EmployerSize'])

orgsize = orgsize.loc[~orgsize['EmployerSize'].isin(["I don't know","I prefer not to answer"])]

#orgsize = orgsize.loc[orgsize['WorkDatasetSize'].isin(["100GB","1TB","10TB","100TB","1PB"])]

#orgsize = orgsize.loc[~orgsize['WorkDatasetSize'].isin(["<1MB","1MB","10MB","100MB","1GB"])]

orgsize = orgsize.loc[orgsize['WorkDatasetSize'].isin(["1TB","10TB","100TB","1PB","10PB","100PB"])]



s1 = orgsize.groupby('EmployerSize').get_group('Fewer than 10 employees')

s10 = orgsize.groupby('EmployerSize').get_group('10 to 19 employees')

s20 = orgsize.groupby('EmployerSize').get_group('20 to 99 employees')

s100 = orgsize.groupby('EmployerSize').get_group('100 to 499 employees')

s500 = orgsize.groupby('EmployerSize').get_group('500 to 999 employees')

s1000 = orgsize.groupby('EmployerSize').get_group('1,000 to 4,999 employees')

s5000 = orgsize.groupby('EmployerSize').get_group('5,000 to 9,999 employees')

s10000 = orgsize.groupby('EmployerSize').get_group('10,000 or more employees')



orgsize_groups = [s1,s10,s20,s100,s500,s1000,s5000,s10000]

orgsizes = pd.concat(orgsize_groups).reset_index(drop=True)
plt.figure(figsize=(20,15))

sns.set(font_scale=2)

sns.countplot(y=orgsizes['EmployerSize'],orient='h', data=orgsizes, hue='WorkDatasetSize')

plt.title('Large Datasets by Employer Size', fontsize=16)

plt.xlabel('Responses', fontsize=16)

plt.ylabel('Dataset Size', fontsize=16)

plt.legend(loc=(1.04,0))

plt.show()
orgsize = answers.dropna(subset = ['EmployerSize'])

orgsize = orgsize.loc[~orgsize['EmployerSize'].isin(["I don't know","I prefer not to answer"])]

orgsize = orgsize.loc[orgsize['WorkDatasetSize'].isin(["10MB","100MB","1GB","10GB"])]



s1 = orgsize.groupby('EmployerSize').get_group('Fewer than 10 employees')

s10 = orgsize.groupby('EmployerSize').get_group('10 to 19 employees')

s20 = orgsize.groupby('EmployerSize').get_group('20 to 99 employees')

s100 = orgsize.groupby('EmployerSize').get_group('100 to 499 employees')

s500 = orgsize.groupby('EmployerSize').get_group('500 to 999 employees')

s1000 = orgsize.groupby('EmployerSize').get_group('1,000 to 4,999 employees')

s5000 = orgsize.groupby('EmployerSize').get_group('5,000 to 9,999 employees')

s10000 = orgsize.groupby('EmployerSize').get_group('10,000 or more employees')



orgsize_groups = [s1,s10,s20,s100,s500,s1000,s5000,s10000]

orgsizes = pd.concat(orgsize_groups).reset_index(drop=True)
# Count and plot 

plt.figure(figsize=(20,15))

sns.set(font_scale=2)

sns.countplot(y=orgsizes['EmployerSize'],orient='h', data=orgsizes, hue='WorkDatasetSize')

plt.title('Medium Datasets by Employer Size', fontsize=16)

plt.xlabel('Responses', fontsize=16)

plt.ylabel('Dataset Size', fontsize=16)

plt.legend(loc=(1.04,0))

plt.show()
# groups

empsize = answers.dropna(subset = ['EmployerSize'])

empsize = empsize.loc[~empsize['EmployerSize'].isin(["I don't know","I prefer not to answer"])]



ltmb1 = empsize.groupby('WorkDatasetSize').get_group('<1MB')

gb100 = empsize.groupby('WorkDatasetSize').get_group('100GB')

mb100 = empsize.groupby('WorkDatasetSize').get_group('100MB')

tb100 = empsize.groupby('WorkDatasetSize').get_group('100TB')

pb100 = empsize.groupby('WorkDatasetSize').get_group('100PB')

mb10 = empsize.groupby('WorkDatasetSize').get_group('10MB')

gb10 = empsize.groupby('WorkDatasetSize').get_group('10GB')

tb10 = empsize.groupby('WorkDatasetSize').get_group('10TB')

pb10 = empsize.groupby('WorkDatasetSize').get_group('10PB')

mb1 = empsize.groupby('WorkDatasetSize').get_group('1MB')

gb1 = empsize.groupby('WorkDatasetSize').get_group('1GB')

tb1 = empsize.groupby('WorkDatasetSize').get_group('1TB')

pb1 = empsize.groupby('WorkDatasetSize').get_group('1PB')

eb1 = empsize.groupby('WorkDatasetSize').get_group('1EB')

gteb1 = empsize.groupby('WorkDatasetSize').get_group('>1EB')



size_groups = [mb10,mb100,gb1,gb10,gb100,tb1,tb10,tb100,pb1]

dsizes = pd.concat(size_groups).reset_index(drop=True)
# Dataset sizes across employer sizes 

plt.figure(figsize=(20,30))

sns.set(font_scale=2)

sns.countplot(y=dsizes['WorkDatasetSize'],orient='h', data=dsizes, hue='EmployerSize')

plt.title('Dataset Sizes', fontsize=16)

plt.xlabel('Count', fontsize=16)

plt.ylabel('Size', fontsize=16)

plt.legend(loc=(1.04,0))

plt.show()
# analyze by employer type

emptypelist = ["Employed by professional services/consulting firm",

               "Employed by a company that performs advanced analytics",

               "Employed by a company that doesn't perform advanced analytics",

               "Employed by college or university",

               "Employed by company that makes advanced analytic software",

               "Self-employed",

               "Employed by government",

               "Employed by non-profit or NGO"]

df1 = answers.loc[answers['CurrentEmployerType'].isin(emptypelist)]

df1.head(10)
# group by dataset sizes for the employer types

ltmb1 = df1.groupby('WorkDatasetSize').get_group('<1MB')

gb100 = df1.groupby('WorkDatasetSize').get_group('100GB')

mb100 = df1.groupby('WorkDatasetSize').get_group('100MB')

tb100 = df1.groupby('WorkDatasetSize').get_group('100TB')

pb100 = df1.groupby('WorkDatasetSize').get_group('100PB')

mb10 = df1.groupby('WorkDatasetSize').get_group('10MB')

gb10 = df1.groupby('WorkDatasetSize').get_group('10GB')

tb10 = df1.groupby('WorkDatasetSize').get_group('10TB')

pb10 = df1.groupby('WorkDatasetSize').get_group('10PB')

mb1 = df1.groupby('WorkDatasetSize').get_group('1MB')

gb1 = df1.groupby('WorkDatasetSize').get_group('1GB')

tb1 = df1.groupby('WorkDatasetSize').get_group('1TB')

pb1 = df1.groupby('WorkDatasetSize').get_group('1PB')

eb1 = df1.groupby('WorkDatasetSize').get_group('1EB')

gteb1 = df1.groupby('WorkDatasetSize').get_group('>1EB')



size_groups = [mb10,mb100,gb1,gb10,gb100,tb1,tb10,tb100]

dsizes = pd.concat(size_groups).reset_index(drop=True)
plt.figure(figsize=(20,30))

sns.set(font_scale=2)

sns.countplot(y=dsizes['WorkDatasetSize'],orient='h', data=dsizes, hue='CurrentEmployerType')

plt.title('Dataset Sizes By Employer Type', fontsize=16)



plt.xlabel('Count', fontsize=16)

plt.ylabel('Size', fontsize=16)

plt.legend(loc=(1.04,0))

plt.show()