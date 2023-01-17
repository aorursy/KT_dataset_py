# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
colleges = pd.read_csv('/kaggle/input/college-salaries/salaries-by-college-type.csv')
colleges
colleges.isnull().sum()
colleges=colleges.drop(['Mid-Career 10th Percentile Salary', 'Mid-Career 90th Percentile Salary'], axis=1)
colleges
def toInt(df, column):
    sal_list = []
    for i in column: 
        a = i.replace('$', '')
        b = float(a.replace(',', ''))
        sal_list.append(b)
    new = pd.DataFrame(sal_list, columns = [column.name])
    df.update(new)
toInt(colleges,colleges['Starting Median Salary'])
toInt(colleges,colleges['Mid-Career Median Salary'])
toInt(colleges,colleges['Mid-Career 25th Percentile Salary'])
toInt(colleges,colleges['Mid-Career 75th Percentile Salary'])
colleges
colleges.sort_values(by='Mid-Career Median Salary', ascending=False).head(10)
colleges.sort_values(by='Starting Median Salary', ascending=False).head(10)
colleges.sort_values(by='Mid-Career 25th Percentile Salary', ascending=False).head(10)
colleges.sort_values(by='Mid-Career 75th Percentile Salary', ascending=False).head(10)
colleges.groupby('School Type')['School Type'].count()
engineering_list = []
party_list = []
lib_list = []
ivy_list = []
state_list = []
for sal, typ in zip(colleges['Starting Median Salary'], colleges['School Type']):
    sal = int(sal)
    if typ == 'Engineering': 
        engineering_list.append(sal)
    elif typ == 'Party':
        party_list.append(sal)
    elif typ == 'Liberal Arts':
        lib_list.append(sal)
    elif typ == 'Ivy League':
        ivy_list.append(sal)
    else: 
        state_list.append(sal)
print('Average Engineering Starting Salary: ', round(sum(engineering_list)/len(engineering_list), 2))
print('Average Party Starting Salary: ', round(sum(party_list)/len(party_list), 2))
print('Average Liberal Arts Starting Salary: ', round(sum(lib_list)/len(lib_list),2))
print('Average Ivy League Starting Salary: ', round(sum(ivy_list)/len(ivy_list),2))
print('Average State Starting Salary: ', round(sum(state_list)/len(state_list),2 ))
    
    
plt.figure(figsize=(10, 10))
data = [colleges['Starting Median Salary'], colleges['Mid-Career Median Salary'], 
        colleges['Mid-Career 25th Percentile Salary'], colleges['Mid-Career 75th Percentile Salary']]
labels = ['Starting Median Salary', 'Mid-Career Median Salary', 'Mid-Career 25th Percentile Salary', 
         'Mid-Career 75th Percentile Salary']
plt.xticks(rotation=45)
plt.boxplot(data,labels=labels)
# identifying the outliers for starting median salary
colleges[colleges['Starting Median Salary'] > 57000]
# identifying lower outlier for mid-career median salary
colleges[colleges['Mid-Career Median Salary'] < 50000]
# identifying lower outlier for mid-career 25th percentile salary
colleges[colleges['Mid-Career 25th Percentile Salary']<35000]
# identifying lower outlier for mid-career 75th percentile salary
colleges[colleges['Mid-Career 75th Percentile Salary']<70000]
majors = pd.read_csv('/kaggle/input/college-salaries/degrees-that-pay-back.csv')
majors
toInt(majors, majors['Starting Median Salary'])
toInt(majors, majors['Mid-Career Median Salary'])
toInt(majors, majors['Mid-Career 90th Percentile Salary'])
plt.figure(figsize=(10, 10))
data = [majors['Starting Median Salary'], majors['Mid-Career Median Salary'],
        majors['Mid-Career 90th Percentile Salary']]
plt.boxplot(data)
majors[majors['Starting Median Salary'] > 70000]
majors.sort_values(by='Starting Median Salary', ascending=False)
majors.sort_values(by='Mid-Career Median Salary', ascending=False)
majors.sort_values(by='Mid-Career 90th Percentile Salary', ascending=False)
majors.sort_values(by='Percent change from Starting to Mid-Career Salary', ascending=False)
regions = pd.read_csv('/kaggle/input/college-salaries/salaries-by-region.csv')
toInt(regions, regions['Starting Median Salary'])
regions=regions.drop(['Mid-Career 10th Percentile Salary', 'Mid-Career 90th Percentile Salary'], axis=1)
regions
regions.groupby('Region').Region.count()
regions.sort_values(by='Region')
cali_list = []
midwest_list = []
northeast_list = []
south_list = []
west_list = []
for sal, reg in zip(regions['Starting Median Salary'], regions['Region']): 
    if reg == 'California': 
        cali_list.append(sal)
    elif reg == 'Midwestern': 
        midwest_list.append(sal)
    elif reg == 'Northeastern': 
        northeast_list.append(sal)
    elif reg == 'Southern': 
        south_list.append(sal)
    else: 
        west_list.append(sal)
print('Average California Starting Median Salary: ', round(sum(cali_list)/len(cali_list), 2))
print('Average Midwestern Starting Median Salary: ', round(sum(midwest_list)/len(midwest_list), 2))
print('Average Northeastern Starting Median Salary: ', round(sum(northeast_list)/len(northeast_list), 2))
print('Average Southern Starting Median Salary: ', round(sum(south_list)/len(south_list), 2))
print('Average Western Starting Median Salary: ', round(sum(west_list)/len(west_list), 2))
data = [regions[regions.Region == 'California']['Starting Median Salary'], regions[regions.Region == 'Midwestern']['Starting Median Salary'], 
        regions[regions.Region=='Northeastern']['Starting Median Salary'], regions[regions.Region == 'Southern']['Starting Median Salary'], 
        regions[regions.Region=='Western']['Starting Median Salary']]
plt.figure(figsize=(10,7))
plt.title('Starting Median Salary Per Region')
labels=['California', 'Midwestern', 'Northeastern', 'Southern', 'Western']
boxplot = plt.boxplot(data, labels=labels)
boxplot.annotate()
for reg, sal in zip(regions.Region, regions['Starting Median Salary']): 
    if reg=='California': 
        if sal > 65000:
            print('C: ', sal)
    if reg == 'Midwestern': 
        if sal > 55000: 
            print('MW: ', sal)
    if reg == 'Northeastern': 
        if sal > 70000: 
            print('NE: ', sal)
    if reg == 'Southern': 
        if sal > 55000: 
            print('S: ', sal)
    if reg == 'Western': 
        if sal > 55000: 
            print('W: ', sal)

        
print('California Outliers: ')
print('\t', regions[regions['Starting Median Salary']==70400]['School Name'])
print('\t', regions[regions['Starting Median Salary']==75500]['School Name'])
print('\t', regions[regions['Starting Median Salary']==71800]['School Name'])

print('\nNorthwestern Outliers: ')
print('\t', regions[regions['Starting Median Salary']==72200]['School Name'])

print('\nMidwestern Outliers: ')
print('\t', regions[regions['Starting Median Salary']==56300]['School Name'])
print('\t', regions[regions['Starting Median Salary']==56000]['School Name'])
print('\t', regions[regions['Starting Median Salary']==56200]['School Name'])
print('\t', regions[regions['Starting Median Salary']==57100]['School Name'])
print('\t', regions[regions['Starting Median Salary']==55800]['School Name'])

print('\nSouthern Outliers: ')
print('\t', regions[regions['Starting Median Salary']==64000]['School Name'])
print('\t', regions[regions['Starting Median Salary']==58900]['School Name'])
print('\t', regions[regions['Starting Median Salary']==58300]['School Name'])

print('\nWestern Outliers: ')
print('\t', regions[regions['Starting Median Salary']==58100]['School Name'])