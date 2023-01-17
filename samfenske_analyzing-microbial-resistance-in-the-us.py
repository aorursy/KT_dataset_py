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
isolate=pd.read_csv('/kaggle/input/isolate/IsolateData.csv')
isolate
isolate['Data Year'].value_counts()
isolate['Data Year'].dtype
for year in range(len(isolate)):
    if isolate.iloc[year,4]=='2018*':
        isolate.iloc[year,4]='2018'
    if isolate.iloc[year,4]=='2019*':
        isolate.iloc[year,4]='2019'
isolate=isolate.astype({'Data Year': int})
isolate['Data Year'].dtype
isolate['Data Year'].value_counts()
#boolan frame showing true/false based on whether the Data Year is 2019
separate=isolate['Data Year'].isin([2019])
#putting seperate in brackets applies boolean condition, the - means to not include any of the 2019 data
isolate_pre_2019=isolate[-separate]
isolate_pre_2019['Data Year'].value_counts()
salmonella=pd.read_csv('/kaggle/input/salmonella2/IsolateData (2).csv')
for year in range(len(salmonella)):
    if salmonella.iloc[year,4]=='2018*':
        salmonella.iloc[year,4]='2018'
salmonella=salmonella.astype({'Data Year': int})
clean=salmonella[['Genus','Species','Serotype','Data Year','Region Name','Age Group','Specimen Source','Resistance Pattern','Resistance Determinants','Predictive Resistance Pattern','NCBI Accession Number','WGS ID']]
clean['NCBI Accession Number'].value_counts()
#there are 1840 entries, about half have accession numbers so let's not use it
clean['WGS ID'].value_counts()
#not enough data, get rid of this too
clean=salmonella[['Genus','Species','Serotype','Data Year','Region Name','Age Group','Specimen Source','Resistance Pattern','Resistance Determinants','Predictive Resistance Pattern']]
salmonella=clean
def clean_data(dataset):
    for year in range(len(dataset)):
        if dataset.iloc[year,4]=='2018*':
            dataset.iloc[year,4]='2018'
    dataset=dataset.astype({'Data Year': int})
    dataset=dataset[['Genus','Species','Serotype','Data Year','Region Name','Age Group','Specimen Source','Resistance Pattern','Resistance Determinants','Predictive Resistance Pattern']]
    return dataset
def fix_region(dataset):
    for row in range(len(dataset)):
        if dataset.iloc[row,4]=='Region 1':
            dataset.iloc[row,4]='Boston'
        if dataset.iloc[row,4]=='Region 2':
            dataset.iloc[row,4]='New York'
        if dataset.iloc[row,4]=='Region 3':
            dataset.iloc[row,4]='Philadelphia'
        if dataset.iloc[row,4]=='Region 4':
            dataset.iloc[row,4]='Atlanta'
        if dataset.iloc[row,4]=='Region 5':
            dataset.iloc[row,4]='Chicago'
        if dataset.iloc[row,4]=='Region 6':
            dataset.iloc[row,4]='Dallas'
        if dataset.iloc[row,4]=='Region 7':
            dataset.iloc[row,4]='Kansas City'
        if dataset.iloc[row,4]=='Region 8':
            dataset.iloc[row,4]='Denver'
        if dataset.iloc[row,4]=='Region 9':
            dataset.iloc[row,4]='San Francisco'
        if dataset.iloc[row,4]=='Region 10':
            dataset.iloc[row,4]='Seattle'
    return dataset
fix_region(salmonella)
#To make things easier, we want to be able to iterate through every region rather
#than manually type out a new script for every region (10x the typing).
#The following code creates a series of region names, of which we can iterate through
#in the next block.
regions=salmonella['Region Name'].value_counts()
regions.index
#create blank dataframe that we may add to as we go
region_data=pd.DataFrame()
#iterate through every region via the regions series created above
for region in regions.index:
    #total data points and total resistant points need to be set to zero before we 
    #add to them
    r1=0
    t1=0
    #iterate through every row in the actaul dataset 
    for row in range(len(salmonella)):
        #find data for our given region
        if salmonella.iloc[row,4]==region:
            #add one to total data points
            t1=t1+1
            #check if data shows resistance pattern
            if salmonella.iloc[row,7]!='No resistance detected':
                #add one resistant point if shows resistance, here by saying it's not
                #non-resistant
                r1=r1+1
    #This part may be tricky for beginners-while we are still in the for loop iterating
    #over a given region, we want to add data to the new dataframe
    d = {'Region': [region], 'Resistance Detected': [r1],'Total Points':[t1],'Percent Resistant':[r1/t1]}
    df = pd.DataFrame(data=d)
    region_data=region_data.append(df)
#all the indices are zero, so let's reset the index and get rid of the column that
#set all the indices to zero
region_data=region_data.reset_index()
region_data=region_data.drop(columns='index')
#sort by percent resiatnt to see which regions had the least resistant strains
salmonella_2018=region_data.sort_values(by='Percent Resistant',ascending=False)
def region_stats(dataset):
    regions=dataset['Region Name'].value_counts()
    region_data=pd.DataFrame()
    for region in regions.index:
        r1=0
        t1=0
        for row in range(len(dataset)):
            if dataset.iloc[row,4]==region:
                t1=t1+1
                if dataset.iloc[row,7]!='No resistance detected':
                    r1=r1+1
        d = {'Region': [region], 'Resistance Detected': [r1],'Total Points':[t1],'Percent Resistant':[r1/t1]}
        df = pd.DataFrame(data=d)
        region_data=region_data.append(df)
    region_data=region_data.reset_index()
    region_data=region_data.drop(columns='index')
    return region_data
total=0
for row in range(len(region_data)):
    total+=region_data.iloc[row,2]
total
ecoli=pd.read_csv('/kaggle/input/ecolio157/IsolateData (3).csv')
ecoli=clean_data(ecoli)
ecoli=fix_region(ecoli)
ecoli_2018=region_stats(ecoli).sort_values(by='Percent Resistant', ascending=False)
salmonella_2018
ecoli_2018