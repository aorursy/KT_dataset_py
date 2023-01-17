# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt #to plot graphs

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.ExcelFile('../input/obes-phys-acti-diet-eng-2017-tab.xlsx')
#to identify all the different sheets available

df.sheet_names
#Table 7 has the relevant data and we also remove all unnecessary info

data=df.parse(u'Table 7', skiprows=108, skipfooter=53)
data.head()
#rename all the relevant columns we need

data=data.rename(columns={'London':'regions',53267:'all',17720:'male',35533:'female'})
#format the data so it only contains relevant columns

data=data[['regions','all','male','female']]
#set the index to 'regions'

data.set_index('regions')
#most obese region as of 2015/16

data[data['all']==data['all'].max()]
#least obese region

data[data['all']==data['all'].min()]
#prepare the data for graphing

data_gen=data[['regions','male','female']]
#stacked bar graph showing the number of obesity admissions by gender

#females have a much higher admission rate than males

data_gen.plot.bar(figsize=(12,6),stacked=True)
#time to examine gender and obesity admission from 2005/6 till 2015/16

#parse new sheet

time=df.parse(u'Table 1', skiprows=8, skipfooter=14)
#rename and select relevant columns and set Year as the index

time.rename(columns={'Unnamed: 0':'Year', 'Unnamed: 2': 'all','Unnamed: 3':'male', 'Unnamed: 4':'female'}, inplace=True)

time=time[['Year','male','female']]

time.set_index('Year', inplace=True)
time
#time to plot the graph

time.plot.line()
#time for some more data analysis

#create new column for the total of male and female per year

time['total']=time['male']+time['female']
#which year had the most obese people

time[time['total']==time['total'].max()]
#which year had the least obese people

time[time['total']==time['total'].min()]