# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/h1b_kaggle.csv")
data.shape
data.head()
data.drop(['Unnamed: 0'],inplace = True,axis = 1)
data.head(1)
data.info()
data.isnull().any()
data.isnull().sum().sort_values(ascending = False)
data['EMPLOYER_NAME'].value_counts()[:10]
import matplotlib.pyplot as plt
import seaborn as sns
emp = data['EMPLOYER_NAME'].value_counts()[:10]

sns.barplot(x= emp.values, y = emp.index)
soc = data['SOC_NAME'].value_counts()[:10]

sns.barplot(x= soc.values, y = soc.index)
title = data['JOB_TITLE'].value_counts()[:10]

sns.barplot(x= title.values, y = title.index)
year = data['YEAR'].value_counts()[:10]

sns.barplot(x= year.index, y = year.values,saturation = 0.2)
## WOW..!! approx. 150K is the Average PREVAILING_WAGE of all the petitions 

data['PREVAILING_WAGE'].sum()/data.shape[0]

data['WORKSITE'].value_counts()[:20]
pd.options.display.float_format = '{:,.2f}'.format

data.groupby('WORKSITE').agg({'PREVAILING_WAGE':'mean'}).sort_values(by = ['PREVAILING_WAGE'],ascending = False)
data[data['WORKSITE'] == 'WASHINGTON, NA']
data['CASE_STATUS'].value_counts()

sns.countplot(data['FULL_TIME_POSITION'])
subset = data[(data['EMPLOYER_NAME'] == 'INFOSYS LIMITED') | (data['EMPLOYER_NAME'] == 'TATA CONSULTANCY SERVICES LIMITED') | (data['EMPLOYER_NAME'] =='WIPRO LIMITED')] 
subset.groupby(['EMPLOYER_NAME','YEAR']).count()['CASE_STATUS']

fig, ax = plt.subplots(figsize = (10,6))



sns.countplot(palette = 'gist_ncar',x= 'YEAR',hue = 'EMPLOYER_NAME',data = subset)
subset.groupby(['YEAR','EMPLOYER_NAME']).agg({'PREVAILING_WAGE':'mean'}).reset_index()
fig, ax = plt.subplots(figsize = (10,6))

sns.barplot(palette = 'YlGnBu_r',x= 'YEAR',hue = 'EMPLOYER_NAME',y='PREVAILING_WAGE', data = subset.groupby(['YEAR','EMPLOYER_NAME']).agg({'PREVAILING_WAGE':'mean'}).reset_index())
fig, ax = plt.subplots(figsize = (10,6))

sns.barplot(x= 'EMPLOYER_NAME',hue = 'SOC_NAME',y = 'CASE_STATUS', data = subset.groupby(['SOC_NAME','EMPLOYER_NAME']).count()['CASE_STATUS'].sort_values(ascending = False)[:20].reset_index())