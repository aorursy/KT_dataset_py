# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading excel files

import pandas as pd

Jan = pd.read_excel('../input/job-market-reports-sept19jan20/Jobs Sep 2019.xlsx')

Jan.head(2)
#get other files

Dec=pd.read_excel('../input/job-market-reports-sept19jan20/Jobs Dec 2019.xlsx')

Nov=pd.read_excel('../input/job-market-reports-sept19jan20/Jobs Nov 2019.xlsx')

Oct=pd.read_excel('../input/job-market-reports-sept19jan20/Jobs oct 2019.xlsx')

Sep=pd.read_excel('../input/job-market-reports-sept19jan20/Jobs Sep 2019.xlsx')

Aug=pd.read_excel('../input/job-market-reports-sept19jan20/Jobs August 2019.xlsx')
Aug.describe()
Aug.info()
Aug['Metro'].unique()
Aug['Dimension Type'].unique()
Aug['Dimension'].unique()
Aug['Measure'].unique()
Aug['Value'].unique()
Sep.head(2)
#Expand on October

Okt=Oct[['Metro', 'Dimension', 'Value']]

Okt.head(2)
import numpy as np

#label encode the metro column

from sklearn.preprocessing import LabelEncoder

Okt['Metro']= Okt['Metro'].astype('category')

Okt['Metro Catgs']=Okt['Metro'].cat.codes

Okt.head(3)
import seaborn as sns

sns.countplot(Okt['Metro Catgs'])
#group by categories to better understand the df data

G1= Okt.groupby('Metro Catgs')

G1.head().head()
Okt.astype(str)
Okt['Value'] = Okt['Value'].str.replace('$', '')
Okt['Value']=Okt['Value'].str.replace(',', '').astype(float)
Okt['Value']=Okt['Value'].fillna(0)
#Verify changes

Okt['Value'].head()
Okt['Metro Catgs'].astype(int)
#correlate

Okt.corr()



##there seem to be no correlation between these columns as they are.
#Visualize it

#Linear-graph for relationships

Okt.plot.scatter("Value", "Metro Catgs", color='green')
Okt.corr().plot.bar()
#The original columns have mixed information.

list(Nov['Dimension'].unique())
##extract job titles in the column

#First, find the location of the desired data

#Nov['Dimension'].head(50) ##job titles start at 42

Nov['Dimension'].tail(80) ##job titles end at 4837
N=Nov[42:4838] #to include the last row go 1 over

N.head(3)
#I want to examine the national data with job titles

National=N[(N['Metro']=='National')]

National =National[(National['Dimension Type']=='Job Title')]

National.head(3)
#Job opening titles with their base salary

Jobs=National[['Dimension','Value']]
#replace NaNs with 0s

Jobs['Value']=Jobs['Value'].fillna(0)
#clean data 

Jobs['Value'] = Jobs['Value'].str.replace('$', '')

Jobs['Value'] = Jobs['Value'].str.replace(',', '').astype(float)
Jobs.head(2)
#Label encode job titles

from sklearn.preprocessing import LabelEncoder

Jobs['Dimension']=Jobs['Dimension'].astype('category')

Jobs['Titles catgs']=Jobs['Dimension'].cat.codes

Jobs.head(2)
Jobs.describe()
Jobs.median()
Jobs.corr()

#There does not seem to be a correlation between the columns
#Job title-to-pay distribution

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

plt.scatter(Jobs['Value'], Jobs['Titles catgs'], color='purple')

plt.show()
Jobs.corr().plot.bar()
#A df of monthly US Job Opening and Median pay across all dfs: Aug 2019 - Jan 2020

months = {'Jan20': Jan[0:2], 'Dec19': Dec[0:2], 'Nov19': Nov[0:2], 'Oct19': Oct[0:2], 'Sep19': Sep[0:2], 'Aug19': Aug[0:2]}



Monthly = pd.concat(months)

JobsNpay= Monthly.drop(['Metro', 'Dimension Type', 'Month', 'Dimension', 'YoY'], axis=1)

JobsNpay
#clean data 

JobsNpay['Value'] = JobsNpay['Value'].str.replace('$', '')

JobsNpay['Value'] = JobsNpay['Value'].str.replace(',', '').astype(float)
#Compare monthly US Job Opening and Median pay across all dfs: Aug 2019 - Jan 2020

Monthly1 = pd.concat([Jan[0:2], Dec[0:2], Nov[0:2], Oct[0:2], Sep[0:2], Aug[0:2]])

Monthly1
JobsNpay1= Monthly1.drop(['Metro', 'Dimension Type', 'Month', 'Dimension', 'YoY'], axis=1)

JobsNpay1
#clean data 

JobsNpay1['Value'] = JobsNpay1['Value'].str.replace('$', '')

JobsNpay1['Value'] = JobsNpay1['Value'].str.replace(',', '').astype(float)
sns.distplot(JobsNpay1['Value'], color='red', bins=3)

#Every other row represent something different.

#Also, the numbers could be normalize to 'even' the data
p=JobsNpay1.iloc[[0,2,4,6,8,10]]

p
USJobs=p.dropna()
USJobs.median()
USJobs.hist()
q=JobsNpay1.iloc[[1,3,5,7,9,11]]

Salary=q.dropna()

Salary
Salary.describe()
import matplotlib.pyplot as plotter 

# Months as label

pieLabels = 'Jan2020', 'Dec19', 'Nov19', 'Oct19', 'Sep19', 'Aug19' 



figureObject, axesObject = plotter.subplots()



# Draw the pie chart

axesObject.pie(USJobs['Value'],

        labels=pieLabels,

        autopct='%1.2f',

        startangle=90)



# Aspect ratio - equal means pie is a circle

axesObject.axis('equal') 



plotter.show()
Salary['Value'].corr
USJobs['Value'].corr
Okt['Dimension']= Okt['Dimension'].astype('category')
OctCityJobs=Okt[(Okt['Dimension']=='Metro Job Openings')]

OctCityJobs=OctCityJobs.drop(['Dimension','Metro Catgs'], axis=1)

OctCityJobs
Dec.head(2)
Dec1= Dec[['Metro','Dimension','Value']]

Dec1['Metro']=Dec1['Metro'].astype('category')

Dec1['Dimension']= Dec1['Dimension'].astype('category')

Dec1.head(2)
#clean data 

Dec1['Value'] = Dec1['Value'].str.replace('$', '')

Dec1['Value'] = Dec1['Value'].str.replace(',', '').astype(float)
DecCityJobs=Dec1[(Dec1['Dimension']=='Metro Job Openings')]

DecCityJobs=DecCityJobs.drop(['Dimension'], axis=1)

DecCityJobs.rename(columns={'Value': 'Dec Jobs'}, inplace=True)

DecCityJobs.head(2)
result = pd.concat([(DecCityJobs['Dec Jobs']), (OctCityJobs['Value'])], axis=1)

result
result.corr()
#Linear correlation 

x=result['Dec Jobs']

y=result['Value']

plt.scatter(x,y)

plt.show()
result.corr(method="spearman")
#heatmap

f, ax = plt.subplots(figsize =(7, 6)) 

sns.heatmap(result, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 