# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import datetime

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import data

data=pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')

data.head(5)
#Detils about the dataframe

data.info()
#Count of NA values for each columns

data.isna().sum()
#We shall drop all the NA records

data.dropna(inplace=True)

#Count of NA values for each columns

data.isna().sum()
# Job Title

data['Job Title'].value_counts()
# Top 10 Job Titles

sns.barplot(x=data['Job Title'].value_counts()[0:10],y=data['Job Title'].value_counts()[0:10].index);
# Lets analyse the top roles in a particular city

data['Location'].value_counts()
#Define a function to retun the top job tiles for a given location

def trending_roles_location(data,location,cnt):

    location_data=data[data['Location'].str.contains(location)]

    top_titles = data['Job Title'].value_counts()[0:cnt]

    return top_titles
# Analyse the Bengaluru data

print(trending_roles_location(data,'Bengaluru',10))
# Find the trending job titles in demand in Bangalore

labels=trending_roles_location(data,'Bengaluru',5).index

colors=['cyan','pink','orange','lightgreen','yellow']

explode=[0.1,0,0,0,0]

values=trending_roles_location(data,'Bengaluru',5)

#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Top trending roles in Bangalore',color='black',fontsize=10)

plt.show()
#Lets extract the skills

skills=data['Key Skills'].str.split("|",expand=True,)

skills_list=skills.stack(0, dropna=True)

skills_list.value_counts()[0:20]

#Skills with the count

top_skills= skills_list.value_counts().rename_axis('skills').reset_index(name='counts')

top_skills.head(10)
import squarify 

# Prepare Data

labels = top_skills[0:9].apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")",axis=1)

sizes = top_skills['counts'].value_counts()[0:9]

colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]



# Draw Plot

plt.figure(figsize=(12,8), dpi= 80)

squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)



# Decorate

plt.title('Treemap of Top 10 Skills in Demand')

plt.axis('off');
#Lets do month wise analysis

#Convert the Timestamp to Pandas Datetime

data['Crawl Timestamp'] = pd.to_datetime(data['Crawl Timestamp'])

#Extract the Year and Month

#data['year'] = pd.DatetimeIndex(data['Crawl Timestamp']).year

#Data belongs to same year

data['month'] = pd.DatetimeIndex(data['Crawl Timestamp']).month

data['day']=pd.DatetimeIndex(data['Crawl Timestamp']).day
print("Month /n",data['month'].value_counts())

print("day /n",data['day'].value_counts())
#Plot the stacked chart for each month crawl

data_month_day = data.groupby(['month', 'day']).size().reset_index().pivot(columns='day', index='month', values=0)

data_month_day.plot(kind='bar', stacked=True)

plt.title("Stacked Histogram for each month crawl details\n", fontsize=12)

plt.xlabel("Month")

plt.ylabel("Number of crawls\n");