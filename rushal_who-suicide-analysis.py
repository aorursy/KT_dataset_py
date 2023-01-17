# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
suicide_analysis = pd.read_csv("../input/who_suicide_statistics.csv")
suicide_analysis.head()
suicide_analysis.shape
suicide_analysis.info()
age_coder = {'5-14 years':0,
            '15-24 years':1,
            '25-34 years':2,
            '35-54 years':3,
            '55-74 years':4,
            '75+ years':5}
gender_coder = {'female':0,'male':1}
suicide_analysis['age_encoder'] = suicide_analysis['age'].map(age_coder)
suicide_analysis['sex_encoder'] = suicide_analysis['sex'].map(gender_coder)
suicide_analysis.head()
suicide_analysis.suicides_no.fillna(0,inplace=True)
suicide = suicide_analysis.groupby('age_encoder')[['suicides_no']].sum()
#suicide.index.to_series().map(en)
#Suicide based on age groups
en = {0:'5-14 years',
      1:'15-24 years',
      2:'25-34 years',
      3:'35-54 years',
      4:'55-74 years',
      5:'75+ years'}
gen = {0:'female',1:'male'}

plt.figure(figsize=(12,5))
sns.barplot(x=suicide.index.map(en.get),y=suicide.suicides_no)
plt.title("Total Suicide based in Age group")
plt.xlabel("Age Group")
plt.ylabel("Number of Suicide")
#Total Suicide male and female
male_suicide = suicide_analysis[suicide_analysis.sex_encoder == 0]['suicides_no'].values.sum()
female_suicide = suicide_analysis[suicide_analysis.sex_encoder == 1]['suicides_no'].values.sum()

age_differance = pd.DataFrame([male_suicide,female_suicide],index=['male','female'])
age_differance.head()
age_differance.plot(kind='bar',title="Total Suicide Based In Gender")
plt.legend()
#Suicide of top most 10 country
plt.figure(figsize=(14,6))
suicide_analysis.groupby('country').sum().sort_values(by='suicides_no',ascending=False)[['suicides_no']][:10].plot(kind='bar',figsize=(16,8),title='Suicide Based in Country')
#Total Suicide on each year in descendin order
suicide_analysis.groupby("year").sum().sort_values(by='suicides_no',ascending=False)[['suicides_no']].plot(kind='bar',figsize=(16,8),title='Suicide Based in Year')
suicide_data = suicide_analysis.groupby(['year','age']).sum()['suicides_no'].reset_index()
#Suicide Based on Year And Age
color=['red','green','blue','orange','gray','#222111']
plt.figure(figsize=(16,12))
sns.swarmplot(x='year',y='suicides_no',hue='age',data=suicide_data,palette=color)
plt.title("Suicide Based On The Year And Age Group")
plt.xticks(rotation=90)
plt.ylabel("Suicide Number")
suicide_country_age = suicide_analysis.groupby(['country','age']).sum()['suicides_no'].reset_index()
suicide_country_age.head()
#Suicide Based on Country And Age
plt.figure(figsize=(18,20))
sns.stripplot(x='suicides_no',y='country',hue='age',data=suicide_country_age,jitter=True)
plt.ylabel("Country")
plt.xlabel("Suicide Number")
plt.title('Suicide Based On The Country and Age')
