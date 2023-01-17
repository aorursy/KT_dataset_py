# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.simplefilter('ignore')

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
suicides = pd.read_csv('../input/suicides-in-india/Suicides in India 2001-2012.csv')
suicides.head()
suicides = suicides[suicides['Year']==2012]
for col in suicides.columns[:-1]:

    print(col,'-',suicides[col].nunique(),'-',suicides[col].unique())

    print('\n')
suicides[suicides['Type_code']=='Means_adopted'].head()
suicides[suicides['Type_code']=='Means_adopted'].Type.nunique()
suicides[suicides['Type_code']=='Means_adopted'].groupby('Type').sum()
means_adopted = suicides[suicides['Type_code']=='Means_adopted'].groupby('Type').sum().sort_values('Total', ascending=False)

plt.figure(figsize=(10,5))

ax = sns.barplot(x=means_adopted.index,y=means_adopted['Total'],data=means_adopted)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()
suicides[suicides['Type_code']=='Education_Status'].head()
educational_status = suicides[suicides['Type_code']=='Education_Status']
for col in educational_status.columns[3:-1]:

    print(col,'-',educational_status[col].nunique(),'-',educational_status[col].unique())

    print('\n')
educational_status_grouped = suicides[suicides['Type_code']=='Education_Status'].groupby('Type').sum().sort_values('Total', ascending=False)

plt.figure(figsize=(10,5))

ax = sns.barplot(x=educational_status_grouped.index,y=educational_status_grouped['Total'],data=educational_status_grouped)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()
plt.figure(figsize=(10,5))

ax = sns.barplot(x='Type',y='Total',hue='Gender',data=educational_status.sort_values('Total',ascending='False'),ci=None)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()
educational_status.head()
educational_status['Type'].unique()
educational_status_statewise = educational_status.groupby('State').sum().sort_values('Total')

educational_status_statewise = educational_status_statewise.drop(['Total (States)','Total (All India)','Total (Uts)'])
plt.figure(figsize=(10,5))

ax = sns.barplot(x=educational_status_statewise.index,y=educational_status_statewise['Total'],data=educational_status_statewise)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

ax.set_title('Absolute number of student suicides in Indian States')
#print(os.listdir("../input/all-census-data/"))

population_data = pd.read_csv('../input/all-census-data/elementary_2015_16.csv')
population_data.head()
population_data = population_data[['STATE NAME','TOTAL POULATION']]
population_data = population_data.groupby('STATE NAME').sum()
population_data.columns = ['population']
#population data is of the year when the state andhra pradesh had been divided into 2 - one itself and the other named telangana

#education data is of the time when it was one whole

#hence the population data had to be manipulated for these 2 states

population_data.at['ANDHRA PRADESH', 'population'] = population_data.ix['ANDHRA PRADESH']['population'] + population_data.ix['TELANGANA']['population']
population_data.drop('TELANGANA',axis=0,inplace=True)
educational_status_statewise = educational_status_statewise.sort_values('State')

population_data.index = population_data.index.str.lower()

educational_status_statewise.index = educational_status_statewise.index.str.lower()
educational_status_statewise.rename(index={'a & n islands': 'andaman and nicobar islands',

                                          'd & n haveli':'dadra and nagar haveli',

                                          'delhi (ut)':'delhi',

                                          'daman & diu':'daman and diu',

                                          'jammu & kashmir':'jammu and kashmir',

                                          'puducherry':'pondicherry'},inplace=True)
new_data = pd.concat([population_data,educational_status_statewise],axis=1)
new_data.head()
new_data.columns
new_data['%suicide'] = (new_data['Total']/new_data['population'])*100
new_data = new_data.sort_values('%suicide',ascending=False)

new_data.head()
plt.figure(figsize=(10,5))

ax = sns.barplot(x=new_data.index,y=new_data['%suicide'],data=new_data)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

ax.set_title('Rate of student suicides in Indian States')
professional_profile = suicides[suicides['Type_code']=='Professional_Profile']
professional_profile.head()
for col in professional_profile.columns[:-1]:

    print(col,'-',professional_profile[col].nunique(),'-',professional_profile[col].unique())
professional_profile = professional_profile[professional_profile['Type']!='Others (Please Specify)']

by_profession = professional_profile.groupby('Type').sum()

by_profession = by_profession.sort_values('Total',ascending=False)

plt.figure(figsize=(10,5))

ax = sns.barplot(x=by_profession.index,y=by_profession['Total'],data=by_profession)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

ax.set_title('Suicides categorised by different profession')
#print((os.listdir("../input/rainfall-in-india/")))

rainfall = pd.read_csv('../input/rainfall-in-india/district wise rainfall normal.csv')

rainfall.head()
rainfall = rainfall.groupby('STATE_UT_NAME').sum()

rainfall = rainfall[['ANNUAL']]

rainfall.columns = ['Annual Rainfall']

rainfall.head()
rainfall.index = rainfall.index.str.lower()
rainfall.rename(index={'andaman and nicobar islands': 'a & n islands',

                        'chatisgarh':'chhattisgarh',

                        'dadar nagar haveli':'d & n haveli',

                        'daman and dui':'daman & diu',

                        'delhi':'delhi (ut)',

                        'himachal':'himachal pradesh',

                        'jammu and kashmir':'jammu & kashmir',

                        'orissa':'odiasha',

                        'pondicherry':'puducherry',

                        'uttaranchal':'uttarakhand'},inplace=True)
agricultural_suicides = professional_profile[professional_profile['Type']=='Farming/Agriculture Activity']

agricultural_suicides = agricultural_suicides.groupby('State').sum()

agricultural_suicides.index = agricultural_suicides.index.str.lower()

agricultural_suicides.head()
agr_suicide_rainfall_data = pd.concat([rainfall,agricultural_suicides],axis=1)

agr_suicide_rainfall_data.head()
sns.lmplot(x='Annual Rainfall',y='Total',data=agr_suicide_rainfall_data)

ax = plt.gca()

ax.set_title("Agricultural suicides vs rainfall received")

ax.set_ylabel('Suicides')
plt.figure(figsize=(10,5))

ax = sns.barplot(x='Type',y='Total',hue='Gender',data=professional_profile.sort_values('Total',ascending=False))

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

ax.set_title('Suicides categorised by different profession')
social_status = suicides[suicides['Type_code']=='Social_Status']

social_status.head()
for col in social_status.columns[:-1]:

    print(col,'-',social_status[col].nunique(),'-',social_status[col].unique())
social_status.head()
ax = sns.barplot(x='Type',y='Total',data=social_status.sort_values('Total'),hue='Gender',

                 ci=None,order=['Married','Never Married','Widowed/Widower','Seperated','Divorcee'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

ax.set_title('Suicides categorised by marital status')
suicide_causes = suicides[suicides['Type_code']=='Causes']

suicide_causes.head()
for col in suicide_causes.columns[:-1]:

    print(col,'-',suicide_causes[col].nunique(),'-',suicide_causes[col].unique())
suicide_causes_bytype = suicide_causes.groupby('Type').sum().sort_values('Total',ascending=False)

suicide_causes_bytype.head()
suicide_causes_bytype.drop(['Other Causes (Please Specity)','Causes Not known'],inplace=True)

suicide_causes_bytype.head()
plt.figure(figsize=(10,5))

ax = sns.barplot(x=suicide_causes_bytype.index,y='Total',data=suicide_causes_bytype,ci=None)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

ax.set_title('Major causes of suicides in India')