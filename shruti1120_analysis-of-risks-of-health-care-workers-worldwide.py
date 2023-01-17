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
# Importing necessary library

import pandas as pd

import numpy as np

import glob  

import matplotlib.pyplot as plt

import plotly as py

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as pyo

pyo.init_notebook_mode()

import seaborn as sns

import sklearn





#Reading the dataset containing information on death worldwide of Healthcare personnels

#made from Memorandum



dataset = pd.read_csv('../input/world-death-in-healthcare/Worldwide_Deaths_In_Healthcare.csv')





#PREPROCCESSING TO HANDLE MISSING AGES WITH MEAN.

#Since in many places age was missing, so I decided to use most_frequent ages for those

#missing data, as that would make lowest bias possible

#however median is also another available option



Pre_Proc = dataset.iloc[:,0:5].values#[:,:-1]means taking all rows and all coloumns except the last one





from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer = imputer.fit(Pre_Proc[:,1:2])

Pre_Proc[:,1:2] = imputer.transform(Pre_Proc[:,1:2])





#Transforming pre_proc to dataframe for further proccessing

#and operation simplicity



Pre_Proc = pd.DataFrame(Pre_Proc)

Pre_Proc.columns = ["Name","Age","Department","Country","Death_Count_of_total_healthcare_individuals_in_that_country"]



#now reading data which speaks of total number of confirmed cases per country

#this includes total population not only healthcare workers



covid = pd.read_csv('../input/worldwide-confirmed-data/confirmeddata.csv').rename(columns={'country_region':'Country'})

Pre_Proc = Pre_Proc.merge(covid[['Country', 'total_confirmed_pop_country_wise']], how='left', on='Country')



#printing number of unique countries where where healthcare workers death due to C19

#have occurred along with number of unique departments throughout the world

#from where healthcare workers death is recorded

unique_countries = Pre_Proc.Country.nunique()

print('total number of unique countries',unique_countries)

unique_depts = Pre_Proc.Department.nunique()

print('total number of unique departments',unique_depts)

#Plotting the age variations of death of HC workers

sns.kdeplot(Pre_Proc.Age)
#  Relation between number of deceased workers by total number of confirmed population worldwide

fig,axes = plt.subplots(1,1,figsize=(20,10))

plot1 = Pre_Proc['total_confirmed_pop_country_wise'].value_counts().plot(kind='bar')

plt.title('Number Of Health Workers who died of COVID-19 by total_confirmed_population_country_wise', fontsize=24)

plt.ylabel('number of Health Care workers', fontsize=20)

plt.xticks(fontsize=20)

plt.tight_layout()

#  Relation between number of deceased workers by Department worldwide

fig,axes = plt.subplots(1,1,figsize=(20,10))

plot1 = Pre_Proc['Department'].value_counts().plot(kind='bar')

plt.title('Number Of Health Workers who died of COVID-19 by Department worldwide', fontsize=24)

plt.ylabel('number of Health Care workers', fontsize=20)

plt.xticks(fontsize=20)

plt.tight_layout()
#  Relation between number of deceased workers by Department in Italy 



temp1 = Pre_Proc[Pre_Proc.Country == 'Italy']

fig,axes = plt.subplots(1,1,figsize=(20,10))

plot1 = temp1['Department'].value_counts().plot(kind='bar')

plt.title('Number Of Health Workers who died of COVID-19 by Department Italy', fontsize=24)

plt.ylabel('number of Health Care workers', fontsize=20)

plt.xticks(fontsize=20)

plt.tight_layout()



#  Relation between number of deceased workers by Department in US 



temp2 = Pre_Proc[Pre_Proc.Country == 'US']

fig,axes = plt.subplots(1,1,figsize=(20,10))

plot1 = temp2['Department'].value_counts().plot(kind='bar')

plt.title('Number Of Health Workers who died of COVID-19 by Department US', fontsize=24)

plt.ylabel('number of Health Care workers', fontsize=20)

plt.xticks(fontsize=20)

plt.tight_layout()



#  Relation between number of deceased workers by Department in United Kingdom 



temp3 = Pre_Proc[Pre_Proc.Country == 'United Kingdom']

fig,axes = plt.subplots(1,1,figsize=(20,10))

plot1 = temp3['Department'].value_counts().plot(kind='bar')

plt.title('Number Of Health Workers who died of COVID-19 by Department United Kingdom', fontsize=24)

plt.ylabel('number of Health Care  workers', fontsize=20)

plt.xticks(fontsize=20)

plt.tight_layout()



#  Relation between number of deceased workers by Department in Iran 



temp4 = Pre_Proc[Pre_Proc.Country == 'Iran']

fig,axes = plt.subplots(1,1,figsize=(20,10))

plot1 = temp4['Department'].value_counts().plot(kind='bar')

plt.title('Number Of Health Workers who died of COVID-19 by Department Iran', fontsize=24)

plt.ylabel('number of Health Care workers', fontsize=20)

plt.xticks(fontsize=20)

plt.tight_layout()
#  Relation between number of deceased workers by Country worldwide



fig,axes = plt.subplots(1,1,figsize=(20,10))

plot1 = Pre_Proc['Country'].value_counts().plot(kind='bar')

plt.title('Number Of Health Care Workers who died of COVID-19 by Country', fontsize=24)

plt.ylabel('number of Health Care workers', fontsize=20)

plt.xticks(fontsize=20)

plt.tight_layout()
#  Relation between number of deceased workers by Age in Italy 



temp1 = Pre_Proc[Pre_Proc.Country == 'Italy']

fig,axes = plt.subplots(1,1,figsize=(20,10))

plot1 = temp1['Age'].value_counts().plot(kind='bar')

plt.title('Number Of Health Workers who died of COVID-19 by Age Italy', fontsize=24)

plt.ylabel('number of Health Care workers', fontsize=20)

plt.xticks(fontsize=20)

plt.tight_layout()



#  Relation between number of deceased workers by Age in US 



temp2 = Pre_Proc[Pre_Proc.Country == 'US']

fig,axes = plt.subplots(1,1,figsize=(20,10))

plot1 = temp2['Age'].value_counts().plot(kind='bar')

plt.title('Number Of Health Workers who died of COVID-19 by Age US', fontsize=24)

plt.ylabel('number of Health Care workers', fontsize=20)

plt.xticks(fontsize=20)

plt.tight_layout()



#  Relation between number of deceased workers by Age in United Kingdom 



temp3 = Pre_Proc[Pre_Proc.Country == 'United Kingdom']

fig,axes = plt.subplots(1,1,figsize=(20,10))

plot1 = temp3['Age'].value_counts().plot(kind='bar')

plt.title('Number Of Health Workers who died of COVID-19 by Age United Kingdom', fontsize=24)

plt.ylabel('number of Health Care  workers', fontsize=20)

plt.xticks(fontsize=20)

plt.tight_layout()



#  Relation between number of deceased workers by Age in Iran 



temp4 = Pre_Proc[Pre_Proc.Country == 'Iran']

fig,axes = plt.subplots(1,1,figsize=(20,10))

plot1 = temp4['Age'].value_counts().plot(kind='bar')

plt.title('Number Of Health Workers who died of COVID-19 by Age Iran', fontsize=24)

plt.ylabel('number of Health Care workers', fontsize=20)

plt.xticks(fontsize=20)

plt.tight_layout()


list_view = Pre_Proc[['Department', 'Death_Count_of_total_healthcare_individuals_in_that_country','Country']].copy()

list_view = list_view.groupby(['Department','Country']).count()
pd.set_option("display.max_rows", None, "display.max_columns", None)

print(list_view)