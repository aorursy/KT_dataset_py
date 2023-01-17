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
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import textwrap

import plotly.express as px
public_schools = pd.read_csv('../input/us-schools-dataset/Public_Schools.csv')

public_schools.head(3)
public_schools.isnull().sum()
public_schools.drop(columns=['X','Y','ZIP','ZIP4','OBJECTID','ADDRESS','SOURCEDATE','WEBSITE','SOURCE','TELEPHONE','NCESID','SHELTER_ID','DISTRICTID','VAL_METHOD','NAICS_CODE','COUNTYFIPS','COUNTRY'],axis=1,inplace=True)

public_schools.head()
public_schools['VAL_DATE'] = public_schools['VAL_DATE'].str.replace(':','').str.rstrip('0')

public_schools['YEAR'] = pd.DatetimeIndex(public_schools['VAL_DATE']).year



#Removing VAL_DATE column

public_schools.drop(columns=['VAL_DATE'],axis=1,inplace=True)
public_schools.head(3)
public_schools = public_schools.rename(columns={'FT_TEACHER':'NO_OF_TEACHERS'})

public_schools.head(3)
public_schools = public_schools.rename(columns={'LEVEL_':'STAGE'})

public_schools.head(3)
public = public_schools.loc[public_schools['STAGE'].isin(['HIGH','ELEMENTARY','MIDDLE','SECONDARY','PREKINDERGARTEN','ADULT EDUCATION'])]

public.head(3)
public.NAME.value_counts()
public = public.replace({'NAME':'LINCOLN ELEMENTARY'},'LINCOLN ELEMENTARY SCHOOL')

public.NAME.value_counts()
public.NO_OF_TEACHERS.value_counts()
public.NO_OF_TEACHERS.replace(-999,999,inplace=True)

public.NO_OF_TEACHERS.value_counts()
public.POPULATION.value_counts()
public.POPULATION.replace(-999,999,inplace=True)

public.POPULATION.value_counts()
state_entop = public.groupby('STATE')['ENROLLMENT'].sum().reset_index()

State_entop = state_entop.sort_values('ENROLLMENT',ascending=False).head(10)

State_entop.reset_index(inplace=True)



State_enbottom = state_entop.sort_values('ENROLLMENT',ascending=True).head(10)

State_enbottom.reset_index(inplace=True)



print(State_entop, '\n')

print(State_enbottom)
max_width = 15

states = [State_entop,State_enbottom]

states_title = ['Top 10', 'Bottom 10']

other_title = ['High Enrollments','Low Enrollments']

fig, ax = plt.subplots(2,1, figsize = (22,14))

fig.subplots_adjust(hspace = 0.5)

for i in range(0,2):

    sns.barplot(ax = ax[i], data = states[i], x = 'STATE', y = 'ENROLLMENT')

    sns.barplot(ax = ax[i], data = states[i], x = 'STATE', y = 'ENROLLMENT')

    ax[i].legend()

    ax[i].set_title(states_title[i]+ ' States with '+ other_title[i], fontsize = 20)

    ax[i].set_ylabel('Enrollment', fontsize = 20)

    ax[i].set_xlabel('States', fontsize = 20)

    ax[i].set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax[i].get_xticklabels())

    ax[i].tick_params(labelsize = 18)

    

plt.show()
name_en = public.groupby(['NAME'])['ENROLLMENT'].sum().reset_index()

Name_en = name_en.sort_values('ENROLLMENT',ascending=False).head(5)

Name_en.reset_index(inplace=True)

Name_en
fig, ax = plt.subplots(figsize = [16,5])

sns.barplot(data = Name_en,x = 'NAME',y = 'ENROLLMENT', ax = ax)

ax.set_ylabel('Enrollment Count')

ax.set_title('Top 5 Public Schools with Enrollment')

for index,Name_en in enumerate(Name_en['ENROLLMENT'].astype(int)):

       ax.text(x=index-0.1 , y =Name_en+2 , s=f"{Name_en}" , fontdict=dict(fontsize=8))

plt.show()
year_en = public.groupby('YEAR')['ENROLLMENT'].sum().reset_index()

Year_en = year_en.sort_values('ENROLLMENT',ascending=False).head(5)

Year_en
plt.subplots(figsize=(10,10))

splot = sns.barplot(x=Year_en['YEAR'],y=Year_en['ENROLLMENT'], palette = 'winter_r')

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 15), textcoords = 'offset points')



plt.xlabel('YEAR',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.xticks(rotation=90)

plt.yticks(fontsize=15)

plt.title('Top 5 ENROLLMENT YEARS',fontsize=15);
public2010 = public.loc[public.YEAR.isin(['2010'])]

public2010.head(3)
public_sch = public2010.groupby('STAGE')['ENROLLMENT'].sum().reset_index()

Public_sch2010 = public_sch.sort_values('ENROLLMENT',ascending=False).head(5)

Public_sch2010
fig = px.pie(Public_sch2010, values=Public_sch2010['ENROLLMENT'], names=Public_sch2010['STAGE'])

fig.update_layout(title = 'Stages with Most Enrollment')

fig.show()
stage_teacher = public.groupby('STAGE')['NO_OF_TEACHERS'].sum().reset_index()

Stage_teacher = stage_teacher.sort_values('NO_OF_TEACHERS',ascending=False).head(5)

Stage_teacher
max_width = 15

fig, ax = plt.subplots(figsize = (10,8))

sns.barplot(ax = ax, data = Stage_teacher, x = 'STAGE', y = 'NO_OF_TEACHERS')

ax.legend()

ax.set_title('STAGES WITH HIGH NUMBER OF TEACHERS', fontsize = 15)

ax.set_ylabel('NO_OF_TEACHERS', fontsize = 15)

ax.set_xlabel('STAGES', fontsize = 15)

    

plt.show()
sch_pop = public.groupby('NAME')['POPULATION'].sum().reset_index()

Sch_pop = sch_pop.sort_values('POPULATION',ascending=False).head(5)

Sch_pop
plt.figure(figsize=(10,5))

chart = sns.barplot(data=Sch_pop,x='NAME',y='POPULATION',palette='Set1')

chart.set_xticklabels(chart.get_xticklabels(), rotation=65, horizontalalignment='right',fontweight='light')

chart.axes.yaxis.label.set_text("Population Count")