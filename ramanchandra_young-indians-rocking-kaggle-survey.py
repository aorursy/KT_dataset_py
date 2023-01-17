import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='ticks',color_codes=True)

from sklearn.preprocessing import LabelEncoder

import os

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

schema_df=pd.read_csv('../input/kaggle-survey-2019/survey_schema.csv');
selected_cols=['Q1','Q2','Q3','Q4','Q5','Q6','Q8','Q11','Q14','Q15']

df=df[selected_cols]
df.head()
#Rename the columns for better understanding

df=df.rename(columns={

             'Q1':'Age',

             'Q2':'Gender',

             'Q3':'Country',

             'Q4':'Education level',

             'Q5':'Job title',

             'Q6':'Company size',

             'Q8':'ML implementaion in company',

            'Q11':'Investment on ML',

            'Q14':'Primary tool used for ML',

            'Q15':'Coding experience'

})
df=df.drop([0])

df=df.reset_index()

df.head()
df=df.drop('index',axis=1)

df.head()
plt.figure(figsize=(10,8))

sns.countplot(x='Gender',data=df)

plt.ylabel('Number of participants')

plt.title('Gender distribution in Survey')

plt.show()
df['Country']=df['Country'].replace({'United States of America':'USA',

                                     'United Kingdom of Great Britain and Northern Ireland':'UK'})

plt.figure(figsize=(10,8))

sns.countplot(x='Country',data=df)

plt.xticks(rotation=90)

plt.ylabel('Number of participants')

plt.title('Country wise distribution in Survey')

plt.show()
#consider top 10 cuntries

Top10_df=df['Country'].value_counts()[:10].reset_index()

Top10_df
Top10_df['Country']=Top10_df['Country']*100/Top10_df['Country'].sum()

#Pie chart for top 10 countries

explode=(0.1,0,0,0,0,0,0,0,0,0)

colors=['g','b','r','c','m','y','#ff9999','#66b3ff','#99ff99','#ffcc99']

fig, ax=plt.subplots(figsize=(10,8))

ax.pie(Top10_df['Country'],explode=explode,labels=Top10_df['index'],shadow=True,startangle=90,colors=colors,autopct='%1.1f%%')

ax.axis('equal')

plt.title('Respondents distribution in top 10 countries')

plt.tight_layout()

plot=plt.show()
# Pi chart for Male respondents in top 10 countries

male_df=df[df['Gender']=='Male']

Top10_male_df=male_df['Country'].value_counts()[:10].reset_index()

Top10_male_df['Country']=Top10_male_df['Country']*100/Top10_male_df['Country'].sum()

explode=(0.1,0,0,0,0,0,0,0,0,0)

colors=['g','b','r','c','m','y','#ff9999','#66b3ff','#99ff99','#ffcc99']

fig, ax=plt.subplots(figsize=(10,8))

ax.pie(Top10_male_df['Country'],explode=explode,labels=Top10_male_df['index'],shadow=True,startangle=90,colors=colors,autopct='%1.1f%%')

ax.axis('equal')

plt.title('Distribution of Male respondents in top 10 countries')

plt.tight_layout()

plot=plt.show()
# Pi chart for Female respondents in top 10 countries

female_df=df[df['Gender']=='Female']

Top10_female_df=female_df['Country'].value_counts()[:10].reset_index()

Top10_male_df['Country']=Top10_female_df['Country']*100/Top10_female_df['Country'].sum()

explode=(0.1,0,0,0,0,0,0,0,0,0)

colors=['g','b','r','c','m','y','#ff9999','#66b3ff','#99ff99','#ffcc99']

fig, ax=plt.subplots(figsize=(10,8))

ax.pie(Top10_female_df['Country'],explode=explode,labels=Top10_female_df['index'],shadow=True,startangle=90,colors=colors,autopct='%1.1f%%')

ax.axis('equal')

plt.title('Distribution of Female respondents in top 10 countries')

plt.tight_layout()

plot=plt.show()
plt.figure(figsize=(14,10))

sns.countplot(x='Age',hue='Gender',data=df)

plt.ylabel('Number of respondents')

plt.title('Age distribution in survey by Gender')

plt.show()
youth_df1=df[df['Age']=='18-21']

youth_df2=df[df['Age']=='22-24']

youth_df=pd.concat([youth_df1,youth_df2]).reset_index()

youth_df=youth_df.drop('index',axis=1)

youth_df.head()
youth_df['Country']=youth_df['Country'].replace({'United States of America':'USA',

                                     'United Kingdom of Great Britain and Northern Ireland':'UK'})

plt.figure(figsize=(14,10))

sns.countplot(x='Country',data=youth_df)

plt.ylabel('Number of youth participants')

plt.xticks(rotation=90)

plt.title('Distribution of youth in survey by country')

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(x='Gender',data=youth_df)

plt.ylabel('Number of participants')

plt.title('Distribution of youth in Survey by Gender')

plt.show()
MaleY_df=youth_df[youth_df['Gender']=='Male']

Top_maleY_df=MaleY_df['Country'].value_counts()[:10].reset_index()

Top_maleY_df['Country']=Top_maleY_df['Country']*100/Top_maleY_df['Country'][0]

FemaleY_df=youth_df[youth_df['Gender']=='Female']

Top_femaleY_df=FemaleY_df['Country'].value_counts()[:10].reset_index()

Top_femaleY_df['Country']=Top_femaleY_df['Country']*100/Top_femaleY_df['Country'][0]

explode=(0.1,0,0,0,0,0,0,0,0,0)

colors=['b','r','g','m','c','y','#ff9999','#66b3ff','#99ff99','#ffcc99']



fig=plt.figure(figsize=(18,10),dpi=1200)

#Pie chart for distribution of young males in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_maleY_df['Country'],explode=explode,labels=Top_maleY_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of young males in survey in top 10 countries')



#Pie chart for distribution of young females in survey

ax1=plt.subplot2grid((1,2),(0,1))

plt.pie(Top_femaleY_df['Country'],explode=explode,labels=Top_femaleY_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of young females in survey in top 10 countries');
plt.figure(figsize=(12,8))

sns.countplot(x='Job title',hue='Gender',data=youth_df)

plt.ylabel('Number of young respondents')

plt.xticks(rotation=90)

plt.title('Distribution of young respondents by Job title');
#Job titel = student

Student_df=youth_df[youth_df['Job title']=='Student']

Top_student_df=Student_df['Country'].value_counts()[:10].reset_index()

Top_student_df['Country']=Top_student_df['Country']*100/Top_student_df['Country'][0]

#Job titel = Software engineer

SWE_df=youth_df[youth_df['Job title']=='Software Engineer']

Top_software_df=SWE_df['Country'].value_counts()[:10].reset_index()

Top_software_df['Country']=Top_software_df['Country']*100/Top_software_df['Country'][0]

#Job titel = Data Scientist

DS_df=youth_df[youth_df['Job title']=='Data Scientist']

Top_DS_df=DS_df['Country'].value_counts()[:10].reset_index()

#Job titel = Not employed

Notemploy_df=youth_df[youth_df['Job title']=='Not employed']

Top_Notemploy_df=Notemploy_df['Country'].value_counts()[:10].reset_index()

Top_Notemploy_df['Country']=Top_Notemploy_df['Country']*100/Top_Notemploy_df['Country'][0]

#Job titel = Data analyst

DA_df=youth_df[youth_df['Job title']=='Data Analyst']

Top_DA_df=DA_df['Country'].value_counts()[:10].reset_index()

Top_DA_df['Country']=Top_DA_df['Country']*100/Top_DA_df['Country'][0]

#Job titel = Business Analyst

BA_df=youth_df[youth_df['Job title']=='Business Analyst']

Top_BA_df=BA_df['Country'].value_counts()[:10].reset_index()

Top_BA_df['Country']=Top_BA_df['Country']*100/Top_BA_df['Country'][0]

#Job titel = other

other_df=youth_df[youth_df['Job title']=='Other']

Top_other_df=other_df['Country'].value_counts()[:10].reset_index()

Top_other_df['Country']=Top_other_df['Country']*100/Top_other_df['Country'][0]

#Job titel = Data Engineer

DA_df=youth_df[youth_df['Job title']=='Data Engineer']

Top_DE_df=DA_df['Country'].value_counts()[:10].reset_index()

Top_DE_df['Country']=Top_DE_df['Country']*100/Top_DE_df['Country'][0]

#Job titel = statistician

Stat_df=youth_df[youth_df['Job title']=='Statistician']

Top_stat_df=Stat_df['Country'].value_counts()[:10].reset_index()

Top_stat_df['Country']=Top_stat_df['Country']*100/Top_stat_df['Country'][0]

#Job titel = Research scientist

RS_df=youth_df[youth_df['Job title']=='Research Scientist']

Top_RS_df=RS_df['Country'].value_counts()[:10].reset_index()

Top_RS_df['Country']=Top_RS_df['Country']*100/Top_RS_df['Country'][0]



explode=(0.1,0,0,0,0,0,0,0,0,0)

colors=['b','r','g','m','c','y','#ff9999','#66b3ff','#99ff99','#ffcc99']

fig=plt.figure(figsize=(20,35),dpi=1200)

#Pie chart for distribution of young students in survey

ax1=plt.subplot2grid((5,2),(0,0))

plt.pie(Top_student_df['Country'],explode=explode,labels=Top_student_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of young students in survey in top 10 countries')



#Pie chart for distribution of young software engineers in survey

ax1=plt.subplot2grid((5,2),(0,1))

plt.pie(Top_software_df['Country'],explode=explode,labels=Top_software_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of young software engineers in survey in top 10 countries');
#Pie chart for distribution of young Data scientists in survey

fig=plt.figure(figsize=(20,35),dpi=1200)

ax1=plt.subplot2grid((5,2),(1,0))

plt.pie(Top_DS_df['Country'],explode=explode,labels=Top_DS_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of young Data scientists in survey in top 10 countries');



#Pie chart for distribution of not employed youth  in survey

ax1=plt.subplot2grid((5,2),(1,1))

plt.pie(Top_Notemploy_df['Country'],explode=explode,labels=Top_Notemploy_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of not employed youth in survey in top 10 countries');
#Pie chart for distribution of young Data analysts  in survey

fig=plt.figure(figsize=(20,35),dpi=1200)

ax1=plt.subplot2grid((5,2),(2,0))

plt.pie(Top_DA_df['Country'],explode=explode,labels=Top_DA_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of young Data analysts in survey in top 10 countries');



#Pie chart for distribution of young Business analysts  in survey

ax1=plt.subplot2grid((5,2),(2,1))

plt.pie(Top_BA_df['Country'],explode=explode,labels=Top_BA_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of young Business analysts in survey in top 10 countries');
#Pie chart for distribution of young others  in survey

fig=plt.figure(figsize=(20,35),dpi=1200)

ax1=plt.subplot2grid((5,2),(3,0))

plt.pie(Top_other_df['Country'],explode=explode,labels=Top_other_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of young others in survey in top 10 countries');



#Pie chart for distribution of young Statisticians  in survey

ax1=plt.subplot2grid((5,2),(3,1))

plt.pie(Top_stat_df['Country'],explode=explode,labels=Top_stat_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of young Statisticians in survey in top 10 countries');

#Pie chart for distribution of young Research scientists  in survey

fig=plt.figure(figsize=(20,35),dpi=1200)

ax1=plt.subplot2grid((5,2),(4,0))

plt.pie(Top_RS_df['Country'],explode=explode,labels=Top_RS_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of young Research scientists in survey in top 10 countries');



#Pie chart for distribution of young Data engineers in survey

ax1=plt.subplot2grid((5,2),(4,1))

plt.pie(Top_DE_df['Country'],explode=explode,labels=Top_DE_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of young Data Engineers in survey in top 10 countries');
youth_df['Education level']= youth_df['Education level'].replace({ 

    "Some college/university study without earning a bachelor’s degree":"No degree ",

    "No formal education past high school":'Till High school',

    "Bachelor’s degree":'Bachelor degree',

    "Master’s degree":'Master degree'

})

plt.figure(figsize=(14,8))

sns.countplot(x='Education level',hue='Gender',data=youth_df)

plt.ylabel('Number of Youngsters')

plt.xticks(rotation=90)

plt.title('Distribution of youngsters in the survey by Education level')

plt.show();
#Remove white spaces

youth_df['Education level']=youth_df['Education level'].str.strip() 
#Education level= college study with no bachelor degree

college_df=youth_df[youth_df['Education level']=='No degree']

Top_college_df=college_df['Country'].value_counts()[:10].reset_index()

Top_college_df['Country']=Top_college_df['Country']*100/Top_college_df['Country'][0]

#Education level= Bachelor degree

bachelor_df=youth_df[youth_df['Education level']=='Bachelor degree']

Top_bachelor_df=bachelor_df['Country'].value_counts()[:10].reset_index()

Top_bachelor_df['Country']=Top_bachelor_df['Country']*100/Top_bachelor_df['Country'][0]

#Education level= Master degree

master_df=youth_df[youth_df['Education level']=='Master degree']

Top_master_df=master_df['Country'].value_counts()[:10].reset_index()

Top_master_df['Country']=Top_master_df['Country']*100/Top_master_df['Country'][0]

#Education level= Professional degree

professional_df=youth_df[youth_df['Education level']=='Professional degree']

Top_professional_df=professional_df['Country'].value_counts()[:10].reset_index()

Top_professional_df['Country']=Top_professional_df['Country']*100/Top_professional_df['Country'][0]

#Education level= Doctoral degree

doctoral_df=youth_df[youth_df['Education level']=='Doctoral degree']

Top_doctoral_df=doctoral_df['Country'].value_counts()[:10].reset_index()

Top_doctoral_df['Country']=Top_doctoral_df['Country']*100/Top_doctoral_df['Country'][0]

#Education level= Till High school

highschool_df=youth_df[youth_df['Education level']=='Till High school']

Top_highschool_df=highschool_df['Country'].value_counts()[:10].reset_index()

Top_highschool_df['Country']=Top_highschool_df['Country']*100/Top_highschool_df['Country'][0]



explode=(0.1,0,0,0,0,0,0,0,0,0)

colors=['b','r','g','m','c','y','#ff9999','#66b3ff','#99ff99','#ffcc99']

fig=plt.figure(figsize=(20,35),dpi=1200)



#Pie chart for distribution of youngasters having college study with no bachelor degree in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_college_df['Country'],explode=explode,labels=Top_college_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters having college study with no bachelor's degree in survey in top 10 countries")



#Pie chart for distribution of youngasters with Bachelor's degree in survey

ax1=plt.subplot2grid((1,2),(0,1))

plt.pie(Top_bachelor_df['Country'],explode=explode,labels=Top_bachelor_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters with Bachelor's degree in survey in top 10 countries");
fig=plt.figure(figsize=(20,35),dpi=1200)



#Pie chart for distribution of youngasters with  Master's degree in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_master_df['Country'],explode=explode,labels=Top_master_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters with  Master's degree in survey in top 10 countries")



#Pie chart for distribution of youngasters with Professional degree in survey

ax1=plt.subplot2grid((1,2),(0,1))

plt.pie(Top_professional_df['Country'],explode=explode,labels=Top_professional_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters with Professional degree in survey in top 10 countries");
fig=plt.figure(figsize=(20,35),dpi=1200)



#Pie chart for distribution of youngasters having education level till high school in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_highschool_df['Country'],explode=explode,labels=Top_highschool_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title('Distribution of youngasters having education level till high school in survey in top 10 countries')



#Pie chart for distribution of youngasters with Doctoral degree in survey

ax1=plt.subplot2grid((1,2),(0,1))

plt.pie(Top_doctoral_df['Country'],explode=explode,labels=Top_doctoral_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters with Doctoral degree in survey in top 10 countries");
plt.figure(figsize=(14,8))

sns.countplot(x='Company size',hue='Gender',data=youth_df)

plt.ylabel('Number of Youngsters')

plt.xticks(rotation=90)

plt.title('Distribution of youngsters in the survey by Company size')

plt.show();
youth_df['Company size']=youth_df['Company size'].str.strip()
#Company size = 0-49 employees

zero_49_df=youth_df[youth_df['Company size']=='0-49 employees']

Top_zero_49_df=zero_49_df['Country'].value_counts()[:10].reset_index()

Top_zero_49_df['Country']=Top_zero_49_df['Country']*100/Top_zero_49_df['Country'][0]

#Company size = 1000-9,999 employees

onek_9k_df=youth_df[youth_df['Company size']=='1000-9,999 employees']

Top_onek_9k_df=onek_9k_df['Country'].value_counts()[:10].reset_index()

Top_onek_9k_df['Country']=Top_onek_9k_df['Country']*100/Top_onek_9k_df['Country'][0]

#Company size = >10,000 employees

tenk_df=youth_df[youth_df['Company size']=='> 10,000 employees']

Top_tenk_df=tenk_df['Country'].value_counts()[:10].reset_index()

Top_tenk_df['Country']=Top_tenk_df['Country']*100/Top_tenk_df['Country'][0]

#Company size = 50-249 employees

fifty_249_df=youth_df[youth_df['Company size']=='50-249 employees']

Top_fifty_249_df=fifty_249_df['Country'].value_counts()[:10].reset_index()

Top_fifty_249_df['Country']=Top_fifty_249_df['Country']*100/Top_fifty_249_df['Country'][0]

#Company size = 250-999 employees

two50_999_df=youth_df[youth_df['Company size']=='250-999 employees']

Top_two50_999_df=two50_999_df['Country'].value_counts()[:10].reset_index()

Top_two50_999_df['Country']=Top_two50_999_df['Country']*100/Top_two50_999_df['Country'][0]



explode=(0.1,0,0,0,0,0,0,0,0,0)

colors=['b','r','g','m','c','y','#ff9999','#66b3ff','#99ff99','#ffcc99']

fig=plt.figure(figsize=(20,35),dpi=1200)



#Pie chart for distribution of young employees whose company size is 0-49 in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_zero_49_df['Country'],explode=explode,labels=Top_zero_49_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of young employees whose company size is 0-49 in survey in top 10 countries")



#Pie chart for distribution of young employees whose company size is 1000-9,999 in survey

ax1=plt.subplot2grid((1,2),(0,1))

plt.pie(Top_onek_9k_df['Country'],explode=explode,labels=Top_onek_9k_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of young employees whose company size is 1000-9,999 in survey in top 10 countries");
fig=plt.figure(figsize=(20,35),dpi=1200)

#Pie chart for distribution of young employees whose company size is >10,000 in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_tenk_df['Country'],explode=explode,labels=Top_tenk_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of young employees whose company size is >10,000 in survey in top 10 countries")



#Pie chart for distribution of young employees whose company size is 50-249 employees in survey

ax1=plt.subplot2grid((1,2),(0,1))

plt.pie(Top_fifty_249_df['Country'],explode=explode,labels=Top_fifty_249_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of young employees whose company size is 50-249 employees in survey in top 10 countries");
fig=plt.figure(figsize=(20,35),dpi=1200)

#Pie chart for distribution of young employees whose company size is 250-999 in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_two50_999_df['Country'],explode=explode,labels=Top_two50_999_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of young employees whose company size is 250-999 in survey in top 10 countries");
plt.figure(figsize=(14,8))

sns.countplot(x='Investment on ML',hue='Gender',data=youth_df)

plt.ylabel('Number of Youngsters')

plt.xticks(rotation=90)

plt.title('Distribution of youngsters in the survey by Investment on ML')

plt.show();
youth_df['Investment on ML']=youth_df['Investment on ML'].str.strip()
#Investment on ML = 0 USD

zero_df=youth_df[youth_df['Investment on ML']=='$0 (USD)']

Top_zero_df=zero_df['Country'].value_counts()[:10].reset_index()

Top_zero_df['Country']=Top_zero_df['Country']*100/Top_zero_df['Country'][0]

#Investment on ML = 100-999 USD

Hund_999_df=youth_df[youth_df['Investment on ML']=='$100-$999']

Top_Hund_999_df=Hund_999_df['Country'].value_counts()[:10].reset_index()

Top_Hund_999_df['Country']=Top_Hund_999_df['Country']*100/Top_Hund_999_df['Country'][0]

#Investment on ML = 1-99 USD

one_99_df=youth_df[youth_df['Investment on ML']=='$1-$99']

Top_one_99_df=one_99_df['Country'].value_counts()[:10].reset_index()

Top_one_99_df['Country']=Top_one_99_df['Country']*100/Top_one_99_df['Country'][0]

#Investment on ML = 1000-9,999 USD

onek_9k_df=youth_df[youth_df['Investment on ML']=='$1000-$9,999']

Top_onek_9k_df=onek_9k_df['Country'].value_counts()[:10].reset_index()

Top_onek_9k_df['Country']=Top_onek_9k_df['Country']*100/Top_onek_9k_df['Country'][0]

#Investment on ML = 10,000-99,999 USD

tenk_99k_df=youth_df[youth_df['Investment on ML']=='$10,000-$99,999']

Top_tenk_99k_df=tenk_99k_df['Country'].value_counts()[:10].reset_index()

Top_tenk_99k_df['Country']=Top_tenk_99k_df['Country']*100/Top_tenk_99k_df['Country'][0]

#Investment on ML = >100,000 USD

hundk_df=youth_df[youth_df['Investment on ML']=='> $100,000 ($USD)']

Top_hundk_df=hundk_df['Country'].value_counts()[:10].reset_index()

Top_hundk_df['Country']=Top_hundk_df['Country']*100/Top_hundk_df['Country'][0]



explode=(0.1,0,0,0,0,0,0,0,0,0)

colors=['b','r','g','m','c','y','#ff9999','#66b3ff','#99ff99','#ffcc99']

fig=plt.figure(figsize=(20,35),dpi=1200)



#Pie chart for distribution of youngasters who have invested 0 USD on ML in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_zero_df['Country'],explode=explode,labels=Top_zero_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters who have invested 0 USD on ML in top 10 countries")



#Pie chart for distribution of youngasters whose company size is 100-999 in survey

ax1=plt.subplot2grid((1,2),(0,1))

plt.pie(Top_Hund_999_df['Country'],explode=explode,labels=Top_Hund_999_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters who have invested 100-999 USD on ML in top 10 countries");
fig=plt.figure(figsize=(20,35),dpi=1200)

#Pie chart for distribution of young employees who have invested 1-99 USD on ML in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_zero_df['Country'],explode=explode,labels=Top_zero_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters who have invested 1-99 USD on ML in top 10 countries")



#Pie chart for distribution of young employees whose company size is 1000-9,999 in survey

ax1=plt.subplot2grid((1,2),(0,1))

plt.pie(Top_onek_9k_df['Country'],explode=explode,labels=Top_onek_9k_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters who have invested 1000-9999 USD on ML in top 10 countries");
fig=plt.figure(figsize=(20,35),dpi=1200)

#Pie chart for distribution of young employees who have invested 10,000-99,999 USD on ML in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_tenk_99k_df['Country'],explode=explode,labels=Top_tenk_99k_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters who have invested 10,000-99,999 USD on ML in top 10 countries")



#Pie chart for distribution of young employees whose company size is >100,000 in survey

ax1=plt.subplot2grid((1,2),(0,1))

plt.pie(Top_hundk_df['Country'],explode=explode,labels=Top_hundk_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters who have invested >100,000 USD on ML in top 10 countries");
youth_df['Primary tool used for ML']=youth_df['Primary tool used for ML'].str.strip()
youth_df['Primary tool used for ML']=youth_df['Primary tool used for ML'].replace({

    'Basic statistical software (Microsoft Excel, Google Sheets, etc.)':'MS Excel/Google sheets',

    'Local development environments (RStudio, JupyterLab, etc.)':'RStudio/JupyterLab',

    'Cloud-based data software & APIs (AWS, GCP, Azure, etc.)':'AWS/GCP/Azure',

    'Business intelligence software (Salesforce, Tableau, Spotfire, etc.)':'Salesforce/Tableau',

    'Advanced statistical software (SPSS, SAS, etc.)':'SPSS/SAS'

})
plt.figure(figsize=(14,8))

sns.countplot(x='Primary tool used for ML',hue='Gender',data=youth_df)

plt.ylabel('Number of Youngsters')

plt.xticks(rotation=90)

plt.title('Distribution of youngsters in the survey by Primary tool used for ML')

plt.show();
#Primary tool used for ML = MS Excel/Google sheets

basic_df=youth_df[youth_df['Primary tool used for ML']=='MS Excel/Google sheets']

Top_basic_df=basic_df['Country'].value_counts()[:10].reset_index()

Top_basic_df['Country']=Top_basic_df['Country']*100/Top_basic_df['Country'][0]

#Primary tool used for ML = RStudio/JupyterLab

framework_df=youth_df[youth_df['Primary tool used for ML']=='RStudio/JupyterLab']

Top_framework_df=framework_df['Country'].value_counts()[:10].reset_index()

Top_framework_df['Country']=Top_framework_df['Country']*100/Top_framework_df['Country'][0]

#Primary tool used for ML = AWS/GCP/Azure

cloud_df=youth_df[youth_df['Primary tool used for ML']=='AWS/GCP/Azure']

Top_cloud_df=cloud_df['Country'].value_counts()[:10].reset_index()

Top_cloud_df['Country']=Top_cloud_df['Country']*100/Top_cloud_df['Country'][0]

#Primary tool used for ML = Salesforce/Tableau/Sp

BI_df=youth_df[youth_df['Primary tool used for ML']=='Salesforce/Tableau']

Top_BI_df=BI_df['Country'].value_counts()[:10].reset_index()

Top_BI_df['Country']=Top_BI_df['Country']*100/Top_BI_df['Country'][0]

#Primary tool used for ML = SPSS/SAS

Advanced_df=youth_df[youth_df['Primary tool used for ML']=='SPSS/SAS']

Top_Advanced_df=Advanced_df['Country'].value_counts()[:10].reset_index()

Top_Advanced_df['Country']=Top_Advanced_df['Country']*100/Top_Advanced_df['Country'][0]

#Primary tool used for ML = Other

other_df=youth_df[youth_df['Primary tool used for ML']=='Other']

Top_other_df=other_df['Country'].value_counts()[:10].reset_index()

Top_other_df['Country']=Top_other_df['Country']*100/Top_other_df['Country'][0]

explode=(0.1,0,0,0,0,0,0,0,0,0)

colors=['b','r','g','m','c','y','#ff9999','#66b3ff','#99ff99','#ffcc99']

fig=plt.figure(figsize=(20,35),dpi=1200)



#Pie chart for distribution of youngasters who have used MS Excel/Google sheets for ML in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_basic_df['Country'],explode=explode,labels=Top_basic_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters who have used MS Excel/Google sheets for ML in top 10 countries")



#Pie chart for distribution of youngasters who have used RStudio/JupyterLab for ML in survey

ax1=plt.subplot2grid((1,2),(0,1))

plt.pie(Top_framework_df['Country'],explode=explode,labels=Top_framework_df['index'],shadow=False,startangle=90,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters who have used RStudio/JupyterLab for ML in top 10 countries");
fig=plt.figure(figsize=(20,35),dpi=1200)

#Pie chart for distribution of youngasters who have used AWS/GCP/Azure for ML in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_cloud_df['Country'],explode=explode,labels=Top_cloud_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters who have used AWS/GCP/Azure for ML in top 10 countries")



#Pie chart for distribution of youngasters who have used Salesforce/Tableau for ML in survey

ax1=plt.subplot2grid((1,2),(0,1))

plt.pie(Top_framework_df['Country'],explode=explode,labels=Top_framework_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters who have used Salesforce/Tableau for ML in top 10 countries");
fig=plt.figure(figsize=(20,35),dpi=1200)

#Pie chart for distribution of youngasters who have used SPSS/SAS for ML in survey

ax1=plt.subplot2grid((1,2),(0,0))

plt.pie(Top_Advanced_df['Country'],explode=explode,labels=Top_Advanced_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters who have used SPSS/SAS for ML in top 10 countries")



#Pie chart for distribution of youngasters who have used Other tools for ML in survey

ax1=plt.subplot2grid((1,2),(0,1))

plt.pie(Top_other_df['Country'],explode=explode,labels=Top_other_df['index'],shadow=False,startangle=80,colors=colors,autopct='%1.1f%%')

plt.title("Distribution of youngasters who have used Other tools for ML for ML in top 10 countries");
plt.figure(figsize=(14,8))

sns.countplot(x='Coding experience',hue='Gender',data=youth_df)

plt.ylabel('Number of Youngsters')

plt.xticks(rotation=90)

plt.title('Distribution of youngsters in the survey by Coding experience')

plt.show();