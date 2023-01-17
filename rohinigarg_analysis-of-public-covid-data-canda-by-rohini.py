#Author Rohini Garg  



%reset -f

import numpy as np

import pandas as pd

# 1.2 For plotting

import matplotlib.pyplot as plt



import seaborn as sns



# 1.3 For data processing

from sklearn.preprocessing import StandardScaler

# 1.4 OS related

import os



import sys

sys.executable



import plotly.express as px



import plotly.graph_objects as go 

%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

pd.options.display.max_columns=100
os.chdir("/kaggle/input/uncover")

os.listdir() 

dfCanda = pd.read_csv("/kaggle/input/uncover/covid_19_canada_open_data_working_group/public-covid-19-cases-canada.csv")
#get Age unique

dfCanda.age.unique()





#Update Month

dfCanda['date_report']=pd.to_datetime(dfCanda['date_report'])

dfCanda['month']=dfCanda['date_report'].dt.month

import calendar

dfCanda['month']=dfCanda['month'].apply(lambda x: calendar.month_abbr[x])

dfCanda['month']=dfCanda['month'] + '-' +  dfCanda['date_report'].dt.year.astype(str)

dfCanda.age.unique()



#update case_id because it will help in count

dfCanda['case_id']=np.arange(1,dfCanda.shape[0]+1,1)

 

#Cleaning of data

dfCanda['locally_acquired']=dfCanda.locally_acquired.str.capitalize()



#change column name

dfCanda.rename({'locally_acquired':'Source_of_transmission'},inplace=True,axis='columns')

dfCanda.rename({'sex':'gender'},inplace=True,axis='columns')



#if travel history is t then update Source_of_transmission = Travel History

dfCanda.loc[(dfCanda.has_travel_history=='t') & (dfCanda.Source_of_transmission.isnull()),'Source_of_transmission']='Travel History'

#set na as Not Reported

dfCanda["Source_of_transmission"].fillna("Not Reported", inplace = True) 

dfCanda["Source_of_transmission"].value_counts()



#update daily_cases respect to date_report,province,health_region

dfCanda['daily_cases']=dfCanda.groupby(['date_report','province','health_region']).case_id.transform('count')

#

dfCanda['no_of_cases']=dfCanda.groupby(['date_report','province','health_region','age','Source_of_transmission','gender']).case_id.transform('count')



#age ,Source_transmission,gender wise count

dfCanda['age_Source_of_transmission_gender_count']=dfCanda.groupby(['age','Source_of_transmission','gender']).case_id.transform('count')







dfCanda.columns

dfCanda.head()

#health_region,monthwise cases

fig = plt.figure(figsize=(12, 5))

gr_health_region_month = dfCanda.groupby(['health_region','month'])

df_health_region_month_count = gr_health_region_month['case_id'].count().unstack()

df_health_region_month_count.sort_index(level=0, ascending=True, inplace=True)

ax=sns.heatmap(df_health_region_month_count, cmap = plt.cm.Blues)

ax.set(xlabel="Month")

plt.title("Number of Cases in Canda(Health Region Wise vs month)")



#observation: Most affected health region is "Montreal" and month is April-2020
#Age wise count

fig = go.Figure()

fig=px.histogram(data_frame=dfCanda.loc[dfCanda.age !='Not Reported',:],x='age',title='Age wise no of COVID-19 Confirmed cases in Canda')

fig.update_layout(

    

    xaxis_title="Age bracket",

    yaxis_title="Count",

    title = {

             'y':0.9,

        'x':0.5,

            'xanchor': 'center'

            ,'yanchor': 'top'

        }

)





#observation: 50-59,60-69 age group is most affected..
#province vs Age  cases

#we will igonre where age ='Not Reported' for better analysis

fig = go.Figure()

fig=px.density_heatmap(

                   data_frame =dfCanda.loc[dfCanda.age !='Not Reported',:],

                   x = 'province',

                   y = 'age'

                   )

fig.update_layout( 

    xaxis_title="Province of Canda",

    yaxis_title="Age bracket",

    title = {

             'text' :'Province vs Age wise no of COVID-19 Confirmed cases in Canda',

             'y':.95,

             'x':0.5,

            'xanchor': 'center'

            ,'yanchor': 'top'

        }

)



#observation  Most affected Province  is  "Ontario" in Canda ,50-59,60-69 age group is most affected



#Source of transmission wise summary







fig = px.violin(data_frame=dfCanda.loc[dfCanda.Source_of_transmission!='Not Reported'],

                x="Source_of_transmission",

                box=True,

                points="all"

                

               )

fig.update_layout( 

    title = {

             'text' :"Source of transmission wise summary",

             'y':.95,

             'x':0.5,

            'xanchor': 'center'

            ,'yanchor': 'top'

        }

)





#Observation : More spread due to  close contact & community
#daily cases respect to 'date_report','province','health_region'

fig = plt.figure(figsize=(17, 5))

sns.catplot(x = 'Source_of_transmission',

            y = 'no_of_cases', 

           col = 'month',

            kind = 'box',

            estimator=np.sum,

            data = dfCanda[dfCanda!='Not Reported'])

plt.suptitle("Source of transmission wise summary of daily cases for each month")

plt.subplots_adjust(top=1, left=0.1)

#Observation:With the time interval community transmission is there.


#which group is more suceptible: ALl type of source_transmission and gender , we will ignore "Not Reported"  for age

#no_of_cases w.r.t 'date_report','province','health_region','age','Source_of_transmission','gender'

g=sns.catplot(x = 'age',

            y = 'no_of_cases', 

              hue='gender',

            kind = 'bar',

              row='Source_of_transmission',

              estimator=np.sum,

            data = dfCanda.loc[(dfCanda.age != 'Not Reported') &  (dfCanda.Source_of_transmission != 'Not Reported') & (dfCanda.gender != 'Not Reported') ] )

g.fig.set_size_inches(17,15)



#Observation : 1)People are infected due to close contact

               #Females are more infected than males s
#day interval: respect to  first day of corona virus reported 

dfCanda.sort_values(by='date_report', inplace=True)

dfCanda['day_interval'] = (dfCanda['date_report'] - dfCanda['date_report'].min()).dt.days  

fig = plt.figure(figsize=(17, 10))

dfCanda_datewise=(dfCanda.groupby(['date_report','day_interval']).agg(date_count_canda=('case_id','count'))).reset_index()

dfCanda_datewise.head()

sns.jointplot(dfCanda_datewise.day_interval, dfCanda_datewise.date_count_canda, kind='scatter')



#Observations: cases are increasing with days.


#gender wise analysis where gender is reported

dfGender=dfCanda.loc[dfCanda.gender != 'Not Reported', :]

dfGender=(dfGender.groupby(['day_interval','gender']).agg(date_gender_count=('case_id','count'))).reset_index()

fig = plt.figure(figsize=(17, 10))

sns.lineplot('day_interval', 'date_gender_count', hue='gender',data=dfGender)

sns.relplot('day_interval', 'date_gender_count', hue='gender',data=dfGender)

#observation: Females  are more infected than male with CO-VID19
##gender wise analysis for all province  where gender is reported

dfprov_Gender=dfCanda.loc[(dfCanda.gender != 'Not Reported'), :]

dfprov_Gender=(dfprov_Gender.groupby(['day_interval','gender','province']).agg(count_date_gender_prov=('case_id','count'))).reset_index()

fig = plt.figure(figsize=(17, 10))

dfprov_Gender

g = sns.relplot(x='day_interval', y='count_date_gender_prov', data=dfprov_Gender, hue='gender', col='province', col_wrap=5, kind='scatter')

g.fig.set_size_inches(30, 10)