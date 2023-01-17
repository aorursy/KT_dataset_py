# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
from IPython.core.display import display, HTML
import time
import datetime
from matplotlib.ticker import PercentFormatter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
print('Files Used:')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data_sum = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data_det = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
data_det2 = pd.read_csv('../input/covid19-patient-precondition-dataset/covid.csv')
#data_population = pd.read_csv('/kaggle/input/ncoronavirus2019dataset-temperature-lifeexp/country_population.csv')
#data_fertility = pd.read_csv('/kaggle/input/ncoronavirus2019dataset-temperature-lifeexp/fertility_rate.csv')
#data_life_exp = pd.read_csv('/kaggle/input/ncoronavirus2019dataset-temperature-lifeexp/life_expectancy.csv')
#data_temprature = pd.read_csv('/kaggle/input/ncoronavirus2019dataset-temperature-lifeexp/temperature _monthly_data_per_country.csv')
#data_curefews_dates = pd.read_csv('/kaggle/input/ncoronavirus2019dataset-temperature-lifeexp/Curfews_and_lockdowns_countries_dates.csv')
#prepare covid_19_data.csv data table
#group data based on country and date
data_s = data_sum.groupby(['ObservationDate','Country/Region']).sum()
data_s=data_s.reset_index(['ObservationDate','Country/Region'])
#correct ObservationDate type and create new columns if needed 
data_s['ObservationDate'] = pd.to_datetime(data_s['ObservationDate'])
data_s['day'] = data_s['ObservationDate'].dt.day
data_s['month'] = data_s['ObservationDate'].dt.month
data_s['Active']=data_s['Confirmed']-(data_s['Deaths']+data_s['Recovered'])
data_s['R+D']=data_s['Recovered']+data_s['Deaths']
#create a new table grouped by counties with final comulative cases 
data_c_p = data_s.groupby('Country/Region').max().reset_index(['Country/Region'])
print('COVID-19_data.csv after preparing info:\n')
data_s.info()
data_d=data_det2.copy()
data_d['sex'] = data_d['sex'].replace({1:'female',2:'male'})
data_d.drop(columns=['patient_type','intubed','pneumonia',],inplace=True)
death = []
for x in data_det2['date_died']:
    if (x=='9999-99-99'):
        death.append(0)
    else:
        death.append(1)
data_d['death']=death
data_d['age_category'] = pd.cut(data_d['age'], 8,labels=['Age:30-45','Age:45-60','Age:15-30','Age:60-75','Age:0-15','Age:75-90','Age:90-105','Age:105-120'])

print('COVID-19_line_list_data.csv after preparing and modifying info:\n')
data_d.info()
# ax = data_d['sex'][data_d['sex']!='empty'].hist(label='infected')
ax =sns.countplot(data_d['sex'])
display(HTML('<h3 style="text-align: center;font-weight: normal;"> <strong> Gender </strong> Statistics Count  </h3>'))
plt.legend()
# plt.yscale('log')
# plt.ylim([0,1000])
# plt.yticks([data_d['sex'][(data_d['sex']!='empty')& (data_d['death']==1)].value_counts()[0],data_d['sex'][data_d['sex']!='empty'].value_counts()[0]],[data_d['sex'][(data_d['sex']!='empty')& (data_d['death']==1)].value_counts()[0],data_d['sex'][data_d['sex']!='empty'].value_counts()[0]],rotation=30)
plt.grid(True)
plt.tight_layout()
ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data_d)))
plt.show()
# ax = data_d['sex'][(data_d['death']==1)].hist(label='Death')
ax =sns.countplot(data_d['sex'][(data_d['death']==1)])
plt.legend()
plt.grid(True)
plt.tight_layout()
ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data_d)))
plt.show()

# ax = data_d['age_category'].hist(figsize=(15,3),label='Age_category Infection')
ax = sns.countplot(data_d['age_category'])
display(HTML('<h3 style="text-align: center;font-weight: normal;"> <strong> All ages </strong>are at risk of <strong> infection </strong> </h3>'))
plt.legend()
# plt.yscale('log')
# plt.ylim([0,250000])
# plt.yticks([data_d['age_category'][data_d['death']==1].value_counts()[0],data_d['age_category'].value_counts()[0]],[data_d['age_category'][data_d['death']==1].value_counts()[0],data_d['age_category'].value_counts()[0]],rotation=30)
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=90)
# ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data_d)))
plt.show()
# ax = data_d['age_category'][data_d['death']==1].hist(figsize=(15,3), label='Age_category Deaths')
ax = sns.countplot(data_d['age_category'][data_d['death']==1])
plt.legend()
plt.grid(True)
plt.xticks(rotation=90)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data_d)))
plt.tight_layout()
plt.show()


# ax = data_d['age_category'].hist(figsize=(15,3),label='Age_category Infection')
ax = sns.countplot(data_d['icu'], palette="Set1")
display(HTML('<h3 style="text-align: center;font-weight: normal;"> <strong> All ages </strong>are at risk of <strong> infection </strong> </h3>'))
plt.legend()
# plt.yscale('log')
# plt.ylim([0,250000])
# plt.yticks([data_d['age_category'][data_d['death']==1].value_counts()[0],data_d['age_category'].value_counts()[0]],[data_d['age_category'][data_d['death']==1].value_counts()[0],data_d['age_category'].value_counts()[0]],rotation=30)
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=90)
# ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data_d)))
plt.show()
# ax = data_d['age_category'][data_d['death']==1].hist(figsize=(15,3), label='Age_category Deaths')
ax = sns.countplot(data_d['icu'][data_d['death']==1])
plt.legend()
plt.grid(True)
plt.xticks(rotation=90)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(data_d)))
plt.tight_layout()
plt.show()
plt.figure(figsize=(20,50))
plt.subplots_adjust(hspace = 0.5,wspace=0.3)
plt.subplot(6, 2, 1)
display(HTML('<h3 style="text-align: center;font-weight: normal;">========<strong> COVID-19</strong> Shows <strong>Strong spread</strong> relation with <strong>Temperature </strong>==========</h3>'))
sns.regplot(x='first_quarter_temp', y='Confirmed', data=data_c_p[(data_c_p['first_quarter_temp']!=0)], fit_reg=True)
plt.title('Fisrt quarter temperature relation with Confirmed', fontsize=20,pad=20)
plt.subplot(6, 2, 2)
sns.regplot(x='first_quarter_temp', y='Confirmed', data=data_c_p[(data_c_p['first_quarter_temp']!=0)&(data_c_p['Confirmed']>20000)&(data_c_p['Confirmed']!=0)], fit_reg=True,label='More than 20000 Confirmed')
plt.title('Fisrt quarter temperature filltered on\n spreaded COVID-19 countries Confirmed cases', fontsize=20,pad=20)
plt.legend()
plt.subplot(6, 2, 3)
sns.regplot(x='first_quarter_temp', y='Confirmed', data=data_c_p[(data_c_p['first_quarter_temp']!=0)&(data_c_p['Confirmed']<10000)&(data_c_p['Confirmed']!=0)], fit_reg=True,label='Less than 10000 Confirmed')
sns.regplot(x='first_quarter_temp', y='Confirmed', data=data_c_p[(data_c_p['first_quarter_temp']!=0)&(data_c_p['Confirmed']<1000)&(data_c_p['Confirmed']!=0)], fit_reg=True,label='Less than 1000 Confirmed')
plt.title('Fisrt quarter temperature filltered on\n low cases countries less than 10000/1000 Confirmed cases', fontsize=20,pad=20)
plt.legend()
plt.subplot(6, 2, 4)
sns.distplot(data_c_p['first_quarter_temp'][(data_c_p['first_quarter_temp']!=0)&(data_c_p['Confirmed']>20000)&(data_c_p['Confirmed']!=0)],label='More than 20000 confirmed cases temp rang',color='tab:red',kde=True)
plt.ylabel('first_quarter_temp1')
plt.legend(loc=(0,0.95))
plt.twinx()
sns.distplot(data_c_p['first_quarter_temp'][(data_c_p['first_quarter_temp']!=0)&(data_c_p['Confirmed']<1000)&(data_c_p['Confirmed']!=0)],label='Less than 1000 confirmed cases temp rang',kde=True)
plt.title('Fisrt quarter temperature Distribution between\n High/low cases COVID-19 countries', fontsize=20,pad=20)
plt.ylabel('first_quarter_temp2')
plt.legend(loc=(0,0.9))
plt.show()
plt.figure(figsize=(20,50))
plt.subplots_adjust(hspace = 0.5,wspace=0.3)
plt.subplot(6, 2, 5)
display(HTML('<h3 style="text-align: center;font-weight: normal;">========<strong> COVID-19 </strong>Shows <strong>Strong spread/Deaths</strong> relation with <strong>Life Expectancy</strong> in Country </strong>==========</h3>'))
sns.regplot(x='life_exp', y='Confirmed', data=data_c_p[(data_c_p['life_exp']!=0)], fit_reg=True)
plt.title('Life Expectancy relation with COVID-19 Confirmed Cases\n And Life Expectancy relation with high spreaded COVID-19 countries', fontsize=20,pad=20)
plt.twiny()
plt.twinx()
sns.regplot(x='life_exp', y='Confirmed', data=data_c_p[(data_c_p['life_exp']!=0)&(data_c_p['Confirmed']>20000)&(data_c_p['Confirmed']!=0)], fit_reg=True,label='More than 20000 Confirmed',color='tab:red')
plt.legend()
plt.subplot(6, 2, 6)
sns.distplot(data_c_p['life_exp'][(data_c_p['life_exp']!=0)&(data_c_p['Confirmed']>20000)&(data_c_p['Confirmed']!=0)],label='More than 10000 confirmed cases life_exp rang',color='tab:red',kde=True)
plt.ylabel('life_exp1')
plt.legend(loc=(0,0.95))
plt.twinx()
sns.distplot(data_c_p['life_exp'][(data_c_p['life_exp']!=0)&(data_c_p['Confirmed']<1000)&(data_c_p['Confirmed']!=0)],label='Less than 1000 confirmed cases life_exp rang',kde=True)
plt.title('Life Expectancy Distribution between\n spreaded vs low cases COVID-19 countries', fontsize=20,pad=20)
plt.ylabel('life_exp2')
plt.legend(loc=(0,0.9))
plt.subplot(6, 2, 7)
sns.regplot(x='life_exp', y='Deaths', data=data_c_p[(data_c_p['life_exp']!=0)], fit_reg=True)
plt.title('Life Expectancy relation with COVID-19 Deaths Cases\n And Life Expectancy relation with high Deaths COVID-19 countries', fontsize=20,pad=20)
plt.twiny()
plt.twinx()
sns.regplot(x='life_exp', y='Deaths', data=data_c_p[(data_c_p['life_exp']!=0)&(data_c_p['Deaths']>3000)&(data_c_p['Deaths']!=0)], fit_reg=True,label='More than 3000 Deaths',color='tab:red')
plt.legend()
plt.subplot(6, 2, 8)
sns.distplot(data_c_p['life_exp'][(data_c_p['life_exp']!=0)&(data_c_p['Deaths']>3000)&(data_c_p['Deaths']!=0)],label='More than 3000 Deaths life_exp rang',color='tab:red',kde=True)
plt.ylabel('life_exp1')
plt.legend(loc=(0,0.95))
plt.twinx()
sns.distplot(data_c_p['life_exp'][(data_c_p['life_exp']!=0)&(data_c_p['Deaths']<1000)&(data_c_p['Deaths']!=0)],label='Less than 1000 Deaths life_exp rang',kde=True)
plt.ylabel('life_exp2')
plt.legend(loc=(0,0.9))
plt.title('Life Expectancy Distribution between\n High/low Deaths COVID-19 countries', fontsize=20,pad=20)
plt.show()
#plt.figure(figsize=(20,50))
#plt.subplots_adjust(hspace = 0.5,wspace=0.3)
#plt.subplot(6, 2, 9)
#sns.regplot(x='population', y='Deaths', data=data_c_p[(data_c_p['population']>0) & (data_c_p['population']<1000000000)], fit_reg=True)
#plt.ylim(-10,500000)
#plt.title('population relation with Deaths with filtered less than bilions values', fontsize=20,pad=20)
#plt.subplot(6, 2, 10)
#sns.regplot(x='population', y='Deaths', data=data_c_p[(data_c_p['population']>0) & (data_c_p['population']<1000000000)], fit_reg=True)
#plt.ylim(-10,100000)
#plt.title('population relation with Deaths with filtered less than 100000 Death values', fontsize=20,pad=20)
#plt.subplot(6, 2, 11)
#sns.regplot(x='population', y='Confirmed', data=data_c_p[(data_c_p['population']>0) & (data_c_p['population']<1000000000)], fit_reg=True)
#plt.ylim(-10,500000)
#plt.title('population relation with Confirmed with filtered less than bilions values', fontsize=20,pad=20)
#plt.subplot(6, 2, 12)
#sns.regplot(x='population', y='Confirmed', data=data_c_p[(data_c_p['population']>0) & (data_c_p['population']<1000000000)], fit_reg=True)
##plt.ylim(-10,100000)
#plt.title(xs, fontsize=20,pad=20)
#plt.show()
cols = ['D+R_percentage_of_A', 'Control_Level','Active']
main_countries = ['Mainland China', 'South Korea', 'Japan', 'Singapore', 'Germany', 'US', 'Spain', 'Iran', 'Italy', 'France', 'UK', 'Canada']
More_than_700Cases_Countires = ['Mainland China','UK', 'US', 'Brazil', 'Italy','Japan','Russia', 'Spain', 'Sweden','Turkey', 'France', 'Germany','India','Canada', 'Australia', 'Austria', 'Belgium', 'Chile', 'Indonesia', 'Iran', 'Ireland', 'Malaysia', 'Mexico','Norway', 'Pakistan','Singapore', 'South Korea', 'Switzerland', 'Ukraine','Saudi Arabia','Kuwait','Lebanon','Egypt','Qatar','Bahrain','Jordan','Oman','Algeria','Tunisia','United Arab Emirates']
Imp_countries = ['Mainland China', 'South Korea', 'Iran', 'Italy', 'Spain', 'Germany','Kuwait','Lebanon','France', 'US', 'Switzerland', 'UK', 'Austria','Belgium', 'Turkey', 'Canada', 'Israel', 'Sweden', 'Australia', 'Russia', 'India', 'Japan','Saudi Arabia','United Arab Emirates', 'Malaysia','Singapore','Qatar']

countries_countrol_level = pd.DataFrame(columns=['Country','Control_Level'])
countries_countrol_level['Country']=data_s[data_s['Confirmed']>700].groupby('Country/Region').max().reset_index(['Country/Region'])['Country/Region']

display(HTML('<h3 style="text-align: left;font-weight: normal;color:#33397d;">  <strong>Summary: </strong> (more details on graphs below) <br></h3>'))
display(HTML('<h3 style="text-align: left;font-weight: normal;color:#33397d;">  <strong>Countries Losing Control of the Outbreak:</strong>  <br></h3><p style="text-align: center;font-weight: normal;"><strong> %s </strong><br> </p>'%(losing_control_countries)))
display(HTML('<h3 style="text-align: left;font-weight: normal;color:#33397d;">  <strong>Countries Manage more Control of the Outbreak:</strong>  <br></h3><p style="text-align: center;font-weight: normal;"><strong> %s </strong><br> </p>'%(more_control_countries)))
display(HTML('<h3 style="text-align: left;font-weight: normal;color:#33397d;">  ============= <br></h3>'))
display(HTML('<h3 style="text-align: left;font-weight: normal;color:#33397d;">  <strong>Countries where the next Outbreak might be:</strong>  <br></h3><p style="text-align: center;font-weight: normal;"><strong> %s </strong><br> </p>'%(next_possible_outbreak_countries)))
display(HTML('<h3 style="text-align: left;font-weight: normal;color:#33397d;">  <strong>Countries Need to keep Control(current procedure) to dodge the outbreak:</strong>  <br></h3><p style="text-align: center;font-weight: normal;"><strong> %s </strong><br> </p>'%(countries_near_controling)))
display(HTML('<h3 style="text-align: left;font-weight: normal;color:#33397d;">  <strong>Countries Manage to Control The out break (Flatten the curve) in current procedures:</strong>  <br></h3><p style="text-align: center;font-weight: normal;"><strong> %s </strong><br><br><br><br> </p>'%(countries_controled_the_outbreak)))

more_control_countries = []
losing_control_countries = []

for country in More_than_700Cases_Countires:
    data = data_s[data_s['Country/Region']==country].groupby(['ObservationDate']).sum()
    data['D+R_percentage_of_A']=((data['Deaths']+data['Recovered'])/data['Active'])*100
    data['A_Divided_on_D+R']=data['Active']/(data['Deaths']+data['Recovered'])
    data['A_Divided_on_D+R'] = ((data['A_Divided_on_D+R'][(data['A_Divided_on_D+R']<1000000)] -  data['A_Divided_on_D+R'].min()) / (data['A_Divided_on_D+R'][(data['A_Divided_on_D+R']<1000000)].max() - data['A_Divided_on_D+R'].min()))*100
    data['Control_Level'] = ((((((data['Active'] - data['R+D'])-(data['Active'] - data['R+D']).min())/((data['Active'] - data['R+D']).max()-(data['Active'] - data['R+D']).min()))*100)-100)*(-1))
    #data['growth_rate_of_control'] = growth_rate(data['Control_Level'])*100
    if data_c_p['Curefew_date'].loc[data_c_p['Country/Region']==country].values:
        Cdd = data_c_p['Curefew_date'].loc[data_c_p['Country/Region']==country].values[0]
        Cdd = pd.to_datetime(Cdd)
        datetime.datetime.timestamp(Cdd)
    else:
        Cdd = 0
    first_case = data['Confirmed'].ne(0).idxmax().date()
    if data['D+R_percentage_of_A'][(data['D+R_percentage_of_A']<1000000)].max()>=100:
        first_controled_date = data['D+R_percentage_of_A'][(data['D+R_percentage_of_A']>=100)&(data['D+R_percentage_of_A']<1000000)].index[0].date()
        #controled_level = 100.0
        if data['D+R_percentage_of_A'].loc[first_controled_date:,][(data['D+R_percentage_of_A']<90)].max()>0:
            lose_control_date = data['D+R_percentage_of_A'].loc[first_controled_date:,][(data['D+R_percentage_of_A']<90)].index[0].date()
            #controled_level = np.round(data['D+R_percentage_of_A'].loc[first_controled_date:,][(data['D+R_percentage_of_A']<90)].describe()[-2],0)
            if data['D+R_percentage_of_A'].loc[lose_control_date:,][(data['D+R_percentage_of_A']>100)].max()>0:
                second_controled_date = data['D+R_percentage_of_A'].loc[lose_control_date:,][(data['D+R_percentage_of_A']>100)].index[0].date()
                #controled_level = 100.0
            else:
                second_controled_date = 'None'
        else:
            lose_control_date = 'Still under Control'
            second_controled_date = 'None'
    else:
        first_controled_date = 'Not yet'
        lose_control_date = 'None'
        second_controled_date = 'None'
        #controled_level = np.round((((data['D+R_percentage_of_A'][(data['D+R_percentage_of_A']<1000000)]) / 100)*100).max(),0)
    controled_level = np.round(data['Control_Level'].tail(n=10).describe()[1],1)
    countries_countrol_level['Control_Level'][countries_countrol_level['Country']==country]=controled_level
    if np.round(data['Control_Level'].tail(n=3).describe()[1],1) - np.round(data['Control_Level'].tail(n=10).describe()[1],1) >= 0:
        control_direction = 'More Control'
        more_control_countries.append(country)
    else:
        control_direction = 'Losing Control'
        losing_control_countries.append(country)
        
    plt.figure(figsize=(20,5))
    plt.subplots_adjust(hspace = 0.5,wspace=0.3)
    display(HTML('<h3 style="text-align: center;font-weight: normal;"> ======== <strong>%s</strong> ======== Current status: <strong>%s</strong> ======== <br></h3><p style="text-align: center;font-weight: normal;"> First case on <strong>%s</strong>   &nbsp;&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;&nbsp;    First Controled on: <strong>%s</strong>   &nbsp;&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;&nbsp;  Lost Control on: <strong>%s</strong> <br>             Second controled on: <strong>%s</strong>     &nbsp;&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;&nbsp;        Control level: <strong>%s%s</strong><br>&nbsp;&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;&nbsp;        Current status: <strong>%s</strong><br> </p>'% (country,control_direction,first_case,first_controled_date,lose_control_date,second_controled_date,controled_level,'%',control_direction)))
    plt.tight_layout()
    for ix, column in enumerate(cols):
        plt.subplot(1, 3, ix+1)       
        if ix+1==3:
            data['Active'].plot()
            data['Confirmed'].plot()
            data['R+D'].plot()
            plt.xlabel('').set_visible(False)
            plt.title('Active, Confirmed, Recovered&Deaths', fontsize=15,pad=10)
            plt.legend()
        elif ix+1==2:
            #plt.setp(plt.subplot(1, 3, ix+1), ylim=(0, 110))
            data[column].plot(legend=True,style='r--')#.get_xaxis().set_visible(False)            
            plt.title('Level_of_control_Direction\nMore Controled up toward 100\nlosing control -down- toward 0', fontsize=15,pad=10)
        else:
            plt.setp(plt.subplot(1, 3, ix+1), ylim=(0, 100))
            data[column].plot(legend=True)#.get_xaxis().set_visible(False)
            plt.title('Controled when it is more than 100\n lossing control when less than 100', fontsize=15,pad=10)
        plt.setp(plt.subplot(1, 3, ix+1), xlim=(data_s['ObservationDate'][0].date(), (data_s['ObservationDate'].loc[data_s['ObservationDate'].index[-1]]+np.timedelta64(2, 'D')).date()))
        plt.text(first_case, 0, '=== First Case %s ==='%first_case,rotation=90,fontsize=10)
        if Cdd!=0:
            plt.text(Cdd.date(), 0, '=== CureFew Started on %s ====='%Cdd.date(),rotation=90,fontsize=10)
    plt.show()
countries_controled_the_outbreak = countries_countrol_level['Country'][countries_countrol_level['Control_Level']>=80].tolist()
countries_near_controling = countries_countrol_level['Country'][(countries_countrol_level['Control_Level']>40)&(countries_countrol_level['Control_Level']<80)].tolist()
next_possible_outbreak_countries = countries_countrol_level['Country'][countries_countrol_level['Control_Level']<=40].tolist()
