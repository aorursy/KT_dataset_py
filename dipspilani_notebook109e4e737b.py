import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
dataset_patientinfo = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')
dataset_seoulfloating = pd.read_csv('/kaggle/input/coronavirusdataset/SeoulFloating.csv')

dataset_patientinfo.tail()
# CHILDREN ARE LESS LIKELY TO CATCH THE INFECTION, WORKING AGE IS HIGHLY VULNERABLE. (FROM JAN TO JUNE DATA)
sns.barplot(x=["0s","10s","20s","30s","40s","50s","60s","70s","80s"] , y = [66,178,899,523,518,667,482,232,170])
sns.barplot(x=["Female","Male"],y=[2218,1825]) #THIS SHOWS THAT FEMALES HAVE A HIGHER INFECTION RATE
dataset_patientinfo.info()
infectionperc_by_area = [1610/4246 , 840/4246 , 703/4246 , 162/4246 , 128/4246]
for i in range(len(infectionperc_by_area)):
    infectionperc_by_area[i] = infectionperc_by_area[i]*100
infectionperc_by_area_label=['contact with patient','overseas inflow','etc','Itaewon Clubs','Richway']


sns.barplot(x=infectionperc_by_area_label[:10],y=infectionperc_by_area[:10]) #TOP 5 SOURCES OF COVID-19 AND THEIR %
#SUPER SPREADERS
sns.barplot(x=["2000000205","4100000008","2000000167","1400000209","4100000006","2000000309"] , y=[51,27,24,24,21,21]) 
dataset_patientinfo['state'].value_counts()
dataset_patientinfo['sex'].value_counts()
sorted_by_state = dataset_patientinfo.sort_values(by='state')   
sorted_by_state.head()
all_deceased = sorted_by_state.iloc[:78,[1,2]]
all_deceased['sex'].value_counts()
# CHANCES OF A MAN DYING OF THE DISEASE IS ALMOST 2X HIGHER
sns.barplot(x=['male','female'],y=[47,28])
all_deceased['age'].value_counts()
#DEATH RATE IS LINEARLY CORRELATED WITH AGE
sns.barplot(x=['80s','70s','60s','50s','90s','40s','30s'],y=[25,21,12,7,7,2,1])
def time_format(x):
    x = str(x)
    if "-" in x:
        z = x.split("-")
        return z
dataset_patientinfo['confirmed_date'] = dataset_patientinfo['confirmed_date'].apply(lambda x:time_format(x))
dataset_patientinfo['released_date'] = dataset_patientinfo['released_date'].apply(lambda x:time_format(x))
dataset_patientinfo.head()
from datetime import date
confirm_cure = dataset_patientinfo.iloc[:,[10,11]].values
duration = []
for i in confirm_cure:
    if i[0] and i[1]:
        start = date(int(i[0][0]),int(i[0][1]),int(i[0][2]))
        end = date(int(i[1][0]),int(i[1][1]),int(i[1][2]))
        duration.append((end-start).days)
# FOR MOST PEOPLE, RECOVERY TIME IS 20-25 DAYS.
sns.distplot(duration,bins=50)
import statistics
statistics.mean(duration)

#average recovery time is 24.71 days.
dataset_seoulfloating.head()
dataset_tg = pd.read_csv('/kaggle/input/coronavirusdataset/TimeGender.csv')
dataset_tg.head()
dataset_tg.shape
import random
days_list5 = []
while len(days_list5)<20:
    xx = random.randint(0,242)
    if xx%2==0:
        days_list5.append(xx)           
male_vs_female=[]
for t in days_list5:
    male_vs_female.append([dataset_tg['confirmed'][t],dataset_tg['confirmed'][t+1]])
male_vs_female
m,f = [],[]
for t in male_vs_female:
    m.append(t[0])
    f.append(t[1])
print(m)
print(f)
#The dashed line shows the line x=y(when the number of males and females affected are equal) but surprisingly, 20 random
#dates were chosen and surprisingly on every date, the number of accumulated infections among females were higher.
bb = [1000,6500]
aa = [1000,6500]
plt.xlabel('Total Males affected')
plt.ylabel('Total Females affected')
plt.plot(aa,bb, 'r--')
plt.scatter(m,f)
days_list5 = []
while len(days_list5)<20:
    xx = random.randint(0,242)
    if xx%2==0:
        days_list5.append(xx)
male_vs_female=[]
for t in days_list5:
    male_vs_female.append([dataset_tg['deceased'][t],dataset_tg['deceased'][t+1]])
male_vs_female
m,f = [],[]
for t in male_vs_female:
    m.append(t[0])
    f.append(t[1])
print(m)
print(f)        
# In contrast to number of infections, the number of accumulated deceased graph shifts closer to men's side.
#IMPORTANT INSIGHT: number of infections is much higher in females but death rate in females is lower.
bb = [1,200]
aa = [1,200]
plt.xlabel('Total Males deceased')
plt.ylabel('Total Females deceased')
plt.plot(aa,bb, 'r--')
plt.scatter(m,f)
dataset_time = pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv')
dataset_time.head() # Data collected over 163 days starting from 20th January 2020
# dataset_time.shape
day_by_day_test = [dataset_time['test'][0]]
day_by_day_confirmed = [dataset_time['confirmed'][0]]
day_by_day_deceased = [dataset_time['deceased'][0]]
all_test,all_confirmed,all_deceased = [dataset_time['test'][0]],[dataset_time['confirmed'][0]],[dataset_time['deceased'][0]]
for i in range(1,163):
    day_by_day_test.append(dataset_time['test'][i]-dataset_time['test'][i-1])
    day_by_day_confirmed.append(dataset_time['confirmed'][i]-dataset_time['confirmed'][i-1])
    day_by_day_deceased.append(dataset_time['deceased'][i]-dataset_time['deceased'][i-1])
    all_test.append(dataset_time['test'][i])
    all_confirmed.append(dataset_time['confirmed'][i])
    all_deceased.append(dataset_time['deceased'][i])
# sns.lineplot(x=range(163),y=day_by_day_deceased)
plt.figure(figsize=(15,5))
plt.plot(day_by_day_deceased)
plt.title('COVID-19 deceased figures (daily)')
plt.ylabel('Deceased')
plt.xticks(ticks=[0,11,40,71,101,132] ,labels = ['Jan','Feb','Mar','Apr','May','Jun'])
plt.show()
# sns.lineplot(x=range(163),y=day_by_day_test)
plt.figure(figsize=(15,5))
plt.plot(day_by_day_test)
plt.title('COVID-19 testing figures (daily)')
plt.ylabel('Number of tests done')
plt.xticks(ticks=[0,11,40,71,101,132] ,labels = ['Jan','Feb','Mar','Apr','May','Jun'])
plt.show()
# sns.lineplot(x=range(163),y=day_by_day_confirmed)
plt.figure(figsize=(15,5))
plt.plot(day_by_day_confirmed)
plt.title('COVID-19 confirmed cases (daily)')
plt.ylabel('New patients')
plt.xticks(ticks=[0,11,40,71,101,132] ,labels = ['Jan','Feb','Mar','Apr','May','Jun'])
plt.show()
dataset_search = pd.read_csv('/kaggle/input/coronavirusdataset/SearchTrend.csv')
dataset_search.head()
dataset_search.shape
dataset_search['date'][1276]
search_2018_cold = dataset_search.iloc[731:911,[1]].values
#search_2018_cold
search_2019_cold = dataset_search.iloc[1096:1276,[1]].values
search_2020_cold = dataset_search.iloc[1461:1642,[1]].values

search_2018_flu = dataset_search.iloc[731:911,[2]].values
search_2019_flu = dataset_search.iloc[1096:1276,[2]].values
search_2020_flu = dataset_search.iloc[1461:1642,[2]].values

search_2018_pneumonia = dataset_search.iloc[731:911,[3]].values
search_2019_pneumonia = dataset_search.iloc[1096:1276,[3]].values
search_2020_pneumonia = dataset_search.iloc[1461:1642,[3]].values

search_2018_corona = dataset_search.iloc[731:911,[4]].values
search_2019_corona = dataset_search.iloc[1096:1276,[4]].values
search_2020_corona = dataset_search.iloc[1461:1642,[4]].values
# NEGLICTING THE MOMENTARY OUTLIER, THE SEARCH VOLUME OF 'COLD' WAS HIGHER IN 2020 SIGNIFICANTLY
plt.plot(search_2018_cold)
plt.plot(search_2019_cold)
plt.plot(search_2020_cold)

plt.title('Internet search volume of the word COLD')
plt.xlabel('Days since start of the year')
plt.ylabel('Search Volume')
plt.xticks(ticks=[0,32,60,91,121,152], labels=['January','February','March','April','May','June'])
plt.legend(labels=['2018','2019','2020'])
# SEARCH VOLUME OF 'FLU' WAS USUAL, NO SIGNIFICANT CHANGE
plt.plot(search_2018_flu)
plt.plot(search_2019_flu)
plt.plot(search_2020_flu)

plt.title('Internet search volume of the word FLU')
plt.xlabel('Days since start of the year')
plt.ylabel('Search Volume')
plt.xticks(ticks=[0,32,60,91,121,152], labels=['January','February','March','April','May','June'])
plt.legend(labels=['2018','2019','2020'])
# 'PNEUMONIA' WAS SEARCHED IN MUCH HIGHER VOLUME, ATLEAST 100x HIGHER
plt.plot(search_2018_pneumonia)
plt.plot(search_2019_pneumonia)
plt.plot(search_2020_pneumonia)

plt.title('Internet search volume of the word PNEUMONIA')
plt.xlabel('Days since start of the year')
plt.ylabel('Search Volume')
plt.xticks(ticks=[0,32,60,91,121,152], labels=['January','February','March','April','May','June'])
plt.legend(labels=['2018','2019','2020'])
# 'CORONAVIRUS' SEARCHES WERE 1000x HIGHER IN VOLUME
plt.plot(search_2018_corona)
plt.plot(search_2019_corona)
plt.plot(search_2020_corona)

plt.title('Internet search volume of the word CORONAVIRUS')
plt.xlabel('Days since start of the year')
plt.ylabel('Search Volume')
plt.xticks(ticks=[0,32,60,91,121,152], labels=['January','February','March','April','May','June'])
plt.legend(labels=['2018','2019','2020'])
dataset_case = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')
dataset_case.head()
dataset_case['group'].value_counts()
# Group infection rates were 2.5x higher.
sns.barplot(x=['Group Infection' , 'Individual Infection'] , y=[124,50])
dataset_policy = pd.read_csv('/kaggle/input/coronavirusdataset/Policy.csv')
dataset_policy.head(10)
# dataset_policy.info()
# dataset_policy['type'].value_counts()
dataset_policy.shape
dataset_policy
#Key Details:
#1. Level 1(Blue) Infectious Disease Alert was announced by the govt as early as 3rd January
#2. Level 2(Yellow) Infectious Disease Alert was announced by the govt on 20th January, l3 on 28th jan
#3. The highest Red Alert was announced on 23rd February
#4. Immigration procedure was started to be heavily scrutinized starting from 4th Feb. It was limited to Chinese immigrants initially.
#5. Special immigration procedure was mandated on all countries on 19th March
#6. Drive through screening centre was first started by local govt on 26 Feb
#7. All Schools were shut on 2nd March
#8. High School online classes started on 9th April, rest of the classes from next week.
dataset_weather = pd.read_csv('/kaggle/input/coronavirusdataset/Weather.csv')
dataset_weather = dataset_weather.sort_values(by='code')

seoul_data = dataset_weather.iloc[:1642,:]
dataset_weather.head()
seoul_data.tail()
seoul_data = seoul_data.sort_values(by='date')
seoul_data_2020 = seoul_data.iloc[1461:,:]
seoul_data_2020.head()
seoul_weather_2020 = seoul_data_2020.iloc[:,[3,9]].values
seoul_weather_2020[0:5]
seoul_2020_temp = []
seoul_2020_humid = []

for i in seoul_weather_2020:
    seoul_2020_temp.append(i[0])
    seoul_2020_humid.append(i[1])
    
plt.figure(figsize=(15,5))
plt.plot(seoul_2020_humid)
plt.plot(day_by_day_confirmed)
plt.xlabel('Month')
plt.title('Correlation between new cases and average temperature')
plt.xticks(ticks=[0,32,60,91,121,152], labels=['January','February','March','April','May','June'])
plt.legend(labels=['Average Temperature','New Cases'])
plt.figure(figsize=(15,5))
plt.plot(all_deceased)
plt.title('COVID-19 deceased figures (so far)')
plt.ylabel('Deceased')
plt.xticks(ticks=[0,11,40,71,101,132] ,labels = ['Jan','Feb','Mar','Apr','May','Jun'])
plt.show()
plt.figure(figsize=(15,5))
plt.plot(all_test)
plt.title('COVID-19 testing figures (cumulative)')
plt.ylabel('Total Tests Done')
plt.xticks(ticks=[0,11,40,71,101,132] ,labels = ['Jan','Feb','Mar','Apr','May','Jun'])
plt.show()
plt.figure(figsize=(15,5))
plt.plot(all_confirmed)
plt.title('COVID-19 confirmed figures (cumulative)')
plt.ylabel('Total Confirmed')
plt.xticks(ticks=[0,11,40,71,101,132] ,labels = ['Jan','Feb','Mar','Apr','May','Jun'])
plt.show()
