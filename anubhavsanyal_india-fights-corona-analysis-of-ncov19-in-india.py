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
age_data=pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
print(age_data)
age_X=age_data.iloc[0:9,1]
age_y=age_data.iloc[0:9,2]
age_group_list=[]
age_cases_list=[]
for i in range(0,9):
    age_group_list.append(age_X[i])
    age_cases_list.append(age_y[i])
print(age_group_list)
print(age_cases_list)
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(8,8))
plt.pie(age_cases_list,labels=age_group_list,autopct = '%1.1f%%')
plt.legend()
plt.title('India-Covid19 Cases vs Age')
data_gender=pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')
data_gender
gender=data_gender.iloc[:,4]
count_male=0
count_female=0
for i in range(0,27889):
    if(gender[i]=='M'):
        count_male=count_male+1
    elif(gender[i]=='F'):
        count_female=count_female+1
print(count_male)
print(count_female)
gender_list=[count_male,count_female]
print(gender_list)
gender_label=['Male','Female']
plt.figure(figsize=(5,5))
plt.pie(gender_list,labels=gender_label,autopct = '%1.1f%%')
plt.legend()
plt.title('India-Covid19 cases vs Gender')

data_cases=pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv')
data_cases
country_name=data_cases.iloc[:,1]
country_confirmed=data_cases.iloc[:,2]
country_recovered=data_cases.iloc[:,3]
country_deaths=data_cases.iloc[:,4]
india_confirmed=[]
india_recovered=[]
india_deaths=[]
for i in range(0,len(data_cases)):
    if(country_name[i]=='India'):
        india_confirmed.append(country_confirmed[i])
        india_recovered.append(country_recovered[i])
        india_deaths.append(country_deaths[i])
print(india_confirmed)
print(india_recovered)
print(india_deaths)
days=[]
day=1
for i in range(0,len(data_cases)):
    if(country_name[i]=='India'):
        days.append(day)
        day=day+1
        
len(days)
plt.figure(figsize=(15,10))
plt.xticks(rotation=90,fontsize=12)

g_1=plt.plot(days,india_confirmed)
g_2=plt.plot(days,india_recovered,color='green')
g_3=plt.plot(days,india_deaths,color='red')
plt.title('Covid-19 India -- Confirmed vs Recovered vs Death')
plt.xlabel('Days',fontsize=18)
plt.ylabel('Total Cases',fontsize=18)
plt.legend()

confirmed_china=[]
confirmed_italy=[]
confirmed_germany=[]
confirmed_uk=[]
confirmed_us=[]
for i in range(0,len(data_cases)):
    if(country_name[i]=='China'):
        confirmed_china.append(country_confirmed[i])
    elif(country_name[i]=='Italy'):
        confirmed_italy.append(country_confirmed[i])
    elif(country_name[i]=='Germany'):
        confirmed_germany.append(country_confirmed[i])
    elif(country_name[i]=='United Kingdom'):
        confirmed_uk.append(country_confirmed[i])
    elif(country_name[i]=='US'):
        confirmed_us.append(country_confirmed[i])
len(confirmed_us)
fig=plt.figure(figsize=(15,10))
plt.plot(days,india_confirmed,label='India')
plt.plot(days,confirmed_china,label='China')
plt.plot(days,confirmed_uk,label='United Kingdom')
plt.plot(days,confirmed_italy,label='Italy')
plt.plot(days,confirmed_germany,label='Germany')
plt.plot(days,confirmed_us,label='USA')
plt.yticks(rotation=90,fontsize=12)
plt.xlabel('Days',fontsize=12)
plt.ylabel('Cases')
plt.legend()
plt.show()
data_test_state=pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')
data_test_state
state_testing_date=data_test_state.iloc[:,0]
state_testing_name=data_test_state.iloc[:,1]
state_testing_samples=data_test_state.iloc[:,2]
testing_name_state=[]
testing_samples=[]
for i in range(0,len(data_test_state)):
    if(state_testing_date[i]=='2020-05-20'):
        testing_samples.append(state_testing_samples[i])
        testing_name_state.append(state_testing_name[i])
       
figure=plt.figure(figsize=(15,10))
plt.bar(testing_name_state,testing_samples,color='orange')
plt.xticks(rotation=90,fontsize=12)
plt.show()
data_states_cases=pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
data_states_cases
data_states_date=data_states_cases.iloc[:,1]
data_states_name=data_states_cases.iloc[:,3]
data_states_cc=data_states_cases.iloc[:,8]
delhi_confirmed=[]
maharashtra_confirmed=[]
tamilnadu_confirmed=[]
uttarpradesh_confirmed=[]
westbengal_confirmed=[]
madhyapradesh_confirmed=[]
chhattisgarh_confirmed=[]
andrapradesh_confirmed=[]
delhi_date=[]
maharashtra_date=[]
tamilnadu_date=[]
uttarpradesh_date=[]
westbengal_date=[]
madhyapradesh_date=[]
chhattisgarh_date=[]
andrapradesh_date=[]
for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='Delhi'):
        delhi_confirmed.append(data_states_cc[i])
        delhi_date.append(data_states_date[i])
    
figure=plt.figure(figsize=(30,10))
plt.plot(delhi_date,delhi_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('Delhi',fontsize=20)
plt.show()


for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='Maharashtra'):
        maharashtra_confirmed.append(data_states_cc[i])
        maharashtra_date.append(data_states_date[i])
figure=plt.figure(figsize=(30,10))
plt.plot(maharashtra_date,maharashtra_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('Maharashtra',fontsize=20)
plt.show()
for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='Uttar Pradesh'):
        uttarpradesh_confirmed.append(data_states_cc[i])
        uttarpradesh_date.append(data_states_date[i])
figure=plt.figure(figsize=(30,10))
plt.plot(uttarpradesh_date,uttarpradesh_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('Uttar Pradesh',fontsize=20)
plt.show()
for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='Madhya Pradesh'):
        madhyapradesh_confirmed.append(data_states_cc[i])
        madhyapradesh_date.append(data_states_date[i])
figure=plt.figure(figsize=(30,10))
plt.plot(madhyapradesh_date,madhyapradesh_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('Madhya Pradesh',fontsize=20)
plt.show()
for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='Chhattisgarh'):
        chhattisgarh_confirmed.append(data_states_cc[i])
        chhattisgarh_date.append(data_states_date[i])
figure=plt.figure(figsize=(30,10))
plt.plot(chhattisgarh_date,chhattisgarh_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('Chhattisgarh',fontsize=20)
plt.show()
for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='Tamil Nadu'):
        tamilnadu_confirmed.append(data_states_cc[i])
        tamilnadu_date.append(data_states_date[i])
figure=plt.figure(figsize=(30,10))
plt.plot(tamilnadu_date,tamilnadu_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('Tamil Nadu',fontsize=20)
plt.show()
for i in range(0,len(data_states_cases)):
    if(data_states_name[i]=='West Bengal'):
        westbengal_confirmed.append(data_states_cc[i])
        westbengal_date.append(data_states_date[i])
figure=plt.figure(figsize=(30,10))
plt.plot(westbengal_date,westbengal_confirmed)
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('Date',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.title('West Bengal',fontsize=20)
plt.show()
state_hos_details=pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')
state_hos_details
hospital_state_name=state_hos_details.iloc[0:35,1]
hospital_state_primary_hos=state_hos_details.iloc[0:35,2]
hospital_state_community_hos=state_hos_details.iloc[0:35,3]
hospital_state_subd_hos=state_hos_details.iloc[0:35,4]
hospital_state_d_hos=state_hos_details.iloc[0:35:,5]
hospital_state_total_hos=state_hos_details.iloc[0:35:,6]

hospital_state_name=np.asarray(hospital_state_name)
hospital_state_primary_hos=np.asarray(hospital_state_primary_hos)
hospital_state_community_hos=np.asarray(hospital_state_community_hos)
hospital_state_subd_hos=np.asarray(hospital_state_subd_hos)
hospital_state_d_hos=np.asarray(hospital_state_d_hos)
hospital_state_total_hos=np.asarray(hospital_state_total_hos)
plt.figure(figsize=(15,10))
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('State Name',fontsize=13)
plt.ylabel('Number of Primary HealthCare centres',fontsize=13)
plt.bar(hospital_state_name,hospital_state_primary_hos)
plt.show()
plt.figure(figsize=(10,5))
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('State Name',fontsize=13)
plt.ylabel('Number of Community HealthCare centres',fontsize=13)
plt.bar(hospital_state_name,hospital_state_community_hos)
plt.show()
plt.figure(figsize=(10,5))
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('State Name',fontsize=13)
plt.ylabel('Number of Sub District Hospitals',fontsize=13)
plt.bar(hospital_state_name,hospital_state_subd_hos)
plt.show()
plt.figure(figsize=(10,5))
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('State Name',fontsize=13)
plt.ylabel('Number of District Hospitals',fontsize=13)
plt.bar(hospital_state_name,hospital_state_d_hos)
plt.show()
plt.figure(figsize=(10,5))
plt.xticks(rotation=90,fontsize=12)
plt.xlabel('State Name',fontsize=13)
plt.ylabel('Number of Total Hospitals',fontsize=13)
plt.bar(hospital_state_name,hospital_state_total_hos)
plt.show()
data_cases
growth_list_india=[]
for i in range(0,len(data_cases)):
    if(country_name[i]=='India'):
        growth_list_india.append(country_confirmed[i])
growth_list_india
growth_india=[]
for i in range(0,len(growth_list_india)-1):
    growth_india.append(growth_list_india[i+1]-growth_list_india[i])
    
growth_india
sum_growth=0

for i in range(0,len(growth_india)):
    sum_growth=sum_growth+growth_india[i]
averagr_growth_rate=sum_growth/len(growth_india)
    
averagr_growth_rate
average_rate=[]
for i in range(len(growth_list_india)-20,len(growth_list_india)-1):
    average_rate.append(growth_list_india[i+1]/growth_list_india[i])
    
sum_growth_mul=0
for i in range(0,len(average_rate)):
    
    sum_growth_mul=sum_growth_mul+average_rate[i]


    
sum_growth_mul=sum_growth_mul/len(average_rate)
sum_growth_mul
prediction_for_next_15_days=[]
prediction_for_next_15_days.append(growth_list_india[len(growth_list_india)-1])
for i in range(1,15):
    prediction_for_next_15_days.append(prediction_for_next_15_days[i-1]*sum_growth_mul)
prediction_for_next_15_days
days_predicted=[]
for i in range(0,15):
    days_predicted.append(i)
plt.plot(days_predicted,prediction_for_next_15_days,color='Orange')
plt.xlabel('Days',fontsize=12)
plt.ylabel('Cases',fontsize=12)
plt.title('Prediction for next 15 days',fontsize=15)
