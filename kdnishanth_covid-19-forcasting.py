from IPython.display import Image

Image("../input/covid19/photo-1584036561566-baf8f5f1b144.jpeg", width = "800px")
import folium

import operator 

import folium as f

import pandas as pd

import seaborn as sns

import plotly.express as px

import plotly.offline as py

import matplotlib.pyplot as plt

from IPython.core.display import HTML
Input_Data1 = pd.read_csv('../input/coronavirusdataset/TimeGender.csv')

Input_Data1.head()
Input_Data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

Input_Data.head()
HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1571387"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')

male = 0

female = 0

male_confirmed = []

male_deceased = []

female_confirmed = []

female_deceased = []



for i in range (len(Input_Data1["sex"])):

    if Input_Data1["sex"][i] == 'male':

        male = male + 1

        male_confirmed.append(Input_Data1["confirmed"][i])

        male_deceased.append(Input_Data1["deceased"][i])

    else:

        female = female  + 1

        female_confirmed.append(Input_Data1["confirmed"][i])

        female_deceased.append(Input_Data1["deceased"][i])
df = pd.DataFrame([sum(male_confirmed),sum(male_deceased)], index=['confirmed case', 'Death case'], columns=['Male Data Analysis'])

df.plot(kind='bar', subplots=True, figsize=(10, 10))

df = pd.DataFrame([sum(female_confirmed),sum(female_deceased)], index=['confirmed case', 'Death case'], columns=['Female Data Analysis'])

df.plot(kind='bar', subplots=True, figsize=(10, 10))
def people_count(Cloumn_name):

    _people_count = 0

    for i in Cloumn_name:

        _people_count = _people_count + i

    return _people_count



def COVID_analysis(Column_name,_Column_name):

    Local_Country = []

    Local_List = []

    for i in range(len(Column_name)):

        if Column_name[i] > 0 :

            Local_List.append(Column_name[i])

            Local_Country.append(_Column_name[i])

            

    return Local_List, Local_Country



def InsertionSort(arr): 

    for i in range(1, len(arr)):

        key = arr[i] 

        j = i-1

        while j >= 0 and key < arr[j] : 

                arr[j + 1] = arr[j] 

                j -= 1

        arr[j + 1] = key  

    return arr

def Remove_Duplication(_Permanent_List, _Permanent_List_, Date_List_):

    _Country = []

    _list = []

    Confirmed_Count = []

    for i in range(len(b)): 

        if not (_Permanent_List[i] in _Country):

            _list.append(_Permanent_List[i])

            Confirmed_Count.append(Permanent_List_[i])

    return _Country, Confirmed_Count

No_confirmed_case = people_count(Input_Data["Confirmed"])

No_Recovered_case = people_count(Input_Data["Recovered"])

No_Death_case = people_count(Input_Data["Deaths"])
print("Number of People Affected by COVID-19:"+" "+str(No_confirmed_case))

print("-------------------------------------------------------------------------------------\n\n")

print("Number of People Recovered From COVID-19:"+" "+str(No_Recovered_case))

print("-------------------------------------------------------------------------------------\n\n")

print("Number of People Died for COVID-19:"+" "+str(No_Death_case))

print("-------------------------------------------------------------------------------------")
country_confirmed_cases = []

country_death_cases = [] 

country_active_cases = []

country_recovery_cases = []

country_mortality_rate = [] 



latest_data = Input_Data

unique_countries =  list(latest_data['Country/Region'].unique())



no_cases = []

for i in unique_countries:

    cases = latest_data[latest_data['Country/Region']==i]['Confirmed'].sum()

    if cases > 0:

        country_confirmed_cases.append(cases)

    else:

        no_cases.append(i)

        

for i in no_cases:

    unique_countries.remove(i)

    

# sort countries by the number of confirmed cases

unique_countries = [k for k, v in sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]

for i in range(len(unique_countries)):

    country_confirmed_cases[i] = latest_data[latest_data['Country/Region']==unique_countries[i]]['Confirmed'].sum()

    country_death_cases.append(latest_data[latest_data['Country/Region']==unique_countries[i]]['Deaths'].sum())

    country_recovery_cases.append(latest_data[latest_data['Country/Region']==unique_countries[i]]['Recovered'].sum())

    country_active_cases.append(country_confirmed_cases[i] - country_death_cases[i] - country_recovery_cases[i])

    country_mortality_rate.append(country_death_cases[i]/country_confirmed_cases[i])

#Data table



country_df = pd.DataFrame({'Country Name': unique_countries, 'Number of Confirmed Cases': country_confirmed_cases,

                          'Number of Deaths': country_death_cases, 'Number of Recoveries' : country_recovery_cases, 

                          'Number of Active Cases' : country_active_cases,

                          'Mortality Rate': country_mortality_rate})



# number of cases per country/region



country_df.style.background_gradient(cmap='Greens')
Confirmed_cases_count, Confirmed_cases_Country = COVID_analysis(Input_Data["Confirmed"],Input_Data["Country/Region"])

Recovered_case_count, Date_Cases = COVID_analysis(Input_Data["Recovered"],Input_Data["Date"])

Death_case_count, Date_Cases = COVID_analysis(Input_Data["Deaths"],Input_Data["Date"])



a_ = InsertionSort(Confirmed_cases_count)

b_ = InsertionSort(Recovered_case_count)

c_ = InsertionSort(Death_case_count)



_Confirmed_List = a_[::-1]

_Recovered_List_ = b_[::-1]

_Death_List_ = c_[::-1]

_Country_List = Confirmed_cases_Country[::-1]

Date_Cases_list = Date_Cases[::-1]



_Confirmed_cases_count = []

_Confirmed_cases_Country = []

_Death_case_count = []

_Date_Cases = []
names='confirmed', 'Recovered', 'Death',

size=[No_confirmed_case,No_Recovered_case,No_Death_case]





my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(size, labels=names, colors=['orange','green','blue'])

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title("Total number of cases", size = 20)

plt.show()
map = folium.Map(location=[20.5937, 78.9629], zoom_start=4,tiles='cartodbpositron')



for lat, lon,state,Confirmed,Recovered,Deaths in zip(Input_Data['Lat'], Input_Data['Long'],Input_Data['Country/Region'],Input_Data['Confirmed'],Input_Data['Recovered'],Input_Data['Deaths']):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        color='YlOrRd',

                      popup =(

                    'State: ' + str(state) + '<br>'

                    'Confirmed: ' + str(Confirmed) + '<br>'

                      'Recovered: ' + str(Recovered) + '<br>'

                      'Deaths: ' + str(Deaths) + '<br>'),



                        fill_color='red',

                        fill_opacity=0.7 ).add_to(map)

map