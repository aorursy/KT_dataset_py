!pip install calmap

import calmap
import numpy as np # storing and anaysis
import pandas as pd
!pip install plotly
import matplotlib.pyplot as plt # visualization
import matplotlib.dates as mdates
from matplotlib import rcParams, pyplot as plt, style as style
import seaborn as sns
!pip install folium
from plotnine import * 
import plotly.express as px
import folium
c = '#393e46' # color pallette
d = '#ff2e63'
r = '#30e3ca'
i = '#f8b400'
cdr = [c, d, r] # grey - red - blue
idr = [i, d, r] # yellow - red - blue
# importing datasets
full = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
full.head()
full.info()
full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')
full_grouped.head()

full_grouped.info()

full['Present_infected'] = full['Confirmed'] - full['Deaths'] -full['Recovered']
full_grouped['Country/Region'].unique() 
full['Country/Region'].unique() 
# checking for missing value
full.isna().sum()
# filling missing values with NA
full[['Province/State']] = full[['Province/State']].fillna('NA')

full[['Confirmed', 'Deaths', 'Recovered', 'Present_infected']] = full[['Confirmed', 'Deaths', 'Recovered', 'Present_infected']].fillna(0)
#full['Country/Region'].unique() 
full = full.rename(columns={"Province/State":"state","Country/Region": "country"})
full_grouped = full_grouped.rename(columns={"Country/Region": "country"})
full['country'].unique() 
full['country'].value_counts().head(10) 

# complete dataset 
# complete = full_table.copy()



full.loc[full['country'] == "US", "country"] = "USA"  
full.loc[full['country'] == "United Kingdom", "country"] = "UK"

full['country'].unique()
full_grouped['country'].unique() 




full["Date1"] = full["Date"].astype(str)  
full['year'] = full['Date1'].str.split('-', expand=True)[0] 
full['month'] = full['Date1'].str.split('-', expand=True)[1] 
full['day'] = full['Date1'].str.split('-', expand=True)[2] 

SouthKore = full[full['country']=='South Korea'] 
usa = full[full['country']=='USA']
uk = full[full['country']=='UK'] 

full_SouthKore = full_grouped[full_grouped['country']=='South Korea'] 

increase_confirm = []
increase_recovered = []
increase_deaths = []

for i in range(SouthKore.shape[0]-1):
    increase_confirm.append((SouthKore["Confirmed"].iloc[i+1]-SouthKore["Confirmed"].iloc[i])) 
    increase_recovered.append((SouthKore["Recovered"].iloc[i+1]-SouthKore["Recovered"].iloc[i])) 
    increase_deaths.append((SouthKore["Deaths"].iloc[i+1]-SouthKore["Deaths"].iloc[i]))

increase_confirm.insert(0,0) 
increase_recovered.insert(0,0)
increase_deaths.insert(0,0)
SouthKore.head()







SouthKore.set_index('Date1', inplace=True) 
SouthKore.info()

plt.figure(figsize=(12,6))
plt.plot(SouthKore["Confirmed"],marker="o",label="Confirmed Cases") 
plt.plot(SouthKore["Recovered"],marker="*",label="Recovered Cases") 
plt.plot(SouthKore["Deaths"],marker="^",label="Death Cases") 
plt.ylabel("Number of Cases", fontsize = 18)
plt.xlabel("Date", fontsize = 18)
plt.xticks(['2020-01-22', '2020-02-01', '2020-02-20', '2020-03-01', '2020-03-20', '2020-04-01', '2020-04-20', '2020-05-01', '2020-05-20', '2020-06-01', '2020-07-01', '2020-07-20'], rotation=90) 
#plt.xticks(rotation=90)
plt.title("Growth of different Types of cumulative Cases over Time in south korea", fontsize = 20)
plt.legend()
plt.figure(figsize=(20,10))
plt.plot(SouthKore.index,increase_confirm,label="Number of Confiremd Cases",marker='o') 
plt.plot(SouthKore.index,increase_recovered,label="Number of Recovered Cases",marker='*')
plt.plot(SouthKore.index,increase_deaths,label="Number of Death Cases",marker='^')
plt.xlabel("Timestamp")
plt.ylabel("Number")
plt.title("Number of different Types of Cases in South korea")
plt.xticks(rotation=90) 
plt.legend()

SouthKore.info()


SouthKore['increase_confirm'] = increase_confirm  
SouthKore['increase_recovered'] = increase_recovered  #
SouthKore['increase_deaths'] = increase_deaths  


SouthKore1 = SouthKore.groupby('month')['increase_confirm'].sum() 


SouthKore1 = SouthKore.groupby('month')['increase_confirm', 'increase_recovered', 'increase_deaths'].sum()


SouthKore1 = pd.DataFrame(SouthKore1, columns=['increase_confirm', 'increase_recovered', 'increase_deaths'])
SouthKore1.reset_index(inplace=True) 
sns.barplot(x="month", y="increase_confirm", data=SouthKore1)  
plt.title("Total number of New cases by month in South korea(COVID-19)  - 2020yr")
style.use('ggplot')  
rcParams['figure.figsize'] = 8,5 
ax = sns.barplot(x = 'month', y="increase_confirm", data=SouthKore1)
plt.title("Total number of New cases by month in South korea(COVID-19)  - 2020yr")
provinces = np.asarray(SouthKore1['increase_confirm']) 
plt.title('Total number of New cases by month in South korea(COVID-19)  - 2020yr ')
plt.bar(SouthKore1['month'],provinces,color='teal') 
plt.ylabel('No. of increase confirm cases')
for j,provinces in enumerate(provinces):
    plt.text(j,provinces+2, provinces,horizontalalignment='center') 
plt.xticks(rotation=45)
plt.show()
provinces = np.asarray(SouthKore1['increase_confirm']) 
provincesi = np.asarray(SouthKore1['increase_recovered']) 
provincesy = np.asarray(SouthKore1['increase_deaths'])
plt.rcParams['figure.figsize'] = (13, 5)
plt.subplot(1, 3, 1) 
sns.barplot(x="month", y="increase_confirm", data=SouthKore1) 
plt.title('increase_confirm by month', fontsize = 15)
plt.xlabel('Month', fontsize = 15)
plt.ylabel('count', fontsize = 15)
for j,provinces in enumerate(provinces):
    plt.text(j,provinces+2, provinces,horizontalalignment='center') 
plt.subplot(1, 3, 2) 
sns.barplot(x="month", y="increase_recovered", data=SouthKore1) 
plt.title('increase_recovered by month', fontsize = 15)
plt.xlabel('Month', fontsize = 15)
plt.ylabel('count', fontsize = 15)
for j,provincesi in enumerate(provincesi):
    plt.text(j,provincesi+2, provincesi,horizontalalignment='center') 
plt.subplot(1, 3, 3)
sns.barplot(x="month", y="increase_deaths", data=SouthKore1) 
plt.title('increase_deaths by month', fontsize = 15)
plt.xlabel('Month', fontsize = 15)
plt.ylabel('count', fontsize = 15)
for j,provincesy in enumerate(provincesy):
    plt.text(j,provincesy+2, provincesy,horizontalalignment='center') 
plt.show()


Confirmedtop = full[['country', 'Confirmed']].groupby(['country'])['Confirmed'].max().reset_index()  

Confirmedtop10 = Confirmedtop.sort_values(by='Confirmed', ascending=False).head(10) 
Confirmedtop10.head(10)

Worldconfirmed10 = np.asarray(Confirmedtop10['Confirmed']) 
plt.figure(figsize=(10,5))
plt.title('present number of cumulative confirmed cases by country(COVID-19)  - TOP 10 ')
plt.bar(Confirmedtop10['country'], Worldconfirmed10, color='teal') 
plt.xlabel('Top 10 Country')
plt.ylabel('No. of cumulative confirmed cases')
for j,Worldconfirmed10 in enumerate(Worldconfirmed10):
    plt.text(j,Worldconfirmed10+2, Worldconfirmed10,horizontalalignment='center') #
plt.xticks(rotation=45)
plt.show()
Topcountry = full[(full['country']=='UK') | (full['country']=='USA') | (full['country']=='Brazil') | (full['country']=='South Korea') | (full['country']=='India') | (full['country']=='Russia') ]
Topcountry.set_index('Date1', inplace=True) 

Topcountry.head()
Topcountry['country'].unique()

southKr = Topcountry[Topcountry['country']=='South Korea'] 
usa = Topcountry[Topcountry['country']=='USA'] 
uk = Topcountry[Topcountry['country']=='UK'] 
bz = Topcountry[Topcountry['country']=='Brazil'] 
india = Topcountry[Topcountry['country']=='India'] 

usa.info()

plt.figure(figsize=(10,5))

plt.rcParams['figure.figsize'] = (15, 7)
plt.subplot(2, 2, 1) 
plt.plot(usa["Confirmed"],marker="o",label="Confirmed Cases") 
plt.plot(usa["Recovered"],marker="*",label="Recovered Cases")  
plt.plot(usa["Deaths"],marker="^",label="Death Cases") 
plt.ylabel("Number of Cases", fontsize = 10)
plt.xlabel("Date", fontsize = 10)
plt.xticks(['2020-01-22', '2020-02-01', '2020-02-20', '2020-03-01', '2020-03-20', '2020-04-01', '2020-04-20', '2020-05-01', '2020-05-20', '2020-06-01', '2020-07-01', '2020-07-20'], rotation=90) #x축 눈금값이 너무 많아 특정 값만 지정하기. 그리고, 값 표시를 90도 회전시켜 표시.
#plt.xticks(rotation=90)
plt.title("Growth of different Types of cumulative Cases over Time in USA", fontsize = 10)
plt.legend()
plt.subplot(2, 2, 2) 
plt.plot(southKr["Confirmed"],marker="o",label="Confirmed Cases") 
plt.plot(southKr["Recovered"],marker="*",label="Recovered Cases")  
plt.plot(southKr["Deaths"],marker="^",label="Death Cases") 
plt.ylabel("Number of Cases", fontsize = 10)
plt.xlabel("Date", fontsize = 10)
plt.xticks(['2020-01-22', '2020-02-01', '2020-02-20', '2020-03-01', '2020-03-20', '2020-04-01', '2020-04-20', '2020-05-01', '2020-05-20', '2020-06-01', '2020-07-01', '2020-07-20'], rotation=90) #x축 눈금값이 너무 많아 특정 값만 지정하기. 그리고, 값 표시를 90도 회전시켜 표시.
#plt.xticks(rotation=90)
plt.title("Growth of different Types of cumulative Cases over Time in south korea", fontsize = 10)
plt.legend()
plt.subplot(2, 2, 3) 
plt.plot(india["Confirmed"],marker="o",label="Confirmed Cases") 
plt.plot(india["Recovered"],marker="*",label="Recovered Cases")  
plt.plot(india["Deaths"],marker="^",label="Death Cases") 
plt.ylabel("Number of Cases", fontsize = 10)
plt.xlabel("Date", fontsize = 10)
plt.xticks(['2020-01-22', '2020-02-01', '2020-02-20', '2020-03-01', '2020-03-20', '2020-04-01', '2020-04-20', '2020-05-01', '2020-05-20', '2020-06-01', '2020-07-01', '2020-07-20'], rotation=90) #x축 눈금값이 너무 많아 특정 값만 지정하기. 그리고, 값 표시를 90도 회전시켜 표시.
#plt.xticks(rotation=90)
plt.title("Growth of different Types of cumulative Cases over Time in south korea", fontsize = 10)
plt.legend()
plt.subplot(2, 2, 4) 
plt.plot(bz["Confirmed"],marker="o",label="Confirmed Cases") 
plt.plot(bz["Recovered"],marker="*",label="Recovered Cases") 
plt.plot(bz["Deaths"],marker="^",label="Death Cases") 
plt.ylabel("Number of Cases", fontsize = 10)
plt.xlabel("Date", fontsize = 10)
plt.xticks(['2020-01-22', '2020-02-01', '2020-02-20', '2020-03-01', '2020-03-20', '2020-04-01', '2020-04-20', '2020-05-01', '2020-05-20', '2020-06-01', '2020-07-01', '2020-07-20'], rotation=90) #x축 눈금값이 너무 많아 특정 값만 지정하기. 그리고, 값 표시를 90도 회전시켜 표시.
#plt.xticks(rotation=90)
plt.title("Growth of different Types of cumulative Cases over Time in south korea", fontsize = 10)
plt.legend()
plt.show()






