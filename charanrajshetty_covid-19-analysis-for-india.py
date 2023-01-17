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
df = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

df.head()
print("Size/Shape of the dataset: ",df.shape)

print("Checking for null values:\n",df.isnull().sum())

print("Checking Data-type of each column:\n",df.dtypes)



#df['Time']
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)

df.head()

#df.drop(['Date'],axis=1)
#pd.set_option('display.max_rows', None)

Analysis = df.groupby(["Date"]).agg({"Confirmed":'sum',"Cured":'sum',"Deaths":'sum'})

Analysis["Days Since"] = Analysis.index-Analysis.index.min()

Analysis
print("Total no of confirmed cases till date :",Analysis['Confirmed'].iloc[-1])

print("Total no of deaths till date :",Analysis['Deaths'].iloc[-1])

print("Total no of cured cases till date :",Analysis['Cured'].iloc[-1])

print("Total no of active cases till date :",Analysis['Confirmed'].iloc[-1]-Analysis['Deaths'].iloc[-1]-Analysis['Cured'].iloc[-1])

print("Total no of confirmed cases per day :",np.round((Analysis['Confirmed'].iloc[-1])/(Analysis.shape[0])))

print("Total no of death cases per day :",np.round((Analysis['Deaths'].iloc[-1])/(Analysis.shape[0])))

print("Total no of cured cases per day :",np.round((Analysis['Cured'].iloc[-1])/(Analysis.shape[0])))

print("Total no of confirmed cases per hour :",np.round((Analysis['Confirmed'].iloc[-1])/((Analysis.shape[0])*24)))

print("Total no of death cases per hour :",np.round((Analysis['Deaths'].iloc[-1])/((Analysis.shape[0])*24)))

print("Total no of cured cases per hour :",np.round((Analysis['Cured'].iloc[-1])/((Analysis.shape[0])*24)))





from matplotlib import pyplot as plt

import seaborn as sns



#Date = Analysis.index.date

#print(Date)

#Total_confirmed = Analysis['Confirmed']

#Total_deaths    = Analysis['Deaths']

#Total_cured     = Analysis['Cured']



#fig,axs = plt.subplots(2,2,figsize=(20,15))

plt.figure(figsize=(15,5))

#plt.plot(Date,Total_confirmed)



plt.plot(Analysis.index.date,Analysis["Confirmed"],linewidth='5',color='b')

plt.plot(Analysis.index.date,Analysis["Cured"],linewidth='5',color='g')

plt.plot(Analysis.index.date,Analysis["Deaths"],linewidth='5',color='r')



plt.legend(['Confirmed','Cured','Deaths'])

plt.title("Curve of Confirmed V/s Cured V/s Death",fontsize=20)

plt.xlabel("Month",fontsize=20)

plt.ylabel("Total no of cases",fontsize=20)

plt.xticks(rotation=90)
Analysis["Mortality Rate"]=(Analysis["Deaths"]/Analysis["Confirmed"])*100

Analysis["Recovery Rate"]=(Analysis["Cured"]/Analysis["Confirmed"])*100



plt.style.use('seaborn')

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,6))

ax1.plot(Analysis["Mortality Rate"],linewidth=1,color='b')

ax1.set_title('Mortality Rates',fontsize=20)

ax1.set_xlabel('Month',fontsize=20)

ax1.set_ylabel('Total count',fontsize=20)



ax2.plot(Analysis["Recovery Rate"],linewidth=1,color='b')

ax2.set_title('Recovery Rates',fontsize=20)

ax2.set_xlabel('Month',fontsize=20)

ax2.set_ylabel('Total count',fontsize=20)

#print(Analysis["Mortality Rate"])

Analysis.head()
import plotly.express as px

fig = px.bar( df.loc[(df['State/UnionTerritory']=='Maharashtra')&(df.Date >= '2020-03-01')].sort_values('Confirmed',ascending = False),x='Date', y='Confirmed',

             color="Confirmed", color_continuous_scale=px.colors.sequential.Brwnyl)

fig.update_layout(title_text='Confirmed COVID-19 cases in Maharashtra')

fig.show()
import plotly.express as px

fig = px.bar( df.loc[(df['State/UnionTerritory']=='Delhi')&(df.Date >= '2020-03-01')].sort_values('Confirmed',ascending = False),x='Date', y='Confirmed',

             color="Confirmed",color_continuous_scale=px.colors.sequential.BuGn)

fig.update_layout(title_text='Confirmed COVID-19 cases in Delhi')

fig.show()
import plotly.express as px

fig = px.bar( df.loc[(df['State/UnionTerritory']=='Tamil Nadu')&(df.Date >= '2020-03-01')].sort_values('Confirmed',ascending = False),x='Date', y='Confirmed',

             color="Confirmed",color_continuous_scale=px.colors.sequential.BuGn)

fig.update_layout(title_text='Confirmed COVID-19 cases in Tamil Nadu')

fig.show()
df
pd.set_option('display.max_rows', None)

Wrst_Afctd_ST= df.groupby(["State/UnionTerritory"]).agg({"Confirmed":'sum'})

Wrst_Afctd_ST



States    = Wrst_Afctd_ST.index

Confirmed = list(Wrst_Afctd_ST['Confirmed'])

#print(Confirmed)

#print(States)

#Wrst_Afctd_ST

plt.figure(figsize=(20,20))

plt.title("Pie Chart of Worst Hit States",fontsize=30)

plt.pie(Confirmed,labels=States,autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.show()


ST_testing = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')





States = (list(ST_testing['State'].unique()))

States

Total_Testing = []



for item in States:

    Total_Testing.append(ST_testing[ST_testing['State']==item]['TotalSamples'].max())



##state_testing =    

#print(Total_Testing)



india_testing = pd.DataFrame(list(zip(States,Total_Testing)),columns=['States','Total_Testing'])



india_testing
plt.figure(figsize=(20,10))

plt.barh(States,Total_Testing)

plt.xlabel("Total test count",fontsize=30)

plt.ylabel("States",fontsize=30)

plt.title("Maximum testing Statewise",fontsize=30)



plt.show()
State = list(df['State/UnionTerritory'].unique())

#print(State)



Confirmed = []

Deaths    = []

Cured = []

for item in State:

    Confirmed.append(df[df['State/UnionTerritory']==item]['Confirmed'].max())

    Deaths.append(df[df['State/UnionTerritory']==item]['Deaths'].max())

    Cured.append(df[df['State/UnionTerritory']==item]['Cured'].max())

    

#print(Confirmed)



Statewise = pd.DataFrame(list(zip(State,Confirmed,Deaths,Cured)),columns=['State','Confirmed','Deaths','Cured'])

Statewise['Mortality'] = (Statewise['Deaths']/Statewise['Confirmed'])*100

Statewise['Recovery'] = (Statewise['Cured']/Statewise['Confirmed'])*100

Statewise
mortality = Statewise['Mortality']

plt.figure(figsize=(20,10))

plt.bar(State,mortality,color='r')

plt.xlabel("States",fontsize=30)

plt.ylabel("Mortality rate",fontsize=30)

plt.title("Mortality Rate Statewise",fontsize=30)

plt.xticks(rotation=90)

plt.show()
recovery = Statewise['Recovery']

plt.figure(figsize=(20,10))

plt.bar(State,recovery,color='g')

plt.xlabel("States",fontsize=30)

plt.ylabel("Recovery rate",fontsize=30)

plt.title("Recovery Rate Statewise",fontsize=30)

plt.xticks(rotation=90)

plt.show()
df_lab = pd.read_csv('../input/icmrtestinglabswithcoords1/datasets_624680_1113799_ICMRTestingLabsWithCoords.csv')

fig = px.scatter_mapbox(df_lab,

                        lat="latitude",

                        lon="longitude",

                        mapbox_style='stamen-watercolor',

                        hover_name='lab',

                        hover_data=['city','state','pincode'],

                        zoom=2.5,

                        size_max=15,

                        title= 'COVID19 Testing Labs in India')

fig.show()
AgeDF = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')

AgeDF



AgeGroup = AgeDF['AgeGroup']

slice    = AgeDF['TotalCases']



plt.figure(figsize=(20,20))

plt.title("Pie Chart of Pandaemic impact Age group wise",fontsize=30)

plt.pie(slice,labels=AgeGroup,autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.show()
