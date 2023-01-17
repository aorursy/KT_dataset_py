

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Reading the dataset

data= pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv",)

data.head()

## Let's look at the various columns

data.info()
new_data=pd.DataFrame(data.groupby(['Country'])['Confirmed'].sum())

new_data['Confirmed_counts']=new_data['Confirmed']

new_data['Recovered_counts']=pd.DataFrame(data.groupby(['Country'])['Recovered'].sum())

new_data['Deaths_counts']=pd.DataFrame(data.groupby(['Country'])['Deaths'].sum())



top20_Country=new_data.sort_values('Confirmed_counts',ascending=False).head(20)

top20_Country=top20_Country.drop('Confirmed',axis=1)

top20_Country
#null Data Control

print(data.isnull().sum().to_frame('nulls'))
#first i don't like 'Province/State' column name so i will change it to 'State'

data['State']=data['Province/State'] 

data=data.drop('Province/State',axis=1)

data.head()
#implement Interpolation method to Complete missing data

data2= data.astype({"State":'category'})

data2["State"]=(data2["State"].cat.codes.replace(-1, np.nan).interpolate().astype(int).astype('category').cat.rename_categories(data2["State"].cat.categories))

print(data2.isnull().sum().to_frame('nulls')) #Control nulls again

new_data2=pd.DataFrame(data2.groupby(['State'])['Confirmed'].sum())

new_data2['Confirmed_counts']=new_data2['Confirmed']

new_data2['Recovered_counts']=pd.DataFrame(data2.groupby(['State'])['Recovered'].sum())

new_data2['Deaths_counts']=pd.DataFrame(data2.groupby(['State'])['Deaths'].sum())



top20_state=new_data2.sort_values('Confirmed_counts',ascending=False).head(20)

x=pd.merge(top20_state,data2,on='State')



x1= x.drop_duplicates(subset=['State'])

most_Infected_State=x1[['State','Confirmed_counts','Recovered_counts','Deaths_counts','Country']]

most_Infected_State
most_Infected_State=most_Infected_State.iloc[0:10,:]



x=most_Infected_State['Confirmed_counts']

y=most_Infected_State['State']

plt.rcParams['figure.figsize'] = (15, 10)



sns.barplot(x,y,order=y ).set_title('Top infected state in china')  #graf çizdir (Most popular)
#Top Recovered cities in China 

top20_state=new_data2.sort_values('Recovered_counts',ascending=False).head(20)

top20_state=top20_state.head(10)

x=top20_state.index

y=top20_state['Recovered_counts']

sns.barplot(x, y, order=x, palette="vlag")
#Top 10 state with deaths case in China

top10_dead_people=new_data2.sort_values('Deaths_counts',ascending=False).head(20)

top10_dead_people=top10_dead_people.drop(['Confirmed_counts','Confirmed','Recovered_counts'],axis=1)

top10_dead_people.head(11)
## Cases of infection according to date

Confirmed_count_date=pd.DataFrame(data2.groupby(['Last Update'])['Confirmed'].sum())

labels=Confirmed_count_date.index

sizes=Confirmed_count_date['Confirmed']

explode = None   # explode 1st slice

plt.subplots(figsize=(8,8))

plt.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%', shadow=False, startangle=0)

plt.axis('equal')



print(Confirmed_count_date)

#infected Cities in every date.

Confirmed_State_date=pd.DataFrame(data.groupby(['Last Update'])['State'].apply(list))

Confirmed_State_date
# which cities infected in this day?

# 1/22/2020 day

day_1_22_2020=Confirmed_State_date.iloc[0:1,:].values

day_1_22_2020
# 1/25/2020 day

day_1_25_2020=Confirmed_State_date.iloc[4:5,:].values

day_1_25_2020