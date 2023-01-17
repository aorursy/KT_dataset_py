import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df=pd.read_csv("../input/covid19-dataset/covid_19_data.csv")
df.head()
df.shape
df.info()
df.isna().sum()
df['Country/Region'].replace('Mainland China','China',inplace=True)
df[["Confirmed","Deaths","Recovered"]] = df[["Confirmed","Deaths","Recovered"]].astype(int) 
df['ObservationDate']=pd.to_datetime(df['ObservationDate'],format="%m/%d/%Y")
df.columns
df.drop(['SNo','Last Update'],inplace=True,axis=1)
df.head()
pd.set_option('display.max_rows', None) # to print all rows in the output

date_wise_data=df.groupby(["ObservationDate"])['Confirmed','Recovered','Deaths'].sum().reset_index()

date_wise_data.head(10)
df['Active_cases'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

df.head()
new_data = df[df['ObservationDate'] == max(df['ObservationDate'])].reset_index()



new_data_details = new_data.groupby(["ObservationDate"])["Confirmed","Active_cases","Recovered","Deaths"].sum().reset_index()

new_data_details
labels = ["Active cases","Recovered","Deaths"]

values = new_data_details.loc[0, ["Active_cases","Recovered","Deaths"]]

plt.pie(values, labels=labels, autopct='%1.2f%%')

plt.title('Total Confirmed Cases:'+str(new_data_details['Confirmed'][0]))

plt.show()
Country_wise_data=new_data.groupby(['Country/Region'])["Confirmed","Active_cases","Recovered","Deaths"].sum().sort_values('Active_cases',ascending=False).reset_index()

Country_wise_data
x=Country_wise_data['Country/Region'][:10]

y=Country_wise_data['Active_cases'][:10]

plt.bar(x,y)

plt.xticks(rotation=90)

plt.title("Corona Active Cases Top 10 Countries")

plt.xlabel('Countries')

plt.ylabel("Active Cases Count")

for x,y in zip(x,y):

    plt.text(x,y,y)

plt.grid(True)

plt.show()
Country_wise_Confirmed_data=new_data.groupby(['Country/Region'])["Confirmed","Active_cases","Recovered","Deaths"].sum().sort_values('Confirmed',ascending=False).reset_index()

Country_wise_Confirmed_data.head(30)
x=Country_wise_Confirmed_data['Country/Region'][:10]

y=Country_wise_Confirmed_data['Confirmed'][:10]

plt.bar(x,y)

plt.xticks(rotation=90)

plt.title("Corona Confirmed Top 10 Countries")

plt.xlabel('Countries')

plt.ylabel("Confirmed Cases Count")

for x,y in zip(x,y):

    plt.text(x,y,y)

plt.grid(True)

plt.show()
Country_wise_Recovered_data=new_data.groupby(['Country/Region'])["Confirmed","Active_cases","Recovered","Deaths"].sum().sort_values('Recovered',ascending=False).reset_index()

Country_wise_Recovered_data.head(30)
x=Country_wise_Recovered_data['Country/Region'][:10]

y=Country_wise_Recovered_data['Recovered'][:10]

plt.bar(x,y)

plt.xticks(rotation=90)

plt.title("Corona Recovered Top 10 Countries")

plt.xlabel('Countries')

plt.ylabel("Recovered Cases Count")

for x,y in zip(x,y):

    plt.text(x,y,y)

plt.grid(True)

plt.show()
Country_wise_Deaths_data=new_data.groupby(['Country/Region'])["Confirmed","Active_cases","Recovered","Deaths"].sum().sort_values('Deaths',ascending=False).reset_index()

Country_wise_Deaths_data.head(30)
x=Country_wise_Deaths_data['Country/Region'][:10]

y=Country_wise_Deaths_data['Deaths'][:10]

plt.bar(x,y)

plt.xticks(rotation=90)

plt.title("Corona Deaths Top 10 Countries")

plt.xlabel('Countries')

plt.ylabel("Death Cases Count")

for x,y in zip(x,y):

    plt.text(x,y,y)

plt.grid(True)

plt.show()
India_data=df[df['Country/Region']=='India'].reset_index(drop=True)

India_data.head(10)
India_data.drop(['Province/State'],inplace=True,axis=1)
India_data['WeekOfYear']=India_data['ObservationDate'].dt.weekofyear

India_data.head()
week_wise_india_data=India_data.groupby(['WeekOfYear'])['Confirmed'].sum().plot(kind='bar', figsize=(12,5), color="indigo", fontsize=13);

week_wise_india_data.set_title("Week Wise Confirmed Cases in India", fontsize=22)

week_wise_india_data.set_ylabel("Confirmed Cases Count", fontsize=15)

week_wise_india_data.set_xlabel("Week of the Year", fontsize=15);



totals = []

for i in week_wise_india_data.patches:

    totals.append(i.get_width())

total = sum(totals)



for i in week_wise_india_data.patches:

    week_wise_india_data.text(i.get_x()+.05, i.get_height()+6.5, \

            str(int(np.ceil((i.get_height()/total)))),

            fontsize=12,

            color='red')

plt.show()
week_wise_india_Active_data=India_data.groupby(['WeekOfYear'])['Active_cases'].sum().plot(kind='bar', figsize=(12,5), color="indigo", fontsize=13);

week_wise_india_Active_data.set_title("Week Wise Active Cases in India", fontsize=22)

week_wise_india_Active_data.set_ylabel("Active Cases Count", fontsize=15)

week_wise_india_Active_data.set_xlabel("Week of the Year", fontsize=15);



totals = []

for i in week_wise_india_Active_data.patches:

    totals.append(i.get_width())

total = sum(totals)



for i in week_wise_india_Active_data.patches:

    week_wise_india_Active_data.text(i.get_x()+.05, i.get_height()+6.5, \

            str(int(np.ceil((i.get_height()/total)))),

            fontsize=12,

            color='red')

plt.show()
week_wise_india_recovering_data=India_data.groupby(['WeekOfYear'])['Recovered'].sum().plot(kind='bar', figsize=(12,5), color="indigo", fontsize=13);

week_wise_india_recovering_data.set_title("Week Wise Recovering Cases in India", fontsize=22)

week_wise_india_recovering_data.set_ylabel("Recovering Cases Count", fontsize=15)

week_wise_india_recovering_data.set_xlabel("Week of the Year", fontsize=15);



totals = []

for i in week_wise_india_recovering_data.patches:

    totals.append(i.get_width())

total = sum(totals)



for i in week_wise_india_recovering_data.patches:

    week_wise_india_recovering_data.text(i.get_x()+.05, i.get_height()+0.5, \

            str(int(np.ceil((i.get_height()/total)))),

            fontsize=12,

            color='red')

plt.show()
week_wise_india_death_data=India_data.groupby(['WeekOfYear'])['Deaths'].sum().plot(kind='bar', figsize=(12,5), color="indigo", fontsize=13);

week_wise_india_death_data.set_title("Week Wise Death Cases in India", fontsize=22)

week_wise_india_death_data.set_ylabel("Death Cases Count", fontsize=15)

week_wise_india_death_data.set_xlabel("Week of the Year", fontsize=15);



totals = []

for i in week_wise_india_death_data.patches:

    totals.append(i.get_width())

total = sum(totals)



for i in week_wise_india_death_data.patches:

    week_wise_india_death_data.text(i.get_x()+.05, i.get_height()+0.2, \

            str(int(np.ceil((i.get_height()/total)))),

            fontsize=12,

            color='red')

plt.show()
dates=India_data['ObservationDate']

c=India_data['Confirmed']

a=India_data['Active_cases']

d=India_data['Deaths']

r=India_data['Recovered']

plt.figure(figsize=(15,7))

plt.plot(dates,c)

plt.plot(dates,a,marker='2')

plt.plot(dates,r,marker='.')

plt.plot(dates,d,marker='>')

plt.xticks(rotation=90)

plt.title("Evolution of Cases over Time In INDIA",fontsize=22)

plt.xlabel('Dates',fontsize=15)

plt.ylabel('Cases Count',fontsize=15)

plt.grid(True)

plt.legend(['Confirmed Cases','Active Cases','Recoverd Cases','Deaths'])

plt.show()
India_latest_data = India_data[India_data['ObservationDate'] == max(India_data['ObservationDate'])].reset_index()



India_latest_data_details = India_latest_data.groupby(["ObservationDate"])["Confirmed","Active_cases","Recovered","Deaths"].sum().reset_index()

India_latest_data_details
labels = ["Active cases","Recovered","Deaths"]

values = India_latest_data_details.loc[0, ["Active_cases","Recovered","Deaths"]]

plt.pie(values, labels=labels, autopct='%1.2f%%')

plt.title('Total Confirmed Cases:'+str(India_latest_data_details['Confirmed'][0]),fontsize=20)

plt.show()