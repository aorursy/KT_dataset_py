import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

%matplotlib notebook
covid19_df = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")

individuals_df = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")
covid19_df.head()
covid19_df.tail()
covid19_df.shape
covid19_df.isna().sum()
covid19_df_latest = covid19_df[covid19_df['Date']=="26/04/20"]

covid19_df_latest.head()
covid19_df_latest['Confirmed'].sum()
covid19_df_latest = covid19_df_latest.sort_values(by=['Confirmed'], ascending = False)

plt.figure(figsize=(12,8), dpi=80)

plt.bar(covid19_df_latest['State/UnionTerritory'][:5], covid19_df_latest['Confirmed'][:5],

        align='center',color='lightgrey')

plt.ylabel('Number of Confirmed Cases', size = 12)

plt.title('States with maximum confirmed cases', size = 16)

plt.show()
covid19_df_latest['Deaths'].sum()
covid19_df_latest = covid19_df_latest.sort_values(by=['Deaths'], ascending = False)

plt.figure(figsize=(12,8), dpi=80)

plt.bar(covid19_df_latest['State/UnionTerritory'][:5], covid19_df_latest['Deaths'][:5], align='center',color='lightgrey')

plt.ylabel('Number of Deaths', size = 12)

plt.title('States with maximum deaths', size = 16)

plt.show()
covid19_df_latest['Deaths/Confirmed Cases'] = (covid19_df_latest['Confirmed']/covid19_df_latest['Deaths']).round(2)

covid19_df_latest['Deaths/Confirmed Cases'] = [np.nan if x == float("inf") else x for x in covid19_df_latest['Deaths/Confirmed Cases']]

covid19_df_latest = covid19_df_latest.sort_values(by=['Deaths/Confirmed Cases'], ascending=True, na_position='last')

covid19_df_latest.iloc[:10]
individuals_df.isna().sum()
individuals_df.iloc[0]
individuals_grouped_district = individuals_df.groupby('detected_district')

individuals_grouped_district = individuals_grouped_district['id']

individuals_grouped_district.columns = ['count']

individuals_grouped_district.count().sort_values(ascending=False).head()
individuals_grouped_gender = individuals_df.groupby('gender')

individuals_grouped_gender = pd.DataFrame(individuals_grouped_gender.size().reset_index(name = "count"))

individuals_grouped_gender.head()



plt.figure(figsize=(10,6), dpi=80)

barlist = plt.bar(individuals_grouped_gender['gender'], individuals_grouped_gender['count'], align = 'center', color='grey', alpha=0.3)

barlist[1].set_color('r')

plt.ylabel('Count', size=12)

plt.title('Count on the basis of gender', size=16)

plt.show()
individuals_grouped_date = individuals_df.groupby('diagnosed_date')

individuals_grouped_date = pd.DataFrame(individuals_grouped_date.size().reset_index(name = "count"))

individuals_grouped_date[['Day','Month','Year']] = individuals_grouped_date.diagnosed_date.apply( 

   lambda x: pd.Series(str(x).split("/")))

individuals_grouped_date.sort_values(by=['Year','Month','Day'], inplace = True, ascending = True)

individuals_grouped_date.reset_index(inplace = True)

individuals_grouped_date['Cumulative Count'] = individuals_grouped_date['count'].cumsum()

individuals_grouped_date = individuals_grouped_date.drop(['index', 'Day', 'Month', 'Year'], axis = 1)

individuals_grouped_date.head()
individuals_grouped_date.tail()
individuals_grouped_date = individuals_grouped_date.iloc[3:]

individuals_grouped_date.reset_index(inplace = True)

individuals_grouped_date.columns = ['Day Number', 'diagnosed_date', 'count', 'Cumulative Count']

individuals_grouped_date['Day Number'] = individuals_grouped_date['Day Number'] - 2

individuals_grouped_date



plt.figure(figsize=(12,8), dpi=80)

plt.plot(individuals_grouped_date['Day Number'], individuals_grouped_date['Cumulative Count'], color="grey", alpha = 0.5)

plt.xlabel('Number of Days', size = 12)

plt.ylabel('Number of Cases', size = 12)

plt.title('How the case count increased in India', size=16)

plt.show()
covid19_maharashtra = covid19_df[covid19_df['State/UnionTerritory'] == "Maharashtra"]

covid19_maharashtra.head()

covid19_maharashtra.reset_index(inplace = True)

covid19_maharashtra = covid19_maharashtra.drop(['index', 'Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'],  axis = 1)

covid19_maharashtra.reset_index(inplace = True)

covid19_maharashtra.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']

covid19_maharashtra['Day Count'] = covid19_maharashtra['Day Count'] + 8

missing_values = pd.DataFrame({"Day Count": [x for x in range(1,8)],

                           "Date": ["0"+str(x)+"/03/20" for x in range(2,9)],

                           "State/UnionTerritory": ["Maharashtra"]*7,

                           "Deaths": [0]*7,

                           "Confirmed": [0]*7})

covid19_maharashtra = covid19_maharashtra.append(missing_values, ignore_index = True)

covid19_maharashtra = covid19_maharashtra.sort_values(by="Day Count", ascending = True)

covid19_maharashtra.reset_index(drop=True, inplace=True)

print(covid19_maharashtra.shape)

covid19_maharashtra.head()
covid19_kerala = covid19_df[covid19_df['State/UnionTerritory'] == "Kerala"]

covid19_kerala = covid19_kerala.iloc[32:]

covid19_kerala.reset_index(inplace = True)

covid19_kerala = covid19_kerala.drop(['index','Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'], axis = 1)

covid19_kerala.reset_index(inplace = True)

covid19_kerala.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']

covid19_kerala['Day Count'] = covid19_kerala['Day Count'] + 1

print(covid19_kerala.shape)

covid19_kerala.head()
covid19_delhi = covid19_df[covid19_df['State/UnionTerritory'] == "Delhi"]

covid19_delhi.reset_index(inplace = True)

covid19_delhi = covid19_delhi.drop(['index','Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'], axis = 1)

covid19_delhi.reset_index(inplace = True)

covid19_delhi.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']

covid19_delhi['Day Count'] = covid19_delhi['Day Count'] + 1

print(covid19_delhi.shape)

covid19_delhi.head()
covid19_rajasthan = covid19_df[covid19_df['State/UnionTerritory'] == "Rajasthan"]

covid19_rajasthan.reset_index(inplace = True)

covid19_rajasthan = covid19_rajasthan.drop(['index','Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'], axis = 1)

covid19_rajasthan.reset_index(inplace = True)

covid19_rajasthan.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']

covid19_rajasthan['Day Count'] = covid19_rajasthan['Day Count'] + 2

missing_values = pd.DataFrame({"Day Count": [1],

                           "Date": ["02/03/20"],

                           "State/UnionTerritory": ["Rajasthan"],

                           "Deaths": [0],

                           "Confirmed": [0]})

covid19_rajasthan = covid19_rajasthan.append(missing_values, ignore_index = True)

covid19_rajasthan = covid19_rajasthan.sort_values(by="Day Count", ascending = True)

covid19_rajasthan.reset_index(drop=True, inplace=True)

print(covid19_rajasthan.shape)

covid19_rajasthan.head()
covid19_gujarat = covid19_df[covid19_df['State/UnionTerritory'] == "Gujarat"]

covid19_gujarat.reset_index(inplace = True)

covid19_gujarat = covid19_gujarat.drop(['index','Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'], axis = 1)

covid19_gujarat.reset_index(inplace = True)

covid19_gujarat.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']

covid19_gujarat['Day Count'] = covid19_gujarat['Day Count'] + 19

missing_values = pd.DataFrame({"Day Count": [x for x in range(1,19)],

                           "Date": [("0" + str(x) if x < 10 else str(x))+"/03/20" for x in range(2,20)],

                           "State/UnionTerritory": ["Gujarat"]*18,

                           "Deaths": [0]*18,

                           "Confirmed": [0]*18})

covid19_gujarat = covid19_gujarat.append(missing_values, ignore_index = True)

covid19_gujarat = covid19_gujarat.sort_values(by="Day Count", ascending = True)

covid19_gujarat.reset_index(drop=True, inplace=True)

print(covid19_gujarat.shape)

covid19_gujarat.head()
plt.figure(figsize=(12,8), dpi=80)

plt.plot(covid19_kerala['Day Count'], covid19_kerala['Confirmed'])

plt.plot(covid19_maharashtra['Day Count'], covid19_maharashtra['Confirmed'])

plt.plot(covid19_delhi['Day Count'], covid19_delhi['Confirmed'])

plt.plot(covid19_rajasthan['Day Count'], covid19_rajasthan['Confirmed'])

plt.plot(covid19_gujarat['Day Count'], covid19_gujarat['Confirmed'])

plt.legend(['Kerala', 'Maharashtra', 'Delhi', 'Rajasthan', 'Gujarat'], loc='upper left')

plt.xlabel('Day Count', size=12)

plt.ylabel('Confirmed Cases Count', size=12)

plt.title('Which states are flattening the curve ?', size = 16)

plt.show()