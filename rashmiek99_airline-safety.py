import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
airline_safety = pd.read_csv("/kaggle/input/fivethirtyeight-airline-safety-dataset/airline-safety.csv")

airline_safety.head()
airline_safety.info()
airline_safety.describe()
airline_safety.nunique()
sns.heatmap(airline_safety.corr());
airline_safety.plot(x = 'airline',y=['incidents_85_99','fatal_accidents_85_99','fatalities_85_99'],kind='bar',figsize=(20,15));

plt.show()


airline_safety.plot(x = 'airline',y=['incidents_00_14','fatal_accidents_00_14','fatalities_00_14'],kind='bar',figsize=(20,15));

plt.show()
incidents8599 = pd.cut(airline_safety['incidents_85_99'],bins=[0,5,10,20,50,100],labels=['0-5','5-10','10-20','20-50','50-100'])

incidents0014 = pd.cut(airline_safety['incidents_00_14'],bins=[0,5,10,20,50,100],labels=['0-5','5-10','10-20','20-50','50-100'])



print("Incidents between 1985-1999\n",incidents8599.value_counts())

print("Incidents between 2000-2014\n",incidents0014.value_counts())
fatal_accidents8599 = pd.cut(airline_safety['fatal_accidents_85_99'],bins=[0,5,10,20,50,100],labels=['0-5','5-10','10-20','20-50','50-100'])

fatal_accidents0014 = pd.cut(airline_safety['fatal_accidents_00_14'],bins=[0,5,10,20,50,100],labels=['0-5','5-10','10-20','20-50','50-100'])



print("Fatal accidents between 1985-1999")

print(fatal_accidents8599.value_counts())

print("\nFatal accidents between 2000-2014")

print(fatal_accidents0014.value_counts())
fatalities8599 = pd.cut(airline_safety['fatalities_85_99'],bins=[0,25,50,100,200,500,800],labels=['0-25','25-50','50-100','100-200','200-500','500-800'])

fatalities0014 = pd.cut(airline_safety['fatalities_00_14'],bins=[0,25,50,100,200,500,800],labels=['0-25','25-50','50-100','100-200','200-500','500-800'])



print("Fatalities between 1985-1999")

print(fatalities8599.value_counts())

print("\nFatalities between 2000-2014")

print(fatalities0014.value_counts())
m = airline_safety['incidents_85_99'].max()

n = airline_safety['incidents_85_99'].min()



print("Airline with maximum incidents between 1985-1999")

print(airline_safety[airline_safety['incidents_85_99'] == m]['airline'].values)

print("\nAirline with minimum incidents between 1985-1999")

print(airline_safety[airline_safety['incidents_85_99'] == n]['airline'].values)

print('\n')



m = airline_safety['fatal_accidents_85_99'].max()

n = airline_safety['fatal_accidents_85_99'].min()



print("Airline with maximum fatal accidents between 1985-1999")

print(airline_safety[airline_safety['fatal_accidents_85_99'] == m]['airline'].values)

print("\nAirline with minimum fatal accidents between 1985-1999")

print(airline_safety[airline_safety['incidents_85_99'] == n]['airline'].values)

print('\n')



m = airline_safety['fatalities_85_99'].max()

n = airline_safety['fatalities_85_99'].min()



print("Airline with maximum fatalities between 1985-1999")

print(airline_safety[airline_safety['fatalities_85_99'] == m]['airline'].values)

print("\nAirline with minimum fatalities between 1985-1999")

print(airline_safety[airline_safety['fatalities_85_99'] == n]['airline'].values)

print('\n')
m = airline_safety['incidents_00_14'].max()

n = airline_safety['incidents_00_14'].min()



print("Airline with maximum incidents between 2000-2014")

print(airline_safety[airline_safety['incidents_00_14'] == m]['airline'].values)

print("\nAirline with minimum incidents between 2000-2014")

print(airline_safety[airline_safety['incidents_00_14'] == n]['airline'].values)

print('\n')



m = airline_safety['fatal_accidents_00_14'].max()

n = airline_safety['fatal_accidents_00_14'].min()



print("Airline with maximum fatal accidents between 2000-2014")

print(airline_safety[airline_safety['fatal_accidents_00_14'] == m]['airline'].values)

print("\nAirline with minimum fatal accidents between 2000-2014")

print(airline_safety[airline_safety['incidents_00_14'] == n]['airline'].values)

print('\n')



m = airline_safety['fatalities_00_14'].max()

n = airline_safety['fatalities_00_14'].min()



print("Airline with maximum fatalities between 2000-2014")

print(airline_safety[airline_safety['fatalities_00_14'] == m]['airline'].values)

print("\nAirline with minimum fatalities between 2000-2014")

print(airline_safety[airline_safety['fatalities_00_14'] == n]['airline'].values)

print('\n')
year_85_99 = airline_safety[['airline','avail_seat_km_per_week','incidents_85_99','fatal_accidents_85_99','fatalities_85_99']]

year_00_14 = airline_safety[['airline','avail_seat_km_per_week','incidents_00_14','fatal_accidents_00_14','fatalities_00_14']]
fig = go.Figure(data=[go.Table(

    header=dict(values=year_85_99.columns,

                line_color='darkslategray',

                fill_color='paleturquoise',

                align='left'),

    cells=dict(values=[year_85_99['airline'],year_85_99['avail_seat_km_per_week'],year_85_99['incidents_85_99'],year_85_99['fatal_accidents_85_99'],year_85_99['fatalities_85_99']], # 2nd column

               line_color='darkslategray',

               fill_color='lavender',

               align='left'))

])



fig.show()
fig = go.Figure(data=[go.Table(

    header=dict(values=year_00_14.columns,

                line_color='darkslategray',

                fill_color='paleturquoise',

                align='left'),

    cells=dict(values=[year_00_14['airline'],year_00_14['avail_seat_km_per_week'],year_00_14['incidents_00_14'],year_00_14['fatal_accidents_00_14'],year_00_14['fatalities_00_14']], # 2nd column

               line_color='darkslategray',

               fill_color='lavender',

               align='left'))

])



fig.show()
print("Enter the airline you want to check")

#user_input = input()

user_input = 'american'

user_input = user_input.lower()



data= airline_safety[airline_safety["airline"].map(lambda x: x.lower().strip('*')) == user_input]



if data.empty:

    print("Please enter the airlines correctly")

else:

    print("\nAvailable Seats Km per week ->" ,data.avail_seat_km_per_week.values)

    print("\nIncidents between       1985-1999 ->",data.incidents_85_99.values)

    print("\nFatal accidents between 1985-1999 ->",data.fatal_accidents_85_99.values)

    print("\nFatalities between      1985-1999 ->",data.fatalities_85_99.values)

    print("\nIncidents between       2000 - 2014 ->",data.incidents_00_14.values)

    print("\nFatal accidents between 2000 - 2014 ->",data.fatal_accidents_00_14.values)

    print("\nFatalities between      2000 - 2014 ->",data.fatalities_00_14.values)