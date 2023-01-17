import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import collections
print(os.listdir("../input"))
crime_data = pd.read_csv('../input/Crime_Column_Description.csv')
population_data = pd.read_csv('../input/Population_by_Borough_NYC.csv')
complaint_data = pd.read_csv('../input/NYPD_Complaint_Data_Historic.csv', dtype=object)
population_data.head(6)
precent_columns = ['1950 - Boro share of NYC total', '1960 - Boro share of NYC total', '1970 - Boro share of NYC total', 
                   '1980 - Boro share of NYC total', '1990 - Boro share of NYC total', '2000 - Boro share of NYC total', 
                   '2010 - Boro share of NYC total', '2020 - Boro share of NYC total', '2030 - Boro share of NYC total', 
                   '2040 - Boro share of NYC total']
years = ['1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020', '2030', '2040']

bronx_precent = []
brooklyn_precent = []
manhattan_precent = []
queens_precent = []
staten_precent = []

for col in precent_columns:
    bronx_precent.append(float(population_data[col][1][:-1]))
    brooklyn_precent.append(float(population_data[col][2][:-1]))  
    manhattan_precent.append(float(population_data[col][3][:-1]))
    queens_precent.append(float(population_data[col][4][:-1]))
    staten_precent.append(float(population_data[col][5][:-1]))


plt.figure(figsize=[15, 7])
plt.bar(years, bronx_precent, label="Bronx")
plt.bar(years, brooklyn_precent, bottom=bronx_precent, label="Brooklyn")
plt.bar(years, manhattan_precent, bottom = [i+j for i,j in zip(bronx_precent, brooklyn_precent)], label="Manhattan")
plt.bar(years, queens_precent, bottom = [i+j+k for i,j,k in zip(bronx_precent, brooklyn_precent, manhattan_precent)], label="Queens")
plt.bar(years, staten_precent, bottom = [i+j+k+n for i,j,k, n in zip(bronx_precent, brooklyn_precent, manhattan_precent, queens_precent)], label="Staten Island")

plt.xlabel('Years', size=15)
plt.ylabel('Precentage of NYC population', size=15)
plt.legend(bbox_to_anchor=(1,0.6), prop={'size': 15})
plt.title('Precent Share of Population for each New York City Borough', size=18)

plt.show()
total_pop = [int(population_data[i][0].replace(',', '')) for i in years]

plt.figure(figsize=[15,7])
plt.bar(years, total_pop)

plt.xlabel('Years', size=15)
plt.ylabel('Population', size=15)
plt.title('Total Population of New York City', size=18)

plt.show()
print(crime_data.to_string())
dig_offense_code = {}
code_meaning = {}

dig_offense_code = complaint_data['KY_CD'].value_counts().to_dict()
code_meaning = complaint_data['OFNS_DESC'].value_counts().to_dict()

plt.figure(figsize=[15,7])
plt.bar(list(dig_offense_code.keys()), list(dig_offense_code.values()))
plt.xlabel('3 digit offense classification code', size=14)
plt.ylabel('Number of Occurances', size=14)
plt.title('Which Offense Occured the Most?', size=16)

plt.show()
updated_dig_offense = {key: value for key, value in dig_offense_code.items() if value > 25000 }

        
plt.figure(figsize=[15,7])
plt.bar(list(updated_dig_offense.keys()), list(updated_dig_offense.values()))
plt.xlabel('3 digit offense classification code', size=14)
plt.ylabel('Number of Occurances', size=14)
plt.title('Which Offense Occured the Most?', size=16)

plt.show()
for key, meaning in zip(updated_dig_offense.keys(), code_meaning.keys()):
    print(key, ':', meaning)
from collections import defaultdict

months_by_count = {i+1: 0 for i in range(12)}
days_by_count = {i+1: 0 for i in range(31)}
years_by_count = defaultdict(int)
day_of_year = {}

for i in complaint_data['CMPLNT_FR_DT']:
    if ( isinstance(i ,str) ):
        dates = i.split('/')
        months_by_count[int(dates[0])] += 1
        days_by_count[int(dates[1])] += 1
        years_by_count[int(dates[2])] += 1
        
        day_of_year[dates[0] + '/' + dates[1] ] = day_of_year.get(dates[0] + '/' + dates[1] ,0) + 1
plt.figure(figsize=[12,15])
plt.subplots_adjust(hspace=0.5)

plt.subplot(3,1,1)
plt.title('Crime by Days')
plt.xlabel('Days')
plt.ylabel('Amount of Crime Incidents')
plt.bar(days_by_count.keys(), days_by_count.values())

plt.subplot(3,1,2)
plt.title('Crime by Months')
plt.xlabel('Months')
plt.ylabel('Amount of Crime Incidents')
plt.bar(months_by_count.keys(), months_by_count.values())

plt.subplot(3,1,3)
plt.title('Crime by Years')
plt.xlabel('Years')
plt.ylabel('Amount of Crime Incidents')
plt.plot(years_by_count.keys(), years_by_count.values())

plt.show()
plt.figure(figsize=[10,7])

day_of_year = dict(collections.Counter(day_of_year).most_common(10))
plt.bar(day_of_year.keys(), day_of_year.values())
plt.gca().set_ylim(ymin=3500)
plt.ylabel('Amount of Crimes')
plt.xlabel('Days to Avoid!')
plt.title('Days with Most Reported Crimes')
plt.show()