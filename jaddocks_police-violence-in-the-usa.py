# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datetime import datetime

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
medianincomedata = pd.read_csv("../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv", encoding = "windows-1252")

graduateframe = pd.read_csv("../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv", encoding = "windows-1252")
graduateframe.replace(['-'], 'NaT', inplace = True)
graduatedata = graduateframe[graduateframe['percent_completed_hs'] != 'NaT']

povertydata = pd.read_csv("../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv", encoding = "windows-1252")

policepolicydata = pd.read_csv("../input/policiesofpolice/police_policies.csv", encoding = "windows-1252")
policepolicydata.City.replace(['Washington DC'], 'Washington', inplace = True)
policepolicydata.City.replace(['Kansas City, MO'], 'Kansas City', inplace = True)
policepolicydata.fillna('0', inplace = True)
policepolicydata.rename(columns = {'Requires\xa0De-Escalation': 'Requires De-Escalation', 'Has Use of Force\xa0Continuum' : 'Has Use of Force Continuum',
                                   'Bans Chokeholds\xa0and Strangleholds' : 'Bans Chokeholds and Strangleholds', 'Requires Warning\xa0Before Shooting' : 'Requires Warning Before Shooting',
                                   'Restricts Shooting\xa0at Moving Vehicles' : 'Restricts Shooting at Moving Vehicles',
                                   'Requires Exhaust All Other\xa0Means Before Shooting' : 'Requires Exhaust All Other Means Before Shooting',
                                    'Requires Comprehensive\xa0Reporting' : 'Requires Comprehensive Reporting'}, inplace = True)

policekillingsdata = pd.read_csv('../input/police-violence-in-the-us/shootings_wash_post.csv')
policekillingsdata2 = pd.read_csv('../input/police-violence-in-the-us/police_killings.csv')
policekillingsdata2.replace({'Physical restraint' : 'Physical Restraint', 'Taser, Pepper spray, beaten' : 'Taser, Pepper Spray, Beaten', 'Unknown race' : 'Unknown Race'}, inplace = True)
policekillingsdata2["Victim's age"].replace({'Unknown' : 'NaN', '40s' : 'NaN'}, inplace = True)
policekillingsdata2.rename(columns = {'Geography (via Trulia methodology based on zipcode population density: http://jedkolko.com/wp-content/uploads/2015/05/full-ZCTA-urban-suburban-rural-classification.xlsx )' 
                                      : 'Human Geography', 'Date of Incident (month/day/year)' : 'date'}, inplace = True)
policekillingsdata2['date'] = pd.to_datetime(policekillingsdata2['date'])
policekillingsdata2['year'] = policekillingsdata2.apply(lambda row: row['date'].year, axis = 1) 

raceframe = pd.read_csv("../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv", encoding = "windows-1252")
raceframe.replace(['(X)'], 'NaT', inplace = True)
racedata = raceframe[raceframe['share_white'] != 'NaT']
racedata.City.replace({' city': '', ' town' : '', 'Urban Honolulu CDP' : 'Honolulu',
                       'New York City' : 'New York', ' municipality' : '', 'Lexington-Fayette urban county' : 'Lexington',
                       'Louisville/Jefferson County metro government' : 'Louisville',
                      'Nashville-Davidson metropolitan government' : 'Nashville',
                      'Washington ' : 'Washington'}, regex = True, inplace = True)

racedata2 = pd.read_csv('../input/racedata/raw_data.csv')
racedata2.columns = racedata2.iloc[1]
racedata2 = racedata2[3:]
racedata2.drop(racedata2.tail(18).index, inplace = True)
racedata2.replace({'<.01': '0'}, inplace = True)
racedata2 = racedata2.fillna('0')

statepop = pd.read_csv("../input/statepop/statepop.csv", thousands = ',')
statepop.dropna(subset = ['Code'], inplace = True)
stategunlaws = pd.read_csv("../input/stategunlaws/stategunlaws.csv")
stategunlaws.rename(columns = {'Code' : 'States'}, inplace = True)

citypop = pd.read_csv("../input/citypop/City pop.csv", thousands = ',')
citypop.rename(columns = {'Unnamed: 1' : 'City', 'Unnamed: 2' : 'State', 'Unnamed: 14' : 'Population'}, inplace = True)
citypop.rename(columns = {'Unnamed: 2' : 'City'}, inplace = True)
citypop.City.replace({' city' : '', ' municipality' : '', 'Urban ': '', ' CDP' : '', ' \(balance\)' : '', '-Fayette urban county' : '', '/Jefferson County metro government' : '',
                     '-Davidson metropolitan government' : ''}, regex = True, inplace = True)

statenames = statepop['State name'].unique().tolist()

politicsdata = pd.read_csv('../input/politics/politics_538.csv', skipinitialspace = True)

budgetdata = pd.read_csv('../input/budgets/budgets.csv', skipinitialspace = True)

violentcrimesdata = pd.read_csv('../input/police-violence-in-the-us/deaths_arrests_race.csv')

juvenilearrestsdata = pd.read_csv('../input/police-violence-in-the-us/juvenile_arrests.csv')

chicagocrimes = pd.read_csv('../input/police-violence-in-the-us/large_metro_areas/Chicago Crimes_-_2001_to_Present.csv')
citystatenamelist = list(citypop['State'])
citystatenamelist = [str(i) for i in citystatenamelist]
citystatenamelist = [i.lstrip() for i in citystatenamelist]
citystatecodelist = []
for name in citystatenamelist:
    if name in list(statepop['State name']):
        citystatecodelist.append(statepop[statepop['State name'] == name]['Code'].item())
    else:
        citystatecodelist.append('NaN')
        
citystatecodearray = np.asarray(citystatecodelist)
citypop['State code'] = citystatecodearray

citystatenamelist2 = list(politicsdata['State'])
citystatenamelist2 = [str(i) for i in citystatenamelist2]
citystatenamelist2 = [i.lstrip() for i in citystatenamelist2]
citystatecodelist2 = []
citystatecodelist2 = []
for name in citystatenamelist2:
    if name in list(statepop['State name']):
        citystatecodelist2.append(statepop[statepop['State name'] == name]['Code'].item())
    else:
        citystatecodelist2.append('NaN')
citystatecodearray2 = np.asarray(citystatecodelist2)
politicsdata['State code'] = citystatecodearray2


citystatenamelist3 = list(racedata2['Location'])
citystatenamelist3 = [str(i) for i in citystatenamelist3]
citystatenamelist3 = [i.lstrip() for i in citystatenamelist3]
citystatecodelist3 = []
citystatecodelist3 = []
for name in citystatenamelist3:
    if name in list(statepop['State name']):
        citystatecodelist3.append(statepop[statepop['State name'] == name]['Code'].item())
    else:
        citystatecodelist3.append('NaN')
citystatecodearray3 = np.asarray(citystatecodelist3)
racedata2['State code'] = citystatecodearray3
killsbystate2 = pd.Series(policekillingsdata2['State'])
killsbystatecount2 = killsbystate2.value_counts()

plt.figure(figsize=(16,10))
sns.barplot(x = killsbystatecount2.index, y = killsbystatecount2.values)
plt.ylabel('Police Killings (Jan 2013 - Jan 2020)')
plt.xlabel('States')
plt.title('Police Killings by State from 2013 - 2020')
statepop.Population = statepop.Population.astype(float)
sortedstatepop = statepop.sort_values(by = 'Population', ascending = False)

plt.figure(figsize=(16,10))
sns.barplot(x = sortedstatepop.Code, y = sortedstatepop.Population)
plt.xlabel('States')
plt.title('State Populations')
plt.ylabel('Estimated Population in 2019 (10,000,000)')
policekillingsagedata = policekillingsdata2[policekillingsdata2["Victim's age"].astype(float) > 10][policekillingsdata2["Victim's age"].astype(float) < 100]["Victim's age"].value_counts()

plt.figure(figsize = (20, 10))
sns.barplot(x = policekillingsagedata.index, y = policekillingsagedata.values)
plt.xlabel('Age')
plt.ylabel('Total Deaths')
plt.title('Total Deaths by Age of Victim')

years = (2013, 2014, 2015, 2016, 2017, 2018, 2019)
b = []
for year in years:
    b.append(len(policekillingsdata2[policekillingsdata2['year'] == year]))
killingsperyear = pd.DataFrame({'Police Killings' : b, 'year' : years})

plt.figure(figsize = (16, 10))
sns.barplot(x = 'year', y = 'Police Killings', data = killingsperyear)
plt.xlabel('Year')
plt.title('Total Police Killings By Year')
dates = pd.date_range(start = '01/01/13', end = '12/31/19')
dayspolicekilled = policekillingsdata2['date'].unique()
killingsday = []

for day in dayspolicekilled:
    killstoday = policekillingsdata2[policekillingsdata2['date'] == day]
    killingsday.append(len(killstoday))

killingseachday = pd.DataFrame({'Killings' : killingsday, 'Date' : dayspolicekilled})
killingsperday = []

for day in dates:
    if day in list(killingseachday['Date'].unique()):
        killingsperday.append(killingseachday[killingseachday['Date'] == day]['Killings'].item())
    else:
        killingsperday.append(0)

killingsperdayframe = pd.DataFrame({'Killings' : killingsperday, 'Date' : dates})
killingsperdayframe['year'] = killingsperdayframe.apply(lambda row: row['Date'].year, axis = 1) 
killingsperdayframe2 = pd.DataFrame(pd.np.empty((11, 0)) * pd.np.nan) 

for year in years:
    killingsperdayframe2[year] = killingsperdayframe[killingsperdayframe['year'] == year]['Killings'].value_counts()
plt.figure(figsize = (8, 5))
sns.barplot(x = killingsperdayframe2.index, y = killingsperdayframe2[2013])
plt.title('2013 Frequency of Deaths from Police Shootings Each Day')
plt.xlabel('Number of Police Killings')
plt.ylabel('Number of Days')
plt.figure(figsize = (8, 5))
sns.barplot(x = killingsperdayframe2.index, y = killingsperdayframe2[2014])
plt.title('2014 Frequency of Deaths from Police Shootings Each Day')
plt.xlabel('Number of Police Killings')
plt.ylabel('Number of Days')
plt.figure(figsize = (8, 5))
sns.barplot(x = killingsperdayframe2.index, y = killingsperdayframe2[2015])
plt.title('2015 Frequency of Deaths from Police Shootings Each Day')
plt.xlabel('Number of Police Killings')
plt.ylabel('Number of Days')
plt.figure(figsize = (8, 5))
sns.barplot(x = killingsperdayframe2.index, y = killingsperdayframe2[2016])
plt.title('2016 Frequency of Deaths from Police Shootings Each Day')
plt.xlabel('Number of Police Killings')
plt.ylabel('Number of Days')
plt.figure(figsize = (8, 5))
sns.barplot(x = killingsperdayframe2.index, y = killingsperdayframe2[2017])
plt.title('2017 Frequency of Deaths from Police Shootings Each Day')
plt.xlabel('Number of Police Killings')
plt.ylabel('Number of Days')
plt.figure(figsize = (8, 5))
sns.barplot(x = killingsperdayframe2.index, y = killingsperdayframe2[2018])
plt.title('2018 Frequency of Deaths from Police Shootings Each Day')
plt.xlabel('Number of Police Killings')
plt.ylabel('Number of Days')
plt.figure(figsize = (8, 5))
sns.barplot(x = killingsperdayframe2.index, y = killingsperdayframe2[2019])
plt.title('2019 Frequency of Deaths from Police Shootings Each Day')
plt.xlabel('Number of Police Killings')
plt.ylabel('Number of Days')
states = list(policekillingsdata2['State'].unique())
states = [x for x in states if str(x) != 'nan']
killsbystatecountframe = killsbystatecount2.to_frame()
b = []

for state in states:
    x = statepop[statepop['Code'] == state]
    y = killsbystatecountframe[killsbystatecountframe.index == state]
    killsmillionyear = (((float(y['State'])/float(x.Population)) / 7) * 1000000)
    b.append(killsmillionyear)

statekillsPM = pd.DataFrame({'States' : states,'shootingsPM': b})
statekillsPM = statekillsPM.sort_values(by = ['shootingsPM'], ascending = False)

plt.figure(figsize=(16,10))
sns.barplot(x = statekillsPM.States, y = statekillsPM.shootingsPM)
plt.ylabel('Deaths from Police Shootings per 1M Residence')
plt.title('Police Shootings per year per 1M Residence by State')
povertydata.poverty_rate.replace("-", 0.0, inplace = True)
povertydata.poverty_rate = povertydata.poverty_rate.astype(float)
povertybystatelist = []

for state in states:
    statepoverty = povertydata[povertydata['Geographic Area'] == state]
    meanpoverty = sum(statepoverty.poverty_rate) / len(statepoverty)
    povertybystatelist.append(meanpoverty)
    
povertybystate = pd.DataFrame({ "States" : states, "Average poverty rate" : povertybystatelist})
sortedpovertybystate = povertybystate.sort_values(by = ['Average poverty rate'], ascending = False)

plt.figure(figsize=(16,10))
sns.barplot(x = sortedpovertybystate['States'], y = sortedpovertybystate['Average poverty rate'])
plt.ylabel('% of People Below Poverty Line')
statepovertyandkills = pd.merge(statekillsPM, sortedpovertybystate, on = 'States', how = 'inner')


plot = sns.lmplot(x = 'Average poverty rate', y = 'shootingsPM', data = statepovertyandkills, size = 8)
plt.xlabel('% of People Below Poverty Line')
plt.ylabel('Deaths from Police Shootings per year per 1M Residence')
plt.title('Deaths from Police Shootings compared with Poverty Rate')

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

label_point(statepovertyandkills['Average poverty rate'], statepovertyandkills['shootingsPM'], statepovertyandkills.States, plt.gca())
hsgraduatebystate = pd.DataFrame(columns = ('States', 'Percent Graduated HighSchool'))
for state in states:
    graduatestatedata = graduatedata[graduatedata['Geographic Area'] == state]
    meangraduatestatedata = ((graduatestatedata['percent_completed_hs'].astype(float).sum()) / len(graduatestatedata))
    hsgraduatebystate = hsgraduatebystate.append({'States' : state, 'Percent Graduated HighSchool' : meangraduatestatedata}, ignore_index = True)
statepovertykillsgraduates = pd.merge(statepovertyandkills, hsgraduatebystate, on = 'States')

sns.lmplot(x = 'Percent Graduated HighSchool', y = 'shootingsPM', data = statepovertykillsgraduates, height = 7)
plt.ylabel('Deaths from Police Shootings per year per 1M Residence')

label_point(statepovertykillsgraduates['Percent Graduated HighSchool'], statepovertykillsgraduates['shootingsPM'], statepovertykillsgraduates.States, plt.gca())
frame = pd.merge(statekillsPM, stategunlaws, on = 'States', how = 'inner')

newframe = frame.drop('Unnamed: 2', axis = 1)
newframe = newframe.dropna()

plt.figure(figsize=(16,10))
sns.barplot(x = newframe.States, y = newframe.shootingsPM, hue = newframe['Gun laws'], dodge = False, palette = 'coolwarm')
plt.ylabel('Deaths from Police Shootings per 1M Residence')
plt.gca().legend(frameon = False,  title = '     Gun Law Strength \n 1 Weakest - 10 Strongest', labelspacing = 1)
plt.title('Relative Police Shootings by State with Gun Law Strength')
racelist = list(policekillingsdata2["Victim's race"].unique())
a = policekillingsdata2["Victim's race"].value_counts()

plt.figure(figsize = (16, 10))
sns.barplot(x = a.index, y = a.values)
plt.ylabel('Total Deaths')
plt.xlabel('Race of Victim')
plt.title('Deaths from Police Physical Force by Race from 2013 - 2020')
physicalforce = ['Physical Restraint', 'Beaten', 'Asphyxiated',
                 'Pepper Spray', 'Taser, Baton', 'Taser, Physical Restraint', 
                 'Baton, Pepper Spray, Physical Restraint', 'Taser, Pepper Spray, Beaten',
                 'Taser, Beaten', 'Beaten/Bludgeoned with instrument',
                 'Taser', 'Tasered', ]

physicalforcedeaths = pd.DataFrame()
for cause in physicalforce:
    a = policekillingsdata2[policekillingsdata2['Cause of death'] == cause]
    physicalforcedeaths = physicalforcedeaths.append(a, ignore_index = True)
physicalforcedeathsbystate = pd.Series(physicalforcedeaths['State'])
physicalforcedeathscountbystate = physicalforcedeathsbystate.value_counts()

plt.figure(figsize=(16,10))
sns.barplot(x = physicalforcedeathscountbystate.index, y = physicalforcedeathscountbystate.values)
plt.ylabel('Deaths from Police Physical Force (Jan 2013 - Dec 2019)')
plt.xlabel('States')
plt.title('Deaths from Police Physical Force by State from 2013 - 2020')
b = []
for state in states:
    if state in list(physicalforcedeaths['State']):
        x = statepop[statepop['Code'] == state]
        y = physicalforcedeathscountbystate[physicalforcedeathscountbystate.index == state]
        killsperhthousand2 = (((float(y.values)/float(x.Population)) / 5) * 1000000)
        b.append(killsperhthousand2)
    else:
        b.append(0)
physicalforcedeathsPM = pd.DataFrame({'States': states,'killsPM':b})
PFkillsPM = physicalforcedeathsPM.sort_values(by = ['killsPM'], ascending = False)



plt.figure(figsize=(16,10))
sns.barplot(x = PFkillsPM.States, y = PFkillsPM.killsPM)
plt.ylabel('Deaths from Police Physical Force per 1M Residence')
plt.title('Relative Police Killings from Physical Force by State')
armedlist = policekillingsdata['armed'].unique()
unarmeddeathsbystate = pd.DataFrame(columns = ('States', 'Unarmed Deaths'))

for state in states:
    unarmeddeaths = policekillingsdata[policekillingsdata['armed'] == 'unarmed'][policekillingsdata['state'] == state]
    unarmeddeathsbystate = unarmeddeathsbystate.append({'States' : state, 'Unarmed Deaths' : len(unarmeddeaths)}, ignore_index = True)
    
unarmeddeathsbystate.sort_values(by = 'Unarmed Deaths', inplace = True, ascending = False)
plt.figure(figsize = (16, 10))
sns.barplot(x = unarmeddeathsbystate['States'], y = unarmeddeathsbystate['Unarmed Deaths'])
plt.title('Total Police Killings of Unarmed Victims by State')
b = []
for state in states:
    x = statepop[statepop['Code'] == state]
    y = unarmeddeathsbystate[unarmeddeathsbystate['States'] == state]
    unarmeddeathspermil = (((float(y['Unarmed Deaths'])/float(x.Population)) / 5) * 1000000)
    b.append(unarmeddeathspermil)
unarmeddeathspermilframe = pd.DataFrame({'States': states,'Unarmed Deaths per 1M Residence':b})
sortedunarmeddeathspermil = unarmeddeathspermilframe.sort_values(by = ['Unarmed Deaths per 1M Residence'], ascending = False)

plt.figure(figsize=(16,10))
sns.barplot(x = sortedunarmeddeathspermil.States, y = sortedunarmeddeathspermil['Unarmed Deaths per 1M Residence'])
plt.ylabel('Deaths of Unarmed Victims from Police Shootings per year per 1M Residence')
plt.title('Police Shootings of Unarmed Victims per year per 1M Residence')

cities = policepolicydata['City'].unique()
states2 = policepolicydata['State']
policekillingspercityframe = pd.DataFrame(columns = ('City', 'Number of Deaths'))

x = 0
for city in cities:
    citykillings = policekillingsdata2[policekillingsdata2['City'] == city][policekillingsdata2['State'] == states2[x]]
    x += 1
    policekillingspercityframe = policekillingspercityframe.append({'City' : city, 'Number of Deaths' : len(citykillings)}, ignore_index = True)

policekillingspercityframe = policekillingspercityframe[policekillingspercityframe['City'] != '0']

policekillingspercityframe.sort_values(by = 'Number of Deaths', inplace = True, ascending = False)
plt.figure(figsize = (16, 10))
sns.barplot(x = policekillingspercityframe['City'], y = policekillingspercityframe['Number of Deaths'])
plt.xticks(rotation=90)
plt.ylabel('Total Police Killings')
plt.title('Total Police Killings by city (2013 - 2020)')
b = []
for city in np.delete(cities, [0]):
    x = citypop[citypop['City'] == city][citypop['State code'] == policepolicydata[policepolicydata['City'] == city]['State'].item()]
    y = policekillingspercityframe[policekillingspercityframe['City'] == city]
    citydeathsperyearper1m = ((y['Number of Deaths'].item()/float(x.Population)) / 5) * 1000000
    b.append(citydeathsperyearper1m)
citydeathsyear1m = pd.DataFrame({'City': np.delete(cities, [0]),'Deaths per year per 1M Residence' : b})
sortedcitydeathsyear1m = citydeathsyear1m.sort_values(by = ['Deaths per year per 1M Residence'], ascending = False)

plt.figure(figsize = (16, 10))
sns.barplot(x = sortedcitydeathsyear1m['City'], y = sortedcitydeathsyear1m['Deaths per year per 1M Residence'])
plt.xticks(rotation = 90)
plt.ylabel('Average Annual Police Killings per 1M Residence ')
plt.title('Relative Police Killings by city (2013 - 2020)')
policekillingspercityframe2 = pd.DataFrame()
x = 0
for city in cities:
    citykillings2 = policekillingsdata2[policekillingsdata2['City'] == city][policekillingsdata2['State'] == states2[x]]
    x += 1
    policekillingspercityframe2 = policekillingspercityframe2.append({'City' : city, 'Number of Deaths' : len(citykillings2)}, ignore_index = True)

b = []
for city in np.delete(cities, [0]):
    x = citypop[citypop['City'] == city][citypop['State code'] == policepolicydata[policepolicydata['City'] == city]['State'].item()]
    y = policekillingspercityframe2[policekillingspercityframe2['City'] == city]
    citydeathsperyearper1m2 = ((y['Number of Deaths'].item()/float(x.Population)) / 7) * 1000000
    b.append(citydeathsperyearper1m2)
citydeathsyear1m = pd.DataFrame({'City' : np.delete(cities, [0]), 'Relative Annual Police Killings' : b})

columns = ['City', 'Violent crimes 2013 (if reported by agency)', 'Violent crimes 2014 (if reported by agency)', 'Violent crimes 2015 (if reported by agency)',
          'Violent crimes 2016 (if reported by agency)', 'Violent crimes 2017 (if reported by agency)', 'Violent crimes 2018 (if reported by agency)']
cityviolent = violentcrimesdata[columns].mean(axis = 1)
columns = ['City', '2013 Total Arrests (UCR Data)', '2014 Total Arrests', '2015 Total Arrests', '2016 Total Arrests', '2017 Total Arrests', '2018 Total Arrests']
cityarrests = violentcrimesdata[columns].mean(axis = 1)
crimecities = list(violentcrimesdata['City'])
arrestsandviolentcrimes = pd.DataFrame({'City' : crimecities, 'Average Violent Crimes': cityviolent , 'Average Arrests' : cityarrests})
arrestsandviolentcrimes.replace(['Kansas City Missouri'], 'Kansas City', inplace = True)

b = []
c = []
for city in np.delete(cities, [0]):
    x = citypop[citypop['City'] == city][citypop['State code'] == policepolicydata[policepolicydata['City'] == city]['State'].item()]
    b.append((arrestsandviolentcrimes[arrestsandviolentcrimes['City'] == city]['Average Violent Crimes'].item() / float(x.Population)) * 100000)
    c.append((arrestsandviolentcrimes[arrestsandviolentcrimes['City'] == city]['Average Arrests'].item() / float(x.Population)) * 100000)

citydeathsyear1m['Pop adjusted violent crimes'] = b
citydeathsyear1m['Pop adjusted arrests'] = c
citydeathsyear1m = citydeathsyear1m.sort_values(by = 'Relative Annual Police Killings', ascending = False)
fig = plt.figure(figsize = (16, 10))
ax =  fig.subplots()
ax.plot(citydeathsyear1m['City'], citydeathsyear1m['Relative Annual Police Killings'], 'o')
plt.xticks(rotation = 90)
ax.set_xlabel('City', fontsize = '10')
ax.set_ylabel('Average annual Police killings per 1M residence', color = 'blue', fontsize = '10')

ax2 = ax.twinx()
ax2.plot(citydeathsyear1m['City'], citydeathsyear1m['Pop adjusted violent crimes'], 'o', color = 'red')
ax2.set_ylabel('Average annual Violent Crimes committed per 100,000 residence', color = 'red', fontsize = '10')
fig = plt.figure(figsize = (16, 10))
ax =  fig.subplots()
ax.plot(citydeathsyear1m['City'], citydeathsyear1m['Relative Annual Police Killings'], 'o')
plt.xticks(rotation = 90)
ax.set_xlabel('City', fontsize = '10')
ax.set_ylabel('Average annual Police killings per 1M residence', color = 'blue', fontsize = '10')

ax2 = ax.twinx()
ax2.plot(citydeathsyear1m['City'], citydeathsyear1m['Pop adjusted arrests'], 'o', color = 'red')
ax2.set_ylabel('Average annual Arrests committed per 100,000 residence', color = 'red', fontsize = '10')
years = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
yearcitydeathsdict = {}

for city in cities:
    b = []
    if city in policekillingsdata2.City.unique():
        for year in years:
            a = len(policekillingsdata2[policekillingsdata2.City == city][policekillingsdata2['year'] == year])
            b.append(a)
    yearcitydeathsdict[city] = b

yearcitydeaths = pd.DataFrame.from_dict(yearcitydeathsdict, orient='index', columns=['2013 Deaths', '2014 Deaths',
                                                                                     '2015 Deaths', '2016 Deaths',
                                                                                     '2017 Deaths', '2018 Deaths',
                                                                                     '2019 Deaths'])
yearcitydeaths.dropna(inplace = True)
yearcitydeaths.reset_index(level = 0, inplace = True)
yearcitydeaths.rename(columns = {'index' : 'City'}, inplace = True)

deathscrimescity = pd.merge(violentcrimesdata, yearcitydeaths, on = 'City', how = 'inner')
columns = ['2013 Deaths', '2014 Deaths','2015 Deaths', '2016 Deaths','2017 Deaths', '2018 Deaths', '2013 Total Arrests (UCR Data)',
           '2014 Total Arrests','2015 Total Arrests','2016 Total Arrests','2017 Total Arrests','2018 Total Arrests', 'Violent crimes 2013 (if reported by agency)',
          'Violent crimes 2014 (if reported by agency)', 'Violent crimes 2015 (if reported by agency)', 'Violent crimes 2016 (if reported by agency)', 
          'Violent crimes 2017 (if reported by agency)', 'Violent crimes 2018 (if reported by agency)',]
deathscrimescity = deathscrimescity[columns].astype(float)
deathscrimescity = deathscrimescity.transpose()
deathscrimescity.convert_dtypes(convert_integer = True)
means = deathscrimescity.mean(axis = 1)

deathscrimescity = pd.DataFrame({'Deaths' : [means['2013 Deaths'], means['2014 Deaths'], means['2015 Deaths'], means['2016 Deaths'], means['2017 Deaths'], means['2018 Deaths']],
                                'Arrests' : [means['2013 Total Arrests (UCR Data)'], means['2014 Total Arrests'], means['2015 Total Arrests'], means['2016 Total Arrests'],
                                                  means['2017 Total Arrests'], means['2018 Total Arrests']],
                                'Violent Crimes' : [means['Violent crimes 2013 (if reported by agency)'], means['Violent crimes 2014 (if reported by agency)'],
                                                   means['Violent crimes 2015 (if reported by agency)'], means['Violent crimes 2016 (if reported by agency)'],
                                                   means['Violent crimes 2017 (if reported by agency)'], means['Violent crimes 2018 (if reported by agency)']]}) 
deathscrimescity.rename({0 : 2013, 1 : 2014, 2 : 2015, 3 : 2016, 4: 2017, 5 : 2018}, axis = 'index', inplace = True)

deathscrimescity.index
fig,ax = plt.subplots()
ax.plot(deathscrimescity.index, deathscrimescity.Deaths, color = "red", marker = "o")
ax.set_xlabel("Year", fontsize = 14)
ax.set_ylabel("Mean Police Killings over 100 Cities", color = "red", fontsize = 14)
ax.set_title('Average Killings and Average Arrests over Time for 100 Cities')

ax2 = ax.twinx()
ax2.plot(deathscrimescity.index, deathscrimescity.Arrests, color = "#1461cc",marker = "o")
ax2.set_ylabel("Mean Arrests over 100 Cities", color = "#1461cc", fontsize = 14)
plt.show()
fig, ax = plt.subplots()
ax.plot(deathscrimescity.index, deathscrimescity.Deaths, color = "red", marker = "o")
ax.set_xlabel("Year", fontsize = 14)
ax.set_ylabel("Mean Police Killings over 100 Cities", color = "red", fontsize = 14)
ax.set_title('Average Killings and Average Violent Crimes over Time for 100 Cities')

ax2 = ax.twinx()
ax2.plot(deathscrimescity.index, deathscrimescity['Violent Crimes'], color = "#1461cc", marker = "o")
ax2.set_ylabel("Mean Violent Crimes over 100 Cities", color = "#1461cc", fontsize = 14)
plt.show()
meanracedata = pd.DataFrame(columns = ['States', 'meanwhite', 'meanblack', 'meanhispanic', 'meanasian', 'meannativeamerican'])
for state in states:
    staterace = racedata[racedata['Geographic area'] == state]
    meanwhitef = staterace.share_white.astype(float).sum() / len(staterace)
    meanblackf = staterace.share_black.astype(float).sum() / len(staterace)
    meanhispanicf = staterace.share_hispanic.astype(float).sum() / len(staterace)
    meanasianf = staterace.share_asian.astype(float).sum() / len(staterace)
    meannativeamericanf = staterace.share_native_american.astype(float).sum() / len(staterace)
    
    meanwhite = (meanwhitef / (meanwhitef + meanblackf + meanhispanicf + meanasianf + meannativeamericanf)) * 100
    meanblack = (meanblackf / (meanwhitef + meanblackf + meanhispanicf + meanasianf + meannativeamericanf)) * 100
    meanhispanic = (meanhispanicf / (meanwhitef + meanblackf + meanhispanicf + meanasianf + meannativeamericanf)) * 100
    meanasian = (meanasianf / (meanwhitef + meanblackf + meanhispanicf + meanasianf + meannativeamericanf)) * 100
    meannativeamerican = (meannativeamericanf / (meanwhitef + meanblackf + meanhispanicf + meanasianf + meannativeamericanf)) * 100
    
    meanracedata = meanracedata.append({'States' : state,'meanwhite' : meanwhite, 'meanblack' : meanblack, 'meanhispanic' : meanhispanic, 'meanasian' : meanasian,
                                        'meannativeamerican' : meannativeamerican}, ignore_index = True)
meanracedata.sort_values(by = 'States', inplace = True)

stateracedata = pd.DataFrame(columns = ['State', 'White', 'Two Or More Races', 'Native Hawaiian/Other Pacific Islander', 'Black', 'Hispanic', 'Asian', 'American Indian/Alaska Native'])
for state in states:
    tempracedata = racedata2[racedata2['State code'] == state]
    white = (stateracedata['White'] / (stateracedata['Two Or More Races'] + stateracedata['Native Hawaiian/Other Pacific Islander'] + stateracedata.White +
                                                   stateracedata.Black + stateracedata.Hispanic + stateracedata.Asian + stateracedata['American Indian/Alaska Native'])) * 100
    mixed = (stateracedata['Two Or More Races'] / (stateracedata['Two Or More Races'] + stateracedata['Native Hawaiian/Other Pacific Islander'] + stateracedata.White +
                                                   stateracedata.Black + stateracedata.Hispanic + stateracedata.Asian + stateracedata['American Indian/Alaska Native'])) * 100
    pi = (stateracedata['Native Hawaiian/Other Pacific Islander'] / (stateracedata['Two Or More Races'] + stateracedata['Native Hawaiian/Other Pacific Islander'] + stateracedata.White +
                                                   stateracedata.Black + stateracedata.Hispanic + stateracedata.Asian + stateracedata['American Indian/Alaska Native'])) * 100
    black = (stateracedata['Black'] / (stateracedata['Two Or More Races'] + stateracedata['Native Hawaiian/Other Pacific Islander'] + stateracedata.White +
                                                   stateracedata.Black + stateracedata.Hispanic + stateracedata.Asian + stateracedata['American Indian/Alaska Native'])) * 100
    hispanic = (stateracedata['Hispanic'] / (stateracedata['Two Or More Races'] + stateracedata['Native Hawaiian/Other Pacific Islander'] + stateracedata.White +
                                                   stateracedata.Black + stateracedata.Hispanic + stateracedata.Asian + stateracedata['American Indian/Alaska Native'])) * 100
    asian = (stateracedata['Asian'] / (stateracedata['Two Or More Races'] + stateracedata['Native Hawaiian/Other Pacific Islander'] + stateracedata.White +
                                                   stateracedata.Black + stateracedata.Hispanic + stateracedata.Asian + stateracedata['American Indian/Alaska Native'])) * 100
    na = (stateracedata['American Indian/Alaska Native'] / (stateracedata['Two Or More Races'] + stateracedata['Native Hawaiian/Other Pacific Islander'] + stateracedata.White +
                                                   stateracedata.Black + stateracedata.Hispanic + stateracedata.Asian + stateracedata['American Indian/Alaska Native'])) * 100
    stateracedata = stateracedata.append({'State' : state, 'White' : white, 'Two Or More Races' : mixed, 'Native Hawaiian/Other Pacific Islander' : pi, 'Black' : black, 'Hispanic' : hispanic,
                                          'Asian' : asian, 'American Indian/Alaska Native' : na}, ignore_index = True)
barWidth = 0.85
names = racedata2['State code']
columns = ['White', 'Black', 'Hispanic', 'Asian', 'American Indian/Alaska Native', 'Native Hawaiian/Other Pacific Islander', 'Two Or More Races']
racedata2[columns] = racedata2[columns].astype(float)

plt.figure(figsize = (16, 10))
plt.bar(names, stateracedata.White, color = '#b5ffb9', edgecolor='white', width = barWidth)
plt.bar(names, stateracedata.Black, bottom = stateracedata.White, color='#f9bc86', edgecolor='white', width = barWidth)
plt.bar(names, stateracedata.Hispanic, bottom = stateracedata.White + stateracedata.Black, color='#a3acff', edgecolor='white', width=barWidth)
plt.bar(names, stateracedata.Asian, bottom = stateracedata.White + stateracedata.Black + stateracedata.Hispanic, color='#eda3ff', edgecolor='white', width=barWidth)
plt.bar(names, stateracedata['American Indian/Alaska Native'], bottom = stateracedata.White + stateracedata.Black + stateracedata.Hispanic + stateracedata.Asian, color='#eb4b4b', edgecolor='white', width=barWidth)
plt.bar(names, stateracedata['Native Hawaiian/Other Pacific Islander'], bottom = stateracedata.White + stateracedata.Black + stateracedata.Hispanic + stateracedata.Asian + stateracedata['American Indian/Alaska Native'], color='#ffb03b', edgecolor='white', width=barWidth)
plt.bar(names, stateracedata['Two Or More Races'], bottom = stateracedata['Native Hawaiian/Other Pacific Islander'] + stateracedata.White + stateracedata.Black + stateracedata.Hispanic + stateracedata.Asian + stateracedata['American Indian/Alaska Native'], color='#db51e8', edgecolor='white', width=barWidth)
races = ['White', 'African American', 'Hispanic', 'Asian', 'Native American', 'Pacific Islander', 'Mixed Race']
plt.legend(races)

plt.ylabel("% of Population by Race")
plt.xticks(names)
plt.xlabel("States")
races = list(policekillingsdata2["Victim's race"].unique())
totalracedeaths = []

for race in races:
    racedeaths = policekillingsdata2[policekillingsdata2["Victim's race"] == race]
    totalracedeaths.append(len(racedeaths))

plt.figure(figsize = (10, 5))    
sns.barplot(x = races, y = totalracedeaths)
plt.xlabel('Race')
plt.ylabel('Total Deaths')
plt.title('Total Police Killings by Race (2013 - 2020)')

racedeathsPM = pd.merge(meanracedata, statekillsPM, on = 'States', how = 'inner')
poc = pd.DataFrame({'States' : racedeathsPM['States'], 'meanPOC':(100 - racedeathsPM['meanwhite'])})
racedeathsPM = pd.merge(racedeathsPM, poc, on = 'States', how = 'inner')


sns.lmplot(x = 'meanPOC', y = 'shootingsPM', data = racedeathsPM, size = 7)
plt.xlabel("% of Population that are People of Colour")
plt.ylabel('Deaths from Police Shootings per year per 1M Residence')
plt.title('')

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

label_point(racedeathsPM['meanPOC'] + .4, racedeathsPM['shootingsPM'] - .02, racedeathsPM.States, plt.gca())

frame3 = pd.merge(racedeathsPM, statepovertyandkills.drop(columns = ['shootingsPM']), on = 'States', how = 'inner')

sns.lmplot(x = 'meanPOC', y = 'Average poverty rate', data = frame3, height = 7)
plt.xlabel("% of Population that are People of Colour")
plt.ylabel('Average % People Below Poverty Line')
plt.title('')

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

label_point(frame3['meanPOC'] + .4, frame3['Average poverty rate'] - .1, frame3.States, plt.gca())
racedeathsbystate = pd.DataFrame(columns = ['States', 'White Deaths', 'Black Deaths', 'Hispanic Deaths', 'Asian Deaths', 'Native American Deaths', 'Deaths of Other Races', 'Pacific Islander Deaths'])
for state in states:
    stateracedeaths = pd.Series()
    statedeaths = policekillingsdata2[policekillingsdata2['State'] == state]
    for race in races:
        stateracedeaths[race] = len(statedeaths[statedeaths["Victim's race"] == race])
    racedeathsbystate = racedeathsbystate.append({'States' : state, 'White Deaths' : stateracedeaths['White'], 'Black Deaths' : stateracedeaths['Black'], 'Hispanic Deaths' : stateracedeaths['Hispanic'],
                                                  'Asian Deaths' : stateracedeaths['Asian'], 'Native American Deaths' : stateracedeaths['Native American'],
                                                  'Pacific Islander Deaths' : stateracedeaths['Pacific Islander'],
                                                  'Deaths of Other Races' : stateracedeaths['Unknown Race'], 'Total Deaths' : len(statedeaths)}, ignore_index = True)
deathproportionpoc = pd.DataFrame(columns = ['States', 'DeathProportionPOC'])
for state in states:
    deathproportionpoc = deathproportionpoc.append({'States' : state, 'DeathProportionPOC' : (100 - ((float(racedeathsbystate[racedeathsbystate['States'] == state]['White Deaths'].item()) / float(killsbystatecount2[state]) * 100)))}, ignore_index = True)

frame4 = pd.merge(deathproportionpoc, racedeathsbystate, on = 'States')
frame4 = pd.merge(frame3, frame4, on = 'States')
frame4.sort_values(by = 'Total Deaths', inplace = True, ascending = False)
barWidth = 0.4
bars1 = frame4['DeathProportionPOC'].mean()
bars2 = frame4['meanPOC'].mean()
 
r1 = 1
r2 = 1.6
 
plt.figure(figsize=(18, 10))
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='% of Deaths were Victim was a POC ')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='% of POC in the Population')

plt.tick_params(axis='x', which='both', bottom = False, top = False, labelbottom = False)
#plt.xlabel('States', fontweight='bold')
plt.legend(fontsize = 'x-large')
barWidth = 0.4
bars1 = frame4['DeathProportionPOC']
bars2 = frame4['meanPOC']
 
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
plt.figure(figsize=(18, 10))
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='% of Deaths were Victim was a POC ')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='% of POC in the Population')

plt.xlabel('States', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], frame4['States'])
plt.legend()
racelist = list(physicalforcedeaths["Victim's race"].unique())
a = physicalforcedeaths["Victim's race"].value_counts()

plt.figure(figsize = (16, 10))
sns.barplot(x = a.index, y = a.values)
plt.ylabel('Total Deaths')
plt.xlabel('Race of Victim')
plt.title('Deaths from Police Physical Force by Race from 2014 - 2020')
a, b, c = [], [], []

for index, row in politicsdata.iterrows():
    if row['City'] in list(citydeathsyear1m['City']):
        a.append(row['City'])
        b.append(citydeathsyear1m[citydeathsyear1m['City'] == row['City']]['Relative Annual Police Killings'].item())
        c.append(row['Republican Vote Share'])
politicsdeaths = pd.DataFrame({'City' : a, 'Deaths per year per 1M Residence' : b, 'Republican Vote Share' : c})

sns.lmplot(x = 'Republican Vote Share', y = 'Deaths per year per 1M Residence', data = politicsdeaths)
citypolicydeaths = pd.merge(citydeathsyear1m, policepolicydata, on = 'City', how = 'inner')

graph1 = pd.merge(citypolicydeaths['Requires De-Escalation'].astype(float), citypolicydeaths['Relative Annual Police Killings'].astype(float), left_index = True, right_index = True)
sns.lmplot(x = 'Relative Annual Police Killings',
           y = 'Requires De-Escalation',
           data = graph1, y_jitter=.01, logistic=True, truncate=False)

graph2 = pd.merge(citypolicydeaths['Has Use of Force Continuum'].astype(float), citypolicydeaths['Relative Annual Police Killings'].astype(float), left_index = True, right_index = True)
sns.lmplot(x = 'Deaths per year per 1M Residence',
           y = 'Has Use of Force Continuum',
           data = graph2, y_jitter=.01, logistic=True, truncate=False)
graph3 = pd.merge(citypolicydeaths['Bans Chokeholds and Strangleholds'].astype(float), citypolicydeaths['Relative Annual Police Killings'].astype(float), left_index = True, right_index = True)
sns.lmplot(x = 'Deaths per year per 1M Residence',
           y = 'Bans Chokeholds and Strangleholds',
           data = graph3, y_jitter=.01, logistic=True, truncate=False)
graph4 = pd.merge(citypolicydeaths['Requires Warning Before Shooting'].astype(float), citypolicydeaths['Relative Annual Police Killings'].astype(float), left_index = True, right_index = True)
sns.lmplot(x = 'Deaths per year per 1M Residence',
           y = 'Requires Warning Before Shooting',
           data = graph4)
graph5 = pd.merge(citypolicydeaths['Restricts Shooting at Moving Vehicles'].astype(float), citypolicydeaths['Relative Annual Police Killings'].astype(float), left_index = True, right_index = True)
sns.lmplot(x = 'Deaths per year per 1M Residence',
           y = 'Restricts Shooting at Moving Vehicles',
           data = graph5, y_jitter=.01, logistic=True, truncate=False)
graph6 = pd.merge(citypolicydeaths['Requires Exhaust All Other Means Before Shooting'].astype(float), citypolicydeaths['Deaths per year per 1M Residence'].astype(float), left_index = True, right_index = True)
sns.lmplot(x = 'Deaths per year per 1M Residence',
           y = 'Requires Exhaust All Other Means Before Shooting',
           data = graph6, y_jitter=.01, logistic=True, truncate=False)
graph7 = pd.merge(citypolicydeaths['Duty to Intervene'].astype(float), citypolicydeaths['Relative Annual Police Killings'].astype(float), left_index = True, right_index = True)
sns.lmplot(x = 'Deaths per year per 1M Residence',
           y = 'Duty to Intervene',
           data = graph7, y_jitter=.01, logistic=True, truncate=False)
graph8 = pd.merge(citypolicydeaths['Requires Comprehensive Reporting'].astype(float), citypolicydeaths['Relative Annual Police Killings'].astype(float), left_index = True, right_index = True)
sns.lmplot(x = 'Deaths per year per 1M Residence',
           y = 'Requires Comprehensive Reporting',
           data = graph8, y_jitter=.01, logistic=True, truncate=False)
graph9 = pd.merge(citypolicydeaths['Bans Chokeholds and Strangleholds'].astype(float), citypolicydeaths['Relative Annual Police Killings'].astype(float), left_index = True, right_index = True)
sns.lmplot(x = 'Deaths per year per 1M Residence',
           y = 'Bans Chokeholds and Strangleholds',
           data = graph9, y_jitter=.1, logistic=True, truncate=False)

list(citypolicydeaths['City'].unique())
banningchokeholds = pd.DataFrame(columns = ('City', 'Deaths by physical force'))
nodata = []
for city in list(citypolicydeaths['City'].unique()):
    if city in policekillingsdata2['City'].unique():
        b = 0
        for cause in physicalforce:
            a = policekillingsdata2[policekillingsdata2['Cause of death'] == cause][policekillingsdata2['City'] == city][policekillingsdata2['State'] == policepolicydata[policepolicydata['City'] == city]['State'].item()]
            b = b + len(a)

        banningchokeholds = banningchokeholds.append({'City' : city, 'Deaths by physical force' : b}, ignore_index = True)
banningchokeholds = pd.merge(banningchokeholds, policepolicydata, on = 'City', how = 'inner')
banningchokeholds['Deaths by physical force'] = banningchokeholds['Deaths by physical force'].astype(float)
banningchokeholds['Bans Chokeholds and Strangleholds'] = banningchokeholds['Bans Chokeholds and Strangleholds'].astype(float)
banningchokeholds['Has Use of Force Continuum'] = banningchokeholds['Has Use of Force Continuum'].astype(float)

sns.lmplot(x = 'Deaths by physical force',
           y = 'Bans Chokeholds and Strangleholds',
           data = banningchokeholds, y_jitter=.1, logistic=True, truncate=False)

sns.lmplot(x = 'Deaths by physical force',
           y = 'Has Use of Force Continuum',
           data = banningchokeholds, y_jitter=.1, logistic=True, truncate=False)
racedata[racedata['City'] == 'Detroit']
policepolicydata[policepolicydata['City'] == 'Milwaukee']
b = []
for city in np.delete(cities, [0]):
    if city in list(racedata.City):
        b.append(100 - float(racedata[racedata['City'] == city][racedata['Geographic area'] == policepolicydata[policepolicydata['City'] == city]['State'].item()]['share_white'].item()))

BMEcity = pd.DataFrame({'City' : np.delete(cities, [0]), '%BME' : b})
BMEcitydeaths = pd.merge(BMEcity, citydeathsyear1m, on = 'City', how = 'inner')

plt.figure(figsize = (16, 10))
sns.lmplot(x = 'Deaths per year per 1M Residence',
           y = '%BME',
           data = BMEcitydeaths,
          size = 7)

label_point(BMEcitydeaths['Deaths per year per 1M Residence'] + .4, BMEcitydeaths['%BME'] - .1, BMEcitydeaths.City, plt.gca())
geolist = list(policekillingsdata2['Human Geography'].unique())
racegeodeaths = pd.DataFrame(columns = ['Human Geography'])
for geography in geolist:
    b = policekillingsdata2[policekillingsdata2['Human Geography'] == geography]["Victim's race"].value_counts()
    racegeodeaths = racegeodeaths.append(b, ignore_index = True)
racegeodeaths['Human Geography'] = geolist

racegeoplot = pd.melt(racegeodeaths, id_vars = 'Human Geography', var_name = 'Race', value_name = 'Deaths')

sns.catplot(x = 'Human Geography', y = 'Deaths', hue = 'Race', data = racegeoplot, kind = 'bar', size = 7)
budgetdata['police_city']
plt.figure(figsize = (18,12))
sns.lineplot(x = 'year', y = 'police_city', data = budgetdata)
plt.ylabel('Police Budget')
alljuvenilecrimearrests = juvenilearrestsdata[juvenilearrestsdata['Offense'] == 'All crimes'][juvenilearrestsdata['Category'] == 'Juvenile Arrest Rates (Arrest of Persons Age 10-17/100,000 Persons Ages 10-17)']
alljuvenilecrimearrests = alljuvenilecrimearrests.transpose().iloc[2:]
alljuvenilecrimearrests.reset_index(level = 0, inplace = True)
alljuvenilecrimearrests.columns = ['year', 'Juvenile arrest rate']
alljuvenilecrimearrests.year = alljuvenilecrimearrests.year.astype(float)
alljuvenilecrimearrests['Juvenile arrest rate'] = alljuvenilecrimearrests['Juvenile arrest rate'].astype(float)

budgetyears = list(budgetdata['year'].unique())
b = []
for year in budgetyears:
    b.append(budgetdata[budgetdata['year'] == year]['police_city'].mean())

budgetyearsdata = pd.DataFrame({'year' : budgetyears, 'mean budget' : b})

fig = plt.figure(figsize = (16, 10))
ax = fig.subplots()
ax.plot(alljuvenilecrimearrests['year'], alljuvenilecrimearrests['Juvenile arrest rate'])
ax.set_xlabel('Year', fontsize = '10')
ax.set_ylabel('Juvenile Arrest Rate per 100,000 Juveniles', color = 'blue')

ax2 = ax.twinx()
ax2.plot(budgetyearsdata['year'], budgetyearsdata['mean budget'], color = 'red')
ax2.set_ylabel('Police Budget', color = 'red')
budgetdatacitylist = list(budgetdata['City'].unique())
comparebudgetdata = pd.DataFrame()
selectcitydeathsyear1m = pd.DataFrame()
for city in np.delete(cities, [0]):
    if city in budgetdatacitylist:
        comparebudgetdata = comparebudgetdata.append(budgetdata[budgetdata['City'] == city], ignore_index = True)
        selectcitydeathsyear1m = selectcitydeathsyear1m.append(citydeathsyear1m[citydeathsyear1m['City'] == city], ignore_index = True)

budgetdata2017 = pd.merge(comparebudgetdata[comparebudgetdata['year'] == 2017], selectcitydeathsyear1m, on = 'City', how = 'inner')

sns.lmplot(x = 'police_city', y = 'Deaths per year per 1M Residence', data = budgetdata2017)
chicagocrimebudget = pd.merge(comparebudgetdata[comparebudgetdata['City'] == 'Chicago'][comparebudgetdata['year'] > 2000],
                              crimesperyearchicago[crimesperyearchicago['year'] < 2018],
                              on = 'year',
                              how = 'inner',)

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(chicagocrimebudget.year, chicagocrimebudget.police_city, color="red", marker="o")
# set x-axis label
ax.set_xlabel("Year",fontsize=14)
# set y-axis label
ax.set_ylabel("Police Budget",color="red",fontsize=14)


# twin object for two different y-axis on the sample plot
ax2 = ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(chicagocrimebudget.year, chicagocrimebudget["reported crimes"],color="blue",marker="o")
ax2.set_ylabel("Reported Crimes",color="blue",fontsize=14)
plt.show()
yearlist = list(chicagocrimes.Year.unique())
a = []
for year in yearlist:
    a.append(len(chicagocrimes[chicagocrimes['Year'] == year]))
crimesperyearchicago = pd.DataFrame({'year' : yearlist, 
                                     'reported crimes' : a
                                    })
plt.figure(figsize = (16, 10))
sns.barplot(x = 'year', y = 'reported crimes', data = crimesperyearchicago)
a = []
for year in yearlist:
    a.append(len(chicagocrimes[chicagocrimes['Year'] == year][chicagocrimes['Arrest'] == True]))
arrestsperyearchicago = pd.DataFrame({'year' : yearlist, 
                                     'arrests' : a
                                    })
plt.figure(figsize = (16, 10))
sns.barplot(x = 'year', y = 'arrests', data = arrestsperyearchicago)
columns = ['City', 'State', 'police_city']
temppolicebudgets2017 = budgetdata[budgetdata['year'] == 2017][columns]
temppolicebudgets2017.dropna()
policebudgets2017 = pd.DataFrame()
for city, state in zip(cities, states2):
    policebudgets2017 = policebudgets2017.append(temppolicebudgets2017[temppolicebudgets2017['City'] == city][temppolicebudgets2017['State'] == state])
budgetcrimes = pd.merge(policebudgets2017, citydeathsyear1m, on = 'City', how = 'inner')
budgetcrimes.sort_values(by = 'Pop adjusted violent crimes', inplace = True, ascending = False)


fig = plt.figure(figsize = (16, 10))
ax = fig.subplots()
ax.plot(budgetcrimes['City'], budgetcrimes['Pop adjusted violent crimes'], 'o', color = 'blue')
plt.xticks(rotation = 90)
ax.set_ylabel('Violent Crime Rate', color = 'blue')


ax2 = ax.twinx()
ax2.plot(budgetcrimes['City'], budgetcrimes['police_city'], 'o', color = 'red')
ax2.set_ylabel('Police Budget (per capita dollars)', color = 'red')
budgetcrimes.sort_values(by = 'Pop adjusted arrests', inplace = True, ascending = False)
fig = plt.figure(figsize = (16, 10))
ax = fig.subplots()
ax.plot(budgetcrimes['City'], budgetcrimes['Pop adjusted arrests'], 'o', color = 'blue')
plt.xticks(rotation = 90)
ax.set_ylabel('Arrest Rate', color = 'blue')

ax2 = ax.twinx()
ax2.plot(budgetcrimes['City'], budgetcrimes['police_city'], 'o', color = 'red')
ax2.set_ylabel('Police Budget (per capita dollars)', color = 'red')
budgetcrimes[budgetcrimes['City'] == 'Las Vegas']