import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# Dataset containing information about crimes in San Francisco, CA

url = '../input/sanfranciso-crime-dataset/Police_Department_Incidents_-_Previous_Year__2016_.csv'
dataframe = pd.read_csv(url)

dataframe
dataframe['Category'] = dataframe['Category'].str.capitalize()

dataframe['Descript'] = dataframe['Descript'].str.capitalize()

dataframe['PdDistrict'] = dataframe['PdDistrict'].str.capitalize()

dataframe['Resolution'] = dataframe['Resolution'].str.capitalize()
dataframe.head(10)
dataframe.isnull().sum()
dataframe[dataframe['PdDistrict'].isnull()].index.tolist()
dataframe.loc[[112851]]
dataframe.dropna(inplace=True)

dataframe.reset_index(drop=True, inplace=True)
dataframe
# Print the values in columns X, Y and Location to confirm that they are the same

print(dataframe['X'][0], dataframe['Y'][0])

print(dataframe['Location'][0])
#Remove columns X and Y

dataframe = dataframe.drop('X', 1)

dataframe = dataframe.drop('Y', 1)
dataframe.head()
for elem in range(len(dataframe)):

    dataframe['Date'][elem] = dataframe['Date'][elem][:10]
dataframe['Date'].head()
# IncidntNum and PdI columns with differents sizes, since IncidntNum has to do with the category and district in which it happened

print(len(np.unique(dataframe['IncidntNum'])))

print(len(np.unique(dataframe['PdId'])))
#Number of occurrences of all categories of crime

dataframe['Category'].value_counts()
categoriasMF =  dataframe['Category'].value_counts()[:10]

categoriasMF
categoriasMF =  dataframe['Category'].value_counts()[:10]



plt.style.use('default')

cmap = plt.cm.YlGnBu

colors = cmap(np.linspace(0., 1., 10))

ylabel = 'Occurrence'



fig, ax = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)

fig.suptitle('The 10 most frequent crimes', fontsize=16)

categoriasMF.plot.pie(autopct="%.1f%%", colors = colors, ax = ax[0])

categoriasMF.plot.bar(color = colors, ylabel = ylabel, ax = ax[1])

#plt.tight_layout()
days = dataframe['DayOfWeek'].value_counts()

days 
days.plot.bar(title = 'Number of occurrences by days of the week', figsize=(5, 5), color = colors, ylabel = ylabel)
friday = dataframe.loc[(dataframe['DayOfWeek'] == 'Friday')]

saturday = dataframe.loc[(dataframe['DayOfWeek'] == 'Saturday')]

thursday = dataframe.loc[(dataframe['DayOfWeek'] == 'Thursday')]

wednesday = dataframe.loc[(dataframe['DayOfWeek'] == 'Wednesday')]

tuesday = dataframe.loc[(dataframe['DayOfWeek'] == 'Tuesday')]

monday = dataframe.loc[(dataframe['DayOfWeek'] == 'Monday')]

sunday = dataframe.loc[(dataframe['DayOfWeek'] == 'Sunday')]
#Every day have the same order of crimes

friday = friday['Category'].value_counts()[:5]

friday
#Except Saturday

saturday = saturday['Category'].value_counts()[:5]

saturday
fig, ax = plt.subplots(1, 2, figsize=(10,5), constrained_layout=True)

fig.suptitle('Occurrences on Friday and Saturday', fontsize=16)

friday.plot.bar(color = colors, ylabel = ylabel, ax=ax[0])

saturday.plot.bar(color = colors, ylabel = ylabel, ax=ax[1])
morning = dataframe.loc[((dataframe['Time'] > '00:00') & (dataframe['Time'] < '12:00'))]

afternoon = dataframe.loc[((dataframe['Time'] > '12:00') & (dataframe['Time'] < '18:00'))]

evening = dataframe.loc[(dataframe['Time'] > '18:00')]
morn = morning['Time'].value_counts()[:5]

morn
after = afternoon['Time'].value_counts()[:5]

after
even = evening['Time'].value_counts()[:5]

even
fig, ax = plt.subplots(1, 3, figsize=(12,5))

morn.plot.bar(title = 'Morning', color = colors, ylabel = ylabel, ax=ax[0])

after.plot.bar(title = 'Afternoon', color = colors, ylabel = ylabel, ax=ax[1])

even.plot.bar(title = 'Evening', color = colors, ylabel = ylabel, ax=ax[2])

plt.tight_layout()
mornCat = morning['Category'].value_counts()[:5]

mornCat
afterCat = afternoon['Category'].value_counts()[:5]

afterCat
evenCat = evening['Category'].value_counts()[:5]

evenCat
fig, ax = plt.subplots(1, 3, figsize=(12,5))

mornCat.plot.bar(title = 'Morning', color = colors, ylabel = ylabel, ax=ax[0])

afterCat.plot.bar(title = 'Afternoon', color = colors, ylabel = ylabel, ax=ax[1])

evenCat.plot.bar(title = 'Evening', color = colors, ylabel = ylabel, ax=ax[2])

plt.tight_layout()
districts = dataframe['PdDistrict'].value_counts()

districts
fig, ax = plt.subplots(1, 2, figsize=(10,5))

districts.plot.pie(autopct ="%.1f%%", colors = colors, title = 'Percentage of cases per district', ax = ax[0])

districts.plot.bar(title = 'Number of ocurrences by district', color = colors, ylabel = ylabel, ax = ax[1])

plt.tight_layout()
resolution = dataframe['Resolution'].value_counts()[:4]

resolution
resolution.plot.bar(title = 'The 4 most frequent resolutions', figsize=(5, 5), color = colors, ylabel = ylabel)
none = dataframe.loc[dataframe['Resolution'] == 'None']

noneCat = none['Category'].value_counts()[:10]

noneCat
booked = dataframe.loc[dataframe['Resolution'] == 'Arrest, booked']

bookedCat = booked['Category'].value_counts()[:10]

bookedCat
unfouded = dataframe.loc[dataframe['Resolution'] == 'Unfounded']

unfoudedCat = unfouded['Category'].value_counts()[:10]

unfoudedCat
juvenile = dataframe.loc[dataframe['Resolution'] == 'Juvenile booked']

juvenileCat = juvenile['Category'].value_counts()[:10]

juvenileCat
fig, ax = plt.subplots(2, 2, figsize=(10,10))

noneCat.plot.bar(title = 'None', color = colors, ylabel = ylabel, ax=ax[0,0])

bookedCat.plot.bar(title = 'Booked', color = colors, ylabel = ylabel, ax=ax[0,1])

unfoudedCat.plot.bar(title = 'Unfouded', color = colors, ylabel = ylabel, ax=ax[1,0])

juvenileCat.plot.bar(title = 'Juvenile', color = colors, ylabel = ylabel, ax=ax[1,1])

plt.tight_layout()