from IPython.display import Image
Image("../input/us-shootings-pic/m1.jpg" ,width="800" )

%matplotlib inline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/modified-dataset/Mass_Shootings.csv')
data.head()
data.columns
del data['S#']
del data['Policeman Killed']
del data['Gender']
del data['Employeed (Y/N)']
del data['Employed at']
data.columns
len(data)
data['Age'].describe() 
years = data['Year'].unique().tolist()
years = sorted(years)
total_victims_groupedby_year = data.groupby('Year')['Total victims'].sum()


sns.set_style("whitegrid")
plt.figure(figsize=(20,5))
plt.title('US mass shootings victim count from 1966 - 2017', fontsize = 18)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Total victims', fontsize = 15)
sns.pointplot(years, total_victims_groupedby_year) #Excluded '2018' as only 2 mass shootings occured in this year
sns.pointplot(years[37:42], total_victims_groupedby_year[37:42])
plt.xlabel('Year')
mask_13 = data['Year'] == 2013
mask_17 = data['Year'] == 2017
data_13 = data[mask_13]; data_17 = data[mask_17]

data_17['Total victims'].sum() / data_13['Total victims'].sum()
data_13to17 = data[(data['Year'] == 2013) | ((data['Year'] == 2014)) | ((data['Year'] == 2015)) | ((data['Year'] == 2016)) |
    ((data['Year'] == 2017))]
plt.figure(figsize=(10,6))
sns.countplot(y = data_13to17['Cause'], order = data_13to17['Cause'].value_counts().index)
len(data_13to17) / len(data) * 100
sns.countplot(y = data_13to17['Mental_Health_Issues'], order = data_13to17['Mental_Health_Issues'].value_counts().index)
Image("../input/us-shootings-pic/m2.jpg" ,width="800" )
top15_mass_shootings = data[['Title', 'Location', 'Year', 'Date','Cause','Mental_Health_Issues', 'Incident Area', 'Injured', 'Fatalities', 'Total victims']].sort_values(
    'Fatalities', ascending = False)[:15]
top15_mass_shootings
top15_mass_shootings[:1]
plt.figure(figsize=(10,6))


sns.countplot(top15_mass_shootings['Cause'], order = top15_mass_shootings['Cause'].value_counts().index,
            palette='RdBu')
plt.title('Causes for the top 15 deadliest shootings', fontsize = 13)

top15_mass_shootings_v = data[['Title', 'Location', 'Year', 'Date','Cause','Mental_Health_Issues', 'Incident Area', 'Injured', 'Fatalities', 'Total victims']].sort_values(
    'Total victims', ascending = False)[:15]
top15_mass_shootings_v
plt.figure(figsize=(10,6))

sns.countplot(y = data['Cause'], order = data['Cause'].value_counts().index)

plt.figure(figsize=(10,6))

sns.countplot(y = data['Race'], order = data['Race'].value_counts().index, palette='Blues_r')

plt.figure(figsize=(15,10))
sns.countplot(y = data['Target'], order = data['Target'].value_counts().index)
plt.figure(figsize=(10,6))
sns.countplot(data['Mental_Health_Issues'], palette='Greens_r', order = data['Mental_Health_Issues'].value_counts().index)
percentage_of_mental_health_shootings = (data.Mental_Health_Issues == 'Yes').sum() / (data['Mental_Health_Issues'].count())
percentage_of_mental_health_shootings * 100
plt.figure(figsize=(10,6))

sns.countplot(y = data['Month'], order = data['Month'].value_counts().index)

from IPython.display import IFrame
powerBiEmbed = 'https://app.powerbi.com/view?r=eyJrIjoiMDMyMmJhMDItZWY3YS00Mzc1LWFiMDUtNzY4YjUwOWFmM2FhIiwidCI6IjZlODUyZjhkLTNlNGItNDRkZC04M2RhLTAyM2M5OGY3ZjdhYSJ9'
IFrame(powerBiEmbed, width=800, height=600)
Image("../input/us-shootings-pic/m3.jpg" ,width="800")