# Importing the Required Libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")

sns.set()
# Loading the Dataset from the Dropbox



!wget https://www.dropbox.com/s/k0wf1avmyq4cars/fatal-police-shootings-data.csv
df = pd.read_csv('fatal-police-shootings-data.csv', encoding='latin1')

df.head(5)
df.shape
# Plot the manner of death



plt.figure(figsize=(15,8))

ax = sns.countplot(x='manner_of_death', data=df)

plt.title('Manner of death')

for p in ax.patches:

    x = p.get_bbox().get_points()[:,0]

    

    y = p.get_bbox().get_points()[1,1]

    

    ax.annotate('{:.2g}%'.format(100.*y/len(df)), (x.mean(), y), ha='center', va='bottom')

plt.show()
# Plot the 20 most common weapons used by individuals shot

plt.figure(figsize=(15, 8))

ax = sns.countplot(y='armed', data=df,

                   order = df.armed.value_counts().iloc[:20].index)

plt.title('Weapon used by person shot')

plt.show()
plt.figure(figsize=(15, 8))



sns.distplot(df.age[~np.isnan(df.age)])

plt.title('Age of individuals shot by police')

plt.show()
len(df[df.age<16])
# Plot police killings by gender

plt.figure(figsize=(15, 8))



ax = sns.countplot(x = "gender", data = df,

                   order = df.gender.value_counts().index)

for p in ax.patches:

    x = p.get_bbox().get_points()[:,0]

    y = p.get_bbox().get_points()[1,1]

    ax.annotate('{:.2g}%'.format(100.*y/len(df)), (x.mean(), y), ha='center', va='bottom')

plt.title('Police Killings by Gender')

plt.show()
def ActualVsPopulation(df, pop, group):

    """Get dataframe with actual per-group percentage vs population group percentage"""

    d = {group: [], 'type': [], 'percent': []}

    tot_pop = float(sum(pop.values()))

    for g in df[group].dropna().unique(): #for each group



        # Actual percentages

        d[group].append(g)

        d['type'].append('Killings')

        d['percent'].append(100*df[df[group]==g].id.count()/df.id.count())



        # Percentages if statistic followed population distribution

        d[group].append(g)

        d['type'].append('Population') #based on population percentage

        d['percent'].append(100*pop[g]/tot_pop)

        

    return pd.DataFrame(data=d)
# Plot percent police killings by gender vs population percentages

pop_g = {'M': 49.2, 'F': 50.8} #percent population by gender https://www.census.gov/quickfacts/fact/table/US

df = ActualVsPopulation(df, pop_g, 'gender')



plt.figure(figsize=(15, 8))

sns.barplot(x="gender", y="percent", hue="type", data=df, palette=["r", "C0"])

plt.title('Actual Police Killings vs Population Distribution (by Gender)')

plt.show()
# Plot police killings by race



df1 = pd.read_csv('fatal-police-shootings-data.csv', encoding='latin1')



plt.figure(figsize=(15,8))

ax = sns.countplot(x="race", data=df1,

                   order = df1.race.value_counts().index)

for p in ax.patches:

    x = p.get_bbox().get_points()[:,0]

    y = p.get_bbox().get_points()[1,1]

    ax.annotate('{:.2g}%'.format(100.*y/len(df1)), (x.mean(), y), ha='center', va='bottom')



plt.title('Police Killings by Race')

plt.show()
# Plot percent police killings by race vs population percentages



#Population (%) by race gathered from https://www.census.gov/quickfacts/fact/table/US



pop_r = {'W': 60.1, # White  

         'B': 13.4, # Black or African american

         'H': 18.5, # Hispanic or Latino

         'A': 5.9,  # Asian

         'N': 1.5,  # American indian, Alaska Native, Native Hawaian, and Other Pacific Islander

         'O': 0.6}  # other







df = ActualVsPopulation(df1, pop_r, 'race')



plt.figure(figsize=(15,8))

sns.barplot(x="race", y="percent", hue="type", data=df,

            order = df1.race.value_counts().index, palette=["r", "C0"])

plt.title('Actual Police Killings vs Population Distribution (by Race)')

plt.show()
plt.figure(figsize=(18, 12))



sns.countplot(y="state", 

              data=df1,

              order=df1.state.value_counts().index)

plt.title('Police Killings By State')

plt.show()
# Convert date from object to datetime

df1.date = pd.to_datetime(df1.date)
# Plot shootings by month

plt.figure(figsize=(15,18))

sns.countplot(y=df1.date.dt.strftime('%Y %m %B'), 

              order=sorted(df1.date.dt.strftime('%Y %m %B').unique()))

plt.title('Police Killings By Month since start of dataset')

plt.show()
# Plot shootings by day of week

dow_map={0:'M', 1:'T', 2:'W', 3:'Th', 4:'F', 5:'Sa', 6:'Su'}



plt.figure(figsize=(15,8))



sns.countplot(x=df1.date.dt.dayofweek.map(dow_map), order=dow_map.values())

plt.title('Police Killings By Day of Week')

plt.show()
# Plot how many individuals were fleeing when shot



plt.figure(figsize=(15,8))



ax = sns.countplot(x='flee', data=df1)

for p in ax.patches:

    x = p.get_bbox().get_points()[:,0]

    y = p.get_bbox().get_points()[1,1]

    ax.annotate('{:.2g}%'.format(100.*y/len(df1)), (x.mean(), y), ha='center', va='bottom')

plt.title('Method of fleeing\nwhen shot by police')

plt.show()
# Count the proportion of shootings with body camera by state

pcPK = df1.groupby('state').agg({'body_camera': 'mean'})







# Plot percent of shootings with body camera by state



plt.figure(figsize=(6, 13))

sns.barplot(y=pcPK.index, 

            x=100.*pcPK.values.flatten(),

            order=pcPK.body_camera.sort_values(ascending=False).index)

plt.title('Percent of Police Killings\nwith body camera, by State')

plt.xlabel('Percent of Shootings with body camera')

plt.xlim([0, 100])

plt.show()