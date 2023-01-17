import pandas as pd

pd.set_option('max_columns', None)

shootings = pd.read_csv('../input/stanford-msa/mass_shooting_events_stanford_msa_release_06142016.csv')

shootings.head(3)
shootings['Date'] = pd.to_datetime(shootings['Date'])
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")



f, axarr = plt.subplots(2, 2, figsize=(14, 8))

# plt.suptitle('Mass Shooting Events Breakdown', fontsize=18)

f.subplots_adjust(hspace=0.5)



kde_kwargs = {'color': 'crimson', 'shade': True}



sns.kdeplot(shootings['Number of Victim Fatalities'], ax=axarr[0][0], **kde_kwargs)

axarr[0][0].set_title("Victim Fatalities", fontsize=14)



sns.kdeplot(shootings['Number of Victims Injured'], ax=axarr[0][1], **kde_kwargs)

axarr[0][1].set_title("Victim Injuries", fontsize=14)



sns.countplot(shootings['Day of Week'], ax=axarr[1][0], color='salmon')

axarr[1][0].set_title("Day of Week of Attack", fontsize=14)



sns.kdeplot(shootings['Date'].dt.year, ax=axarr[1][1], **kde_kwargs)

axarr[1][1].set_title("Year of Attack", fontsize=14)



sns.despine()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")



f, axarr = plt.subplots(3, 2, figsize=(14, 12))

# plt.suptitle('Mass Shooting Events Breakdown', fontsize=18)

f.subplots_adjust(hspace=0.5)



kde_kwargs = {'color': 'crimson', 'shade': True}



sns.countplot((shootings['Total Number of Fatalities'] - 

               shootings['Number of Victim Fatalities']) > 0, ax=axarr[0][0])

axarr[0][0].set_title("Shooter(s) Killed At Scene", fontsize=14)



sns.countplot(shootings['Shooter Sex'], ax=axarr[0][0])

axarr[0][0].set_title("Sex of Shooter(s)", fontsize=14)



sns.kdeplot(

    shootings[shootings['Average Shooter Age'].map(

        lambda v: pd.notnull(v) and "Unknown" not in str(v) and str(v).isdigit())

             ]['Average Shooter Age'], 

    ax=axarr[0][1], **kde_kwargs)

axarr[0][1].set_title("Average Age of Shooter(s)", fontsize=14)



sns.countplot(shootings['Fate of Shooter at the scene'], ax=axarr[1][0])

axarr[1][0].set_title("Fate of Shooter(s) at Scene", fontsize=14)



sns.countplot(shootings['Shooter\'s Cause of Death'], ax=axarr[1][1])

axarr[1][1].set_title("Shooter(s) Cause of Death", fontsize=14)



sns.countplot(shootings['History of Mental Illness - General'], ax=axarr[2][0])

axarr[2][0].set_title("History of Mental Illness?", fontsize=14)



sns.countplot(shootings['School Related'].map(lambda v: v if pd.isnull(v) else str(v).replace('no', 'No').replace('Killed', 'Unknown').replace('UnkNown', 'Unknown')), ax=axarr[2][1])

axarr[2][1].set_title("School Related?", fontsize=14)



sns.despine()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 8))

ax = fig.gca()



sns.countplot(

    shootings[shootings['Shooter Race'].isin(

            shootings['Shooter Race'].value_counts()[(shootings['Shooter Race'].value_counts() > 3)].index.values

    )]['Shooter Race'], ax=ax, color='salmon'

)

sns.despine()
import geoplot as gplt

import geoplot.crs as gcrs

import geopandas as gpd

import matplotlib.pyplot as plt



us_states = gpd.read_file('../input/united-states-state-shapes/us-states.json')

shootings_by_state = shootings.groupby('State').count().join(us_states.set_index('NAME10'))

shootings_by_state = gpd.GeoDataFrame(shootings_by_state.loc[:, ['Location', 'geometry']])

shootings_by_state = shootings_by_state.rename(columns={'Location': 'Incidents'})

shootings_by_state = shootings_by_state.drop(['Alaska', 'Hawaii'])



gplt.choropleth(shootings_by_state, hue='Incidents', 

                projection=gcrs.AlbersEqualArea(central_longitude=-98, central_latitude=39.5),

                k=None, cmap='YlOrRd', figsize=(16, 10), legend=True)

plt.gca().set_ylim((-1647757.3894385984, 1457718.4893930717))

pass