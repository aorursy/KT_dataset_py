%config IPCompleter.greedy=True

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
data18 = pd.read_csv('../input/schengen-visa-stats/2018-data-for-consulates.csv')

data17 = pd.read_csv('../input/schengen-visa-stats/2017-data-for-consulates.csv')



data17.info()
# Drop summary rows at the end of the dataset

data18.drop(1906, inplace=True)

data17.drop(range(1871,1883), inplace=True)

# Function which removes percent symboles and converts columns to numeric types

def makePretty(train_data):

    train_data['Share of MEVs on total number of uniform visas issued'].fillna('0%', inplace=True)

    train_data['Share of MEVs on total number of uniform visas issued'] = train_data['Share of MEVs on total number of uniform visas issued'].str.replace(r'\%$', '').astype('float') / 100

    train_data['Not issued rate for uniform visas'].fillna('0%', inplace=True)

    train_data['Not issued rate for uniform visas'] = train_data['Not issued rate for uniform visas'].str.replace(r'\%$', '').astype('float') / 100

    train_data['Not issued rate for ATVs and uniform visas '].fillna('0%', inplace=True)

    train_data['Not issued rate for ATVs and uniform visas'] = train_data['Not issued rate for ATVs and uniform visas '].str.replace(r'\%$', '').astype('float') / 100

    train_data.drop(['Not issued rate for ATVs and uniform visas '], axis=1, inplace=True)

    

    int_cols = ['Uniform visas applied for','Multiple entry uniform visas (MEVs) issued',

            'Total LTVs issued','Uniform visas not issued',

            'Total ATVs and uniform visas issued  (including multiple ATVs, MEVs and LTVs) ']

    for col in int_cols:

        train_data[col] = train_data[col].str.replace(r',', '')#.astype('int')

        train_data[col].fillna(0, inplace=True)   

        train_data[col] = train_data[col].astype('int')

makePretty(data18)

makePretty(data17)
grouped_target17 = data17.groupby(['Schengen State']).sum().sort_values(by=['Total ATVs and uniform visas issued  (including multiple ATVs, MEVs and LTVs) '])

grouped_target17['RejectionRate'] = grouped_target17['Uniform visas not issued'] / grouped_target17['Uniform visas applied for']

grouped_target18 = data18.groupby(['Schengen State']).sum().reindex(grouped_target17.index)

grouped_target18['RejectionRate'] = grouped_target18['Uniform visas not issued'] / grouped_target18['Uniform visas applied for']



fig, axs = plt.subplots(1,2, figsize=(12,10))

axs = axs.flatten()

h1 = axs[0].barh(y=np.arange(0,len(grouped_target17))+0.3, height = 0.3, tick_label = grouped_target17.index,

               width=grouped_target17['Total ATVs and uniform visas issued  (including multiple ATVs, MEVs and LTVs) '])

h2 = axs[0].barh(y=np.arange(0,len(grouped_target18)), height = 0.3, tick_label = grouped_target18.index,

               width=grouped_target18['Total ATVs and uniform visas issued  (including multiple ATVs, MEVs and LTVs) '])



h = axs[1].barh(y=np.arange(0,len(grouped_target17))+0.3, height = 0.3, 

                tick_label=grouped_target17.index, width=grouped_target17['RejectionRate'])

h = axs[1].barh(y=np.arange(0,len(grouped_target18)), height = 0.3, 

                tick_label=grouped_target18.index, width=grouped_target18['RejectionRate'])



axs[0].set_xlabel('Total ATVs and uniform visas issued')

#axs[1].set_ylabel('Uniform visas not issued')

axs[1].set_xlabel('Rejection rate')





h = axs[0].tick_params(axis='x', rotation=45)

h = axs[1].tick_params(axis='x', rotation=45)



axs[0].legend((h1[0], h2[0]), ('2017', '2018'))



fig.tight_layout()
grouped_target_source17 = data17.groupby(['Schengen State', 'Country where consulate is located']).sum().sort_values(by=['Total ATVs and uniform visas issued  (including multiple ATVs, MEVs and LTVs) '])

grouped_target_source17['RejectionRate'] = grouped_target_source17['Uniform visas not issued'] / grouped_target17['Uniform visas applied for']

grouped_target_source18 = data18.groupby(['Schengen State', 'Country where consulate is located']).sum().reindex(grouped_target_source17.index)

grouped_target_source18['RejectionRate'] = grouped_target_source18['Uniform visas not issued'] / grouped_target_source18['Uniform visas applied for']

(grouped_target_source18.loc['Poland',:] - grouped_target_source17.loc['Poland',:]).sort_values(by='Uniform visas applied for').head()

import geoplot

import geopandas

pd.options.mode.chained_assignment = None # Suppress SettingWithCopy warning just for this example



from mpl_toolkits.axes_grid1 import make_axes_locatable

world = geopandas.read_file(

    geopandas.datasets.get_path('naturalearth_lowres')

)

# Fix different naming 

world.at[153,'name'] = 'Czech Republic'

eu = world[world['name'].isin(grouped_target17.index)]

# Set index as name to insert the column

eu.index = eu['name']

eu['TotalVisas2018'] = grouped_target18['Total ATVs and uniform visas issued  (including multiple ATVs, MEVs and LTVs) ']
fig, axs = plt.subplots(1,1, figsize=(12,10))

divider = make_axes_locatable(axs)

cax = divider.append_axes("right", size="5%", pad=0.1)

eu.plot('TotalVisas2018', ax=axs, legend=True, cax=cax, cmap='Blues')

axs.set_ylim([30,75])

axs.set_xlim([-30,35])

axs.set_facecolor('xkcd:light blue')

axs.tick_params(

    axis='both',          # changes apply to the x-axis

    which='both',      # both major and minor ticks are affected

    bottom=False,      # ticks along the bottom edge are off

    top=False,         # ticks along the top edge are off

    labelleft=False,   # labels along the left edge are off

    left=False,        # ticks along the left edge are off

    labelbottom=False) # labels along the bottom edge are off

fig.tight_layout()