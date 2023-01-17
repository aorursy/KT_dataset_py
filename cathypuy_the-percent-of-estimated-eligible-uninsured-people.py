from IPython.display import Image

Image("/kaggle/input/estimateduninsured/Estimated-uninsured.png")
import pandas as pd

import geopandas as gpd

import numpy as np

import matplotlib as plt
# Load shapefile & preview it

file_path  =  '/kaggle/input/us-counties-map-shapefile/cb_2018_us_county_500k.shx'

map_dataframe = gpd.read_file(file_path)



# Clean map dataframe

map_dataframe.rename(columns = {'STATEFP':'StateFIPS'}, inplace = True)

map_dataframe['StateFIPS'] = map_dataframe['StateFIPS'].astype(dtype='int')

map_dataframe['NAME'] = map_dataframe['NAME'].astype(dtype='str')

for i in range(len(map_dataframe)):

    string = map_dataframe.loc[i, 'NAME']  

    string = string.replace(' ', '')

    map_dataframe.loc[i, 'NAME'] = string 

    

# Load  Data Frame & Clean

df = pd.read_csv('/kaggle/input/cms-estimated-uninsured-people/the-percent-of-estimated-eligible-uninsured-people-for-outreach-targeting.csv')

df.drop(df.columns.difference(['Counties Within PUMA',

                               'StateFIPS', 'uninsured_percent',

                               'uninsured_total']), 1, inplace=True)

df['map_ID'] = np.nan # feature mapping PUMA regions to county map



# clean 'Counties Within PUMA' column, rows with multiple counties

# are separated

clean_df = pd.DataFrame()

for i in range(len(df)):

    string = df['Counties Within PUMA'][i]  

    string = string.replace('(part)', '')

    string = string.replace(' ', '')

    string = string.replace('County',  '')

    df.loc[i, 'Counties Within PUMA'] = string

    if ',' in string:

        county_list = string.split(',')

        copy = (df.loc[i,:]).copy()

        for j in range(len(county_list)):

            copy['Counties Within PUMA'] = county_list[j]

            copy['map_ID'] = i

            clean_df = clean_df.append(copy)

    else:

        copy = (df.loc[i,:]).copy()

        copy['map_ID'] = i

        clean_df = clean_df.append(copy)   





# clean 'Counties within PUMA' column, multiple rows of the same county are combined

counts = clean_df.groupby(['StateFIPS','Counties Within PUMA']).size().reset_index().rename(columns={0:'count'})

clean_df.reset_index(drop=True, inplace=True)

for i in range(len(counts)):

    # separate data by counties

    if counts['count'][i] > 1:

        rows_in_df = clean_df.loc[(clean_df['StateFIPS'] == counts['StateFIPS'][i]) & 

                            (clean_df['Counties Within PUMA'] == counts['Counties Within PUMA'][i])].reset_index(drop=True)

        # to aggregate the total percent uninsured in county,

        # want to find uninsured_total/total_ppl

        uninsured_total = np.sum(rows_in_df['uninsured_total'])

        # calculate total population of each county

        # formula is: sum(uninsured_total*multiplier)

        # where multiplier is the magnitude of size the county_total

        # is compared to the uninsured_total

        multiplier = 100/(rows_in_df['uninsured_percent'])

        total_ppl = np.dot(multiplier.T, rows_in_df['uninsured_total'])

        uninsured_percent = (uninsured_total/total_ppl)*100

        # delete multiple rows of the same county, but save one 

        # for aggregate stats

        indexNames = clean_df[(clean_df['StateFIPS'] == counts['StateFIPS'][i]) & 

                            (clean_df['Counties Within PUMA'] == counts['Counties Within PUMA'][i])].index

        copy = rows_in_df.loc[0,:].copy()

        clean_df.drop(indexNames, inplace=True)

        copy['uninsured_percent'] = uninsured_percent

        copy['uninsured_total'] = uninsured_total

        clean_df = clean_df.append(copy)

#        print('StateFIPS: ', counts['StateFIPS'][i],

#                'County: ', counts['Counties Within PUMA'][i], 

#              ', uninsured_percent: ', uninsured_percent)



# Since we combined rows, reset index. also make sure map_dataframe and 

# clean_df have same types for comparison and merging        

clean_df.reset_index(drop=True, inplace=True)

clean_df['StateFIPS'] = clean_df['StateFIPS'].astype(dtype='int')

clean_df['Counties Within PUMA'] = clean_df['Counties Within PUMA'].astype(dtype='str')

clean_df.rename(columns={'Counties Within PUMA': 'NAME'}, inplace=True)

  



# merge table data and map data, preserving counties with no data

merged = pd.merge(map_dataframe, clean_df, on=['StateFIPS','NAME'],  how='left',)

merged['map_ID'].fillna(value=-1, inplace=True) # counties with no data

merged = merged.dissolve(by='map_ID') # merge PUMA regions



# prepare figure for plotting

variable = 'uninsured_percent'

fig, ax = plt.pyplot.subplots(figsize=(11,8.5))

plt.pyplot.xlim([-190,-50])

plt.pyplot.ylim([10,80])



# plot data, fill missing data as black

merged[merged.uninsured_percent.notna()].plot(column=variable, 

            cmap='Purples', figsize=(11, 8.5), ax=ax, legend=True, 

            legend_kwds={'label': 'Percent', 'orientation': 'horizontal',

                         'pad': 0})

merged[merged.uninsured_percent.isna()].plot(color='black', ax=ax)



# get rid of geographic axes, label graph, and save

ax.get_xaxis().set_ticks([])

ax.get_yaxis().set_ticks([])

ax.set_title('The Percent of Estimated Eligible Uninsured People by PUMA Region')

ax.annotate('Areas with missing data are blacked out',xy=(0.15, 0.1),

            xycoords='figure fraction', horizontalalignment='left', 

            verticalalignment='top', color='#555555')

plt.pyplot.savefig('Estimated-uninsured.png', dpi=600,

        orientation='landscape')