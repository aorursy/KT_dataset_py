# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(11,8)})

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/metal-by-nation"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

world = pd.read_csv('/kaggle/input/metal-by-nation/world_population_1960_2015.csv', encoding='latin-1')

bands = pd.read_csv('/kaggle/input/metal-by-nation/metal_bands_2017.csv', encoding='latin-1')
# Getting number of bands per country

bands_country = bands['origin'].value_counts()

plt.title('Counts of bands per country')

plt.xticks(rotation=45)

sns.barplot(x=bands_country[:10].keys(), y=bands_country[:10].values)
# Getting More Descriptive information, Bands per capita (related to country's population)

# Results are per 10 000 people (ex: 1.2 represents that 1.2 persons out of 10 000 were in a band)

year = '1970'

band_country_info = pd.DataFrame(columns=['Country Name', 'Population', 'Band Count', 'Bands Per Capita'])

for index, country in world.iterrows():

    country_name = country['Country Name']

    if(country_name in bands_country):

        country_pop = country[year]

        bands_in_country = bands_country[country['Country Name']]

        bands_per_capita = (bands_in_country/country_pop) * 10000

        band_country_info = band_country_info.append({'Country Name': country_name, 'Population': country_pop, 'Band Count': bands_in_country , 'Bands Per Capita': bands_per_capita }, ignore_index=True)

band_country_info = band_country_info.sort_values(by = 'Bands Per Capita', ascending=False)





plt.title("Bands per Capita in " + year)

plt.xticks(rotation=45)

sns.barplot(x = band_country_info[:10]['Country Name'],y = band_country_info[:10]['Bands Per Capita'])

# Evaluating most famous bands based in the number of fans

# Data needs cleaning (there are duplicates for band names)

bands_sorted_fans = bands.sort_values(by = 'fans', ascending=False)





sns.set(rc={'figure.figsize':(44,32)})

sns.set(font_scale=3)



plt.title('Top Fans per band')

plt.xticks(rotation=90)

sns.barplot(x=bands_sorted_fans[:100]['band_name'], y = bands_sorted_fans[:100]['fans'].values)

#bands_sorted_fans.loc[bands_sorted_fans['band_name'] == 'Tool']
pure = {}

mixed = {}

# Comparing Style of playing between pure style or mixed styles

style_df = pd.DataFrame(columns=['Style', 'Pure', 'Mixed'])

for index, band in bands.iterrows():

    cell_style = band['style']

    styles = cell_style.split(',')

    if (len(styles) == 1):

        pure[styles[0]] = pure.get(styles[0], 0) + 1

    else:

        for style in styles:

            mixed[style] = mixed.get(style, 0) + 1



style_df['Style']= mixed.keys()

style_df['Mixed']= style_df['Style'].map(mixed)

style_df['Pure'] = style_df['Style'].map(pure)



style_df = style_df.sort_values(by='Mixed', ascending=False)[:50]



ax = style_df[['Pure', 'Mixed']].plot(kind='bar', color=['#DD2763', '#74BBA4'])

ax.set_xticklabels(style_df['Style'], rotation=90)

plt.title('Counts of the top Metal Genres')

mixed_legend = mpatches.Patch(color='#DD2763', label='Pure')

pure_legend = mpatches.Patch(color='#74BBA4', label='Mixed')

plt.legend(handles=[mixed_legend, pure_legend])
# Is metal music still alive?

# Evaluate Band Number per year



#Key is year value is number of bands formed

bands_formed = {}

#Key is year value is number of bands splitted

bands_split = {}

#Key is year value is number of bands active

bands_active = {}



for index, band in bands.iterrows():

    formed = bands_formed.get(band['formed'], 0) + 1 

    bands_formed[band['formed']] = formed

    split = bands_split.get(band['split'], 0) + 1

    bands_split[band['split']] = split

    active = formed - split

    bands_active[band['formed']] = active if active > 0 else 0



metal_alive = pd.DataFrame(columns = ['Year', 'formed','split'])

metal_alive['Year'] = bands_formed.keys()

metal_alive['formed'] = metal_alive['Year'].map(bands_formed)

metal_alive['split'] = metal_alive['Year'].map(bands_split)



metal_alive['Year'] = pd.to_numeric(metal_alive['Year'], errors='coerce')

metal_alive['formed'] = pd.to_numeric(metal_alive['formed'], errors='coerce')



invalid_years = metal_alive[np.isnan(metal_alive['Year'])].index

metal_alive.drop(invalid_years,inplace=True)

#metal_alive = metal_alive.sort_values(by='Year', ascending=True)





ax = metal_alive[['formed', 'split']].plot(kind='bar', color=['#DD2763', '#74BBA4'])

ax.set_xticklabels(metal_alive['Year'], rotation=90)

plt.title('Is Metal Still Alive?')

formed_legend = mpatches.Patch(color='#DD2763', label='Bands Formed')

split_legend = mpatches.Patch(color='#74BBA4', label='Bands Split')

plt.legend(handles=[formed_legend, split_legend])



sns.set(rc={'figure.figsize':(11,8)})

sns.set(font_scale=1)

#Based on Before, visualize current number of active metal bands per year



metal_active = pd.DataFrame(columns= ['Year','Active'])

metal_active['Year'] = bands_active.keys()

metal_active['Active'] = metal_active['Year'].map(bands_active)





metal_active['Year'] = pd.to_numeric(metal_active['Year'], errors='coerce')

metal_active['Active'] = pd.to_numeric(metal_active['Active'], errors='coerce')



invalid_years = metal_active[np.isnan(metal_active['Year'])].index

metal_active.drop(invalid_years,inplace=True)

metal_active = metal_active.sort_values(by='Year', ascending=True)



plt.title("Active Bands Yearly")

ax = sns.violinplot(y = metal_active['Year'], data=metal_active[['Active']], inner='quartile')
