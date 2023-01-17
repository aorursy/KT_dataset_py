import numpy as np

import pandas as pd

import folium

import geopy

import time

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



from folium import FeatureGroup, LayerControl, Map, Marker

from geopy.geocoders import Nominatim

from pandas.plotting import register_matplotlib_converters



pd.plotting.register_matplotlib_converters()



nom=Nominatim(user_agent="my-application")

p = sns.cubehelix_palette(15)

p2 = sns.cubehelix_palette(10)

p3 = sns.cubehelix_palette(24)

p4 = sns.cubehelix_palette(5)



%matplotlib inline
# Take in raw data and examine it...

dataFile = '../input/nys-spill-incidents/spill-incidents.csv'
NYSspillsRAW_df = pd.read_csv(dataFile)
# Clean up raw data - set date format, re-order into ascending date order, remove old codes

NYSspillsRAW_df['Spill Date'] = pd.to_datetime(NYSspillsRAW_df['Spill Date'])

NYSspillsRAW_df.index = pd.DatetimeIndex(NYSspillsRAW_df['Spill Date'])

NYSspillsRAW_df2 = NYSspillsRAW_df.sort_index().copy()
# do some clean up: narrow down, clean up the date range (early entries are sparse, some are blanks), remove unclassified entries

# https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value

NYSspillsRAW_df3 = NYSspillsRAW_df2[NYSspillsRAW_df2['Source']!='Missing Code in Old Data - Must be fixed']

NYSspillsRAW_df4 = NYSspillsRAW_df3[['Street 1', 'Locality', 'County', 'Spill Date', 'Contributing Factor',

                                'Source', 'Material Name', 'Material Family', 'Quantity', 'Units']].dropna()

NYSspillsRAW_df5 = NYSspillsRAW_df4[NYSspillsRAW_df4['Spill Date'] >= '1984-01-01' ]

len(NYSspillsRAW_df5)
# Let's visualize some of the basic content

spillLocalities = NYSspillsRAW_df5['Locality'].value_counts()

top20SpillLocalities = spillLocalities[:20]

matCat_freq = NYSspillsRAW_df5['Material Family'].value_counts()
f, axes = plt.subplots(1,2)

f.set_figheight(8)

f.set_figwidth(15)

plt.subplots_adjust(wspace=.7)

axes[0].set_title("Most common spill localities")

sns.countplot(y="Locality", data=NYSspillsRAW_df5, order=top20SpillLocalities.index, palette=p3, ax=axes[0])

axes[1].set_title("Material Families")

sns.countplot(y="Material Family", data=NYSspillsRAW_df5, order=matCat_freq.index, palette=p4, ax=axes[1])
# source vs cause

# http://www.datasciencemadesimple.com/cross-tab-cross-table-python-pandas/

sourcesVScauses_dfX = pd.crosstab(NYSspillsRAW_df5['Source'], NYSspillsRAW_df5['Contributing Factor'],margins=True)

sourcesVScauses_df = sourcesVScauses_dfX.drop('Missing Code in Old Data - Must be fixed',1).copy()

sourcesVScauses_df
# take out the 'All' row/column

sourcesVScauses_df2 = sourcesVScauses_df.iloc[0:15,0:13].copy()
# ...and make a heat map

plt.title('Sources of leaks and their causes', fontsize=14)

sns.heatmap(sourcesVScauses_df2, cmap='YlGnBu')
# Let's explore the nature of the hazardous liquid spills

# where do the most liquid spills come from (source)

liquidSpills_df = NYSspillsRAW_df5[NYSspillsRAW_df5['Units'] == 'Gallons'].copy()

hazLiquidSpills_df = liquidSpills_df[

    (liquidSpills_df['Material Family']=='Petroleum') | (liquidSpills_df['Material Family']=='Hazardous Material')]

volSpilledVSsources_dfX = hazLiquidSpills_df.groupby(['Source','Material Family']).sum()

volSpilledVSsources_dfX2 = volSpilledVSsources_dfX.sort_values('Quantity', ascending=False).copy()
volSpilledVSsources_dfX2
hazLiquidSpills_df.head()
# sources of petroleum, hazardous material

hazLiqSources_df = pd.crosstab(hazLiquidSpills_df['Source'], hazLiquidSpills_df['Material Family'],margins=True)
hazLiqSources_df2 = hazLiqSources_df.iloc[0:15,0:2] # drops the 'All' row/column

hazLiqSources_df2
ax = hazLiqSources_df2.plot(kind='bar', title ="Hazardous liquid spill sources", figsize=(8, 6), legend=True, fontsize=12)

ax.set_xlabel("Source", fontsize=12)

ax.set_ylabel("Frequency", fontsize=12)

plt.show()
# what are the commonest liquids spilled?

commonestliquidsSpilled_df = pd.crosstab(hazLiquidSpills_df['Material Name'],hazLiquidSpills_df['Quantity'].sum(),margins=True)

commonestliquidSpills_df2 = hazLiquidSpills_df.groupby(['Material Name']).agg({'Quantity':sum})

commonestliquidSpills_df3 = commonestliquidSpills_df2.sort_values('Quantity', ascending=False).copy()

commonestliquidSpills_df3.head(10)
# lets visualise the first 15

commonestliquidSpills_df4 = commonestliquidSpills_df3.iloc[0:14].copy()
# https://stackoverflow.com/questions/25973581/how-do-i-format-axis-number-format-to-thousands-with-a-comma-in-matplotlib

ax = commonestliquidSpills_df4['Quantity'].plot(kind='bar', title ="Large spills", figsize=(8, 6), legend=False, fontsize=12)

ax.set_xlabel("Material", fontsize=12)



y_labels = ax.get_yticks()

ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))



ax.set_ylabel("Vol (gal)", fontsize=12)

plt.show()
# As a former incident investigator in the oil industry, I want to know causes and sources of hydrocarbon spills

matFamVScauses_df = pd.crosstab(hazLiquidSpills_df['Contributing Factor'], hazLiquidSpills_df['Material Family'],margins=True)

matFamVSsources_df = pd.crosstab(hazLiquidSpills_df['Source'], hazLiquidSpills_df['Material Family'],margins=True)
# drop the 'All' row/column

matFamVScauses_df2 = matFamVScauses_df.iloc[0:14,0:2].copy()

matFamVSsources_df2 = matFamVSsources_df.iloc[0:15,0:2].copy()
# the causes

matFamVScauses_df3 = matFamVScauses_df2.drop('Missing Code in Old Data - Must be fixed',0).copy()

matFamVScauses_df4 = matFamVScauses_df3.sort_values('Petroleum', ascending=False).copy()

matFamVScauses_df5 = matFamVScauses_df4['Petroleum'].copy()
# the sources

matFamVSsources_df3 = matFamVSsources_df2.sort_values('Petroleum', ascending=False).copy()

matFamVSsources_df4 = matFamVSsources_df3['Petroleum'].copy()
# visually

f, axes = plt.subplots(1,2)

f.set_figheight(8)

f.set_figwidth(15)

plt.subplots_adjust(wspace=.7)

axes[0].set_title("Hydrocarbon spill sources")

sns.countplot(y="Contributing Factor", data=NYSspillsRAW_df5, order=matFamVScauses_df5.index, palette=p, ax=axes[1])

axes[1].set_title("Hydrocarbon spill causes")

sns.countplot(y="Source", data=NYSspillsRAW_df5, order=matFamVSsources_df4.index, palette=p3, ax=axes[0])
# Also of interest (from a public health aspect) are sources/causes of sewage spills

sewageVSsources_dfX = liquidSpills_df.loc[liquidSpills_df['Material Name'] == 'raw sewage']

sewageVSsources_df1 = sewageVSsources_dfX.groupby(['Source']).agg({'Quantity':sum}).sort_values('Quantity', ascending=False).copy()
sewageVSsources_df1
# commonest sewage spill sources

sewageVSsources_df2 = sewageVSsources_dfX.groupby(['Source']).count().sort_values('Quantity', ascending=False).copy()

sewageVSsources_df3 = sewageVSsources_df2['Quantity'].copy()

sewageVSsources_df3
# prepare for plotting

sourceNames1 = list(sewageVSsources_df1.index.values)

sourceNames2 = list(sewageVSsources_df3.index.values)

values1 = list(sewageVSsources_df1.Quantity.values)

values2 = list(sewageVSsources_df3.values)
# https://stackoverflow.com/questions/11264521/date-ticks-and-rotation-in-matplotlib

# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html

# https://stackoverflow.com/questions/7066121/how-to-set-a-single-main-title-above-all-the-subplots-with-pyplot

fig, axs = plt.subplots(1, 2, figsize=(14, 10))



plt.setp( axs[0].xaxis.get_majorticklabels(), rotation=90 )

axs[0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

axs[0].bar(sourceNames1, values1)

axs[0].set_xlabel("Source", fontsize=12)

axs[0].set_ylabel("Volume (gals)", fontsize=12)



plt.setp( axs[1].xaxis.get_majorticklabels(), rotation=90 )

axs[1].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

axs[1].bar(sourceNames2, values2)

axs[1].set_xlabel("Source", fontsize=12)

axs[1].set_ylabel("Count", fontsize=12)



fig.tight_layout()

fig.subplots_adjust(top=0.90)



fig.suptitle('Raw sewage release sources in NY', fontsize=20)
# where did the big spills come from? Let's look at the two things that get spilled the most: sewage and kerosene

# big spills - sewage

bigSpills_df1 = liquidSpills_df[['Source','Material Name','Quantity']].copy()

# isolate sewage spills from the rest

bigSpills_df2 = bigSpills_df1[bigSpills_df1['Material Name']=='raw sewage'].copy()

# list sewage spills in descending order

# https://stackoverflow.com/questions/37787698/how-to-sort-pandas-dataframe-from-one-column

bigSpills_df4 = bigSpills_df2.sort_values('Quantity', ascending=False).copy()

bigSpills_df4.head(10)
# aggregate and sort by source

sewageSpilled_df = bigSpills_df4.groupby(['Source']).agg({'Quantity':sum})

sewageSpilled_df1 = sewageSpilled_df.sort_values('Quantity', ascending=False).copy()

sewageSpilled_df1.head(10)
# kerosene

bigSpills_df3 = bigSpills_df1[bigSpills_df1['Material Name']=='kerosene'].copy()

bigSpills_df5 = bigSpills_df3.sort_values('Quantity', ascending=False).copy()

keroseneSpilled_df = bigSpills_df5.groupby(['Source']).agg({'Quantity':sum})

keroseneSpilled_df1 = keroseneSpilled_df.sort_values('Quantity', ascending=False).copy()

keroseneSpilled_df1.head(10)
# prepare for plotting

sourceNames3 = list(sewageSpilled_df1.index[0:9].values)

sourceNames4 = list(keroseneSpilled_df1.index[0:9].values)

values3 = list(sewageSpilled_df1.Quantity[0:9].values)

values4 = list(keroseneSpilled_df1.Quantity[0:9].values)
fig, axs = plt.subplots(1, 2, figsize=(14, 10))



plt.setp( axs[0].xaxis.get_majorticklabels(), rotation=90 )

axs[0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))



axs[0].bar(sourceNames3, values3)

axs[0].set_title("Sewage")

axs[0].set_xlabel("Source", fontsize=12)

axs[0].set_ylabel("Volume (gals)", fontsize=12)



plt.setp( axs[1].xaxis.get_majorticklabels(), rotation=90 )

axs[1].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))



axs[1].bar(sourceNames4, values4)

axs[1].set_title("Kerosene")

axs[1].set_xlabel("Source", fontsize=12)

axs[1].set_ylabel("Volume (gals)", fontsize=12)



fig.tight_layout()

fig.subplots_adjust(top=0.90)



fig.suptitle('Major spill sources in NY', fontsize=20)
# when do most industrial spills occur?

spillTimes_df = NYSspillsRAW_df5[['Street 1', 'Locality', 'County', 'Spill Date', 'Contributing Factor',

                                'Source', 'Material Name', 'Material Family', 'Quantity', 'Units']]

spillTimes_df["Year"] = spillTimes_df['Spill Date'].dt.year

spillTimes_df["Day"] = spillTimes_df['Spill Date'].dt.day

spillTimes_df["Day_of_week"] = spillTimes_df['Spill Date'].dt.dayofweek

spillTimes_df["Month"] = spillTimes_df['Spill Date'].dt.month

spillTimes_df.head()
sewageInstEd_df = spillTimes_df[spillTimes_df['Source'] == 'Institutional, Educational, Gov., Other']

sewageMajFac_df = spillTimes_df[spillTimes_df['Source'] == 'Major Facility (MOSF) > 400,000 gal']
f, axes = plt.subplots(1,2)

f.set_figheight(6)

f.set_figwidth(13)

f.suptitle("Annual sewage spills - main sources", fontsize=20)

plt.subplots_adjust(wspace=.5)

axes[0].set_title("Institutional sources")

sns.countplot(x="Month", data=sewageInstEd_df, palette=p, ax=axes[0])

axes[1].set_title("Industrial sources")

sns.countplot(x="Month", data=sewageMajFac_df, palette=p, ax=axes[1])
# what regulated chemicals e.g. OSHA highly hazardoussubstances (HHS) were spilled and from where (geopy location + folium)

# https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.119AppA

hazardousChems_df = pd.read_csv('../input/spilldata-supporting-files/OSHA_highly_hazardous_chemicals.csv')
# make into a list we can compare to NY spill data

hazardousChemsX = list(hazardousChems_df['CHEMICAL name'])

hazardousChems = [chem.lower() for chem in hazardousChemsX]
# find the hazardous NY spill substances

chemsSpilledX = list(NYSspillsRAW_df5['Material Name'])

chemsSpilled = [chem.lower() for chem in chemsSpilledX]

hazChemsSpilled = [chem for chem in chemsSpilled if chem in hazardousChems]
# make a list of the unique hazardous substances spilled

hazChemsSpilledUnique = []

for chem in hazChemsSpilled:

    if chem not in hazChemsSpilledUnique:

        hazChemsSpilledUnique.append(chem)
len(hazChemsSpilledUnique)
# isolate the highly hazardous substances (HHS) released

# https://cmsdk.com/python/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas.html

hazChemSpilled_dfX = spillTimes_df.loc[spillTimes_df['Material Name'].isin(hazChemsSpilledUnique)].dropna()
# and drop the zero entries

hazChemSpilled_df = hazChemSpilled_dfX[hazChemSpilled_dfX.Quantity > 0]

hazChemSpilled_df.head(10)
# let's explore where these all are, using geopy...

street = list(hazChemSpilled_df['Street 1'])

locality = list(hazChemSpilled_df['Locality'])

county = list(hazChemSpilled_df['County'])



# make a list of locations for which to obtain coordinates from geopy

locForGeopy = []

for i in range(len(street)):

    inter = street[i] + ', ' + locality[i] + ', ' + county[i] + ', ' + 'New York'

    locForGeopy.append(inter)
hazChemSpilled_dfX = pd.read_csv('../input/spilldata-supporting-files/StateOfNY_HazchemSpills_coords.csv')
hazChemSpilled_dfX.head()
# find the top 10 most frequently occurring locations

topTenHazSpilllLox_df = hazChemSpilled_dfX['Street 1'].value_counts()[:10].sort_values(ascending=False)

topTenHazSpilllLox_df
# find the top 10 most frequently spilled HHS's

topTenHazChemsSpilled_df = hazChemSpilled_dfX['Material Name'].value_counts()[:10].sort_values(ascending=False)

topTenHazChemsSpilled_df
# lets map these

NYhazSpillMap_v0 = folium.Map(location=[42.996466, -74.473269], zoom_start=8)



# folium marker colors:

markerColor=['lightgreen','green', 'darkgreen', 'lightblue', 'blue', 'darkblue',

             'cadetblue', 'purple', 'darkpurple', 'orange','beige',

             'lightred', 'pink', 'gray', 'black', 'lightgray', 'red', 'darkred', 'white']



# add towns, incomes, coords



chemNames = list(hazChemSpilled_dfX['Material Name'].values)

latChem = list(hazChemSpilled_dfX.Lat.values)

lonChem = list(hazChemSpilled_dfX.Lon.values)



for i in range(0,len(chemNames)):

    folium.Circle(

       location = [latChem[i], lonChem[i]],

       popup = chemNames[i],

       radius = 50,

       color = markerColor[7],

       fill = True,

       fill_color = markerColor[7]

    ).add_to(NYhazSpillMap_v0)

    

# Save it as html

# NYhazSpillMap_v0.save('../input/spilldata-supporting-files/NY_HazChemSpills.html')
# lastly, plot petroleum product spills vs time: needs consistent units (gallons), non-zero entries, dates in ascending order

petroleumSpills_dfX = liquidSpills_df[liquidSpills_df.Quantity > 0].copy()

petroleumSpills_df = petroleumSpills_dfX.sort_index().copy()
# set a date range

# https://chrisalbon.com/machine_learning/preprocessing_dates_and_times/select_date_and_time_ranges/

petroleumSpills_df2 = petroleumSpills_df[petroleumSpills_df['Spill Date'] > '1984-01-01']

petroleumSpills_ts = pd.Series(petroleumSpills_df2.Quantity)
plt.figure(figsize = (15,6))



plt.title('Overall trend of petroleum-type spills in NY', fontsize=16)

plt.ylabel('Spill volume (gals)')

plt.xlabel('Year')

plt.plot(petroleumSpills_ts, label='Petroleum-type')

plt.grid(True)

plt.legend()