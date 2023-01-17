

import matplotlib.pyplot as plt 

import matplotlib.ticker as ticker

from matplotlib import animation

import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None

import seaborn as sns



from geopy.geocoders import Nominatim # # convert an address into latitude and longitude values

import folium

from mpl_toolkits.basemap import Basemap
terror_df = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')

terror_df.head(3)
print ('Dataframe Shape: ', terror_df.shape)
columns_select = ['eventid', 'iyear', 'country_txt', 'region_txt', 'city', 'latitude', 'longitude', 'success', 'attacktype1_txt', 

                  'targtype1_txt', 'weaptype1_txt', 'nkill', 'nwound', 'gname']



columns_names = ['Id', 'Year', 'Country', 'Region', 'City', 'Lat', 'Lon', 'Success', 'AttackType', 

                 'TargetType', 'WeaponType', 'Kill', 'Wounded',  'TerroristGroup']



terror_df_selected = terror_df[columns_select]



terror_df_selected.columns = columns_names



print ('Selected Dataframe Shape: ', terror_df_selected.shape)



terror_df_selected.head(3)
print ('Are there null values in the dataframe: ', terror_df_selected.isnull().values.any())

print ('NaNs in every column: \n', terror_df_selected.isna().sum())
f = plt.figure(figsize=(14, 7))



sns.set(font_scale=1.1)

year_count = sns.countplot(x='Year', data=terror_df_selected, palette='YlOrRd')

year_count.set_xticklabels(year_count.get_xticklabels(), rotation=70)

plt.ylabel('Count', fontsize=12)

plt.xlabel('Year', fontsize=12)

plt.title('Fig. 1: Number of Terrorist Attacks Year by Year', fontsize=12)
f = plt.figure(figsize=(13, 8))



sns.set(font_scale=0.9)

year_count = sns.countplot(x='Region', data=terror_df_selected,)

year_count.set_xticklabels(year_count.get_xticklabels(), rotation=70)

plt.xlabel('Region', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.title('Fig. 2: Number of Terrorist Attacks by Region', fontsize=12)
f = plt.figure(figsize=(13, 8))



sns.set(font_scale=0.9)

year_count = sns.countplot(x='AttackType', data=terror_df_selected,)

year_count.set_xticklabels(year_count.get_xticklabels(), rotation=70)

plt.xlabel('Methods of Attack', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.title('Fig. 3: Types of Terrorist Attack ', fontsize=12)
f = plt.figure(figsize=(14, 8))



sns.set(font_scale=0.9)

year_count = sns.countplot(y='TargetType', data=terror_df_selected,)

# year_count.set_xticklabels(year_count.get_xticklabels(), rotation=70)

year_count.set_xlabel('Count', fontsize=12)

year_count.set_ylabel('Target Type', fontsize=12)

plt.xlabel('Count', fontsize=12)

target_type_xticks  = [0, 1e3, 5e3, 1e4, 2e4, 3e4, 4e4]

target_type_xlabels = ['0', '1e3', '5e3', '1e4', '2e4', '3e4', '4e4']

plt.xticks(target_type_xticks, target_type_xlabels, fontsize=11, rotation=70)

plt.ylabel('Target Type', fontsize=12)

plt.title('Fig. 4: Types of Targets of Terrorist Attack ', fontsize=12)
fig= plt.figure(figsize=(13, 7))

terror_country = sns.barplot(x=terror_df_selected['Country'].value_counts()[0:15].index, y=terror_df_selected['Country'].value_counts()[0:15], palette='RdYlGn')

terror_country.set_xticklabels(terror_country.get_xticklabels(), rotation=70)

terror_country.set_xlabel('Country', fontsize=12)

terror_country.set_ylabel('Counts', fontsize=12)

plt.title('Fig. 5: Top 15 Countries: Most Attacks by Terrorist Groups', fontsize=12)

plt.show()
f = plt.figure(figsize=(14, 8))

sns.set(font_scale=0.9)



terror_df_selected['WeaponType'] = terror_df_selected['WeaponType'].replace(['Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)'], 'Vehicle')



ncount = len(terror_df_selected)



year_count = sns.countplot(x='WeaponType', data=terror_df_selected,)

year_count.set_xticklabels(year_count.get_xticklabels(), rotation=70)

plt.title('Fig. 6: Types of Terrorist Attack ', fontsize=12)



ax2=year_count.twinx()



ax2.yaxis.tick_left()

year_count.yaxis.tick_right()



# Also switch the labels over

year_count.yaxis.set_label_position('right')

ax2.yaxis.set_label_position('left')



ax2.set_ylabel('Frequency [%]', fontsize=11)



for p in year_count.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    year_count.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

            ha='center', va='bottom') # set the alignment of the text



# Use a LinearLocator to ensure the correct number of ticks

year_count.yaxis.set_major_locator(ticker.LinearLocator(11))



# Fix the frequency range to 0-100

ax2.set_ylim(0,100)

# year_count.set_ylim(0, int(ncount/1e3))

year_count.set_ylabel('Count', fontsize=11)



ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))



ax2.grid(None)
region_year = pd.crosstab(terror_df_selected.Year, terror_df_selected.Region)



region_year.tail(3)
color_list_reg_yr = ['palegreen', 'lime', 'green', 'Aqua', 'skyblue', 'darkred', 'darkgray', 'tan', 'orangered', 'plum', 'salmon', 'mistyrose']

region_year.plot(figsize=(14, 10), fontsize=12, color=color_list_reg_yr)

plt.xlabel('Year', fontsize=12)

plt.ylabel('Number of Attacks in a Year', fontsize=12)

plt.title('Fig. 7: Region Based Terrorist Attacks over Years', fontsize=12)

plt.show()
terror_df_selected['Year10'] = pd.cut(terror_df_selected['Year'], bins=[1969, 1980, 1990, 2000, 2010, 2020], 

               labels=['1970-1980', '1980-1990', '1990-2000', '2000-2010', '2010-2020'])



terror_df_selected.head(3)
region_year10 = pd.crosstab(terror_df_selected.Year10, terror_df_selected.Region,)

region_year10.head(6)
color_list_reg_yr = ['palegreen', 'lime', 'green', 'Aqua', 'skyblue', 'darkred', 'darkgray', 'tan', 'orangered', 'plum', 'salmon', 'mistyrose']



region_year10.plot(figsize=(14, 10), fontsize=12, color=color_list_reg_yr)

plt.xlabel('Year', fontsize=12)

plt.ylabel('Number of Attacks in Each Decade', fontsize=12)

plt.title('Fig. 8: Region Based Terrorist Attack over Decades', fontsize=12)

plt.show()
weapon_type_Region_year10 = pd.crosstab(terror_df_selected.Year10, terror_df_selected.WeaponType)

weapon_type_Region_year10.head(6)
color_list_weapon_yr = ['lime', 'palegreen', 'darkred', 'Aqua', 'orangered', 'salmon', 'tan', 'lavender', 'plum', 'darkgray', 'magenta', 'mistyrose']



weapon_type_Region_year10.plot(figsize=(14, 10), fontsize=12, color=color_list_weapon_yr)

plt.xlabel('Year', fontsize=12)

plt.ylabel('Types of Weapon Used in Each Decade', fontsize=12)

plt.title('Fig. 9: Weapons Used over Decades by Terrorists', fontsize=12)

plt.show()
### replace NaNs with 0s

terror_df_selected['Kill'] = terror_df_selected['Kill'].fillna(0).astype(int)

terror_df_selected['Wounded'] = terror_df_selected['Wounded'].fillna(0).astype(int)



### groupby dataframe y region and select south asia as region 



reg_groupby = terror_df_selected.groupby(['Region'])

reg_groupby_SA_df = reg_groupby.get_group('South Asia')



reg_groupby_SA_df.head(3)
### need to re-arrange the index 



reg_groupby_SA_df.index = range(len(reg_groupby_SA_df.index))



reg_groupby_SA_df.head(3)
print ('South asia dataframe shape: ', reg_groupby_SA_df.shape)
f = plt.figure(figsize=(14, 8))



sns.set(font_scale=0.9)



Country_count_SA = sns.countplot(y='Country', data=reg_groupby_SA_df, palette='RdGy')

# year_count.set_xticklabels(year_count.get_xticklabels(), rotation=70)

Country_count_SA.set_xlabel('Count', fontsize=12)

Country_count_SA.set_ylabel('South Asian Countries', fontsize=12)

plt.title('Fig. 10: South Asian Countries Affected in Terrorist Attack ', fontsize=12)
### check with value_counts



print ('number of attacks in respective countries: \n')

print (reg_groupby_SA_df['Country'].value_counts())
print ('There are {} unique number of terror groups in Souh East Asia'.format(len(reg_groupby_SA_df['TerroristGroup'].unique())) )



print ('\n')



reg_groupby_SA_df_top10TerrorG = reg_groupby_SA_df['TerroristGroup'].value_counts()[1:11].to_frame(name='Counts')



# drop 0 because that is 'unknown' group

# reg_groupby_SA_df_top10TerrorG.head(3)



reg_groupby_SA_df_top10TerrorG['Terror Group'] = reg_groupby_SA_df_top10TerrorG.index

reg_groupby_SA_df_top10TerrorG.reset_index(drop=True, inplace=True)

reg_groupby_SA_df_top10TerrorG.head(3)
f = plt.figure(figsize=(13, 8))

bar_SA_top10TG = sns.barplot(y='Terror Group', x='Counts', data=reg_groupby_SA_df_top10TerrorG)

# bar_SA_top10TG.set_xticklabels(bar_SA_top10TG.get_xticklabels(), rotation=70)

plt.xscale('linear')



xticks = [0, 500, 1e3, 1.5e3, 2e3, 3e3, 4e3, 5e3, 6e3,  7e3]

xlabels = ['0', '500', '1e3', '1.5e3', '2e3', '3e3',  '4e3',  '5e3',  '6e3',  '7e3']

plt.xticks(xticks, xlabels, fontsize=11)

plt.xlabel('Counts', fontsize=12)

plt.ylabel('Terror Group', fontsize=12)



plt.title('Fig. 11: Top 10 Terror Groups in South Asia', fontsize=12)
### create a new column combining killed and wounded 

reg_groupby_SA_df['Affected'] = reg_groupby_SA_df['Kill'] + reg_groupby_SA_df['Wounded']



### plot the mean number of people injured + killed over all the attacks in respective countries  

f = plt.figure(figsize=(10, 7))

reg_groupby_SA_df_grp_country_affected = reg_groupby_SA_df.groupby(['Country'])['Affected'].mean()

reg_groupby_SA_df_grp_country_affected.plot.bar(figsize=(12, 8), color=['orangered', 'red', 'palegreen', 'magenta', 'salmon', 'lime', 'gray', 'coral', 'darkred'])

plt.title('Fig. 12: Mean Number of People Killed + Wounded in Each Attack for Respective Countries', fontsize=12)

plt.xlabel('Country', fontsize=12)



plt.text(8, 10.3, 'Total Attacks \n {}'.format(reg_groupby_SA_df['Country'].value_counts()['Sri Lanka']), horizontalalignment='center', fontsize=11)

plt.text(7, 5, 'Total Attacks \n {}'.format(reg_groupby_SA_df['Country'].value_counts()['Pakistan']), horizontalalignment='center', fontsize=11)

plt.text(6, 3.8, 'Total Attacks \n {}'.format(reg_groupby_SA_df['Country'].value_counts()['Nepal']), horizontalalignment='center', fontsize=11)

plt.text(5, 0.8, 'Total Attacks \n {}'.format(reg_groupby_SA_df['Country'].value_counts()['Mauritius']), horizontalalignment='center', fontsize=11)

plt.text(4, 6.7, 'Total Attacks \n {}'.format(reg_groupby_SA_df['Country'].value_counts()['Maldives']), horizontalalignment='center', fontsize=11)

plt.text(3, 4.3, 'Total Attacks \n {}'.format(reg_groupby_SA_df['Country'].value_counts()['India']), horizontalalignment='center', fontsize=11)

plt.text(2, 2.6, 'Total Attacks \n {}'.format(reg_groupby_SA_df['Country'].value_counts()['Bhutan']), horizontalalignment='center', fontsize=11)

plt.text(1, 6.0, 'Total Attacks \n {}'.format(reg_groupby_SA_df['Country'].value_counts()['Bangladesh']), horizontalalignment='center', fontsize=11)

plt.text(0.05, 6.8, 'Total Attacks \n {}'.format(reg_groupby_SA_df['Country'].value_counts()['Afghanistan']), horizontalalignment='center', fontsize=11)



plt.show()
reg_groupby_SA_df_most_affected5 = reg_groupby_SA_df.sort_values('Affected', ascending=False).head(5)

reg_groupby_SA_df_most_affected5
fig = plt.figure(figsize=(10, 7))

sns.set(font_scale=1.3)

sns.barplot(y='Affected', x='Country', data=reg_groupby_SA_df_most_affected5, hue='Year')

plt.xlabel('Country', fontsize=12)

plt.ylabel('Killed + Wounded', fontsize=12)

plt.title('Fig. 13: Top 5: Fatalitites (Killed + Wounded) in a Single Attack (South Asia)', fontsize=12)

plt.show()
reg_groupby_SA_df_most_affected5_grp_country = reg_groupby_SA_df_most_affected5.groupby(['Country'])['Affected'].sum()

print ('top 5 attacks and people affected in South asia region: \n', reg_groupby_SA_df_most_affected5_grp_country)

print ('Among top 5 attacks Afghanistan suffered 2 heavy blows')
### we drop the rows where lat and lon are NaN (495 of them)

reg_groupby_SA_df.dropna(subset=['Lat','Lon'],inplace=True)

print ('new dataframe shape: ', reg_groupby_SA_df.shape)
### try to plot folium map 

address = 'Kabul'



geolocator = Nominatim(user_agent="Kabul_explorer")

location = geolocator.geocode(address)

Kabul_latitude = location.latitude

Kabul_longitude = location.longitude

print('The geograpical coordinates of Kabul are {}, {}.'.format(Kabul_latitude, Kabul_longitude))



reg_groupby_SA_df_most_affected100 = reg_groupby_SA_df.sort_values('Affected', ascending=False).head(100)



AroundKabul = folium.Map(location=[Kabul_latitude, Kabul_longitude], zoom_start=4.2, tiles='Stamen Terrain')



for lat, lon, label, fat in zip(reg_groupby_SA_df_most_affected100['Lat'], reg_groupby_SA_df_most_affected100['Lon'], 

                                reg_groupby_SA_df_most_affected100['City'], reg_groupby_SA_df_most_affected100['Affected']):

    label = folium.Popup('{}\n Affected: {}'.format(label, fat), parse_html=True)

    folium.CircleMarker([lat, lon], radius=fat*0.02, popup=label, 

                      color='magenta', fill=True, fill_color='#3186cc', 

                      fill_opacity=0.7).add_to(AroundKabul)



loc = 'Based on Number of Affected (Killed + Wounded) People: Top 100 Attacks in South Asia'

title_html = '''

             <h3 align="center" style="font-size:16px"><b>{}</b></h3>

             '''.format(loc)   



AroundKabul.get_root().html.add_child(folium.Element(title_html))



AroundKabul  
## boundary of south asia (lat and lon coordinates)



llon = 58.8

ulon = 98.9

llat = 5.04

ulat = 41.06



years=list(reg_groupby_SA_df.Year.unique())



fig = plt.figure(figsize=(14, 7), dpi=150)



def year_attack(Year):

    plt.clf() # clear the previous plot before new year info is plotted 

    plt.title('Terrorism In South Asia '+'\n'+'Year:' +str(Year))

    my_map = Basemap(projection='merc',resolution='l', llcrnrlon=llon, llcrnrlat=llat-1, urcrnrlon=ulon, urcrnrlat=ulat+3, area_thresh=100, ) # low resolution

    lat_gif=list(reg_groupby_SA_df[reg_groupby_SA_df['Year']==Year].Lat)

    long_gif=list(reg_groupby_SA_df[reg_groupby_SA_df['Year']==Year].Lon)

    x_gif,y_gif=my_map(long_gif, lat_gif)



    my_map.scatter(x_gif, y_gif,s=[fat*1.10 for fat in reg_groupby_SA_df[reg_groupby_SA_df['Year']==Year].Affected], color = 'OrangeRed')



    my_map.drawcoastlines()



    my_map.drawcountries()

          

    my_map.shadedrelief()





year_ani_SA = animation.FuncAnimation(fig, year_attack, years, interval=1500)





year_ani_SA.save('year_attack_ani.gif', writer='imagemagick', fps=1, dpi=150)

plt.clf()
from IPython.display import Image

Image("../working/year_attack_ani.gif")
import io

import base64

from IPython.display import HTML



filename = 'year_attack_ani.gif'



video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))

country_selected = ['Pakistan', 'Afghanistan']



reg_groupby_SA_df_sel_country = reg_groupby_SA_df[reg_groupby_SA_df['Country'].isin(country_selected)]

print ('new selected dataframe: ', reg_groupby_SA_df_sel_country.shape)
f = plt.figure(figsize=(14, 7))



sns.set(font_scale=1.1)

year_count = sns.countplot(x='Year', data=reg_groupby_SA_df_sel_country, hue='Country', palette="Spectral")

year_count.set_xticklabels(year_count.get_xticklabels(), rotation=60)

plt.title('Number of Terrorist Attack Year by Year (Afg & Pak)', fontsize=12)