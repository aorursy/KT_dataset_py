import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import folium

from PIL import Image

from wordcloud import WordCloud, STOPWORDS



# Read excel

df= pd.read_excel('../input/immigration-to-canada-ibm-dataset/Canada.xlsx',

                        sheet_name="Canada by Citizenship",

                        skiprows=range(20),

                        skipfooter=2)
df.info()
df.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

df.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
df['Total'] = df.sum(axis=1)



#Years that we will be using in this notebook

years=list(range(1980,2014))



#Set the country name as index.

df.set_index('Country', inplace=True)



df.head()
df_tot = pd.DataFrame(df[years].sum())



#reset index

df_tot.reset_index(inplace=True)



#rename columns

df_tot.columns = ['year', 'total']



df_tot.head()
# generate scatter plot

ax = df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')



# add title and label to axes

plt.title('Total immigration to Canada from 1980 - 2013',size=18)

plt.xlabel('Year', size=12)

plt.ylabel('Number of Immigrants', size=12)



# plot line of best fit

x = df_tot['year']

fit = np.polyfit(x, df_tot['total'], deg=1)

plt.plot(x, fit[0] * x + fit[1], color='red')

plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))



plt.show()
df_france = df.loc[['France'],years].transpose()

df_france.head()
df_france.plot(kind='bar', figsize=(15, 8), color='#03b6fc')



plt.xlabel('Year', size=12)

plt.ylabel('Number of immigrants', size=12)

plt.title('French immigrants to Canada from 1980 to 2013', size=18)



# Annotate arrow

plt.annotate('',                   # s: str. will leave it blank for no text

             xy=(32, 6500),        # head of the arrow

             xytext=(0, 2400),     # base of the arrow

             xycoords='data',      # coordinate system of the object being annotated 

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', lw=2)

            )



plt.show()
df.sort_values('Total', ascending=False, inplace=True)

df_top5 = df[years].head(5).transpose()

df_top5.head(10)
df_top5.plot(kind='area', 

             alpha=0.25, # 0-1, default value a= 0.5

             figsize=(20, 10),

            )



plt.title('Immigration Trend of Top 5 Countries', size=18)

plt.ylabel('Number of Immigrants', size=12)

plt.xlabel('Years', size=12)



plt.show()
# Extract data from India and China (Top2)

df_CI = df.loc[['India','China'], years].transpose()

df_CI.reset_index(inplace=True)

df_CI.rename(columns={'index':'Year'},inplace=True)

df_CI.head()
#Normalization of the data

norm_china = (df_CI['China'] - df_CI['China'].min()) / (df_CI['China'].max() - df_CI['China'].min())

norm_india = (df_CI['India'] - df_CI['India'].min()) / (df_CI['India'].max() - df_CI['India'].min())
# China

ax0 = df_CI.plot(kind='scatter',

                    x='Year',

                    y='China',

                    figsize=(14, 8),

                    alpha=0.5,                  # transparency

                    color='orange',

                    s=norm_china * 2000 + 10,  # pass in weights 

                    xlim=(1975, 2015)

                   )



# India

ax1 = df_CI.plot(kind='scatter',

                    x='Year',

                    y='India',

                    alpha=0.5,

                    color="purple",

                    s=norm_india * 2000 + 10,

                    ax = ax0

                   )



ax0.set_ylabel('Number of Immigrants', fontsize=12)

ax0.set_title('Immigration from China and India from 1980 - 2013', fontsize=18)

ax0.legend(['China', 'India'], loc='upper left', fontsize=12)
df_top10 = df.loc[:,years].head(10)

df_top10.head(5)
years_80s = list(range(1980, 1990)) 

years_90s = list(range(1990, 2000)) 

years_00s = list(range(2000, 2010)) 



# slice the original dataframe df_can to create a series for each decade

df_80s = df_top10.loc[:, years_80s].sum(axis=1) 

df_90s = df_top10.loc[:, years_90s].sum(axis=1) 

df_00s = df_top10.loc[:, years_00s].sum(axis=1)



# merge the three series into a new data frame

new_df = pd.DataFrame({'1980s': df_80s, '1990s': df_90s, '2000s':df_00s}) 

# display dataframe

new_df.head()
new_df.describe()
color_list=['#c8d5a0','#922e74','#b877af','#e2b5de','#d0c9ca','#7B4B94','#7D82B8','#B7E3CC','#C4FFB2','#D6F7A3']



new_df.transpose().plot(kind='barh', figsize=(15, 10), color=color_list)



plt.xlabel('Year', size=12)

plt.ylabel('Number of immigrants', size=12)

plt.title('Immigration from top 15 countries for decades 80s, 90s and 2000s', size=18)
fig = plt.figure(1, figsize=(12,8))



box = plt.boxplot(new_df.transpose(), patch_artist=True)



# fill with colors

colors = ['#c8d5a0','#922e74','#b877af']



for patch, color in zip(box['boxes'], colors):

    patch.set_facecolor(color)     

          

plt.title('Immigration from top 15 countries for decades 80s, 90s and 2000s', size=18)

plt.ylabel('Number of Immigrants', size=12)

plt.xlabel('Decades', size=12)

plt.xticks([1, 2, 3], ['1980s','1990s','2000s'])

plt.show()
world_geo = '../input/world-countries/world-countries.json'
# reset the index to select the countries as column

df.reset_index(inplace=True)



world_map = folium.Map(location=[0,0],zoom_start=2)



# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013

world_map.choropleth(

    geo_data=world_geo,

    data=df,

    columns=['Country', 'Total'],

    key_on='feature.properties.name',

    fill_color='YlOrRd',

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Immigration to Canada'

)



# display map

world_map
mapleLeaf_mask = np.array(Image.open('../input/resource/maple_leaf.jpeg'))

fig = plt.figure()

fig.set_figwidth(7) # set width

fig.set_figheight(9) # set height



plt.imshow(mapleLeaf_mask, cmap=plt.cm.gray, interpolation='bilinear')

plt.axis('off')

plt.show()
total_immigration = df['Total'].sum()



df.set_index('Country', inplace=True)



max_words = 1000

word_string = ''

for country in df.index.values:

    # check if country's name is a single-word name

    if len(country.split(' ')) == 1:

        repeat_num_times = int(df.loc[country, 'Total']/float(total_immigration)*max_words)

        word_string = word_string + ((country + ' ') * repeat_num_times)
# instantiate a word cloud object

canada_wc = WordCloud(background_color='white', mask=mapleLeaf_mask)



# generate the word cloud

canada_wc.generate(word_string)



# display the word cloud

fig = plt.figure()

fig.set_figwidth(14) # set width

fig.set_figheight(18) # set height



plt.imshow(canada_wc, interpolation='bilinear')

plt.axis('off')

plt.show()