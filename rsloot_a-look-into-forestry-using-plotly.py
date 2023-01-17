# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df_country = pd.read_csv('../input/Country.csv')

df_indicators = pd.read_csv('../input/Indicators.csv')

df_indicators.describe()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
dat = df_indicators.copy()

dat.shape
dat.head()
dat.dtypes
# helper function

def filter_data(df, feat, filter0, reset_index=False):

    d = df[df[feat]==filter0]

    if reset_index:

        d.reset_index(inplace=True, drop=True)

    return d
forest_area = filter_data(dat, 'IndicatorCode', 'AG.LND.FRST.K2', reset_index=True)

forest_area.tail(15)
us = filter_data(forest_area, 'CountryName', 'United States', True)

us
plt.figure(figsize=(5,5))

plt.scatter(us.Year, us.Value) 

plt.ylim(us.Value.min()-10000, us.Value.max()+10000)

plt.title('U.S. Forest Area since 1990 in sq. km')

plt.xlabel('Year')

plt.ylabel('sq. km')

plt.show()
brazil = filter_data(forest_area, 'CountryName', 'Brazil', True)

plt.figure(figsize=(5,5))

plt.scatter(brazil.Year, brazil.Value)

plt.title('Brazil Forest Area since 1990 in sq. km')

plt.xlabel('Year')

plt.ylabel('sq. km')

plt.show()
# import plotly.tools

# import plotly.plotly as py

# import plotly.graph_objs as go

# plotly.tools.set_credentials_file(#username=USER_NAME, api_key=API_KEY)

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
forest_1990 = filter_data(forest_area, 'Year', 1990)

forest_2012 = filter_data(forest_area, 'Year', 2012)
print('1990 Countries: ', forest_1990.shape[0])

print('2012 Countries: ', forest_2012.shape[0])
for country in forest_1990.CountryName:

    if country not in forest_1990.CountryName:

        print(country)

        break
forest_1990 = forest_1990[forest_1990.CountryName != 'Arab World']

print('1990 Countries: ', forest_1990.shape[0])
forest_merged = pd.merge(forest_1990, forest_2012, how='left', on=['CountryName', 'CountryCode', 'IndicatorName', 'IndicatorCode'],

                     suffixes=('1990','2012'))

forest_merged.head()
forest_merged['DifferenceValue'] = forest_merged['Value2012'] - forest_merged['Value1990']

forest_merged.head()
## Bunch of the first rows are aggregates, let's get rid of these for now

forest_merged[:50]
## first 31 are aggragates, will filter out when visualizing.

trace = dict(type='choropleth',

            locations = forest_merged['CountryCode'],

                      z = forest_merged['DifferenceValue'],

                      text = forest_merged['CountryName'],

                      colorscale = [[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], 

                                    [0.2222222222222222, 'rgb(244,109,67)'], [0.3333333333333333, 'rgb(253,174,97)'],

                                    [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'], 

                                    [0.6666666666666666, 'rgb(171,217,233)'], [0.7777777777777778, 'rgb(116,173,209)'], 

                                    [0.8888888888888888, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']],

                      autocolorscale = False,

                      reversescale = False,

                      marker = dict(line = dict (color = 'rgb(180,180,180)',width = 0.5) ),

                      colorbar = dict(autotick = True, title = 'Gain / Loss'),

                      zauto = False,#True,

                      zmin=forest_merged[32:]['DifferenceValue'].min()-1000,

                      zmax=forest_merged[32:]['DifferenceValue'].max()+1000)

data = [trace]



layout = dict(

    title = 'Forest Area Gain/Loss Since 1960',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(type = 'Mercator')

    )

)

[0.034482758620689655, 'rgb(165,42,42)'],

# py.iplot(colorscale_plot(colorscale=chlorophyll, title='Chlorophyll'))



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False, filename='forest gain-world-map' )
forest_merge2 = forest_merged[(forest_merged.CountryName != 'Brazil') & (forest_merged.CountryName != 'China')][32:].reset_index(drop=True)

# this puts holes in the map, let's change china and brazil to NULLs

forest_merge2 = forest_merged.loc[32:,:].reset_index(drop=True)

forest_merge2.loc[forest_merge2.CountryName == 'Brazil', 'DifferenceValue'] = np.NaN #Brazil

forest_merge2.loc[forest_merge2.CountryName == 'China', 'DifferenceValue'] = np.NaN #China
## COLOR Scale; can be ignored

## converting hex to rbg int('b4', 16) ==> 180

import re

colors = '#a52a2a #a23029 #9f3628 #9c3b28 #994027 #954426 #934725 #8f4b24 #8b4f23 #885222 #845521 #815820 #7d5b1f #7b5d1e #75601c #71631b #6b6619 #696819 #636a17 #5f6d16 #5b6f14 #567013 #507311 #4a740f #44760d #3c790b #337a08 #287d05 #187e02 #008000'.split()

colors_rgb = [0]*len(colors)

for idx, h in enumerate(colors):

    rgb = 'rgb('

    for i in re.findall('..', h.lstrip('#')):

        rgb += str(int(i,16))+','

    colors_rgb[idx] = rgb[:-1]+')'

# colors_rgb = ['rgb(str(for h in colors for i in re.findall('..', h.lstrip('#'))]

# re.findall('..',hex)

c_scale = [[i/(len(colors_rgb)-1),colors_rgb[i-1]] for i in range(1,len(colors_rgb)+1)]



c_scale.insert(0,[0.0,0])

# c_scale[:-1]
c_scale = [[i/(len(colors_rgb)-1),colors_rgb[i-1]] for i in range(1,len(colors_rgb))]

c_scale.insert(0,[0,0])



trace = dict(type='choropleth',

            locations = forest_merged['CountryCode'],

                      z = forest_merged['DifferenceValue'],

                      text = forest_merged['CountryName'],

                      colorscale = 'Greens',

                      autocolorscale = False,

                      reversescale = True,

                      marker = dict(line = dict (color = 'rgb(180,180,180)',width = 0.5) ),

                      colorbar = dict(autotick = True, title = 'Gain / Loss'),

                      zauto = False,#True,

                      zmin=forest_merged[32:].loc[((forest_merged.CountryName!='Indonesia')&(forest_merged.CountryName!='Brazil')),\

                                          'DifferenceValue'].min(),

                      zmax=forest_merged[32:].loc[((forest_merged.CountryName!='United States')&(forest_merged.CountryName!='China')),\

                                          'DifferenceValue'].max()+25000)

data = [trace]



layout = dict(

    title = 'Forest Area Gain/Loss Since 1960',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(type = 'Mercator')

    )

)



# py.iplot(colorscale_plot(colorscale=chlorophyll, title='Chlorophyll'))



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False, filename='forest gain-world-map without')# china-or-brazil' )
## Countries with largest gains and losses

biggest_loser = forest_merged[forest_merged.DifferenceValue == forest_merged[32:]['DifferenceValue'].min()].get(['CountryName', 'DifferenceValue']).values

biggest_gainer = forest_merged[forest_merged.DifferenceValue == forest_merged[32:]['DifferenceValue'].max()].get(['CountryName', 'DifferenceValue']).values

print('{} lost {:.2f} square KM of forest\n{} gained {:.2f} sq. KM of forest since 1990 (to 2012)'.format(biggest_loser[0][0],

                                                                                                  abs(biggest_loser[0][1]),

                                                                                                  biggest_gainer[0][0],

                                                                                                  biggest_gainer[0][1]))