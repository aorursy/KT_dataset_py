import pandas as pd

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.set_option('display.float_format', lambda x: '%.6f' % x)

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

plt.style.use('ggplot')

import geopandas as gpd

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import sys

import warnings

if not sys.warnoptions: warnings.simplefilter("ignore")

from sklearn.linear_model import LinearRegression

from io import StringIO



df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv').reset_index()

df.columns = [x.replace('($)', '').rstrip(' ').lstrip(' ') for x in df.columns]

df.columns = [x.replace(' ', '_').replace('/', '_').replace('-', '_').lower() for x in df.columns]

df['gdp_for_year'] = df['gdp_for_year'].str.replace(',', '').astype(float) 

df.head()
df = df[df.year<2000]



# https://en.wikipedia.org/wiki/Importance_of_religion_by_country

# https://en.wikipedia.org/wiki/Demographics_of_atheism

# https://en.wikipedia.org/wiki/Jewish_population_by_country

# https://en.wikipedia.org/wiki/Protestantism_by_country

# https://en.wikipedia.org/wiki/Catholic_Church_by_country

# https://en.wikipedia.org/wiki/Islam_by_country

    

def religion_data():



    csv= """,country,atheism_pct,judaism_pct,protestantism_pct,catholic_pct,islam_pct

    0,Albania,,,0.0023,0.1,0.588

    1,Austria,0.56,0.0105,0.034,0.569,0.08

    2,Belarus,,0.0109,0.05,0.071,0.01

    3,Belgium,0.63,0.0263,0.013500000000000002,0.58,0.076

    4,Bulgaria,0.64,0.0028000000000000004,0.01,0.005,0.134

    5,Croatia,0.31000000000000005,0.004,0.02,0.863,0.015

    6,Czech Republic,0.84,0.0037,0.011000000000000001,0.102,0.002

    7,Denmark,0.72,0.011200000000000002,0.82,0.006999999999999999,0.054000000000000006

    8,Estonia,0.8200000000000001,0.0154,0.11,0.003,0.001

    9,Finland,0.6699999999999999,0.0024,0.7,0.002,0.027000000000000003

    10,France,0.73,0.07150000000000001,0.02,0.51,0.08800000000000001

    11,Germany,0.56,0.0144,0.275,0.282,0.06

    12,Greece,0.20999999999999996,0.0037,0.0028000000000000004,0.004,0.057

    13,Hungary,0.55,0.048600000000000004,0.14,0.37200000000000005,0.006

    14,Ireland,0.30000000000000004,0.0034999999999999996,0.042,0.7829999999999999,0.013999999999999999

    15,Italy,0.26,0.0044,0.013000000000000001,0.83,0.048

    16,Latvia,0.62,0.025,0.35,0.191,0.002

    17,Lithuania,0.53,0.009300000000000001,0.01,0.772,0.001

    18,Netherlands,0.72,0.0177,0.11,0.233,0.071

    19,Norway,0.78,0.0025,0.737,0.024,0.057

    20,Poland,0.20999999999999996,0.0008,0.0034000000000000002,0.858,0.0002

    21,Portugal,0.30000000000000004,0.0006,0.033,0.81,0.004

    22,Romania,0.07999999999999996,0.004699999999999999,0.06,0.047,0.01

    23,Russian Federation,,0.0124,0.0029,0.005,0.17

    24,Serbia,,0.002,0.012,0.061,0.031

    25,Slovakia,0.37,0.0048,0.08900000000000001,0.62,0.002

    26,Slovenia,0.6799999999999999,0.0005,0.008,0.732,0.036000000000000004

    27,Spain,0.41000000000000003,0.0025,0.037000000000000005,0.66,0.026000000000000002

    28,Sweden,0.8200000000000001,0.015300000000000001,0.6,0.018000000000000002,0.081

    29,Switzerland,0.56,0.0227,0.27,0.359,0.052000000000000005

    30,Ukraine,,0.0131,0.023,0.055999999999999994,0.025

    31,United Kingdom,0.63,0.0444,0.16,0.09,0.063"""



    csv = StringIO(csv)

    df = pd.read_csv(csv, index_col=None).iloc[:, 1:]

    return df



dfrel = religion_data()

dfrel.head()


def geographical_data():

    fp = '../input/world-shapefile/world_shapefile.shp'

    df = gpd.read_file(fp) #dfwm for df world map



    #renaming some countries from world map dataset to fit the original dataset

    to_replace = {'Russia':'Russian Federation'}

    df['NAME'] = df['NAME'].replace(to_replace)

    df = df.rename(columns={'NAME': 'country'})



    #defining continents

    df['continent'] = ''

    df.loc[df.REGION==150, 'continent'] = 'europe'

    df.loc[df.REGION.isin([9, 142]), 'continent'] = 'asia'

    df.loc[(df.REGION==19), 'continent'] = 'america'



    #importing flag URLs from iconfinder

    df['flag_url'] = '../input/european-flags/' + df['country'] + '.png'

    

    return df





#merging with original data

dfgeo = geographical_data()

df = pd.merge(df, dfgeo, on='country', how='left').set_index('country')
#computing average suicide rates in Europe since 1985

df = df[df.continent=='europe']

dfg = df.groupby('country').agg({'suicides_no':'sum', 'population':'sum'}).reset_index()

dfg['suicide_rate'] = dfg['suicides_no'] / dfg['population']

dfg.sort_values('suicide_rate', ascending=False).head()[['country', 'suicide_rate']]
#merging geographical data with average suicide rates

dfmap = pd.merge(dfgeo, dfg, on='country', how='left').set_index('country')

dfmap = dfmap[dfmap.continent=='europe']

dfmap = dfmap[dfmap.suicide_rate.notnull()]





#drawing world map

fig, ax = plt.subplots(1, figsize=(15, 10))

dfmap.plot(column='suicide_rate', cmap='Blues', linewidth=0.8, edgecolor='0.8', ax=ax)



# ax.axis('off')

ax.set_aspect(1.2)

ax.set_title('Average suicide rates in Europe since 2000')

# ax.annotate('Data not available for countries in gray', xy=(0.1, .08),  xycoords='figure fraction', 

#             horizontalalignment='left', verticalalignment='top', fontsize=8)



sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=dfmap.suicide_rate.min()*100, 

                            vmax=dfmap.suicide_rate.max()))

sm._A = []

fig.colorbar(sm, ax=ax, fraction=0.015)



plt.tight_layout()

ax.set_xlim(-15, 50)

plt.subplots_adjust(top=0.7)

plt.show()
dfrel = religion_data().set_index('country')

dff = pd.merge(dfg, dfrel, on='country', how='left').set_index('country')



def compute_correlations(dff):

    dff = dff.drop(['suicides_no', 'population'], axis=1)

    dff_corr = dff.corr()[['suicide_rate']]

    dff_corr['suicide_rate_abs'] = dff_corr['suicide_rate'].abs()

    dff_corr =  dff_corr.sort_values('suicide_rate_abs', ascending=False)

    dff_corr = dff_corr[['suicide_rate']]

    return dff_corr



compute_correlations(dff)
def imscatter(x, y, image, ax=None, zoom=1):

    if ax is None:

        ax = plt.gca()

    try:

        image = plt.imread(image)

    except TypeError:

        pass



    im = OffsetImage(image, zoom=zoom)

    x, y = np.atleast_1d(x, y)

    artists = []

    for x0, y0 in zip(x, y):    

        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)

        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))

    return artists



fig, ax = plt.subplots(1, figsize=(15, 7))

sns.scatterplot('suicide_rate', 'islam_pct', data=dff)

ax.set_xlim(dff.suicide_rate.min()*0.1, dff.suicide_rate.max()*1.05);





#adding country flags

dfff = pd.merge(dff.reset_index(), dfgeo[['country', 'flag_url']], how='left').set_index('country')



for i, r in dfff.iterrows():

    try:

        imscatter(r.suicide_rate, r.islam_pct, r.flag_url, zoom=0.15, ax=ax)

    except:

        print(r.flag_url)



ax.set_title('Islam percentage vs. Suicide Rate');

compute_correlations(dff[dff.index!='Albania'])
fig, ax = plt.subplots(1, figsize=(15, 7))

sns.scatterplot('suicide_rate', 'atheism_pct', data=dff)

ax.set_xlim(dff.suicide_rate.min()*0.1, dff.suicide_rate.max()*1.05);



for i, r in dfff.iterrows():

    imscatter(r.suicide_rate, r.atheism_pct, r.flag_url, zoom=0.15, ax=ax)



ax.set_title('Atheism percentage vs. Suicide Rate');