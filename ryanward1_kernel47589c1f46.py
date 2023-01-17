# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import geopandas as gpd
import pycountry

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


os.popen('cd ../input/big-five-personality-test/IPIP-FFM-data-8Nov2018; ls').read()
path = r'../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv'
df_full = pd.read_csv(path, sep='\t')
pd.options.display.max_columns = 999
df_full.head()

df = df_full.copy() # reset df





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

country_dict = {i.alpha_2: i.alpha_3 for i in pycountry.countries}
countries = pd.DataFrame(df.country.value_counts()).T\
              .drop('NONE', axis=1)\
              .rename(columns=country_dict, index={'country': 'count'})

countries_rank = countries.T.rename_axis('iso_a3').reset_index()
countries_rank['rank'] = countries_rank['count'].rank()
countries_rank.T







pos_questions = [ # positive questions adding to the trait.
    'EXT1','EXT3','EXT5','EXT7','EXT9',                       # 5 Extroversion
    'EST1','EST3','EST5','EST6','EST7','EST8','EST9','EST10', # 8 Neuroticism
    'AGR2','AGR4','AGR6','AGR8','AGR9','AGR10',               # 6 Agreeableness
    'CSN1','CSN3','CSN5','CSN7','CSN9','CSN10',               # 6 Conscientiousness
    'OPN1','OPN3','OPN5','OPN7','OPN8','OPN9','OPN10',        # 7 Openness
]
neg_questions = [ # negative (negating) questions subtracting from the trait.
    'EXT2','EXT4','EXT6','EXT8','EXT10', # 5 Extroversion
    'EST2','EST4',                       # 2 Neuroticism
    'AGR1','AGR3','AGR5','AGR7',         # 4 Agreeableness
    'CSN2','CSN4','CSN6','CSN8',         # 4 Conscientiousness
    'OPN2','OPN4','OPN6',                # 3 Openness
]

df[pos_questions] = df[pos_questions].replace({1:-2, 2:-1, 3:0, 4:1, 5:2})
df[neg_questions] = df[neg_questions].replace({1:2, 2:1, 3:0, 4:-1, 5:-2})
# cols = pos_questions + neg_questions
# df = df[sorted(cols)]
# df.head()

traits = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
trait_labels = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']

country_EXT = {}
for c in countries_rank['iso_a3']:
    country_EXT[c] = 0.0

for i in range(len(df.get('EXT1'))):
    total = 0.0
    total += df.get('EXT1')[i]
    total += df.get('EXT2')[i]
    total += df.get('EXT3')[i]
    total += df.get('EXT4')[i]
    total += df.get('EXT5')[i]
    total += df.get('EXT6')[i]
    total += df.get('EXT7')[i]
    total += df.get('EXT8')[i]
    total += df.get('EXT9')[i]
    total += df.get('EXT10')[i]
    if df.get('country')[i] != None and df.get('country')[i] in country_dict and country_dict[df.get('country')[i]] in country_EXT :
        country_EXT[country_dict[df.get('country')[i]]] += total
               
# print('country_EXT: ' + str(country_EXT))
'''
for trait in traits:
    trait_cols = sorted([col for col in df.columns if trait in col and '_E' not in col])
    df[trait] = df[trait_cols].sum(axis=1)
df[traits].head()
'''


# EXTs = [0.0] * 221
# EXTs[0] = 1.0

countries_rank['EXT'] = country_EXT.values()

portion_EXT = []
for i in range(len(countries_rank['EXT'])):
    portion_EXT.append(countries_rank['EXT'][i]/countries_rank['count'][i])
    
countries_rank['portion_EXT'] = portion_EXT



print('countries_rank: ' + str(countries_rank))
# print('countries: ' + str(countries))


sns.set_style("white")

file = gpd.datasets.get_path('naturalearth_lowres')
world = gpd.read_file(file)
world = pd.merge(world, right=countries_rank, how='left', on='iso_a3').fillna(0)

fig, ax = plt.subplots(figsize=(20,10))
# sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=world['rank'].min(), vmax=world['rank'].max()))
# sm.set_array([])
# fig.colorbar(sm)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Amount of Extroversion in country', size=16)
world.drop(159).plot(column='portion_EXT', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8'); sns.set()
# plt.box(on=None)

'''
import plotly.graph_objects as go

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')

fig = go.Figure(data=go.Choropleth(
    locations = df['CODE'],
    z = df['GDP (BILLIONS)'],
    text = df['COUNTRY'],
    colorscale = 'Blues',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix = '$',
    colorbar_title = 'GDP<br>Billions US$',
))

fig.update_layout(
    title_text='2014 Global GDP',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0.55,
        y=0.1,
        xref='paper',
        yref='paper',
        text='Source: <a href="https://www.cia.gov/library/publications/the-world-factbook/fields/2195.html">\
            CIA World Factbook</a>',
        showarrow = False
    )]
)

fig.show()
'''
