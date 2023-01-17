import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from scipy import stats
from wordcloud import WordCloud

import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as offline
offline.init_notebook_mode()
# from plotly import tools
# import plotly.tools as tls
# import squarify
# from mpl_toolkits.basemap import Basemap
# from numpy import array
# from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

import warnings
warnings.filterwarnings("ignore")
#Loading Excel file and saving into dataframe
df = pd.read_csv('../input/population-data/population.csv')
df['female_majority'] = 0
df.loc[(df['2015_male']>df['2015_female']),['female_majority']] = 1
data = [ dict(
        type = 'choropleth',
        locations = df['CODE'],
        z = df['female_majority'],
        text = df['name'],
        colorscale = [[0.0, 'rgb(0,150,150)'],[1.0, 'rgb(240,39,43)']],
        autocolorscale = False,
        legend_title='Population by County',
        showscale = False
        ,name = {0:'male', 1:'female'}
#         colorbar = dict(
#             title = "More")
      ) ]

layout = dict(
    title = 'World Pop 2015',
#     legend_title='Population by County',
#     fips = list([0,1]),
#     values = list(['male', 'female']),
    #showlegend = True,
    geo = dict(
        scope='world',
        showframe = False,
        showcoastlines = False
        )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d1-world-map' )
population_df = df[['name', '2015_total']]
population_df = population_df.sort_values(by=['2015_total'], ascending=False)

from subprocess import check_output
from wordcloud import WordCloud

population_wordcloud = {}
for a, x in population_df.values:
    population_wordcloud[a] = x
wordcloud = WordCloud(
                          background_color='white',
                          max_font_size=70, 
                          random_state=42,
                         ).generate_from_frequencies(frequencies=population_wordcloud)

print(wordcloud)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
