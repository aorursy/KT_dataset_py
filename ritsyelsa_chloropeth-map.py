#Importing Required Modules
import pandas as pd
import numpy as np

#Importing Data
df = pd.read_excel('../input/population.xlsx')
df['greater'] = np.where(df['2015_female']>df['2015_male'],1,0)
df.head(5)
#Listing the ISO codes of countries
import pycountry

countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3

ISO=pd.DataFrame([countries]).T.reset_index()
ISO.rename(columns={'index': 'name', 0: 'ISO'}, inplace=True)
ISO.head(5)
#Merging ISO codes to the given data 
dat=pd.merge(df, ISO, how='left', on='name')
dat.head(5)
#Choropleth map in offline notebook mode
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()

data = [ dict(
        type = 'choropleth',
        locations = dat['ISO'],
        z = dat['greater'],
        text =dat['name'],
        colorscale = [[1,"rgb(255,0,0)"],[0,"rgb(0,0,255)"]],
        autocolorscale = False,
        reversescale = True,
        colorbar = dict(
            title = 'female count'),
      ) ]

lay = go.Layout(title = 'Female population spread')

fig = go.Figure(data=data, layout=lay)
plotly.offline.iplot(fig)
# Percentage of total world population
dat['perc_total']= round((dat['2015_total']/dat['2015_total'].sum())*100,0).astype('int')
dat.head(3)
#convert country names to dictionary with values and its occurences
from wordcloud import WordCloud
import matplotlib.pyplot as plt
x=np.array(dat.name)
a=np.repeat(x,dat.perc_total,axis=0)

from collections import Counter
word_could_dict=Counter(a)
wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
