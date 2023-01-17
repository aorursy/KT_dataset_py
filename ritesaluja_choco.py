# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from termcolor import colored
from PIL import Image

#for word cloud
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

import matplotlib.pyplot as plt
import seaborn as sns

import random

import warnings
warnings.filterwarnings("ignore")

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import plotly.graph_objs as py

import seaborn as sns
# Any results you write to the current directory are saved as output.
choco = pd.read_csv("../input/chocolate-bar-ratings/flavors_of_cacao.csv")
codes = pd.read_csv("../input/country-codes/Country_code.csv")
#Sanitization of Data as per need, lets leave the original file untouched --Note: this step was acheived later after analysis of Dataset
chocotemp = choco
chocotemp["Cocoa\nPercent"] = chocotemp["Cocoa\nPercent"].str.replace('%','').astype('float')
chocotemp = chocotemp.rename(columns={'Company \n(Maker-if known)': 'Maker', 'Specific Bean Origin\nor Bar Name': 'Bar Name', \
                                     'Review\nDate':'ReviewDate','Cocoa\nPercent':'CocoaPercent','Company\nLocation':'CompanyLocation',\
                                     'Bean\nType':'BeanType','Broad Bean\nOrigin':'Broad Bean Origin'})
chocotemp.columns.values[0] = 'Maker'
chocotemp["CompanyLocation"] = chocotemp["CompanyLocation"].str.replace('Domincan Republic','Dominican Republic')
chocotemp["CompanyLocation"] = chocotemp["CompanyLocation"].str.replace('Eucador','Ecuador')
chocotemp["CompanyLocation"] = chocotemp["CompanyLocation"].str.replace('Niacragua','Nicaragua')
chocotemp["CompanyLocation"] = chocotemp["CompanyLocation"].str.replace('Amsterdam','Netherlands')
chocotemp["CompanyLocation"] = chocotemp["CompanyLocation"].str.replace('Sao Tome','Sao Tome and Principe')
chocotemp["CompanyLocation"] = chocotemp["CompanyLocation"].str.replace('St. Lucia','Saint Lucia')
chocotemp["CompanyLocation"] = chocotemp["CompanyLocation"].str.replace('Martinique','Puerto Rico')

#chocotemp.groupby("CompanyLocation")["CompanyLocation"].count()
chcocomaxP = chocotemp.groupby("CompanyLocation")["CocoaPercent"].mean().sort_values(ascending=False).head(10)
print(colored("Top 10 Locations that produces Darkest Chocolates:",'blue'))

plt.figure(figsize=(18,9))
plt.bar(x=chcocomaxP.index,height=chcocomaxP.values,color="#674321",hatch='+',edgecolor=["black"]*10)
plt.xlabel("Locations",color="grey")
#plt.title("Top 10 Locations that produces Darkest Chocolates",color="blue")
plt.ylabel("Avg. Cocoa %",color="grey")
plt.show()
#Data Prep
top6coco = chocotemp.groupby('BeanType')["CocoaPercent"].mean().sort_values(ascending=False).head(6)
x=pd.Series(top6coco.index)
y=pd.Series(top6coco.values)
#plot
fig = plt.figure(figsize=(15,9))
plt.bar(x=x.values,height=y , color="#654321",hatch='||',edgecolor=['black']*6)
plt.xlabel('Bean Types',color='grey')
plt.ylabel('Cocoa Percent (%)',color='grey')
plt.ylim(ymin=60) 
plt.yticks(np.arange(60, 82, 1.0))
plt.show()
#Grey color function
def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    col = ['rgb(210,105,30)','rgb(139,69,19)','rgb(160,82,45)','rgb(165,42,42)','rgb(128,0,0)']
    return random.choice(col)
    #return "hsl(0, %d%%, %d%%)" % (np.random.randint(60, 80),np.random.randint(60, 100))

#Main WordCloud Code
wc = (WordCloud(width=1440, height=1080, relative_scaling=0.5, stopwords=stopwords).generate_from_frequencies(chocotemp['Maker'].value_counts()))

fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")
plt.axis('off')
plt.show()
#fetching needed data and Data Prep
chocoCountry = chocotemp.groupby("CompanyLocation")["CompanyLocation"].count().sort_values(ascending=False)
chocoWorld = pd.DataFrame()
chocoWorld["Country"] = pd.Series(chocoCountry.index)
chocoWorld["Counts"] = pd.Series(chocoCountry.values)
chocoWorld["Code"] = pd.Series(['USA']) #initializing first element
for i in range(1,55):
    chocoWorld["Code"][i] = str(codes[codes.COUNTRY == chocoWorld["Country"][i]]["CODE"].values).replace('[','').replace(']','').replace('\'','')
#left manual work
chocoWorld["Code"][3] = 'GBR'
chocoWorld["Code"][22] = 'GBR' #Scotland falls under United Kingdom (Great Britain)
chocoWorld["Code"][55] = 'GBR'#Wales falls under United Kingdom (Great Britain)
chocoWorld["Code"][33] = 'KOR'

data = [ dict(
        type = 'choropleth',
        locations = chocoWorld["Code"],
        z = chocoWorld["Counts"],
        text = chocoWorld["Country"],
        colorscale = [[0,"rgb(89, 78, 66)"],[0.35,"rgb(128,0,0)"],[0.5,"rgb(165,42,42)"],\
            [0.6,"rgb(160,82,45)"],[0.7,"rgb(139,69,19)"],[1,"rgb(210,105,30)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '#',
            title = '#Companies'),
      ) ]

layout = dict(
    title = 'Where are the Chocolates producing Companies located?',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        showland=True,
        showocean=True,
        projection = dict(
            type = 'natural earth'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )
sns.jointplot(x='Rating', y='CocoaPercent', 
              data=chocotemp, color ='#FA8C00', kind ='reg', 
              size = 8.0)
plt.show()

top10 = chocotemp.groupby("Broad Bean Origin")["Broad Bean Origin"].count().sort_values(ascending=False).head(10)
names = list(top10.index)
names[5] = "unknown"

plt.figure(figsize=(18,9))
plt.bar(x=names,height=top10.values,color="#654321",edgecolor=["black"]*10,hatch='|/')
plt.xlabel("Broad Beans Origin",color="grey")
plt.title("Where does the Broad Beans come from (Top 10 Locations)?",color="blue")
plt.ylabel("Count",color="grey")
plt.show()
cm = sns.light_palette("blue", as_cmap=True)
a = chocotemp.groupby("Broad Bean Origin")["Rating"].mean().sort_values(ascending = False).head(20)
temp = pd.DataFrame()
temp["Broad Bean Origin"] = a.index
temp["Average Rating"] = a.values
temp.style.background_gradient(cmap=cm)
