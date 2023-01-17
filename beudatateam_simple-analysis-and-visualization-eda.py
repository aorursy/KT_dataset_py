

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "

import seaborn as sns

import missingno as msno

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# import warnings library

import warnings        

# ignore filters

warnings.filterwarnings("ignore") # if there is a warning after some codes, this will avoid us to see them.

plt.style.use('ggplot') # style of plots. ggplot is one of the most used style, I also like it.

# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")
dataset.head()
dataset.columns
dataset.isnull().sum()
def bar_plot(variable):

    var=dataset[variable]

    varValue=var.value_counts()[:7]

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index,varValue.index.values,rotation=60)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    
cat_value=["type","country","duration","rating","listed_in"]

for c in cat_value:

    bar_plot(c)
def hist_plt(variable):

    fig=plt.figure(figsize=(9,3))

    x=dataset[variable]

    plt.hist(x,bins=25)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()
num_val=["release_year"]

for c in num_val:

    hist_plt(c)
msno.matrix(dataset)

plt.show()
xdata=dataset.country[(dataset.country.isnull()!=True)&(dataset.type=="Movie")]

wordcloud=WordCloud(

               background_color="white",

               width=1200,

               height=600 ).generate(" ".join(xdata))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
from matplotlib import gridspec

top7_country=dataset[dataset.country.isnull()!=True].country.value_counts()[:7]

print(top7_country)



fig = plt.figure(figsize=(20, 6))

gs = gridspec.GridSpec(nrows=1, ncols=2,

                       height_ratios=[6], 

                       width_ratios=[10, 5])



ax = plt.subplot(gs[0])

sns.barplot(top7_country.index, top7_country, ax=ax, palette="GnBu_d")

ax.set_xticklabels(top7_country.index, rotation='90')

ax.set_title('Top 7 producing countries', fontsize=15, fontweight='bold')



explode = [0 for i in range(7)]

explode[0] = 0.06



ax2 = plt.subplot(gs[1])

ax2.pie(top7_country, labels=top7_country.index,

        shadow=True, startangle=0, explode=explode,

        colors=sns.color_palette("GnBu_d", n_colors=7)

       )

ax2.axis('equal') 



plt.show()

import plotly.express as px

data1 = dataset.groupby('release_year')['country'].value_counts().reset_index(name='counts')

fig = px.choropleth(data1, locations="country", color="counts", 

                    locationmode='country names',

                    animation_frame='release_year',

                    range_color=[0,200],

                    color_continuous_scale=px.colors.sequential.OrRd

                   )



fig.update_layout(title='Comparison by country')

fig.show()
ad=dataset.director[dataset.director.isnull()!=True].value_counts()[:15]

plt.figure(figsize=(15,10))

sns.barplot(x=ad.index,y=ad.values,palette = sns.cubehelix_palette(len(ad.index)))

plt.xlabel('Name')

plt.xticks(rotation=90)

plt.ylabel('Frequency')

plt.title("Most Common 15 Name")
rating_order =  ['TV-MA', 'TV-14', 'R', 'TV-PG', 'NR', 'TV-G', 'TV-Y7', 'TV-Y' , 'TV-Y7-FV']

x=dataset[dataset.type=="Movie"].rating.value_counts()[rating_order]

y=dataset[dataset.type=="TV Show"].rating.value_counts()[rating_order]

print(x)

print(y)

tv_show=y.values

movie=x.iloc[:10].values

tv=[x for x in tv_show]

mv=[x for x in movie]

print(mv)

a=np.arange(len(rating_order))

width=0.36

fig, ax = plt.subplots()

rects1 = ax.bar(a - width/2, tv, width, label='TV Show')

rects2 = ax.bar(a + width/2, mv, width, label='Movie')

ax.set_ylabel('Frequency')

ax.set_title('Frequency by Tv Show and Movie')

ax.set_xticks(a)

ax.set_xticklabels(rating_order)

ax.legend()



def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3), 

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)



fig.tight_layout()



plt.show()


