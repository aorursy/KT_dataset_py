import pandas as pd

from afinn import Afinn

import numpy as np

import re

from nltk.corpus import stopwords

afinn = Afinn()



%matplotlib inline

import matplotlib.pyplot as plt



from random import randint as rdint
dfTED = pd.read_csv('ted_main.csv')

dfTranscripts = pd.read_csv('transcripts.csv')
dfTED = dfTED.get(['main_speaker','title','tags','views', 'url'])

dfTED = dfTED.merge(dfTranscripts, how = 'inner', sort = ['views'])
def clean_n_get(text):

    

    """This function receives a text, clean it and rate the sentiments in it"""

    

    instance = re.sub("[^a-zA-Z]", " ", text).lower().split()

    stops = set(stopwords.words("english"))

    

    cleaned_text = [w for w in instance if w not in stops]

    sentiments = [afinn.score(x) for x in cleaned_text]

    

    return sum(sentiments)



def random_color():

    

    """This function creates an random RGB color string"""

    

    color = str()

    letters = ['A','B','C','D','E','F','0','1','2','3','4','5','6','7','8','9']

    for _ in range(6):

        color += letters[rdint(0,15)]

    return '#' + color
dfTED['sentiments'] = dfTED['transcript'].apply(clean_n_get)

# This cell creates the setiments columns, enabling us to create the plot afterwards
dfTED.head(6) # This is what the DataFrame looks like
dfTED['views'] = dfTED['views'].div(10 ** 6) 

# I am dividng the views columns by 10^6 so it will be more understandable in the plot
sViews = pd.Series()

sSentiments = pd.Series()

for i, element in enumerate(dfTED['tags']):

    lista = element.strip()[1:-1].split(",")

    for el in lista:

        tag = el.split("'")[1]

        sViews[tag] = sViews.get(tag, 0) + dfTED['views'][i]

        sSentiments[tag] = sSentiments.get(tag, 0) + dfTED['sentiments'][i]

sSentiments = sSentiments.div(sSentiments.max())
sViews['TEDx'] = 0

sSentiments['TEDx'] = 0



# we zero 'TEDx' because it appears in many talks but isn't a proper tag



sViews.sort_values(ascending = False, inplace = True)

sSentiments.sort_values(ascending = False, inplace = True)
qtdTags = 20

axisSIZE = 2.5 * qtdTags

labelSIZE = 1.2 * axisSIZE

titleSIZE = 1.6 * axisSIZE

COLOR1 = list()

COLOR2 = list()

for _ in range(qtdTags):

    COLOR1.append(random_color())

    COLOR2.append(random_color())
fig, (axes1, axes2) = plt.subplots(nrows = 2, ncols = 1, figsize = (3.2 * qtdTags, 3.2 * qtdTags))



dataViews = sViews.head(qtdTags)

dataSentiments = sSentiments.head(qtdTags)



dataViews.plot.bar(ax = axes1, color = COLOR1)

dataSentiments.plot.bar(ax = axes2, color = COLOR2)



x1, y1 = dataViews.index, dataWords

x2, y2 = dataSentiments.index, dataSentiments



axes1.set_title("Tags x Views", fontsize = titleSIZE)

axes1.set_ylabel('Views (10^6)', fontsize = labelSIZE)

axes1.set_xlabel('Tags', fontsize = labelSIZE)

axes1.tick_params(labelsize = axisSIZE)



axes2.set_title("Tags X Sentiments",fontsize = titleSIZE)

axes2.set_ylabel('Sentiments', fontsize = labelSIZE)

axes2.set_xlabel('Tags', fontsize = labelSIZE)

axes2.tick_params(labelsize = axisSIZE)



fig.tight_layout()
# fig.savefig('TED_plots.png')

# This saves the plot above