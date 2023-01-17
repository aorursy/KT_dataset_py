import nltk

from nltk.corpus import stopwords

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

RESPONSES = '../input/sg-public-feedback/CitizensAgenda_201909_English.csv'
#read in the raw data

df = pd.read_csv(RESPONSES,encoding="ISO-8859-1")
topics = {'climate change':['climate change','climate crisis','environment'],

         '377a':['377a','lgbt','gay','queer'],

         'pofma':['pofma','fake news','freedom of speech'],

         'cpf':['cpf','retirement','gic','temasek'],

         'education':['education'],

         'healthcare':['healthcare','medishield'],

         'democracy':['democracy','transparency','nepotism'],

         'economy':['economy','jobs','income','inflation','wage'],

         'cost of living':['cost of living'],

          'inequality':['inequality'],

          'race':['racial','racism','races','ethnicity','religion','religious'],

         }
topic_count = {}

for row in df.iterrows():

    for t in topics:

        tk_matches = [1 for tk in topics[t] if tk in row[1].response.lower()]

        if len(tk_matches)>0:

            if t in topic_count: 

                topic_count[t] +=1

            else:

                topic_count[t] = 1
tp = pd.DataFrame({'count':pd.Series(topic_count)}).sort_values('count')

tp['count'] = tp['count']/sum(tp['count'])
import matplotlib.pyplot as plt

tp.plot.barh()

plt.title('top Singapore issues')