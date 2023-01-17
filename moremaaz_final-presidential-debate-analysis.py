# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS



plt.style.use('ggplot')



from plotly.offline import init_notebook_mode, iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')

import plotly.graph_objs as go

import plotly

import plotly.express as px

import plotly.figure_factory as ff
df= pd.read_csv('/kaggle/input/us-presidential-debatefinal-october-2020/Presidential_ debate_USA_ final_October_ 2020.csv')
df.info()
print(f'There are {df.shape[0]} rows and {df.shape[1]} columns in our dataset')
df.head()
df.rename(columns= {'TOPIC':'topic'}, inplace= True)

df.drop(columns='Unnamed: 0', axis= 1, inplace= True)
df['topic']= df.topic.apply(lambda x: x.split(':')[1])

df['topic']= df.topic.apply(lambda x: x.strip())

df['speech']= df.speech.apply(lambda x: x.strip())

df['speech']= df.speech.apply(lambda x: x.replace(',', ''))
df['speech2']= df.speech.apply(lambda x: x.split(' '))

df['length']= df.speech2.apply(lambda x: len(x))
color_pal= ['rgb(221, 204, 119)','rgb(136, 204, 238)', 'rgb(204, 102, 119)']

color_pal2 = ['rgb(136, 204, 238)', 'rgb(204, 102, 119)', 'rgb(221, 204, 119)', 'rgb(17, 119, 51)', 'rgb(51, 34, 136)', 'rgb(170, 68, 153)', 'rgb(68, 170, 153)', 'rgb(153, 153, 51)']
speaker_count= df.speaker.value_counts().sort_values()

px.bar(speaker_count, x=speaker_count.index, y= speaker_count.values, title= 'Speaker Instance Count', 

       color= color_pal, color_discrete_map="identity",

      labels= {'index':'Speaker', 'y':'Count'})
topic_count= df.topic.value_counts().sort_values()

px.bar(speaker_count, x=topic_count.index, y= topic_count.values, 

       title= 'Topic Count Instance', labels= {'x':'Topic', 'y':'Count'}, 

       color= color_pal2, 

       color_discrete_map="identity")
b = df.groupby(['topic', 'speaker']).sum().reset_index().sort_values(by='length')

px.bar(b, x='topic', y='length', color= 'speaker', 

       labels={'topic': 'Topic', 'length':'Words Spoken', 'speaker':'Speaker'},

       title= 'Words Spoken Per Topic', 

       color_discrete_sequence= color_pal)
trump = df.loc[df.speaker == 'TRUMP', :]

biden = df.loc[df.speaker == 'BIDEN', :]
print(pd.Series(trump.speech2.sum()).nunique())

print(pd.Series(biden.speech2.sum()).nunique())
labels = ['Trump','Biden']

values = [1649, 1743]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_layout(title_text='Unique Word Count Per Candidate')

fig.show()
plt.figure(figsize=(15,8))

plt.title('Histogram Per Speaker')

sns.kdeplot(trump.length, label= 'Trump')

sns.kdeplot(biden.length, label= 'Biden')
def get_text(column):

    words = ''

    for text in column:

        words += text

    return words
text1 = get_text(trump.speech)



stopwords = set(STOPWORDS)

wc = WordCloud(background_color= 'black', stopwords= stopwords,

              width=1600, height=800)



wc.generate(text1)

plt.figure(figsize=(14,10), facecolor='k')

plt.axis('off')

plt.tight_layout(pad=0)

plt.imshow(wc)

plt.show()
text2 = get_text(biden.speech)



stopwords = set(STOPWORDS)

wc = WordCloud(background_color= 'black', stopwords= stopwords,

              width=1600, height=800)



wc.generate(text2)

plt.figure(figsize=(14,10), facecolor='k')

plt.axis('off')

plt.tight_layout(pad=0)

plt.imshow(wc)

plt.show()