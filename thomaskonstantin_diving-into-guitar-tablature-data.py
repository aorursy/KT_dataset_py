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
import plotly.express as ex

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')
g_data = pd.read_csv('/kaggle/input/top-850-guitar-tabs/gutiarDB.csv')

g_data.head(3)
g_data['Song Rating'] = g_data['Song Rating'].apply(lambda x: int(''.join(x.split(','))))

g_data['Song Hits'] = g_data['Song Hits'].apply(lambda x: int(''.join(x.split(','))))



g_data.head(3)
g_data['Difficulty'].replace({'advance':'advanced','intermediat':'intermediate','novic':'novice'},inplace=True)

diff_dum = pd.get_dummies(g_data['Difficulty'],prefix='Difficulty')

diff_dum.drop(columns=['Difficulty_intermediate'],inplace=True)

g_data = pd.concat([g_data,diff_dum],axis=1)

g_data.drop(columns=['Difficulty'],inplace=True)

plt.figure(figsize=(20,11))

ax = sns.countplot(g_data['Page Type'])

ax.set_title('Distribution Of Different Page Types',fontsize=18)

plt.show()
ex.scatter(x=g_data['Song Hits'],y=g_data['Song Rating'],color=g_data['Page Type'])
plt.figure(figsize=(20,11))

ax = sns.kdeplot(g_data['Song Hits'])

ax.set_xlabel("Number Of Views",fontsize=20)

ax.set_ylabel("Density",fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (g_data['Song Hits'].mean(),), r'$\mathrm{median}=%.2f$' % (g_data['Song Hits'].median(),),

         r'$\sigma=%.2f$' % (g_data['Song Hits'].std(),)))

props = dict(boxstyle='round', facecolor='green', alpha=0.5)

ax.text(0.85, 0.95, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='top', bbox=props)



ax.set_title('Distribution Of View Coutns Across Our Samples',fontsize=21)

plt.show()
plt.figure(figsize=(20,11))

ax = sns.kdeplot(g_data['Song Rating'])

ax.set_xlabel("Number Of Views",fontsize=20)

ax.set_ylabel("Density",fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (g_data['Song Rating'].mean(),), r'$\mathrm{median}=%.2f$' % (g_data['Song Rating'].median(),),

         r'$\sigma=%.2f$' % (g_data['Song Hits'].std(),)))

props = dict(boxstyle='round', facecolor='green', alpha=0.5)

ax.text(0.85, 0.95, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='top', bbox=props)



ax.set_title('Distribution Of View Coutns Across Our Samples',fontsize=21)

plt.show()
from sklearn.preprocessing import LabelEncoder



capo_encoder = LabelEncoder()

page_type_encoder = LabelEncoder()

key_encoder  = LabelEncoder()

tuning_encoder  = LabelEncoder()

difficulty_encoder  = LabelEncoder()



ge_data = g_data.copy()



ge_data['Capo'] =  capo_encoder.fit_transform(g_data['Capo'])

ge_data['Page Type'] =  page_type_encoder.fit_transform(g_data['Page Type'])

ge_data['Key'] =  key_encoder.fit_transform(g_data['Key'])

ge_data['Tuning'] =  tuning_encoder.fit_transform(g_data['Tuning'])



ge_data.head(3)
artists = ge_data.groupby(by='Artist').count()

artists = artists.sort_values(by= 'Song Name',ascending=False)

artists = artists[:30]

artists = artists.rename(columns={'Song Name':'Number Of Songs'})

ex.pie(artists,values='Number Of Songs',names=artists.index,title='Top 30 Artists')
gez_data = ge_data.copy()

gez_data['Key'] = key_encoder.inverse_transform(ge_data['Key'])

gez_data.head(3)
artists = gez_data.groupby(by='Key').count()

artists = artists.sort_values(by= 'Artist',ascending=False)

artists = artists[:5]

artists = artists.rename(columns={'Song Name':'Number Of Songs'})

ex.pie(artists,values='Number Of Songs',names=artists.index,title='Top 30 Keys')

from wordcloud import WordCloud,STOPWORDS

stopwords = list(STOPWORDS)



words = ''

for name in ge_data['Song Name']:

    tokens = name.lower().split(' ')

    words += ' '.join(tokens)+' '





wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(words) 

  

plt.figure(figsize = (25, 15), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()



word_count = WordCloud().process_text(words)

word_count =  {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1])}

word_list = list(word_count.items())[-30:]

word_list = [word for word,count in word_list]

sent = ' '.join(word_list)

pscores = sid.polarity_scores(sent)

dfs = pd.DataFrame(pscores,index=[1])

dfs = dfs.T

dfs= dfs.reset_index()

dfs = dfs.rename(columns={'index':'Type',1:'Value'})

dfs = dfs.drop(3)

ex.line_polar(dfs,r='Value',theta='Type',line_close=True)
def get_pos_sentiment(sir):

    return sid.polarity_scores(sir)['pos']

def get_neg_sentiment(sir):

    return sid.polarity_scores(sir)['neg']

def get_neu_sentiment(sir):

    return sid.polarity_scores(sir)['neu']



ge_data['Positive_Sentiment'] = ge_data['Song Name'].apply(get_pos_sentiment)

ge_data['Negative_Sentiment'] = ge_data['Song Name'].apply(get_neg_sentiment)

ge_data['Neutral_Sentiment'] = ge_data['Song Name'].apply(get_neu_sentiment)

ge_data['Song_Name_Length'] = ge_data['Song Name'].apply(lambda x : len(x))
cors = ge_data.corr('pearson')

plt.figure(figsize=(20,11))

ax = sns.heatmap(cors,annot=True,cmap='mako')

plt.show()