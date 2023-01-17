!pip install  -q wordcloud
import numpy as np

import pandas as pd

import os

print('Inside Input we have:')

for i, (dirname, _, filenames) in enumerate(os.walk('/kaggle/input/contradictory-my-dear-watson')):

    print('\t '* i, '{}) {} folder. It has:-'.format(i+1, dirname.split('/')[-1]))

    for idx,filename in enumerate(filenames):

        print('\t '* (i+1),f'{idx+1}. {filename}' )

train_df = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')

test_df = pd.read_csv('../input/contradictory-my-dear-watson/test.csv')

train_df.shape, test_df.shape
train_df.head()
test_df.head()
import seaborn as sns

sns.countplot(train_df.label);
print('Different types of language are', train_df['language'].unique(), '\nTotal number of Languages are:-',len((train_df['language'].unique())))
import plotly.express as px

import matplotlib.pyplot as plt



name, count = np.unique(train_df['language'], return_counts = True)

fig = px.pie( values= count, names=name, title='Languages Available to us.')

fig.update_traces(hoverinfo='value+label+percent', textposition='inside', textfont_size=15,textinfo = 'value + label',

                  marker=dict( line=dict(color='#000100', width=2)))

fig.show()
name, count = np.unique(train_df[train_df['language'] != 'English'].language, return_counts = True)



fig = px.bar(x=name, y=count)

fig.update_traces(texttemplate='%{y:.2s}',  textposition='outside')

fig.update_layout(uniformtext_minsize=15, uniformtext_mode='hide', xaxis_tickangle=-80)

fig.show()
fig = plt.figure(figsize = (25,18))

for i,n in enumerate(train_df.language.unique()):

    ax1 = plt.subplot(5,3,i+1)

    sns.countplot(train_df[train_df.language == n].label, ax =ax1)

    ax1.set_title(n)

    ax1.set_xlabel('')

import wordcloud
text = train_df[train_df.language == 'English'].premise.to_string()
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

stopwords.update(["many", "alway", "you", "many", "well", 'time, mean', 'much'])

wordcloud = WordCloud(stopwords=stopwords,max_font_size=50, max_words=800, background_color="white").generate(text)



# Display the generated image:

plt.figure(figsize = (15,15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
print('max length of sentence in premise', max(train_df.premise.apply(lambda x:len(x.split(' ')))))

print('min length of sentence in premise',min(train_df.premise.apply(lambda x:len(x.split(' ')))))

print('max length of sentence in hypothesis',max(train_df.hypothesis.apply(lambda x:len(x.split(' ')))))

print('min length of sentence in hypothesis',min(train_df.hypothesis.apply(lambda x:len(x.split(' ')))))