import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib_venn import venn2



import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample_sub = pd.read_csv("../input/sample_submission.csv")
train.head()
train.drop(['id'], axis=1).describe()
test.drop(['id'], axis=1).describe()
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'clean']
rowsums=train.iloc[:,2:].sum(axis=1)

train['clean']=(rowsums==0)

train['clean'] = train['clean'].map(lambda x: 1 if x == True else 0)
clean = train[(train['toxic'] != 1) & (train['severe_toxic'] != 1) & (train['obscene'] != 1) & 

                            (train['threat'] != 1) & (train['insult'] != 1) & (train['identity_hate'] != 1)]

print('Percentage of clean comment ', len(clean)/len(train)*100)
print(train[labels].sum())
data = [go.Bar(

            x=labels,

            y=train[labels].sum(),

    )]

layout = dict(title='Histogram of counts', xaxis=dict(title='Typy of toxicity'))

fig = dict(data=data, layout=layout)

iplot(fig)
print(labels[0])

print(train[train[labels[0]]==1].iloc[4, 1])

print()

print(labels[1])

print(train[train[labels[1]]==1].iloc[4, 1])

print()

print(labels[2])

print(train[train[labels[2]]==1].iloc[6, 1])

print()

print(labels[3])

print(train[train[labels[3]]==1].iloc[4, 1])

print()

print(labels[4])

print(train[train[labels[4]]==1].iloc[4, 1])

print()

print(labels[5])

print(train[train[labels[5]]==1].iloc[10, 1])

print()
from wordcloud import WordCloud ,STOPWORDS

import matplotlib.pyplot as plt

import nltk

stopwords = nltk.corpus.stopwords.words('english')

def print_wordcloud(name):

    subset=train[train[name]==True]

    text=subset.comment_text.values

    wc= WordCloud(background_color="black",max_words=2000,stopwords=stopwords)

    wc.generate(" ".join(text))

    plt.figure(figsize=(10,10))

    plt.title("Popular words in " + name, fontsize=10)

    plt.imshow(wc.recolor(colormap= 'viridis'))

    plt.show()
print_wordcloud(labels[0])
print_wordcloud(labels[1])
print_wordcloud(labels[2])
print_wordcloud(labels[3])
print_wordcloud(labels[4])
print_wordcloud(labels[5])
data = [go.Bar(

            x=list(range(0, 7)),

            y=train[labels[:6]].sum(axis=1).value_counts()

    )]

layout = dict(title='Multilabel', xaxis=dict(title='Counts of labeled columns'))

fig = dict(data=data, layout=layout)

iplot(fig)
print(train[(train[labels[0]] == 1) & (train[labels[1]] == 1) & (train[labels[2]] == 1) & (train[labels[3]] == 1) & (train[labels[4]] == 1) & (train[labels[5]] == 1)]['comment_text'][114])

print()

print(train[(train[labels[0]] == 1) & (train[labels[1]] == 1) & (train[labels[2]] == 1) & (train[labels[3]] == 1) & (train[labels[4]] == 1) & (train[labels[5]] == 1)]['comment_text'][4294])

print()

print(train[(train[labels[0]] == 1) & (train[labels[1]] == 1) & (train[labels[2]] == 1) & (train[labels[3]] == 1) & (train[labels[4]] == 1) & (train[labels[5]] == 1)]['comment_text'][43913])

print()

print(train[(train[labels[0]] == 1) & (train[labels[1]] == 1) & (train[labels[2]] == 1) & (train[labels[3]] == 1) & (train[labels[4]] == 1) & (train[labels[5]] == 1)]['comment_text'][67493])
train['char_length'] = train['comment_text'].apply(lambda x: len(str(x)))
trace = go.Histogram(x=train['char_length'])

layout = dict(title='Histogram of length', xaxis=dict(title='Length'))

fig = dict(data=[trace], layout=layout)

iplot(fig)
trace = go.Heatmap(z=train[labels[:6]].corr().values, 

                  x=labels[:6], 

                  y=labels[:6])

iplot([trace])
t = train[(train['toxic'] == 1) & (train['severe_toxic'] == 0)].shape[0]

s = train[(train['toxic'] == 0) & (train['severe_toxic'] == 1)].shape[0]

t_s = train[(train['toxic'] == 1) & (train['severe_toxic'] == 1)].shape[0]

plt.figure(figsize=(8, 8))

plt.title("Venn diagram for 'toxic' and 'severe_toxic'")

venn2(subsets = (t, s, t_s), set_labels=('toxic', 'severe_toxic'))

plt.show()