import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from nltk.corpus import stopwords as sw

from nltk.corpus import wordnet as wn

import plotly.offline as py

import plotly.graph_objs as go

%matplotlib inline

py.init_notebook_mode(connected=True)
# Reading the train and test set with pandas

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()
fig_sizes = {'S' : (6.5,4),

             'M' : (9.75,6),

             'L' : (13,8)}

    

#function that prepares plot 

#input:

#    ax:seaborn plot

#    f_size:figure size (defaulted as medium size)

#    plot_title: plot title

#    x_title: x title

#    y_title: y title

def show_plot(f_size=fig_sizes['M'],plot_title="",x_title="",y_title=""):

    plt.figure(figsize=f_size)

    plt.xlabel(x_title)

    plt.ylabel(y_title)

    plt.title(plot_title)



auth_counts = train.author.value_counts()

ax_bp = show_plot(fig_sizes['M'],'Author Barplot','Author','Count')

sns.barplot(x=auth_counts.index, y=auth_counts.values,ax=ax_bp)
#Edgar Allan Poe Sample

train.loc[train.author=='EAP','text'].iloc[:5]
#Mary Shelley Sample

train.loc[train.author=='MWS','text'].iloc[:5]
#Mary Shelley Sample

train.loc[train.author=='HPL','text'].iloc[:5]
w_counts = train['text'].str.split(expand=True).unstack().value_counts()
ax_bp = show_plot(fig_sizes['M'],'Word Frequency','Word','Count')

sns.barplot(x=w_counts.index[:20], y=w_counts.values[:20],ax=ax_bp)
sw_en = sw.words('english')

#we will generate a regex that captures any stopword

reg_match_sw = '|'.join(sw_en)

reg_match_sw
#to lower so the regex can handle properly

train.loc[:,'text_lower'] = train['text'].str.lower()

train.loc[:,'text_lower'] = train.text_lower.str.replace(r'(\b)('+(reg_match_sw)+r')(\b)','')

train.loc[:,'text_lower'] = train.text_lower.str.replace(r'[,\.\'\"-?;!]','')
train.text_lower.iloc[:10]
w_counts_wosw = train['text_lower'].str.split(expand=True).unstack().value_counts()

ax_bp = show_plot(fig_sizes['L'],'Word Frequency - WO Stopwords','Word','Count')

sns.barplot(x=w_counts_wosw.index[:20], y=w_counts_wosw.values[:20],ax=ax_bp)
counts_eap = train.loc[train.author=="EAP",'text_lower'].str.split(expand=True).unstack().value_counts()

counts_wms = train.loc[train.author=="MWS",'text_lower'].str.split(expand=True).unstack().value_counts()

counts_hpl =  train.loc[train.author=="HPL",'text_lower'].str.split(expand=True).unstack().value_counts()

counts_eap = counts_eap.loc[[idx for idx in counts_eap.index if (idx in w_counts_wosw.iloc[:250].index)]].sort_index()

counts_wms = counts_wms.loc[[idx for idx in counts_wms.index if (idx in w_counts_wosw.iloc[:250].index)]].sort_index()

counts_hpl = counts_hpl.loc[[idx for idx in counts_hpl.index if (idx in w_counts_wosw.iloc[:250].index)]].sort_index()
common_words = set(counts_eap.index).intersection(set(counts_wms.index)).intersection(set(counts_hpl.index))

counts_eap = counts_eap.loc[counts_eap.index.isin(common_words)]

counts_wms = counts_wms.loc[counts_wms.index.isin(common_words)]

counts_hpl = counts_hpl.loc[counts_hpl.index.isin(common_words)]

print(counts_eap.shape)

print(counts_wms.shape)

print(counts_hpl.shape)
freqs_eqp = 100*counts_eap/np.sum(counts_eap)

freqs_wms = 100*counts_wms/np.sum(counts_wms)

freqs_hpl = 100*counts_hpl/np.sum(counts_hpl)



#distance from the average 

mean_freq = (freqs_eqp+freqs_wms+freqs_hpl)/3

dist_mean = (np.abs(freqs_eqp-mean_freq)+np.abs(freqs_wms-mean_freq)+np.abs(freqs_hpl-mean_freq))/mean_freq
%matplotlib inline

trace1 = go.Scatter3d(

    x=np.log(freqs_eqp),

    y=np.log(freqs_wms),

    z=np.log(freqs_hpl),

    text=counts_eap.index,

    mode='markers',

    marker=dict(

        size=np.log2(counts_eap+counts_wms+counts_hpl),

        color=dist_mean,

        colorscale='RdBu',  

        opacity=0.8

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
for sample in train.loc[train.loc[:,'text_lower'].str.contains('\sde\s'),'text_lower'].iloc[:10]:

    print(sample)
import re

def gen_meta_features(df,text_col):

    #N chars in text

    df["n_chars"+"_"+text_col] = df[text_col].str.len()

    #N words in text

    df["n_words"+"_"+text_col] = df[text_col].str.split().apply(lambda x:len(x))

    #N punctuations chars in text

    df["n_punct"+"_"+text_col] = df[text_col].apply(lambda x: len(re.findall(r'[,.:\'\"-?;!]',str(x))))

    return df
#generate features for raw text

train = gen_meta_features(train,'text')

#generate features for lower wo stopwords text

train = gen_meta_features(train,'text_lower')
ax_box = show_plot(fig_sizes['M'],'N. Char for raw text','Author','N. Char distribution')

sns.boxplot(x=train.loc[train.n_chars_text<1000,'author'], y=train.loc[train.n_chars_text<1000,'n_chars_text'],ax=ax_box)

ax_box = show_plot(fig_sizes['M'],'N. Char for lower text','Author','N. Char distribution')

sns.boxplot(x=train.loc[train.n_chars_text_lower<1000,'author'], y=train.loc[train.n_chars_text_lower<1000,'n_chars_text_lower'],ax=ax_box)
ax_box = show_plot(fig_sizes['M'],'N. Words for raw text','Author','N. Words distribution')

sns.boxplot(x=train.loc[train.n_words_text<1000,'author'], y=train.loc[train.n_chars_text<1000,'n_words_text'],ax=ax_box)

ax_box = show_plot(fig_sizes['M'],'N. Words for lower text','Author','N. Char distribution')

sns.boxplot(x=train.loc[train.n_chars_text_lower<1000,'author'], y=train.loc[train.n_chars_text_lower<1000,'n_words_text_lower'],ax=ax_box)
ax_box = show_plot(fig_sizes['M'],'N. Punctuation chars for raw text','Author','N. Puntctuation chars distribution')

sns.boxplot(x=train.loc[train.n_punct_text<20,'author'], y=train.loc[train.n_punct_text<20,'n_punct_text'],ax=ax_box)
#set of english words

english_vocab = set(w.lower() for w in nltk.corpus.words.words())

#set of words that do not appear in the nltk engluish vocabulary

not_in_vocab = set(train.loc[:,'text_lower'].str.split(expand=True).unstack()).difference(english_vocab)

print(len(not_in_vocab))

list(not_in_vocab)[:15]
train["n_non_voc"] = train['text_lower'].str.split().apply(lambda x:len(set(x).intersection(not_in_vocab)))

ax_box = show_plot(fig_sizes['M'],'N. Words not in nltk corpus','Author','N. Words not in nltk corpus distribution')

sns.boxplot(x=train.loc[train.n_non_voc<50,'author'], y=train.loc[train.n_non_voc<50,'n_non_voc'],ax=ax_box)
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

#example with the plural sufix

print(wordnet_lemmatizer.lemmatize('planets'))

not_in_vocab_lem = set([wordnet_lemmatizer.lemmatize(w) for w in set(train.loc[:,'text_lower'].str.split(expand=True).unstack()) if w is not None]).difference(english_vocab)

print(len(not_in_vocab_lem))

list(not_in_vocab_lem)[:15]
train["n_non_voc_lem"] = train['text_lower'].str.split().apply(lambda x:len(set(x).intersection(not_in_vocab_lem)))

ax_box = show_plot(fig_sizes['M'],'N. Words not in lemmatized nltk corpus','Author','N. Words not in lemmatized nltk corpus distribution')

sns.violinplot(x=train.loc[train.n_non_voc_lem<10,'author'], y=train.loc[train.n_non_voc_lem<10,'n_non_voc_lem'],ax=ax_box)
from nltk.stem.porter import *

stemmer = PorterStemmer()

not_in_vocab_stem = set([stemmer.stem(w) for w in set(train.loc[:,'text_lower'].str.split(expand=True).unstack()) if w is not None]).difference(set([stemmer.stem(v)for v in english_vocab]))

print(len(not_in_vocab_stem))

list(not_in_vocab_stem)[:15]
train["n_non_voc_stem"] = train['text_lower'].str.split().apply(lambda x:len(set(x).intersection(not_in_vocab_stem)))

ax_box = show_plot(fig_sizes['M'],'N. Words not in stemmed nltk corpus','Author','N. Words not in stemmed nltk corpus distribution V2')

sns.violinplot(x=train.loc[train.n_non_voc_stem<5,'author'], y=train.loc[train.n_non_voc_stem<5,'n_non_voc_stem'],ax=ax_box)
train.loc[train.n_non_voc_stem<5].groupby(['author','n_non_voc_stem']).size()/train.shape[0]
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()



print(analyser.polarity_scores("this is a very bad day")['compound'])

print(analyser.polarity_scores("this is a very nice day")['compound'])
train.loc[:,'text_sent'] = train['text_lower'].apply(lambda x:analyser.polarity_scores(x)['compound'])

ax_box = show_plot(fig_sizes['M'],'Text sentiment','Author','Text sentiment distribution')

sns.violinplot(x=train['author'], y=train['text_sent'],ax=ax_box)
ax_sent = show_plot(fig_sizes['L'],'Sentiment Distribution','','')

sns.distplot([train.loc[train.author=='EAP','text_sent']],ax=ax_sent)

sns.distplot([train.loc[train.author=='HPL','text_sent']],ax=ax_sent)

sns.distplot([train.loc[train.author=='MWS','text_sent']],ax=ax_sent)