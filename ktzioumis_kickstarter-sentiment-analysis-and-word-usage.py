import pandas as pd

import numpy as np
df_ks=pd.read_csv('../input/kickstarter-nlp/df_text_eng.csv',index_col='Unnamed: 0')

df_ks.dropna(inplace=True)

df_ks.head()
import nltk

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

sid.polarity_scores(df_ks['blurb'].iloc[2])
def tag_conf_gen(df):

    neg=[]

    neu=[]

    pos=[]

    compound=[]

    for text in df['blurb']:

        result = sid.polarity_scores(text)

        neg.append(result['neg'])

        neu.append(result['neu'])

        pos.append(result['pos'])

        compound.append(result['compound'])

    df['neg']=neg

    df['neu']=neu

    df['pos']=pos

    df['compound']=compound

    return df
df_ks_sent=tag_conf_gen(df_ks)
df_ks_sent.head()
df_ks_sent.describe()
df_ks.loc[df_ks.state=='successful'].describe()
df_ks.loc[df_ks.state=='failed'].describe()
import plotly.graph_objects as go

labels=['Positive','Neutral','Negative']

values=[np.sum([df_ks_sent['compound']>0]),np.sum([df_ks_sent['compound']==0]),np.sum([df_ks_sent['compound']<0])]

fig = go.Figure(data=[go.Pie(labels=labels, values=values,marker_colors=['blue','yellow','red'])])

fig.update_layout(title_text="Sentiment of Kickstarter blurbs",)

fig.show()
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])



for n,s in enumerate(['successful','failed']):

    df=df_ks_sent.loc[df_ks_sent['state']==s]

    labels=['Positive','Neutral','Negative']

    values=[np.sum([df['compound']>0]),np.sum([df['compound']==0]),np.sum([df['compound']<0])]

    colors=['blue','yellow','red']

    fig.add_trace(go.Pie(labels=labels, values=values, name=s, marker_colors=colors),

              1, n+1)



# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.4, hoverinfo="label+percent")



fig.update_layout(

    title_text="Sentiment of Successful and Failed Kickstarter blurbs",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='Successful', x=0.16, y=0.5, font_size=12, showarrow=False),

                 dict(text='Failed', x=0.8, y=0.5, font_size=12, showarrow=False)])

fig.show()
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])



labels=['Successful','Failed']

colors=['green','darkred']



df=df_ks_sent.loc[df_ks_sent['compound']>0]

values=[np.sum([df['state']=='successful']),np.sum([df['state']=='failed'])]

fig.add_trace(go.Pie(labels=labels, values=values, name=s, marker_colors=colors),

              1, 1)



df=df_ks_sent.loc[df_ks_sent['compound']==0]

values=[np.sum([df['state']=='successful']),np.sum([df['state']=='failed'])]

fig.add_trace(go.Pie(labels=labels, values=values, name=s, marker_colors=colors),

              1, 2)



df=df_ks_sent.loc[df_ks_sent['compound']<0]



values=[np.sum([df['state']=='successful']),np.sum([df['state']=='failed'])]

fig.add_trace(go.Pie(labels=labels, values=values, name=s, marker_colors=colors),

              1, 3)



# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.4, hoverinfo="label+percent")



fig.update_layout(

    title_text="Sentiment of Successful and Failed Kickstarter blurbs",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='Positive', x=0.1, y=0.5, font_size=10, showarrow=False),

                 dict(text='Neutral', x=0.5, y=0.5, font_size=10, showarrow=False),

                 dict(text='Negative', x=0.9, y=0.5, font_size=10, showarrow=False)])

fig.show()
fig=go.Figure(data=[go.Histogram(x=df_ks_sent['compound'])])

fig.update_layout(title_text='Sentiment Histogram')

fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(x=df_ks_sent.loc[df_ks_sent['state']=='successful']['compound'],nbinsx=50,name='successful'))

fig.add_trace(go.Histogram(x=df_ks_sent.loc[df_ks_sent['state']=='failed']['compound'],nbinsx=50,name='failed'))

# Overlay both histograms

fig.update_layout(barmode='overlay',title_text='Successful and Failed Sentiment Histograms')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.35)

fig.show()
import plotly.figure_factory as ff

data=[df_ks_sent.loc[df_ks_sent['state']=='successful']['compound'],df_ks_sent.loc[df_ks_sent['state']=='failed']['compound']]

labels=['successful','failed']

fig = ff.create_distplot(data, labels, bin_size=.1)

fig.update_layout(title_text='Successful and Failed Sentiment Distplots')

fig.show()
from scipy import stats

t_stat, p= stats.ttest_ind(df_ks_sent.loc[df_ks_sent['state']=='successful']['compound'],df_ks_sent.loc[df_ks_sent['state']=='failed']['compound'],equal_var=False)

print('T-Statistic: ',round(t_stat,2))

print('P-value: ',p)
df_ks_sent['state_binary']=df_ks_sent['state']=='successful'

df_ks_sent.corr()
fig=go.Figure(data=go.Heatmap(z=df_ks_sent.corr(),

                             x=df_ks_sent.corr().columns,

                             y=df_ks_sent.corr().index,

                             xgap=5,

                             ygap=5,

                             colorscale=[[0.0, "rgb(300,100,100)"],

                [0.4, "lightpink"],

                [0.45, "white"],

                [0.5,"lightblue"],

                [1.0, "rgb(100,100,300)"]]))

fig.update_layout(title_text='Sentiment Correlation Heatmap')

fig.show()
import nltk

from nltk.corpus import gutenberg, stopwords

from nltk.collocations import *

from nltk import FreqDist, word_tokenize

import string

import re

 
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"

nltk.download('stopwords')

stopwords_list = stopwords.words('english')

stopwords_list += [string.punctuation]

stopwords_list += ['0','1','2','3','4','5','6','7','8','9']

succ_blurbs=df_ks_sent.loc[df_ks_sent['state']=='successful']['blurb'].values

succ_flat=' '.join(succ_blurbs)



succ_words= nltk.regexp_tokenize(succ_flat,pattern)

succ_tokens = [word.lower() for word in succ_words]





succ_tokens_stopped = [word for word in succ_tokens if word not in stopwords_list]



succ_freqdist = FreqDist(succ_tokens_stopped)

succ_freqdist.most_common(50)
fail_blurbs=df_ks_sent.loc[df_ks_sent['state']=='failed']['blurb'].values

fail_flat=' '.join(fail_blurbs)



fail_words= nltk.regexp_tokenize(fail_flat,pattern)

fail_tokens = [word.lower() for word in fail_words]



fail_tokens_stopped = [word for word in fail_tokens if word not in stopwords_list]



fail_freqdist = FreqDist(fail_tokens_stopped)

fail_freqdist.most_common(50)
succall_df=pd.DataFrame.from_dict(dict(succ_freqdist),orient='index',columns=['freq_success'])

failall_df=pd.DataFrame.from_dict(dict(fail_freqdist),orient='index',columns=['freq_fail'])

df_freqdist_all=succall_df.join(failall_df,how='left').fillna(0)

df_freqdist_all['diff']=df_freqdist_all['freq_success']-df_freqdist_all['freq_fail']

df_freqdist_all['ratio']=df_freqdist_all['freq_success']/df_freqdist_all['freq_fail']

df_freqdist_all['freq_fail']=np.negative(df_freqdist_all['freq_fail'])

df_freqdist_all.head()
df_freq_succ20=df_freqdist_all.sort_values(by='freq_success',ascending=False).head(20)

df_freq_succ20
fig=go.Figure()

fig.add_trace(go.Bar(x=df_freq_succ20['freq_success'],y=df_freq_succ20.index,orientation='h',marker_color='blue',name='Usage - successful'))

fig.add_trace(go.Bar(x=df_freq_succ20['freq_fail'],y=df_freq_succ20.index,orientation='h',marker_color='red',name='Usage - failed'))

fig.add_trace(go.Bar(x=df_freq_succ20['diff'],y=df_freq_succ20.index,orientation='h',marker_color='green',name='Usage - net difference'))

fig.update_traces(opacity=.6)

fig.update_layout(barmode='overlay',yaxis=dict(autorange="reversed"),title_text='Most Commonly Used Words In Successful Blurbs Totals and Difference')

fig.show()
df_freq_diff20=df_freqdist_all.sort_values(by='diff',ascending=False).head(20)

df_freq_diff20
fig=go.Figure()

fig.add_trace(go.Bar(x=df_freq_diff20['freq_success'],y=df_freq_diff20.index,orientation='h',marker_color='blue',name='Usage - successful'))

fig.add_trace(go.Bar(x=df_freq_diff20['freq_fail'],y=df_freq_diff20.index,orientation='h',marker_color='red',name='Usage - failed'))

fig.add_trace(go.Bar(x=df_freq_diff20['diff'],y=df_freq_diff20.index,orientation='h',marker_color='green',name='Usage - net difference'))

fig.update_traces(opacity=.6)

fig.update_layout(barmode='overlay',yaxis=dict(autorange="reversed"),title_text='Words With Largest Usage Difference Between Successful and Failed Blurbs')

fig.show()
df_freq_ratio20=df_freqdist_all.loc[df_freqdist_all['freq_success']>500].sort_values(by='ratio',ascending=False).head(20)

df_freq_ratio20
fig=go.Figure()

fig.add_trace(go.Bar(x=df_freq_ratio20['freq_success'],y=df_freq_ratio20.index,orientation='h',marker_color='blue',name='Usage - successful'))

fig.add_trace(go.Bar(x=df_freq_ratio20['freq_fail'],y=df_freq_ratio20.index,orientation='h',marker_color='red',name='Usage - failed'))

fig.add_trace(go.Bar(x=df_freq_ratio20['diff'],y=df_freq_ratio20.index,orientation='h',marker_color='green',name='Usage - net difference'))

fig.update_traces(opacity=.6)

fig.update_layout(barmode='overlay',yaxis=dict(autorange="reversed"),title_text='Words With Largest Usage Ratio Between Successful and Failed Blurbs (Minimum 500 Usages)')

fig.show()