import pandas as pd

import seaborn as sns

import numpy as np

import nltk

import re

import functools



from matplotlib import pyplot as plt

from plotly import graph_objects as go

from plotly.subplots import make_subplots



from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from nltk.tokenize import TweetTokenizer, RegexpTokenizer



from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



from collections import Counter

plt.style.use('seaborn')
# Define some colors for the plots

COLOR_DISASTER = '#DB2B44'

COLOR_DISASTER_DARK = '#61131f'



COLOR_NOT_DISASTER = '#65A6B2'

COLOR_NOT_DISASTER_DARK = "#23393d"
PATH_CSV_TRAIN = '/kaggle/input/nlp-getting-started/train.csv'

dataf = pd.read_csv(PATH_CSV_TRAIN)
dataf.head(5)
# Used Hashtags in each tweet

dataf['hashtags'] = dataf.apply(lambda r: [e.lower() for e in re.findall('#[a-zA-Z]+', r['text'])], axis=1)



# Number of words in each tweet

dataf['nb_words'] = dataf.apply(lambda r: len(r['text'].split()), axis=1)



# Number of URLs in each tweet

#reg = re.compile('https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

reg = re.compile('https?\S+(?=\s|$)')

dataf['nb_urls'] = dataf.apply(lambda r: len(re.findall(reg, r['text'])), axis=1)



# Number of hashtags1 in each tweet

dataf['nb_hashtags'] = dataf.apply(lambda r: len(r['hashtags']), axis=1)



# TEXT CLEANING

STOPWORDS = set(nltk.corpus.stopwords.words('english'))

STOPWORDS = STOPWORDS.union(['\x89û_', 'û_', '-', '\&'])



#Remove emojis and special chars

reg = re.compile('\\.+?(?=\B|$)')

dataf['clean_text'] = dataf.apply(lambda r: re.sub(reg, string=r['text'], repl=''), axis=1)

reg = re.compile('\x89Û_')

dataf['clean_text'] = dataf.apply(lambda r: re.sub(reg, string=r['clean_text'], repl=' '), axis=1)

reg = re.compile('\&amp')

dataf['clean_text'] = dataf.apply(lambda r: re.sub(reg, string=r['clean_text'], repl='&'), axis=1)

reg = re.compile('\\n')

dataf['clean_text'] = dataf.apply(lambda r: re.sub(reg, string=r['clean_text'], repl=' '), axis=1)



#Remove hashtag symbol (#)

dataf['clean_text'] = dataf.apply(lambda r: r['clean_text'].replace('#', ''), axis=1)



#Remove user names

reg = re.compile('@[a-zA-Z0-9\_]+')

dataf['clean_text'] = dataf.apply(lambda r: re.sub(reg, string=r['clean_text'], repl=''), axis=1)



#Remove URLs

reg = re.compile('https?\S+(?=\s|$)')

dataf['clean_text'] = dataf.apply(lambda r: re.sub(reg, string=r['clean_text'], repl=''), axis=1)
tokenizer = RegexpTokenizer(r'[a-zA-Z]+\b')

lemmatizer = WordNetLemmatizer()

stemmer = PorterStemmer()



def tknize(string, tokenizer=tokenizer, lemmatizer=lemmatizer):

    words = tokenizer.tokenize(string)

    return [lemmatizer.lemmatize(w) for w in words if w not in STOPWORDS]
fig = go.Figure(data=[go.Pie(labels=['No disaster', 'Disaster'], 

                             values=dataf.groupby('target')['target'].count(), 

                             marker_colors=[COLOR_NOT_DISASTER, COLOR_DISASTER],

                             marker_line_color=[COLOR_NOT_DISASTER_DARK, COLOR_DISASTER_DARK],

                             marker_line_width=[2,2]

                            )])

fig.update_layout(title='Percentage of each tweet by type', template='plotly_dark')

fig.show()
hashtags = nltk.flatten(list(dataf['hashtags'].loc[dataf['target'] == 1]))

cnt = Counter()

for h in hashtags:

    cnt[h] += 1

hashtags_disaster = pd.DataFrame.from_dict(dict(cnt), orient='index', columns=['times']).sort_values('times', ascending=False)#.iloc[:15]



hashtags = nltk.flatten(list(dataf['hashtags'].loc[dataf['target'] == 0]))

cnt = Counter()

for h in hashtags:

    cnt[h] += 1

hashtags_not_disaster = pd.DataFrame.from_dict(dict(cnt), orient='index', columns=['times']).sort_values('times', ascending=False)#.iloc[:15]
TOP_N = 15

fig = go.Figure()



hover_disaster = [f"{i} Tweets" for i in hashtags_disaster.values.flatten()[:TOP_N]]

hover_not_disaster = [f"{i} Tweets" for i in hashtags_not_disaster.values.flatten()[:TOP_N]]



fig.add_trace(go.Bar(y=hashtags_disaster.index[:TOP_N],

                            x=hashtags_disaster.values.flatten()[:TOP_N] * 100 / hashtags_disaster.values.sum(),

                            alignmentgroup='a',

                            showlegend=True,

                            orientation='h',

                            legendgroup='Dis',

                            name='Disaster',

                            text=hashtags_disaster.index,

                            textposition='inside',

                            textfont=dict(size=15, color='white'),

                            hovertext=hover_disaster,

                            hoverinfo='text',

                            marker_color=COLOR_DISASTER,

                            marker_line_color=COLOR_DISASTER_DARK,

                            marker_line_width=1,

                            opacity=0.7

                        )

)

             

fig.add_trace(go.Bar(y=hashtags_not_disaster.index[:TOP_N],

                            x=-hashtags_not_disaster.values.flatten()[:TOP_N] * 100 / hashtags_not_disaster.values.sum(),

                            alignmentgroup='a',

                            showlegend=True,

                            orientation='h',

                            legendgroup='NDis',

                            name='Not Disaster',

                            text=hashtags_not_disaster.index,

                            textposition='inside',

                            textfont=dict(size=15, color='white'),

                            hovertext=hover_not_disaster,

                            hoverinfo='text',

                            marker_color=COLOR_NOT_DISASTER,

                            marker_line_color=COLOR_NOT_DISASTER_DARK,

                            marker_line_width=1,

                            opacity=0.7

                        )

)



fig.update_layout(

    yaxis=dict(showticklabels=False),

    xaxis=dict(showticklabels=False),

    title='Use of hashtags per tweet type (normalized)',

    barmode='relative',

    bargap=0.1,

    template='plotly_dark',

    height=800

)

fig_violin = go.Figure()

fig_violin.add_trace(go.Violin(x=["Number of words" for i in range(len(dataf[dataf['target'] == 1]))],

                        y=dataf[dataf['target'] == 1]['nb_words'],

                        name='Disaster',

                        legendgroup='Dis', scalegroup='Y', scalemode='width',

                        side='negative',

                        line_color=COLOR_DISASTER)

             )

fig_violin.add_trace(go.Violin(x=["Number of words" for i in range(len(dataf[dataf['target'] == 0]))],

                        y=dataf[dataf['target'] == 0]['nb_words'],

                        name='Non disaster',

                        legendgroup='NotDis', scalegroup='Y', scalemode='width',

                        side='positive',

                        line_color=COLOR_NOT_DISASTER)

             )











d = dataf.groupby(['target', 'nb_hashtags'])['nb_hashtags'].count()

fig_hashtags = go.Figure()

fig_hashtags.add_trace(go.Bar(y=d[1].index[:6],

                                x=d[1]*100/d[1].sum(),

                                alignmentgroup='b',

                                showlegend=False,

                                legendgroup='Dis',

                                orientation='h',

                                name='Disaster',

                                dy=5,

                                marker_color=COLOR_DISASTER,

                                marker_line_color=COLOR_DISASTER_DARK,

                                marker_line_width=2,

                                opacity=0.6

                            )

)

             

fig_hashtags.add_trace(go.Bar(y=d[0].index[:6],

                                x=d[0]*100/d[0].sum(),

                                alignmentgroup='b',

                                showlegend=False,

                                legendgroup='NotDis',

                                orientation='h',

                                name='Non Disaster',

                                dy=5,

                                marker_color=COLOR_NOT_DISASTER,

                                marker_line_color=COLOR_NOT_DISASTER_DARK,

                                marker_line_width=2,

                                opacity=0.6

                             )

)













d = dataf.groupby(['target', 'nb_urls'])['nb_urls'].count()

fig_urls = go.Figure()

fig_urls.add_trace(go.Bar(y=d[1].index[:6],

                            x=d[1]*100/d[1].sum(),

                            alignmentgroup='a',

                            showlegend=False,

                            orientation='h',

                            legendgroup='Dis',

                            name='Disaster',

                            dy=5,

                            marker_color=COLOR_DISASTER,

                            marker_line_color=COLOR_DISASTER_DARK,

                            marker_line_width=2,

                            opacity=0.6

                        )

)

             

fig_urls.add_trace(go.Bar(y=d[0].index[:6],

                            x=d[0]*100/d[0].sum(),

                            alignmentgroup='a',

                            showlegend=False,

                            legendgroup='NotDis',

                            orientation='h',

                            name='Non Disaster',

                            dy=5,

                            marker_color=COLOR_NOT_DISASTER,

                            marker_line_color=COLOR_NOT_DISASTER_DARK,

                            marker_line_width=2,

                            opacity=0.6

                         )

)



fig_urls.update_layout(dict(

    yaxis=dict(autorange="reversed", dtick = 1, ticklen=50),

    title='Percentage of Number of URLS for each tweet type'

))





container = make_subplots(rows=1, cols=2, subplot_titles=('Percentage of number of hashtags',

                                                          'Percentage of number of URLs'))



#for f in fig_violin.data:

#    container.append_trace(f, 1, 1)



for f in fig_hashtags.data:

    container.append_trace(f, 1, 1)

    

for f in fig_urls.data:

    container.append_trace(f, 1, 2)



container.update_yaxes(autorange="reversed", title="Number of hashtags",row=1, col=1)

container.update_yaxes(autorange="reversed", title="Number of URLs", row=1, col=2)



container.update_xaxes(title="Percentage", ticksuffix='%', row=1, col=1)

container.update_xaxes(title="Percentage", ticksuffix='%', row=1, col=2)



container.update_layout(title='Textual stats for each tweet type', template='plotly_dark')



fig_violin.update_layout(title='Number of words distribution per type', template='plotly_dark', height=500)



fig_violin.show()

container.show()
text_disasters = functools.reduce(lambda a,b: a + " " + b, dataf[dataf['target']==1]['clean_text'].tolist()).lower()

text_disasters = [w for w in tknize(text_disasters) if w not in STOPWORDS]



text_not_disasters = functools.reduce(lambda a,b: a + " " + b, dataf[dataf['target']==0]['clean_text'].tolist()).lower()

text_not_disasters = [w for w in tknize(text_not_disasters) if w not in STOPWORDS]



N_TOP=10



unigrams_disaster =  pd.DataFrame(nltk.FreqDist(text_disasters).most_common()[:N_TOP], columns=['term', 'times'])

bigrams_disaster = pd.DataFrame([(functools.reduce(lambda a,b: a+" "+b,r[0]), r[1]) for r in nltk.FreqDist(nltk.ngrams(text_disasters, 2)).most_common()[:N_TOP]], columns=['term', 'times'])

trigrams_disaster = pd.DataFrame([(functools.reduce(lambda a,b: a+" "+b,r[0]), r[1]) for r in nltk.FreqDist(nltk.ngrams(text_disasters, 3)).most_common()[:N_TOP]], columns=['term', 'times'])



unigrams_not_disaster =  pd.DataFrame(nltk.FreqDist(text_not_disasters).most_common()[:N_TOP], columns=['term', 'times'])

bigrams_not_disaster = pd.DataFrame([(functools.reduce(lambda a,b: a+" "+b,r[0]), r[1]) for r in nltk.FreqDist(nltk.ngrams(text_not_disasters, 2)).most_common()[:N_TOP]], columns=['term', 'times'])

trigrams_not_disaster = pd.DataFrame([(functools.reduce(lambda a,b: a+" "+b,r[0]), r[1]) for r in nltk.FreqDist(nltk.ngrams(text_not_disasters, 3)).most_common()[:N_TOP]], columns=['term', 'times'])
#UNIGRAMS Fig

container = make_subplots(rows=1, cols=2, subplot_titles=('Disaster', 'Not Disaster'), horizontal_spacing=0.1)



container.add_trace(go.Bar(y=unigrams_disaster['term'].iloc[:15],

                            x=unigrams_disaster['times'].iloc[:15],

                            alignmentgroup='a',

                            showlegend=True,

                            legendgroup='Dis',

                            orientation='h',

                            name='Disaster',

                            marker_color=COLOR_DISASTER,

                            marker_line_color=COLOR_DISASTER_DARK,

                            marker_line_width=2,

                            opacity=0.6,

                         ), row=1, col=1

)



container.add_trace(go.Bar(y=unigrams_not_disaster['term'].iloc[:15],

                            x=unigrams_not_disaster['times'].iloc[:15],

                            alignmentgroup='a',

                            showlegend=True,

                            legendgroup='NotDis',

                            orientation='h',

                            name='Not Disaster',

                            marker_color=COLOR_NOT_DISASTER,

                            marker_line_color=COLOR_NOT_DISASTER_DARK,

                            marker_line_width=2,

                            opacity=0.6,

                         ), row=1, col=2

)



container.update_layout(title='Most common words for each tweet type', template='plotly_dark', height=500)



container.update_yaxes(tickfont=dict(color='white', size=12), autorange="reversed")



container.show()



# NGRAMS Fig

container = make_subplots(rows=2, cols=2, subplot_titles=('Bigrams','Trigrams'), 

                          horizontal_spacing=0.4, 

                          vertical_spacing=0.09)

    

container.add_trace(go.Bar(y=bigrams_disaster['term'].iloc[:15],

                            x=bigrams_disaster['times'].iloc[:15],

                            alignmentgroup='a',

                            showlegend=True,

                            legendgroup='Dis',

                            orientation='h',

                            name='Disaster',

                            marker_color=COLOR_DISASTER,

                            marker_line_color=COLOR_DISASTER_DARK,

                            marker_line_width=2,

                            opacity=0.6,

                         ), row=1, col=1

)



container.add_trace(go.Bar(y=trigrams_disaster['term'].iloc[:15],

                            x=trigrams_disaster['times'].iloc[:15],

                            alignmentgroup='a',

                            showlegend=False,

                            legendgroup='Dis',

                            orientation='h',

                            name='Disaster',

                            marker_color=COLOR_DISASTER,

                            marker_line_color=COLOR_DISASTER_DARK,

                            marker_line_width=2,

                            opacity=0.6,

                         ), row=1, col=2

)



####

# Not disaster

####







container.add_trace(go.Bar(y=bigrams_not_disaster['term'].iloc[:15],

                            x=bigrams_not_disaster['times'].iloc[:15],

                            alignmentgroup='a',

                            showlegend=True,

                            legendgroup='NotDis',

                            orientation='h',

                            name='Not Disaster',

                            marker_color=COLOR_NOT_DISASTER,

                            marker_line_color=COLOR_NOT_DISASTER_DARK,

                            marker_line_width=2,

                            opacity=0.6,

                         ), row=2, col=1

)



container.add_trace(go.Bar(y=trigrams_not_disaster['term'].iloc[:15],

                            x=trigrams_not_disaster['times'].iloc[:15],

                            alignmentgroup='a',

                            showlegend=False,

                            legendgroup='NotDis',

                            orientation='h',

                            name='Not Disaster',

                            marker_color=COLOR_NOT_DISASTER,

                            marker_line_color=COLOR_NOT_DISASTER_DARK,

                            marker_line_width=2,

                            opacity=0.6

                         ), row=2, col=2

)



container.update_layout(title='Most common N-grams for each tweet type', template='plotly_dark', height=750)



container.update_yaxes(tickangle=0, tickfont=dict(color='white', size=12))



container.show()
reg = re.compile('\d*\s')



TWEETS_DISASTER = [re.sub(reg, ' ', tweet) for tweet in dataf[dataf['target']==1]['clean_text'].tolist()]

cvec = CountVectorizer(stop_words=STOPWORDS, min_df=10, max_df=0.7, ngram_range=(2,4), tokenizer=tknize)

sf = cvec.fit_transform(TWEETS_DISASTER)



transformer = TfidfTransformer()

transformed_weights = transformer.fit_transform(sf)

weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()

weights_df_disaster = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights}).sort_values('weight', ascending=False)





TWEETS_NOT_DISASTER = [re.sub(reg, ' ', tweet) for tweet in dataf[dataf['target']==0]['clean_text'].tolist()]

cvec = CountVectorizer(stop_words=STOPWORDS, min_df=10, max_df=0.7, ngram_range=(2,4), tokenizer=tknize)

sf = cvec.fit_transform(TWEETS_NOT_DISASTER)



transformer = TfidfTransformer()

transformed_weights = transformer.fit_transform(sf)

weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()

weights_df_not_disaster = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights}).sort_values('weight', ascending=False)
container = make_subplots(rows=1, cols=2, subplot_titles=('Disaster', 'Not disaster'), horizontal_spacing=.3)



top_n = 40

container.add_trace(go.Bar(y=weights_df_disaster['term'].iloc[:top_n],

                            x=weights_df_disaster['weight'].iloc[:top_n],

                            alignmentgroup='a',

                            showlegend=True, 

                            legendgroup='Dis',

                            orientation='h',

                            name='Disaster',

                            marker_color=COLOR_DISASTER,

                            marker_line_color=COLOR_DISASTER_DARK,

                            marker_line_width=2,

                            opacity=0.6,

                         ), row=1, col=1

)



container.add_trace(go.Bar(y=weights_df_not_disaster['term'].iloc[:top_n],

                            x=weights_df_not_disaster['weight'].iloc[:top_n],

                            alignmentgroup='a',

                            showlegend=True,

                            legendgroup='NotDis',

                            orientation='h',

                            name='Not Disaster',

                            marker_color=COLOR_NOT_DISASTER,

                            marker_line_color=COLOR_NOT_DISASTER_DARK,

                            marker_line_width=2,

                            opacity=0.6,

                         ), row=1, col=2

)



container.update_layout(title='Most weighted (important) terms for each tweet type', template='plotly_dark', height=1000)



container.update_yaxes(tickangle=0, tickfont=dict(color='white', size=14), autorange="reversed")

container.update_xaxes(showticklabels=False, title='weight', titlefont=dict(size=10), color='gray')



container.show()
# This part is optional, but for making the things faster I will pre-build a vocabulary with the 700 most common words for each type and 

# then use it with the TfidfVectorizer so it doesn't build a huge vocabulary. 

# If you wish to use all the words, you simply have to remove "vocabulary=unigrams" from TfidfVectorizer instantation

unigrams_disaster =  pd.DataFrame(nltk.FreqDist(text_disasters).most_common()[:700], columns=['term', 'times'])

unigrams_not_disaster =  pd.DataFrame(nltk.FreqDist(text_not_disasters).most_common()[:700], columns=['term', 'times'])

unigrams = pd.concat([unigrams_disaster, unigrams_not_disaster])['term'].unique().tolist()
# remove "vocabulary=unigrams" if you wish to use all the words

vectorizer = TfidfVectorizer(tokenizer=tknize, vocabulary=unigrams, decode_error='replace')



X = vectorizer.fit_transform(TWEETS_DISASTER + TWEETS_NOT_DISASTER)

X_D = vectorizer.transform(TWEETS_DISASTER) # Matrix of only the disaster tweets

X_ND = vectorizer.transform(TWEETS_NOT_DISASTER) # Matrix of only the not disaster tweets
# Train PCA with all tweet embeddings

pca = PCA(n_components=2).fit(X.toarray())
# Get the 2D coordinates + original tweets together.

dis = pd.DataFrame(pca.transform(X_D.toarray()))

dis['tweet'] = TWEETS_DISASTER



notdis = pd.DataFrame(pca.transform(X_ND.toarray()))

notdis['tweet'] = TWEETS_NOT_DISASTER
fig = go.Figure()



fig.add_trace(go.Scatter(x=dis.iloc[:,0], y=dis.iloc[:,1], text=dis['tweet'].values,marker=dict(color=COLOR_DISASTER, size=2.3), mode='markers', opacity=1, name='Disaster'))

fig.add_trace(go.Scatter(x=notdis.iloc[:,0], y=notdis.iloc[:,1], text=notdis['tweet'].values,marker=dict(color=COLOR_NOT_DISASTER, size=2.3), mode='markers', opacity=.7, name='Not Disaster'))



fig.update_layout(title='2D projection of tweets', template='plotly_dark', height=900)

container.update_yaxes(showticklabels=False)

container.update_xaxes(showticklabels=False)



fig.show()