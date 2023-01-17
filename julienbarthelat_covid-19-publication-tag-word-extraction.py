import pandas as pd
import glob
import datetime
import json
import re
import numpy as np

# bokeh
from bokeh.io import output_notebook, push_notebook
from bokeh.io import show, save, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, DatetimeTickFormatter, NumeralTickFormatter, SingleIntervalTicker, LinearAxis
from bokeh.palettes import Set1_9 as palette
from ipywidgets import interact, IntSlider
import ipywidgets as widget
output_notebook()

import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
root_path = '/kaggle/input/CORD-19-research-challenge'
df_metadata = pd.read_csv('%s/metadata.csv' % root_path)
def publish_time_to_datetime(publish_time):
    if(str(publish_time) == 'nan'):
        return_date = None
        
    else:
        list_publish_time = re.split('[ -]',publish_time)
        if len(list_publish_time) >2 :
            try:
                #'2020 Jan 27'
                #'2017 Apr 7 May-Jun'
                return_date = datetime.datetime.strptime('-'.join(list_publish_time[:3]), '%Y-%b-%d')

            except :                
                try :
                    #'2020 03 16'
                    return_date = datetime.datetime.strptime('-'.join(list_publish_time[:3]), '%Y-%m-%d')
                    
                except:
                    #'2015 Jul-Aug'
                    return_date = datetime.datetime.strptime('-'.join(list_publish_time[:2]), '%Y-%b')

        elif len(list_publish_time) == 2:
            #'2015 Winter' -> 1 fev            
            if(list_publish_time[1] == 'Winter'):
                return_date = datetime.datetime(int(list_publish_time[0]), 2, 1)

            #'2015 Spring' -> 1 may            
            elif(list_publish_time[1] == 'Spring'):
                return_date = datetime.datetime(int(list_publish_time[0]), 5, 1)
                
            #'2015 Autumn' -> 1 nov
            elif(list_publish_time[1] in ['Autumn','Fall']):
                return_date = datetime.datetime(int(list_publish_time[0]), 11, 1)            
            else:
                #"2015 Oct"
                return_date = datetime.datetime.strptime('-'.join(list_publish_time), '%Y-%b')

        elif len(list_publish_time) == 1:
            #'2020'
            return_date = datetime.datetime.strptime('-'.join(list_publish_time), '%Y')

    return return_date
%%time
# thanks to Frank Mitchell
json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)
df_data = pd.DataFrame()

# set a break_limit for quick test (-1 for off)
break_limit = 1000
print_debug = False

for i,file_name in enumerate(json_filenames):
    if(print_debug):print(file_name)
    
    # get the sha
    sha = file_name.split('/')[7][:-5]
    if(print_debug):print(sha)
    
    # get the all_sources information
    df_metadata_sha = df_metadata[df_metadata['sha'] == sha]
   
    if(df_metadata_sha.shape[0] > 0):
        s_metadata_sha = df_metadata_sha.iloc[0]
                    
        dict_to_append = {}
        dict_to_append['sha'] = sha
        dict_to_append['dir'] = file_name.split('/')[4]

        # publish time into datetime format

        datetime_publish_time = publish_time_to_datetime(s_metadata_sha['publish_time'])

        if(datetime_publish_time is not None):
            dict_to_append['publish_time'] = datetime_publish_time
            dict_to_append['title'] = s_metadata_sha['title']
            dict_to_append['doi'] = 'https://doi.org/' + str(s_metadata_sha['doi'])

            # thanks to Frank Mitchell
            with open(file_name) as json_data:
                data = json.load(json_data)

                # get abstract
                abstract_list = [str(data['abstract'][x]['text']) for x in range(len(data['abstract']))]            
                abstract = "\n ".join(abstract_list)
                dict_to_append['abstract'] = abstract


                # get body
                body_list = [str(data['body_text'][x]['text']) for x in range(len(data['body_text']))]            
                body = "\n ".join(body_list)
                dict_to_append['body'] = body


            df_data = df_data.append(dict_to_append, ignore_index=True)

    else:
        if(print_debug):print('not found')
                
    if (break_limit != -1):
        if (i>break_limit):
            break
            
# no more need for the df_metadata : bye bye
del df_metadata
# set sha as index
df_data.index = df_data['sha']
df_data = df_data.drop(['sha'], axis =1)
df_publish_month = df_data.title.groupby(df_data['publish_time'].dt.to_period("M")).count()

source = ColumnDataSource(data=dict(
    month = df_publish_month.index,
    month_tooltips = df_publish_month.index.strftime('%Y/%m'),
    publication_count = df_publish_month.values
))

tooltips = [('month','@month_tooltips'),('publication_count','@publication_count')]
tools = ['pan', 'box_zoom', 'wheel_zoom', 'reset', HoverTool(tooltips=tooltips, names=['hover_tool'])]
p = figure(plot_height=600,  plot_width=800,tooltips=tooltips, active_drag="pan", active_scroll='wheel_zoom')
p.line('month','publication_count',source=source)
p.xaxis.formatter=DatetimeTickFormatter(months=["%Y/%m"])
p.title.text = 'Publication count per Month'
p.xaxis[0].axis_label = 'Months'
p.yaxis[0].axis_label = 'Publication count'
show(p)
%%time
title_weight = 4
abstract_weight = 2
body_weight = 1

def concat(s_publication):
    s_return = ''
    
    # title
    if(str(s_publication['title']) != 'nan'):
        for i in range(title_weight + 1):
            s_return = s_return + s_publication['title'] + ' '

    # abstract
    for i in range(abstract_weight + 1):
        s_return = s_return + s_publication['abstract'] + ' '
        
    # body
    for i in range(body_weight + 1):
        s_return = s_return + s_publication['body'] + ' '
        
    return s_return

df_data['publication_processing'] = df_data.apply(concat, axis=1)
%%time
df_data['publication_processing'] = df_data['publication_processing'].str.lower()
df_data['publication_processing'] = df_data['publication_processing'].str.replace('\n',' ')
!pip install langid
from langid.langid import LanguageIdentifier, model
language_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

def detect_language(s_data):
    result = language_identifier.classify(s_data['publication_processing'])
    s_data['language'] = result[0]
    s_data['language_probabilty'] = result[1]
    return s_data
%%time
df_data = df_data.apply(detect_language, axis=1)
print('language use rate :')
df_data['language'].value_counts()/df_data.shape[0]
print('Number of publication in all languages = %s' % df_data.shape[0])
df_data = df_data[df_data['language']=='en']
print('Number of publication in English = %s' % df_data.shape[0])
list_to_remove  = [
    'the copyright holder for this preprint',
    'https://doi.org',
    'author/funder',
    '\(which was not peer-reviewed\)',
    'international license',
    'doi: biorxiv preprint',
    'it is made available under',
    'who has granted medrxiv a license to display the preprint in perpetuity',
    'doi: medrxiv preprint',    
    'cc-by-nc-nd 4.0',
    'all rights reserved',
    'no reuse allowed without permission',    
    'cc-by 4.0',
    'cc-by-nc 4.0',
    'fig',
    'fig.',
    'al.',
    'peer-reviewed'
    
]
for to_remove in list_to_remove:
    to_remove_count = (100* df_data['publication_processing'][df_data['publication_processing'].str.contains(to_remove)].shape[0]/
        df_data['publication_processing'].shape[0])
    print('"%s" is use in %i%% of the publication' % (to_remove, to_remove_count))
# prepare the regex
# separate word with alpha numerique (\w+)
# keep the words with one or two dash, such as covid-19...

# words composed with two dash
regex = '\w+-\w+-\w+'
# or words composed with one dash
regex += '|\w+-\w+'
# or single words
regex += '|\w+'
%%time
tokenizer = nltk.RegexpTokenizer(regex)
list_stopwords_english = list(nltk.corpus.stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def lemmatize_list(list_word):
    list_return = []
    for str_word in list_word:
        list_return.append(lemmatizer.lemmatize(str_word))
    return list_return

# by step to prevent memory error
step = 1000
stop = int(df_data.shape[0]/step)+1

for i in range(stop):
    if(i == stop - 1):
        print('tokenize publication %s to %s' % (i*step,df_data.shape[0]))
    else:
        print('tokenize publication %s to %s' % (i*step,(i+1)*step -1))

    for to_remove in list_to_remove:
        df_data['publication_processing'].iloc[i*step:(i+1)*step] = \
            df_data['publication_processing'].iloc[i*step:(i+1)*step].str.replace(to_remove,'',regex=False)
        
    # tokenize
    df_data['publication_processing'].iloc[i*step:(i+1)*step] = \
        df_data['publication_processing'].iloc[i*step:(i+1)*step].apply(lambda x:tokenizer.tokenize(x))
    
    # remove stop word
    df_data['publication_processing'].iloc[i*step:(i+1)*step] = \
        df_data['publication_processing'].iloc[i*step:(i+1)*step].apply(
        lambda x:[w for w in x if not w in list_stopwords_english]
    )
   
    # lemmatize
    df_data['publication_processing'].iloc[i*step:(i+1)*step] = \
        df_data['publication_processing'].iloc[i*step:(i+1)*step].apply(lemmatize_list)
%%time
cv = CountVectorizer(analyzer=lambda x: x, dtype='uint8')
counted_values = cv.fit_transform(df_data['publication_processing']).toarray()
df_tf = pd.DataFrame(
    counted_values,
    columns=cv.get_feature_names(),
    index=df_data['publication_processing'].index
)

# to sparse in order to preserve the precious computer memory ;)
df_tf = df_tf.astype(pd.SparseDtype('uint32', 0))
# CLEANING
df_data = df_data[['doi','publish_time','title','publication_processing']]
print('number of word = %s' % df_tf.shape[1])
print('number of word / publication ratio = %i' % (df_tf.shape[1]/df_tf.shape[0]))
df_tf.sum().sort_values(ascending=False).head(30)
list_word = list(df_tf.columns)

def remove_numerical(list_word):
    list_return = []
    for word in list_word:
        try:
            test = int(word.replace('-',''))
        except:
            list_return.append(word)
    return list_return

list_word_without_numerical = remove_numerical(list_word)
df_tf = df_tf[list_word_without_numerical]
print('number of word = %s' % df_tf.shape[1])
print('number of word / publication ratio = %i' % (df_tf.shape[1]/df_tf.shape[0]))
df_tf.sum().sort_values(ascending=False).head(30)
list_word = list(df_tf.columns)

def remove_too_short(list_word):
    list_return = []
    for word in list_word:
        if(len(word)>2):
            list_return.append(word)
    return list_return

list_word_without_short_word = remove_too_short(list_word)
df_tf = df_tf[list_word_without_short_word]
print('number of word = %s' % df_tf.shape[1])
print('number of word / publication ratio = %i' % (df_tf.shape[1]/df_tf.shape[0]))
df_tf.sum().sort_values(ascending=False).head(30)
s_words_use = df_tf[df_tf!=0].astype(pd.SparseDtype('int64', 0)).sum(axis=0)
word_usage_threshold = 2
s_fairly_used_words = s_words_use[s_words_use > word_usage_threshold]
df_tf = df_tf[list(s_fairly_used_words.index)]
print('number of word = %s' % df_tf.shape[1])
print('number of word / publication ratio = %i' % (df_tf.shape[1]/df_tf.shape[0]))
df_tf.sum().sort_values(ascending=False).head(30)
df_tfidf = pd.DataFrame(
    TfidfTransformer().fit_transform(df_tf).toarray(),
    columns = df_tf.columns,
    index = df_tf.index
)
df_tfidf = df_tfidf.astype(pd.SparseDtype())
df_tfidf.mean().sort_values(ascending=False).head(30)
s_discrimination_scores = df_tfidf[df_tfidf >0].mean().sort_values(ascending=False)
print('-The winner as the less discriminating word is %s with a discrimination score of %0.3f' % (
    s_discrimination_scores.index[-1],
    s_discrimination_scores[-1]
))
print('-The winner as the best discriminating word is %s with a discrimination score of %0.3f' % (
    s_discrimination_scores.index[0],
    s_discrimination_scores[0]
))
print('-The word with a mean discrimination score is %s with a score of %0.3f' % (
    s_discrimination_scores[s_discrimination_scores < s_discrimination_scores.mean()].index[0],
    s_discrimination_scores[s_discrimination_scores < s_discrimination_scores.mean()][0]
))
print('-For example : the interesting word covid-19 has as discriminating score of %0.3f' % s_discrimination_scores['covid-19'])
print('-And the common word "also" has as discriminating score of %0.3f' % s_discrimination_scores['also'])
list_words_to_keep = list(s_discrimination_scores[s_discrimination_scores>0.02].index)
df_tfidf = df_tfidf[list_words_to_keep]
print('Results after removing the less discriminating word :')
print('number of word = %s' % df_tfidf.shape[1])
print('number of word / publication ratio = %i' % (df_tfidf.shape[1]/df_tfidf.shape[0]))
print('top30 most used words (with the use rate) :')
(df_tfidf[df_tfidf>0].count()/df_tfidf.shape[0]).sort_values(ascending=False).head(30)
from sklearn.metrics import pairwise_distances
word_distance_matrix = pairwise_distances(df_tfidf.T, metric = 'cosine')
from sklearn.manifold import TSNE
%%time
tsne = TSNE(n_components=2,
            metric='precomputed',
            verbose=1,
            random_state=0
           )
word_tsne = tsne.fit_transform(word_distance_matrix)
%%time
use_rate = df_tfidf[df_tfidf>0].count()/df_tfidf.shape[0]
s_discrimination_scores = df_tfidf[df_tfidf >0].mean()
source = ColumnDataSource(data=dict(
    x = word_tsne.T[0],
    y = word_tsne.T[1],
    word = df_tfidf.columns,
    size = 50*use_rate,
    use_percent = use_rate.apply(lambda x:'%0.3f' % x),
    discrimination_score = s_discrimination_scores.apply(lambda x:'%0.3f' % x)
))

tooltips = [('word','@word'),('use rate','@use_percent'),('discrimination score','@discrimination_score')]
tools = ['pan', 'box_zoom', 'wheel_zoom', 'reset', HoverTool(tooltips=tooltips, names=['hover_tool'])]
p = figure(plot_height=800,  plot_width=800,tooltips=tooltips, active_drag="pan", active_scroll='wheel_zoom',
           x_range=(-100, 100),y_range=(-100, 100))
p.circle('x','y',source=source, size = 'size',alpha=0.5)
show(p)
publication_distance_matrix = pairwise_distances(df_tfidf, metric = 'cosine')
%%time
tsne = TSNE(n_components=2,
            metric='precomputed',
            verbose=1,
            random_state=0
           )
publication_tsne = tsne.fit_transform(publication_distance_matrix)
%%time
source = ColumnDataSource(data=dict(
    x = publication_tsne.T[0],
    y = publication_tsne.T[1],
    doi = df_data['doi'],
    title = df_data['title']
))

tooltips = [('doi','@doi'),('title','@title')]
tools = ['pan', 'box_zoom', 'wheel_zoom', 'reset', HoverTool(tooltips=tooltips, names=['hover_tool'])]
p = figure(plot_height=800,  plot_width=800,tooltips=tooltips, active_drag="pan", active_scroll='wheel_zoom',
          x_range=(-100, 100),y_range=(-100, 100))
p.circle('x','y',source=source, size = 5,alpha=0.5)
show(p)
df_word_distance_matrix = pd.DataFrame(
    word_distance_matrix,
    columns = df_tfidf.columns,
    index = df_tfidf.columns
    
)
df_word_distance_matrix
def get_close_word(word):
    df_result = pd.DataFrame(df_word_distance_matrix[word].sort_values())
    return df_result
get_close_word('covid-19')
from sklearn.metrics.pairwise import paired_cosine_distances
def find_publication(list_keyword):
    s_word_vector = pd.Series(
        data = [0] * df_tfidf.shape[1],
        index = df_tfidf.columns,
    )
    for keyword in list_keyword:
        s_word_vector[keyword] = 1
    s_word_vector = s_word_vector/s_word_vector.sum()
    
    df_publication_distance = pd.DataFrame(
        paired_cosine_distances(df_tfidf,[s_word_vector] * df_tfidf.shape[0]),
        columns = ['distance'],
        index = df_tfidf.index
    )
    df_publication_distance[['title','publish_time','doi']] = df_data[['title','publish_time','doi']]
    df_publication_distance = df_publication_distance.sort_values(by='distance').iloc[:10]
    df_publication_distance = df_publication_distance.reset_index()
    df_publication_distance = df_publication_distance.drop(['sha'], axis = 1)
    return df_publication_distance    
%%time
print(find_publication(['cell','pneumonia','covid-19']))
