import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import json

from datetime import datetime

import matplotlib.pyplot as plt

plt.style.use('ggplot')
!ls /kaggle/input/CORD-19-research-challenge/
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})
# Conversion of publish_time column to Datetime and add a column of year of publication

publish_dates = []

for d in meta_df['publish_time']:

    try:

        publish_dates.append(datetime.strptime(d,"%Y-%m-%d"))

    except Exception as e:

        if isinstance(d, str):

            publish_dates.append(datetime(int(d),1,1))

        else:

            publish_dates.append(datetime(1900,1,1))



meta_df.drop('publish_time',axis=1)

meta_df['publish_time'] = publish_dates

meta_df['year'] = meta_df['publish_time'].apply(lambda x: x.year)



meta_df.info()
meta_df[meta_df['sha'].isnull()]
len(meta_df['cord_uid'].unique())
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
# dictionary to build df_covid19 dataset for the unsupervised clustering

dict_1={'cord_uid': [], 'title':[], 'abstract':[], 'body_text':[]}



# Dictionary that will include [paper_id and its cited_papers] useful for the most cited papers calculation

dict_2={'cord_uid':[], 'year' :[], 'title':[]}



# function that gets paper id based on its sha code or title if sha code does not exit

def get_paper_id(meta_df,sha,title):

    # try to get metadata information thru sha

    meta_data = meta_df.loc[meta_df['sha'] == sha]['cord_uid']

    # no metadata, get cord_uid thru the title

    if len(meta_data) == 0:

        p_id = meta_df.loc[meta_df['title'] == title]['cord_uid']

    else:

        p_id = meta_df.loc[meta_df['sha'] == sha]['cord_uid']

    return p_id



for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    # Open the content of each json file

    content = json.load(open(entry))

    

    ################ get the paper id from meta_df

    paper_id = get_paper_id(meta_df, content['paper_id'], content['metadata']['title'])

    

    if paper_id.empty:

        continue

    else:

        paper_id = paper_id.iloc[0]

    

    ## get meta_data

    meta_data = meta_df.loc[meta_df['cord_uid'] == paper_id]



    ################# get title, abstract and body_text

    # Append paper id

    dict_1['cord_uid'].append(paper_id)

    # get paper title

    paper_title = content['metadata']['title']

    # if title is empty in json data, get it from meta data

    if len(paper_title)==0:

        dict_1['title'].append(meta_data['title'].values[0])

    else:

        dict_1['title'].append(paper_title)

    

    abstract=[]

    body_text =[]

    

    # if abstract is not provided in json data, get it from meta data

    try:

        # try getting data from the json file

        for a in content['abstract']:

            abstract.append(a['text'])

        

        if len(abstract)==0:

            abstract = meta_data['abstract'].values[0]

        else:

            # join abstract 

            abstract = ''.join(str(abstract))

    # otherwise get it from meta data

    except Exception as e:

        abstract= meta_data['abstract'].values[0]

    

    

    # get body text

    for t in content['body_text']:

        body_text.append(t['text'])

    

    # join body text

    body_text = ''.join(body_text)

    

    # cap body_text length to 1 million characters (spacy constraint)

    if len(body_text) > 1000000:

        body_text = body_text[0:999999]

    

    dict_1['abstract'].append(abstract)

    dict_1['body_text'].append(body_text)

    

    ################## get all biliographic references (BIBREF)

    refs = []

    for e in content['bib_entries']:

        refs.append(e)

        

    # get cited references of this paper

    bib_entries = content['bib_entries']

    

    for ref in refs:

        dict_2['cord_uid'].append(paper_id)

        dict_2['year'].append(str(bib_entries[ref]['year']))

        dict_2['title'].append(bib_entries[ref]['title'])

        



df_covid19 = pd.DataFrame(dict_1,columns=['cord_uid','title','abstract','body_text'])

df_bibrefs = pd.DataFrame(dict_2, columns=['cord_uid','year','title'])

df_covid19.head()
df_covid19.head()
df_bibrefs.head()
# Function to create a foreign key combining the year and the title of each cited article

def f(x,y):

    return str(x) + '-' + str(y)



df_bibrefs['key'] = df_bibrefs.apply(lambda x: f(x.year, x.title), axis = 1)
# groupby and count to estimate the number of occuerence of each cited paper within the corpus

new_df_bibrefs = df_bibrefs.groupby('key').count()
# drop year and title columns to keep only cord_uid and key as index

new_df_bibrefs = new_df_bibrefs.drop(['year', 'title'] , axis = 1)
# creating the foreign key (year - title) within meta_df

meta_df['key'] = meta_df.apply(lambda x: f(x.year, x.title), axis =1)
# we reset key index

new_df_bibrefs= new_df_bibrefs.reset_index()
df = pd.merge(meta_df,new_df_bibrefs, on='key', suffixes=('','_y'))

df = df.sort_values(by=['cord_uid_y'], ascending = False)

df = df.rename(columns= {'cord_uid_y' : 'nb_citations'})
df.shape
df.groupby(by=['year']).count()['cord_uid'].plot(kind='bar')
dict_={'cord_uid' :[], 'authors' : []}



for i in range(0, len(df)):

    

    authors = str(df['authors'].iloc[i]).split(';')

    paper_id = df['cord_uid'].iloc[i]

        

    for author in authors:

        dict_['cord_uid'].append(paper_id)

        dict_['authors'].append(author)

        

df_authors = pd.DataFrame(dict_, columns=['cord_uid','authors'])
df_authors.head()
df_authors_pub = df_authors.groupby('authors').count()
# we reset key index

df_authors_pub= df_authors_pub.reset_index()

# filter only on rows without NaN

df_authors_pub = df_authors_pub.sort_values(by=['cord_uid'], ascending = False)[['authors','cord_uid']]
# remove nan line

df_authors_pub= df_authors_pub.drop(index=0)
# Merge df_authors with df to get the number of citations per paper

df_authors = pd.merge(df_authors, df, on= 'cord_uid', suffixes =('','_y'))
df_authors = df_authors[df_authors['authors'].notna()]
# Groupby on authors and sum of the number of citations

df_authors_cited = df_authors.groupby(by=["authors"])['nb_citations'].sum().reset_index().sort_values("nb_citations", ascending=False)
df_authors_cited.head()
df_plot = df_authors_pub.merge(df_authors_cited, on='authors')

df_plot = df_plot.drop_duplicates('authors')

df_plot = df_plot.rename(columns={'cord_uid': 'nb_publications'})
df_plot = df_plot[df_plot['nb_publications']>10]
import bokeh

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS, Slider, TapTool, TextInput, RadioButtonGroup

from bokeh.palettes import Category20

from bokeh.transform import linear_cmap

from bokeh.io import output_file, show

from bokeh.transform import transform

from bokeh.io import output_notebook

from bokeh.plotting import figure

from bokeh.layouts import column

from bokeh.models import RadioButtonGroup

from bokeh.models import TextInput

from bokeh.layouts import gridplot

from bokeh.models import Div

from bokeh.models import Paragraph

from bokeh.layouts import column, widgetbox



output_notebook()

source = ColumnDataSource(df_plot)

tools = "hover, box_zoom, undo, crosshair"

p = figure()

p.scatter('nb_publications', 'nb_citations', source = source,alpha=1)



p.add_tools(

    HoverTool(

        tooltips=[('Author', '@authors'), ('Nb citations','@nb_citations'), ('Nb publications', '@nb_publications')]

    )

)

show(p)
df_scope = df[(df.nb_citations >10) & (df.year >2000)]

df_scope = df_scope[['cord_uid','nb_citations']]

df_cluster = pd.merge(df_scope,df_covid19, on='cord_uid')

df_cluster = df_cluster.sort_values(by=['nb_citations'], ascending = False)

df_cluster.head()
df_cluster.drop_duplicates(subset ='cord_uid', keep='first',inplace=True)

df_cluster['abstract'] = df_cluster['abstract'].apply(lambda x: str(x))

df_cluster.shape
import spacy

import string

import re

import time

from collections import Counter



# Spacy imports

from spacy.lang.en.examples import sentences

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English



#Skit learn imports

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.base import BaseEstimator,TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans

from sklearn.decomposition import TruncatedSVD, PCA

from sklearn.preprocessing import Normalizer

from sklearn.compose import ColumnTransformer



# Basic function to clean the text

def clean_text(text):

    # Removing spaces and converting text into lowercase

    # using str to avoid float values error

    return str(text).strip().lower()



# Named entities extractors

# Load English NLP object (long size for entities recognition and GloVe vectorisation)

nlp= spacy.load('en_core_web_lg')





# Create our list of punctuation marks

punctuations = string.punctuation

stop_words = spacy.lang.en.stop_words.STOP_WORDS

parser = English()



# Creating our tokenizer function (w/o named entities)

def spacy_tokenizer(sentence):

    # Creating our token object, which is used to create documents with linguistic annotations.

    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations  and len(word) > 2]

    # return preprocessed list of tokens

    return mytokens



# Custom transformer using spaCy

class Cleaner(TransformerMixin):

    def transform(self, X, **transform_params):

        # Cleaning Text

        return [clean_text(text) for text in X]



    def fit(self, X, y=None, **fit_params):

        return self



    def get_params(self, deep=True):

        return {}



clusterer = KMeans(n_clusters=10, random_state = 42)



pipeline = Pipeline([

    # Use ColumnTransformer to combine the features from title, abstract and body text

    ('union', ColumnTransformer(

        [

            # Pulling features from the article's title line (first column)

            ('title', TfidfVectorizer(tokenizer=spacy_tokenizer), 0),

            

            # Pipeline for standard bag-of-words model for article (second column)

            ('abstract_bow', Pipeline([

                ('clean', Cleaner()),

                ('tfidf_abstract', TfidfVectorizer(tokenizer=spacy_tokenizer, max_features=2**9, max_df=0.6, min_df=10)),

                ('best', TruncatedSVD(n_components=50)),

            ]), 1),



            # Pipeline for standard bag-of-words model for article (second column)

            ('body_bow', Pipeline([

                ('clean', Cleaner()),

                ('tfidf_body', TfidfVectorizer(tokenizer=spacy_tokenizer, max_features=2**12, max_df=0.6, min_df=10)),

                ('best', TruncatedSVD(n_components=50)),

            ]), 2),



        ],



        # weight components in ColumnTransformer

        transformer_weights={

            'title': 0.25,

            'abstract_bow':0.15,

            'body_bow': 0.60,

            #'named_entities': 1,

        }

    )),



    # Use a k-mean clusterer on the combined features

    ('kmean', clusterer ),

])
data_a = df_cluster[['title','abstract','body_text']]

X_a = pipeline.fit_transform(data_a)

labels_a = pipeline.predict(data_a)

data_a['labels'] = labels_a
from sklearn.manifold import TSNE

import seaborn as sns

y = labels_a

tsne = TSNE(verbose=1)

X_embedded_a = tsne.fit_transform(X_a)
# Prepare data for display 

def get_breaks(content, length):

    data = ""

    words = str(content).split(' ')

    total_chars = 0



    # add break every length characters

    for i in range(len(words)):

        total_chars += len(words[i])

        if total_chars > length:

            data = data + "<br>" + words[i]

            total_chars = 0

        else:

            data = data + " " + words[i]

    return data



df_cluster['title_display'] = df_cluster['title'].apply(lambda x: get_breaks(x,40))

df_cluster['abstract_display'] = df_cluster['abstract'].apply(lambda x: ' '.join(x.split(' ')[:100])) 

df_cluster['abstract_display'] = df_cluster['abstract'].apply(lambda x: get_breaks(x,40)) 
df_temp = df_cluster.merge(meta_df, on='cord_uid', suffixes = ('','_y'))

df_cluster['journal'] = df_temp.drop_duplicates('cord_uid')['journal']

df_cluster['authors'] = df_temp.drop_duplicates('cord_uid')['authors']
# Generating the clustering display

output_notebook()

y_labels = labels_a



# data sources

source = ColumnDataSource(data=dict(

    x= X_embedded_a[:,0], 

    y= X_embedded_a[:,1],

    x_backup = X_embedded_a[:,0],

    y_backup = X_embedded_a[:,1],

    desc= y_labels, 

    titles= df_cluster['title_display'],

    authors = df_cluster['authors'],

    journal =df_cluster['journal'],

    abstract = df_cluster['abstract_display'],

    labels = ["C-" + str(x) for x in y_labels]

    ))



# hover over information

hover = HoverTool(tooltips=[

    ("Title", "@titles{safe}"),

    ("Author(s)", "@authors"),

    ("Journal", "@journal"),

    ("Abstract", "@abstract{safe}"),

], point_policy="follow_mouse")



# map colors

mapper = linear_cmap(field_name='desc', 

                     palette=Category20[20],

                     low=min(y_labels) ,high=max(y_labels))



# prepare the figure

p = figure(plot_width=800, plot_height=800, 

           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'], 

           title="t-SNE Most cited papers last 20 years, Clustered(K-Means), Tf-idf with Title, Abstract and  Plain Text", 

           toolbar_location="right")



# plot

p.scatter('x', 'y', size=5, 

          source=source,

          fill_color=mapper,

          line_alpha=0.3,

          line_color="black",

          legend_group = 'labels')





#header

header = Div(text="""<h1> Most cited papers clustering 2000 - 2020 </h1>""")



# show

show(column(header,p))
# How to save clustering data

data_a.to_csv('clustering_A.csv', index=False)

outcome = pd.DataFrame(X_a)

outcome.to_csv('X_clustering_A.csv',encoding='utf-8', index = False)



# How to load a saved clustering later

# data_a = pd.read_csv('clustering_A.csv')

# labels_a = data['labels']

#X_a = pd.read_csv('X_clustering_A.csv').to_numpy()
def find_virus(x):

    count = 0

    x = str(x)

    count += x.lower().count('covid-19')

    count += x.lower().count('sars-cov-2')

    return count



df_covid19['covid-19']= df_covid19[['title','abstract','body_text']].apply(lambda x: find_virus(x.title)+find_virus(x.abstract)+find_virus(x.body_text),axis=1)
df_only_covid19 = df_covid19.loc[df_covid19['covid-19']>10].sort_values(by=['covid-19'], ascending=False)

# Removing duplicates

df_only_covid19.drop_duplicates(subset ='cord_uid', keep='first',inplace=True)
df_only_covid19.shape
df_only_covid19 = df_only_covid19.merge(meta_df, on='cord_uid', suffixes=('', '_y'))

df_only_covid19 = df_only_covid19[df_only_covid19['year']>2018]

df_only_covid19.drop_duplicates(subset ='cord_uid', keep='first',inplace=True)
# populating titles that say "Comment" with title from meta_df

df_only_covid19['title'] = df_only_covid19[['title','title_y']].apply(lambda x: x.title_y if x.title=='Comment' else x.title, axis=1)
# populating null abstracts by the top 200 words of the body text (it is just a workaround to avoid having a cluster with only papers with null abstract)

df_only_covid19['abstract'] = df_only_covid19['abstract'].fillna('missing')

df_only_covid19['abstract'] = df_only_covid19[['abstract','body_text']].apply(lambda x: ' '.join(x.body_text.split(' ')[:200]) if x.abstract =='missing' else x.abstract, axis=1)
clusterer = KMeans(n_clusters=6, random_state = 42)





pipeline = Pipeline([

    # Use ColumnTransformer to combine the features from title, abstract and body text

    ('union', ColumnTransformer(

        [

            # Pulling features from the article's title line (first column)

            ('title', TfidfVectorizer(tokenizer=spacy_tokenizer), 0),

            

            # Pipeline for standard bag-of-words model for article (second column)

            ('abstract_bow', Pipeline([

                ('clean', Cleaner()),

                ('tfidf_abstract', TfidfVectorizer(tokenizer=spacy_tokenizer, max_features=2**8, max_df=0.6, min_df=10)),

                ('best', TruncatedSVD(n_components=50)),

            ]), 1),



            # Pipeline for standard bag-of-words model for article (second column)

            ('body_bow', Pipeline([

                ('clean', Cleaner()),

                ('tfidf_body', TfidfVectorizer(tokenizer=spacy_tokenizer, max_features=2**12, max_df=0.6, min_df=10)),

                ('best', TruncatedSVD(n_components=50)),

            ]), 2),



        ],



        # weight components in ColumnTransformer

        transformer_weights={

            'title': 0.4,

            'abstract_bow':0.4,

            'body_bow': 0.2,

        }

    )),



    # Use a k-mean clusterer on the combined features

    ('kmean', clusterer ),

])
data = df_only_covid19[['title','abstract','body_text']]

X = pipeline.fit_transform(data)

labels = pipeline.predict(data)

data['labels'] = labels

X_embedded = tsne.fit_transform(X)
df_only_covid19['title_display'] = df_only_covid19['title'].apply(lambda x: get_breaks(str(x),40))

df_only_covid19['abstract_display'] = df_only_covid19['abstract'].apply(lambda x: ' '.join(str(x).split(' ')[:100])) 

df_only_covid19['abstract_display'] = df_only_covid19['abstract'].apply(lambda x: get_breaks(str(x),40)) 
output_notebook()

y_labels = labels



title = df_only_covid19['title']

title = [text[0:40] for text in title]



# data sources

source = ColumnDataSource(data=dict(

    x= X_embedded[:,0], 

    y= X_embedded[:,1],

    x_backup = X_embedded[:,0],

    y_backup = X_embedded[:,1],

    desc= y_labels, 

    titles= df_only_covid19['title_display'],

    authors = title,

    journal =title,

    abstract = df_only_covid19['abstract_display'],

    labels = ["C-" + str(x) for x in y_labels]

    ))



# hover over information

hover = HoverTool(tooltips=[

    ("Title", "@titles{safe}"),

    ("Author(s)", "@authors"),

    ("Journal", "@journal"),

    ("Abstract", "@abstract{safe}"),

],

                 point_policy="follow_mouse")



# map colors

mapper = linear_cmap(field_name='desc', 

                     palette=Category20[20],

                     low=min(y_labels) ,high=max(y_labels))



# prepare the figure

p = figure(plot_width=800, plot_height=800, 

           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'], 

           title="t-SNE Covid-19 Articles, Clustered(K-Means), Tf-idf with Title, Abstract & Plain Text", 

           toolbar_location="right")



# plot

p.scatter('x', 'y', size=5, 

          source=source,

          fill_color=mapper,

          line_alpha=0.3,

          line_color="black",

          legend = 'labels')



#header

header = Div(text="""<h1>COVID-19 Research Papers Cluster - 2019/2020 </h1>""")



# show

show(column(header,p))
# How to save clustering data

data.to_csv('clustering_B_final.csv', index=False)

outcome = pd.DataFrame(X)

outcome.to_csv('X_clustering_B_final.csv',encoding='utf-8', index = False)

df_plot = df_plot.loc[df_plot['nb_publications']> 10]
df_experts = df_plot.loc[df_plot['nb_citations']> 50]
experts = df_experts['authors'].tolist()
def check_author(x):

    count = 0

    authors = str(x).split(';')

    for author in authors:

        if author in experts:

            count+=1

    return count



df_only_covid19['by_expert'] = df_only_covid19['authors'].apply(lambda x: check_author(x))
pd.set_option('display.max_colwidth', -1)

df_top_papers = df_only_covid19[df_only_covid19['by_expert']>0][['title','abstract','authors', 'cord_uid']]

df_top_papers.to_excel('100_top_papers.xlsx')
root_path = '/kaggle/input/100-top-papers-labelling/'

labelling_path = f'{root_path}/100_top_papers_labelling.xlsx'

df_labels = pd.read_excel(labelling_path)
df_labels = df_labels.rename(columns = {'Title' : 'title'})

df_labels = df_labels.drop(['index'], axis=1)
df_top_papers = df_top_papers.merge(df_labels, on='title')
df_top_papers
df_top_papers.groupby('label').count()['title'].plot(kind ='bar')