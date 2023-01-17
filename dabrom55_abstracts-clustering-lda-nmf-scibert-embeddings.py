import numpy as np

import pandas as pd

import os

import json

import glob

import sys

sys.path.insert(0, "../")



#root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'

root_path = '/kaggle/input/CORD-19-research-challenge'



# Get all the files saved into a list and then iterate over them like below to extract relevant information

# hold this information in a dataframe and then move forward from there. 



# Just set up a quick blank dataframe to hold all these medical papers. 



corona_features = {"doc_id": [None], "source": [None], "title": [None],

                  "abstract": [None], "text_body": [None]}

corona_df = pd.DataFrame.from_dict(corona_features)



# Cool so dataframe now set up, lets grab all the json file names. 



# For this we can use the very handy glob library



json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)



# Now we just iterate over the files and populate the data frame. 



def return_corona_df(json_filenames, df, source):



    for file_name in json_filenames:



        row = {"doc_id": None, "source": None, "title": None,

              "abstract": None, "text_body": None}



        with open(file_name) as json_data:

            data = json.load(json_data)



            row['doc_id'] = data['paper_id']

            row['title'] = data['metadata']['title']



            # Now need all of abstract. Put it all in 

            # a list then use str.join() to split it

            # into paragraphs. 



            abstract_list = [data['abstract'][x]['text'] for x in range(len(data['abstract']) - 1)]

            abstract = "\n ".join(abstract_list)



            row['abstract'] = abstract



            # And lastly the body of the text. For some reason I am getting an index error

            # In one of the Json files, so rather than have it wrapped in a lovely list

            # comprehension I've had to use a for loop like a neanderthal. 

            

            # Needless to say this bug will be revisited and conquered. 

            

            body_list = []

            for _ in range(len(data['body_text'])):

                try:

                    body_list.append(data['body_text'][_]['text'])

                except:

                    pass



            body = "\n ".join(body_list)

            

            row['text_body'] = body

            

            # Now just add to the dataframe. 

            

            if source == 'b':

                row['source'] = "BIORXIV"

            elif source == "c":

                row['source'] = "COMMON_USE_SUB"

            elif source == "n":

                row['source'] = "NON_COMMON_USE"

            elif source == "p":

                row['source'] = "PMC_CUSTOM_LICENSE"

            

            df = df.append(row, ignore_index=True)

    

    return df



corona_df = return_corona_df(json_filenames, corona_df, 'b')

corona_out = corona_df.to_csv('kaggle_covid-19_open_csv_format.csv')
import pandas as pd

corona_df = pd.read_csv('kaggle_covid-19_open_csv_format.csv')
!ls 
corona_df.head()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans
abstract = corona_df['abstract']

abstract.fillna("",inplace=True)

tfidf = TfidfVectorizer(max_features=20000, stop_words='english')

X = tfidf.fit_transform(abstract)



clustered = KMeans(n_clusters=6, random_state=0).fit_predict(X)



#with_clusters=pd.DataFrame(clusters)

corona_df['cluster_abstract']=clustered



grouped=corona_df.groupby('cluster_abstract')

for gp_name, gp in grouped:

    display(gp)



grouped.describe()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')

X_tfidf = tfidf.fit_transform(abstract)

tfidf_feature_names = tfidf.get_feature_names()



vectorizer = CountVectorizer(stop_words='english', max_features=1000)

X_tf = vectorizer.fit_transform(abstract)

tf_feature_names = vectorizer.get_feature_names()
no_topics = 15



# Run NMF

nmf = NMF(n_components=no_topics).fit(X_tfidf)



# Run LDA

lda = LatentDirichletAllocation(n_components=no_topics).fit(X_tf)

#extract topics

def display_topics(model, feature_names, no_top_words):

    topics=[]

    for topic_idx, topic in enumerate(model.components_):

        #rint ("Topic %d:" % (topic_idx))

        topic_words=" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])

        #rint(topic_words)

        topics.append(topic_words)

    return topics



no_top_words = 5

#rint("NMF: ")

topics_nmf=display_topics(nmf, tfidf_feature_names, no_top_words)

#rint("\nLDA: ")

topics_lda=display_topics(lda, tf_feature_names, no_top_words)



#rint(topics_nmf)

#rint(topics_lda)



pred_lda=lda.transform(X_tf)

pred_nmf=nmf.transform(X_tfidf)



res_lda=[topics_lda[np.argmax(r)] for r in pred_lda]

res_nmf=[topics_nmf[np.argmax(r)] for r in pred_nmf]



corona_df['topic_lda']=res_lda

corona_df['topic_nmf']=res_nmf
grouped=corona_df.groupby('topic_lda')

for gp_name, gp in grouped:

    display(gp)
grouped.describe()
grouped_nmf=corona_df.groupby('topic_nmf')

for gp_name, gp in grouped_nmf:

    display(gp)
grouped_nmf.describe()


corona_df.drop(corona_df.index[0], inplace=True)

corona_df.dropna(subset=['title'],inplace=True)
from transformers import *

import tensorflow as tf
model_version = 'allenai/scibert_scivocab_uncased'

do_lower_case = True

model = TFBertModel.from_pretrained(model_version, from_pt=True)

tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
model.summary()
list_titles = corona_df['title'].values

titles_encoded=[tokenizer.encode(l ,return_tensors='tf',max_length=200, pad_to_max_length=True) for l in list_titles]



batch_size=100

batched=[]

for i in range(int(len(titles_encoded)/batch_size)+1):

  b = titles_encoded[i*batch_size:(i+1)*batch_size]

  batched.append(tf.concat(b,axis=0))
from tqdm import tqdm

embed =[]

for i in tqdm(range(len(batched))):

#for i in range(200):

  #print(i)

  e = model(batched[i])[0]

  mean_e = tf.reduce_mean(e, axis=1)

  embed.append(mean_e)
import numpy as np

embed_titles = np.concatenate(embed)
#!pip install umap-learn

import umap

reducer = umap.UMAP()
from bokeh.plotting import figure, show, output_notebook

from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper

from bokeh.palettes import Spectral10, Category20c

from bokeh.palettes import magma

import pandas as pd

output_notebook()
red = reducer.fit_transform(embed_titles[:200])
def make_plot(red, title_list, number, color = True):

    digits_df = pd.DataFrame(red, columns=('x', 'y'))

    digits_df['digit'] = title_list

    datasource = ColumnDataSource(digits_df)

    plot_figure = figure(

    title='UMAP projection of the article title embeddings',

    plot_width=890,

    plot_height=600,

    tools=('pan, wheel_zoom, reset')

    )



    plot_figure.add_tools(HoverTool(tooltips="""

    <div>

    <div>

        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>

    </div>

    <div>

        <span style='font-size: 10px; color: #224499'></span>

        <span style='font-size: 10px'>@digit</span>

    </div>

    </div>

    """))

    if color:

        color_mapping = CategoricalColorMapper(factors=title_list, palette=magma(number))

        plot_figure.circle(

            'x',

            'y',

            source=datasource,

            color=dict(field='digit', transform=color_mapping),

            line_alpha=0.6,

            fill_alpha=0.6,

            size=7

        )

        show(plot_figure)

    else:

        

        plot_figure.circle(

            'x',

            'y',

            source=datasource,

            color=dict(field='digit'),

            line_alpha=0.6,

            fill_alpha=0.6,

            size=7

        )

        show(plot_figure)
make_plot(red, list_titles[:200], 200)