## General Utilities

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import json

import re

import os

import warnings 

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

plt.style.use('ggplot')



## Sklearn Utilities

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.metrics.pairwise import cosine_similarity



## Tqdm Utilities

from tqdm import tqdm_notebook, tnrange

from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')



## Bokeh Utilities

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS

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



## IPython Utilities

from IPython.display import HTML



import notebook as widgets

from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output, VBox



from IPython.html import widgets

from IPython.display import display, Image, HTML, Markdown, clear_output
## Install flair library

!pip install flair
## Install allennlp library



!pip install allennlp
!python -m spacy download en_core_web_md
## Load Spacy Utilities:

import spacy

import en_core_web_md

nlp = en_core_web_md.load()
## Flair Utilities

from flair.embeddings import ELMoEmbeddings, PooledFlairEmbeddings, Sentence, DocumentPoolEmbeddings

from typing import List
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
## Information about Metadata:

meta_df.info()
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

print(len(all_json))
def cstr(s, color='blue'):

    return "<text style=color:{}>{}</text>".format(color, s)



def printmd(string):

    display(Markdown(cstr(string)))
## JSON File Reader Class

class FileReader:

    """FileReader adds break after every words when character length reach to certain amount."""

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])

print(first_row)
def get_breaks(content, length):

    data = ""

    words = content.split(' ')

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
dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    content = FileReader(entry)

    

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    # no metadata, skip this paper

    if len(meta_data) == 0:

        continue

    

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

    

    # also create a column for the summary of abstract to be used in a plot

    if len(content.abstract) == 0: 

        # no abstract provided

        dict_['abstract_summary'].append("Not provided.")

    elif len(content.abstract.split(' ')) > 100:

        # abstract provided is too long for plot, take first 300 words append with ...

        info = content.abstract.split(' ')[:100]

        summary = get_breaks(' '.join(info), 40)

        dict_['abstract_summary'].append(summary + "...")

    else:

        # abstract is short enough

        summary = get_breaks(content.abstract, 40)

        dict_['abstract_summary'].append(summary)

        

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    

    try:

        # if more than one author

        authors = meta_data['authors'].values[0].split(';')

        if len(authors) > 2:

            # more than 2 authors, may be problem when plotting, so take first 2 append with ...

            dict_['authors'].append(". ".join(authors[:2]) + "...")

        else:

            # authors will fit in plot

            dict_['authors'].append(". ".join(authors))

    except Exception as e:

        # if only one author - or Null valie

        dict_['authors'].append(meta_data['authors'].values[0])

    

    # add the title information, add breaks when needed

    try:

        title = get_breaks(meta_data['title'].values[0], 40)

        dict_['title'].append(title)

    # if title was not provided

    except Exception as e:

        dict_['title'].append(meta_data['title'].values[0])

    

    # add the journal information

    dict_['journal'].append(meta_data['journal'].values[0])
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])

df_covid.head()
dict_ = None
## Adding word count columns for both abstract and body_text

df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))

df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))
df_covid.head()
## Remove Duplicates

df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)
## Remove NA's from data

df_covid.dropna(inplace=True)

df_covid.info()
## Taking only 12000 articles for analysis:

df_covid = df_covid.head(12000)
## Remove punctuation from each text:

df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

df_covid['title'] = df_covid['title'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
## Convert each text to lower case:

def lower_case(input_str):

    input_str = input_str.lower()

    return input_str



df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: lower_case(x))

df_covid['title'] = df_covid['title'].apply(lambda x: lower_case(x))
## Considering body of articles only:

text = df_covid[["title"]]
text.head()
## Converting text dataframe into array:

text_arr = text.stack().tolist()

len(text_arr)
## Considering only 500 articles for analysis:

require_text = text_arr[:500]
## Using Spacy module for Sentence Tokenization:

sentences = []

for body in tqdm(require_text):

    doc = nlp(body)

    for i in doc.sents:

        if len(i)>10:

            ## Taking those sentences only which have length more than 10

            sentences.append(i.string.strip())



print(len(sentences))
## Creating Document Pool Embeddings using Stacked of PooledFlairEmbeddings('pubmed-forward'), PooledFlairEmbeddings('pubmed-backward') & ELMoEmbeddings('pubmed')

document_embeddings = DocumentPoolEmbeddings([PooledFlairEmbeddings('pubmed-forward'),

                                             PooledFlairEmbeddings('pubmed-backward'),

                                             ELMoEmbeddings('pubmed')],

                                             pooling='min')
## Getting sentence embeddings for each sentence and storing those into flair_elmo_ls:

flair_elmo_ls = []



for _sent in tqdm(sentences):

    example = Sentence(_sent)

    document_embeddings.embed(example)

    flair_elmo_ls.append(example.get_embedding())
## Converting embeddings into numpy array :

flair_elmo_arr = [emb.cpu().detach().numpy() for emb in flair_elmo_ls]
tsne = TSNE(verbose=1, perplexity=5)

X_embedded = tsne.fit_transform(flair_elmo_arr)
from sklearn.cluster import MiniBatchKMeans



k = 20

kmeans = MiniBatchKMeans(n_clusters=k)

y_pred = kmeans.fit_predict(flair_elmo_arr)

y = y_pred
from matplotlib import pyplot as plt

import seaborn as sns

import random 



# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# let's shuffle the list so distinct colors stay next to each other

palette = sns.hls_palette(20, l=.4, s=.9)

random.shuffle(palette)



# plot

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)

plt.title("t-SNE Covid-19 Articles - Clustered (K-Means) - Flair & Elmo Biomedical Embeddings")

plt.show()
output_notebook()

y_labels = y_pred



# data sources

source = ColumnDataSource(data=dict(

    x= X_embedded[:,0], 

    y= X_embedded[:,1],

    x_backup = X_embedded[:,0],

    y_backup = X_embedded[:,1],

    desc= y_labels, 

    titles= df_covid['title'],

    authors = df_covid['authors'],

    journal = df_covid['journal'],

    abstract = df_covid['abstract_summary'],

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

           title="t-SNE Covid-19 Articles, Clustered(K-Means), Flair & Elmo Biomedical Embeddings", 

           toolbar_location="right")



# plot

p.scatter('x', 'y', size=5, 

          source=source,

          fill_color=mapper,

          line_alpha=0.3,

          line_color="black",

          legend = 'labels')



# add callback to control 

callback = CustomJS(args=dict(p=p, source=source), code="""

            

            var radio_value = cb_obj.active;

            var data = source.data; 

            

            x = data['x'];

            y = data['y'];

            

            x_backup = data['x_backup'];

            y_backup = data['y_backup'];

            

            labels = data['desc'];

            

            if (radio_value == '20') {

                for (i = 0; i < x.length; i++) {

                    x[i] = x_backup[i];

                    y[i] = y_backup[i];

                }

            }

            else {

                for (i = 0; i < x.length; i++) {

                    if(labels[i] == radio_value) {

                        x[i] = x_backup[i];

                        y[i] = y_backup[i];

                    } else {

                        x[i] = undefined;

                        y[i] = undefined;

                    }

                }

            }





        source.change.emit();

        """)



# callback for searchbar

keyword_callback = CustomJS(args=dict(p=p, source=source), code="""

            

            var text_value = cb_obj.value;

            var data = source.data; 

            

            x = data['x'];

            y = data['y'];

            

            x_backup = data['x_backup'];

            y_backup = data['y_backup'];

            

            abstract = data['abstract'];

            titles = data['titles'];

            authors = data['authors'];

            journal = data['journal'];



            for (i = 0; i < x.length; i++) {

                if(abstract[i].includes(text_value) || 

                   titles[i].includes(text_value) || 

                   authors[i].includes(text_value) || 

                   journal[i].includes(text_value)) {

                    x[i] = x_backup[i];

                    y[i] = y_backup[i];

                } else {

                    x[i] = undefined;

                    y[i] = undefined;

                }

            }

            





        source.change.emit();

        """)



# option

option = RadioButtonGroup(labels=["C-0", "C-1", "C-2",

                                  "C-3", "C-4", "C-5",

                                  "C-6", "C-7", "C-8",

                                  "C-9", "C-10", "C-11",

                                  "C-12", "C-13", "C-14",

                                  "C-15", "C-16", "C-17",

                                  "C-18", "C-19", "All"], 

                          active=20, callback=callback)



# search box

keyword = TextInput(title="Search:", callback=keyword_callback)



#header

header = Div(text="""<h1>COVID-19 Articles Cluster</h1>""")



# show

show(column(header, widgetbox(option, keyword),p))
def get_similarity(search_string, results_returned = 3):

    example_text = Sentence(search_string)

    document_embeddings.embed(example_text)

    search_vect = example_text.get_embedding()

    search_vect = search_vect.cpu().detach().numpy()

    cosine_similarities = pd.Series(cosine_similarity([search_vect], flair_elmo_arr).flatten())

    output =""

    for i,j in cosine_similarities.nlargest(int(results_returned)).iteritems():

        output +='<p style="font-family:verdana; font-size:110%;"> '

        for i in sentences[i].split():

            if i.lower() in search_string:

                output += " <b>"+str(i)+"</b>"

            else:

                output += " "+str(i)

        output += "</p><hr>"



    output = '<h3>Results:</h3>'+output

    display(HTML(output))



text = widgets.Text(

    value='virus genetics, origin, and evolution',

    placeholder='Paste ticket description here!',

    description='Query:',

    disabled=False,

    layout=widgets.Layout(width='50%', height='50px')

)



out = widgets.Output()



def callback(_):

    with out:

        clear_output()

        # what happens when we press the button

        printmd("**<font color=orange> -------------------------------------------------------------------------------------------------------- </font>**")        

        printmd(f"**<font color=blue>Semantic Search has Started </font>**")

        get_similarity(text.value)

        printmd("**<font color=orange> -------------------------------------------------------------------------------------------------------- </font>**")        



text.on_submit(callback)

# displaying button and its output together

widgets.VBox([text, out])
# Install an End-To-End Closed Domain Question Answering System

!pip install cdqa
## Load Cdqa Utilities:

from ast import literal_eval



from cdqa.utils.filters import filter_paragraphs

from cdqa.pipeline import QAPipeline

from cdqa.utils.download import download_model
## Download BERT Squad 1.1 Pretrained Q&A Model

download_model(model='bert-squad_1.1', dir='./models')
## Converting body_text into different paragraphs :

df_covid["paragraphs"] = [x.split('\n') for x in df_covid["body_text"]]
df = filter_paragraphs(df_covid)

df.head()
cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib')

cdqa_pipeline.fit_retriever(df=df)
def get_cdqa_prediction(x):

    prediction = cdqa_pipeline.predict(x)

    question = '<h3>Question:</h3>'+x

    answer = '<h3>Answer:</h3>'+prediction[0]

    title = '<h3>Title:</h3>'+prediction[1]    

    paragraph = '<h3>Paragraph:</h3>'+prediction[2]    

    

    display(HTML(question))

    display(HTML(answer))

    display(HTML(title))

    display(HTML(paragraph))
text = widgets.Text(

    value='What do we know about diagnostics and surveillance?',

    placeholder='Paste ticket description here!',

    description='Question:',

    disabled=False,

    layout=widgets.Layout(width='50%', height='50px')

)



out = widgets.Output()



def callback(_):

    with out:

        clear_output()

        # what happens when we press the button

        printmd("**<font color=orange> ------------------------------------------------------------------------------------------------------------------------------- </font>**")        

        printmd(f"**<font color=blue>COVID-19 (Question & Answering System)</font>**")

        get_cdqa_prediction(text.value)

        printmd("**<font color=orange> ------------------------------------------------------------------------------------------------------------------------------- </font>**")        



text.on_submit(callback)

# displaying button and its output together

widgets.VBox([text, out])