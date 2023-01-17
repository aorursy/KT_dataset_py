import numpy as np 
import pandas as pd
import glob
import json
import re
import os
import warnings 

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
from bokeh.models import Div
from bokeh.models import Paragraph
from bokeh.layouts import column, widgetbox

## IPython Utilities
from IPython.display import HTML

import notebook as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output, VBox

from IPython.html import widgets
from IPython.display import display, Image, HTML, Markdown, clear_output
!python -m spacy download en_core_web_md
## Load Spacy Utilities:
import spacy
import en_core_web_md
nlp = en_core_web_md.load()
#Load metadata 
root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()
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
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
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
## Taking only 5000 articles for analysis:
df_covid = df_covid.head(5000)
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
## Considering only 200 articles for analysis:
require_text = text_arr[:200]
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
import notebook as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output, VBox

from IPython.html import widgets
from IPython.display import display, Image, HTML, Markdown, clear_output
text = widgets.Text(
    value='What do we know about diagnostics and surveillance?',
    placeholder='your question here',
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
