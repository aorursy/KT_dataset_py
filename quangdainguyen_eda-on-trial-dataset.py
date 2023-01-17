import re
import string
import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


import nltk
from nltk.corpus import stopwords

from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
trial = pd.read_csv('../input/dataset-for-toxic-span-detection/tsd_trial.csv')
spans = trial.spans
text = trial.text
spans[0]
import json
for i in range(0, len(spans)):
    spans[i] = json.loads(spans[i])
selected = []
for n in range(0, len(spans)):
    temp = ""
    for i in range(0, len(spans[n])):
        if i>0 and spans[n][i]-spans[n][i-1] != 1:
            temp += ' '
        temp += text[n][spans[n][i]]
    selected.append(temp)
selected_text = pd.DataFrame(selected, columns = ['selected_text'])
trial2 = pd.concat([trial, selected_text], axis = 1, sort = False)
trial2
trial2 = trial2.drop(['spans'], axis = 1)
#Bỏ cột spans
"""
trial2['temp_list'] = trial2['Selected text'].apply(lambda x:str(x).split())
top = Counter([item for sublist in trial2['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')
"""
def save_model(output_dir, nlp, new_model_name):
    ''' This Function Saves model to 
    given output directory'''
    
    output_dir = f'../working/{output_dir}'
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
# pass model = nlp if you want to train on top of existing model 

def train(train_data, output_dir, n_iter=20, model=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    ""
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
    
    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()


        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts,  # batch of texts
                            annotations,  # batch of annotations
                            drop=0.5,   # dropout - make it harder to memorise data
                            losses=losses, 
                            )
            print("Losses", losses)
    save_model(output_dir, nlp, 'st_ner')
def get_model_out_path():
    model_out_path = 'models/model_neg'
    return model_out_path
#modify here
def get_training_data():
    '''
    Returns Trainong data in the format needed to train spacy NER
    '''
    train_data = []
    for index, row in trial2.iterrows():
        selected_text = row.selected_text
        text = row.text
        start = text.find(selected_text)
        end = start + len(selected_text)
        train_data.append((text, {"entities": [[start, end, 'selected_text']]}))
    return train_data
train_data = get_training_data()
model_path = get_model_out_path()

train(train_data, model_path, n_iter=30, model=None)
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text
MODELS_BASE_PATH = './models/'
model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')
selected_texts = []
for index, row in trial2.iterrows():
        text = row.text
        output_str = ""
        selected_texts.append(predict_entities(text, model_neg))
selected_texts = pd.DataFrame(selected_texts, columns = ['output3'])
trial2 = pd.concat([trial2, selected_texts], axis = 1, sort = False)
trial2.head(n = 30)
trial2.head(n = 30)
