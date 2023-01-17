# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

print("Training Examples : {}".format(len(train)))

print("Testing Examples : {}".format(len(test)))
train.head()
#Importing libraries for EDA

import plotly.graph_objects as go

from plotly.subplots import make_subplots
#Data preparation

class_dist = train.groupby(["sentiment"]).count()['textID'].reset_index()

#Lets look at the distribution of classes

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {}]])

fig.add_trace(go.Pie(labels=class_dist.sentiment, values=class_dist.textID, name="% Distribution",hole=.5),

              1, 1)

fig.add_trace(go.Bar(x = class_dist.sentiment, y=class_dist.textID, name="Frequency Distribution"),

              1, 2)

fig.update_layout(title="% Distribution and Frequency Disstibution")
import math

import re

from collections import Counter



WORD = re.compile(r"\w+")



def text_to_vector(text):

    words = WORD.findall(text)

    return Counter(words)



def get_cosine(text1, text2):

    vec1 = text_to_vector(text1)

    vec2 = text_to_vector(text2)   

    intersection = set(vec1.keys()) & set(vec2.keys())

    numerator = sum([vec1[x] * vec2[x] for x in intersection])



    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])

    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])

    denominator = math.sqrt(sum1) * math.sqrt(sum2)



    if not denominator:

        return 0.0

    else:

        return float(numerator) / denominator



#testing the cosine similarity

str1 = train.text[0]

str2 = train.selected_text[0]



print("String 1 : {}".format(str1))

print("String 2 : {}".format(str2))

cosine_score = get_cosine(str1,str2) 

print("Cosine Similarity of the above two sentences : {}%".format(np.round(cosine_score*100)))
train['COSINE_Score'] = train.apply(lambda row:get_cosine(str(row['text']),str(row['selected_text'])),axis=1)

train.head()
import plotly.express as px

fig = px.box(train,x='sentiment',y='COSINE_Score',color='sentiment')

fig.update_traces(quartilemethod="inclusive") # or "inclusive", or "linear" by default

fig.show()
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
#Training Model



# pass model = nlp if you want to train on top of existing model 



from tqdm import tqdm

import os

import nltk

import spacy

import random

from spacy.util import compounding

from spacy.util import minibatch



def train_model(train_data, output_dir, n_iter=20, model=None):

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
def get_training_data(df_train):

    '''

    Returns Trainong data in the format needed to train spacy NER

    '''

    train_data = []

    for index, row in df_train.iterrows():

        sentiment = row.sentiment #Store sentiment here

        selected_text = str(row.selected_text)

        text = str(row.text)

        start = text.find(selected_text)

        end = start + len(selected_text)

        train_data.append((text, {"entities": [[start, end, sentiment]]})) #sentiment as training

    return train_data
model_path = '/models/model'

train_data = get_training_data(train)

train_model(train_data,model_path,n_iter=3,model=None)
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
selected_texts = []

MODELS_BASE_PATH = '/kaggle/working/models/model'



if MODELS_BASE_PATH is not None:

    print("Loading Models  from ", MODELS_BASE_PATH)

    model = spacy.load(MODELS_BASE_PATH)        

    for index, row in test.iterrows():

        text = row.text

        output_str = ""

        selected_texts.append(predict_entities(text, model))

          

test['selected_text'] = selected_texts
df_submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

df_submission['selected_text'] = test['selected_text']

df_submission.to_csv("submission.csv", index=False)

display(df_submission.head(10))