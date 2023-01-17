import pandas as pd

import re

from textblob import TextBlob



#test process lib

import re

import string

import nltk

from nltk.corpus import stopwords



import spacy



data= pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

data.head()



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#visualize lib

stop = stopwords.words('english')

from nltk.tokenize import word_tokenize

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go



from tqdm import tqdm

import random

import plotly.express as px

import plotly.figure_factory as ff

from plotly.offline import iplot

from collections import Counter

from string import *

import cufflinks



import tokenizers

import torch

from spacy.util import compounding



#transformers to tokenization

from transformers import BertTokenizer

from tqdm import trange



from spacy.util import minibatch

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#file sys mangmnt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd



data= pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

print(data.shape)

data.head()
data.info()
data.describe()
data['sentiment'].value_counts()
data['NoOfSelectedTextWords'] = data['text'].apply(lambda x:len(str(x).split()))                #Number Of words in Selected Text           

data['NoOfTextWords'] = data['text'].apply(lambda x:len(str(x).split()))                                 #Number Of words in main text

data['DifferenceOfTextWordsToSelectedTextWords'] = data['NoOfTextWords'] - data['NoOfSelectedTextWords'] #Difference in Number of words text and Selected Text
data.head()
import matplotlib.pyplot as pt

activities= ['nutral', 'positive', 'negative']

slices=[1430,1103,1001]

colors=['c','b','m']

pt.pie(slices, labels=activities, colors=colors, startangle=90, radius=2.0, autopct= '%1.1f%%')

pt.legend()

pt.show()
display= pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

print(display.shape)

display.head()
display.info()
display.describe()
display['sentiment'].value_counts()
import matplotlib.pyplot as pt

activities= ['nutral', 'positive', 'negative']

slices=[11118,8582,7781]

colors=['r','y','g']

pt.pie(slices, labels=activities, colors=colors, startangle=90, radius=2.0, autopct= '%1.1f%%')

pt.legend()

pt.show()
display['NoOfSelectedTextWords'] = display['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text

display['NoOfTextWords'] = display['text'].apply(lambda x:len(str(x).split()))                  #Number Of words in main text

display['DifferenceOfTextWordsToSelectedTextWords'] = display['NoOfTextWords'] - display['NoOfSelectedTextWords'] #Difference in Number of words text and Selected Text
display.head()
display['temp_list'] = display['selected_text'].apply(lambda x:str(x).split())
def remove_stopword(x):

    return [y for y in x if y not in stopwords.words('english')]

display['temp_list'] = display['temp_list'].apply(lambda x:remove_stopword(x))
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

sns.countplot(data['sentiment'])
plt.figure(figsize=(10,6))

sns.countplot(display['sentiment'])
from collections import Counter

data['text'].apply(lambda x:str(x).split())

top = Counter([item for sublist in data['sentiment'] for item in sublist])

temp = pd.DataFrame(top.most_common(30))

temp.columns = ['Similar!!letters','count']

temp.style.background_gradient(cmap='Purples')
from collections import Counter

display['temp_list'] = display['selected_text'].apply(lambda x:str(x).split())

top = Counter([item for sublist in display['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(30))

temp.columns = ['Similar!!words','count']

temp.style.background_gradient(cmap='Reds')
from plotly import graph_objs as gpy

import plotly.express as ppy

import plotly.figure_factory as fpy

fig = ppy.bar(temp, x="Similar!!words", y="count", title='Similar!!words in Selected_Text', orientation='v', 

             width=700, height=700, color='Similar!!words')

fig.show()
neutral = display[display['sentiment'] == 'neutral']

positive = display[display['sentiment'] == 'positive']

negative = display[display['sentiment'] == 'negative']
tweets_length = display['text'].apply(lambda x:len(str(x)))



sns.distplot(tweets_length)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def wc(df, text = 'text'):

    

    # Join all tweets in one string

    corpus = " ".join(str(review) for review in df[text])

    print (f"There are {len(corpus)} words in the combination of all review.")

    

    wc = WordCloud(max_font_size=50, 

                          max_words=100, 

                          background_color="white").generate(corpus)

    

    plt.figure(figsize=(15,15))

    plt.imshow(wc, interpolation="bilinear")

    plt.axis("off")

    plt.show()



wc(df = display)
wc(df = display, text = 'selected_text')
wc(df = neutral, text = 'text')
wc(df = positive, text = 'text')
wc(df = negative, text = 'text')
def clean_text(text):

    '''Convert text to lowercase,remove punctuation, remove words containing numbers, ,remove links and remove text in square brackets,.'''

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    return text
display['text'] = display['text'].apply(lambda x:clean_text(x))

display['selected_text'] = display['selected_text'].apply(lambda x:clean_text(x))
display.head()
def CLEAN_TEXT(text):



    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\s+|www\.\s+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', ' ', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
stop = stopwords.words('english')

def CLEAN_TEXT1(text):



    # TOKENIZE TEXT AND REMOVE PUNCUTATION

    text = [word.strip(string.punctuation) for word in text.split(" ")]

    # REMOVE WORDS THAT CONTAIN NUMBERS

    text = [word for word in text if not any(c.isdigit() for c in word)]

    # REMOVE STOP WORDS

    text = [x for x in text if x not in stop]

    # REMOVE EMPTY TOKENS

    text = [t for t in text if len(t) > 0]

    # REMOVE WORDS WITH ONLY ONE LETTER

    text = [t for t in text if len(t) > 1]

    # JOIN ALL

    text = " ".join(text)

    return(text)
display['text'] = display['text'].apply(str).apply(lambda x: CLEAN_TEXT1(x))

display['selected_text'] = display.selected_text.apply(str).apply(lambda x: CLEAN_TEXT1(x))
display['CLEANED_TEXT'] = display['text'].apply(lambda x: CLEAN_TEXT1(x))

display['CLEANED_SELECTED_TEXT'] = display.selected_text.apply(lambda x: CLEAN_TEXT1(x))
display.head(90)
data['text'] = display['text'].apply(str).apply(lambda x: CLEAN_TEXT1(x))

data['CLEANED_TEXT'] = data['text'].apply(lambda x: CLEAN_TEXT1(x))

data.head(50)
submission = '../input/tweet-sentiment-extraction/sample_submission.csv'

submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
model_path = '/kaggle/working/models/'

model_path_pos = model_path + 'model_pos'

model_path_neg = model_path + 'model_neg'
def predict(text,model):

    docx = model(text)

    ent_arr=[]

    for ent in docx.ents:

        #PRINT(ENT.TEXT)

        start = text.find(ent.text)

        end = start + len(ent.text)

        entity_arr = [start,end,ent.label_]

        if entity_arr not in ent_arr:

            ent_arr.append(entity_arr)

    selected_text = text[ent_arr[0][0]:ent_arr[0][1]] if len(ent_arr)>0 else text

    return selected_text

from IPython.core.display import HTML

def multi_table(table_list):

    ''' ACCEPS A LIST OF IPYTABLE OBJECTS AND RETURNS A TABLE WHICH CONTAINS EACH IPYTABLE IN A CELL

    '''

    return HTML(

        '<TABLE><TR STYLE="BACKGROUND-COLOR:WHITE;">' + 

        ''.join(['<TD>' + table._repr_html_() + '</TD>' for table in table_list]) +

        '</TR></TABLE>'

    )

multi_table([data.head(10),submission.head(10)])
temp = pd.DataFrame(top.most_common(20))

temp = temp.iloc[1:,:]

temp.columns = ['Common_words','count']





from palettable.colorbrewer.qualitative import Pastel1_7

plt.figure(figsize=(16,10))

my_circle=plt.Circle((0,0), 0.7, color='red')

plt.rcParams['text.color'] = 'blue'

plt.pie(temp['count'], labels=temp['Common_words'], colors=Pastel1_7.hex_colors)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title('Common_words')

plt.show()
from plotly import graph_objs as go



fig = go.Figure(go.Funnelarea(

    text =display['sentiment'].value_counts().index,

    values = display['sentiment'].value_counts().values,

    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}

    ))

fig.show()

def get_training_data(sentiment):

    display_data=[]

    

    '''

    RETURNS TRAINING DATA IN THE FORMAT NEEDED TO TRAIN SPACY NER

    '''

    for index,row in display.iterrows():

        if row.sentiment == sentiment:

            selected_text = row.CLEANED_SELECTED_TEXT

            text = row.text

            start = text.find(selected_text)

            end = start + len(selected_text)

            display_data.append((text, {"entities": [[start,end,'selected_text']]}))

    return display_data
def training(display_data, output_dir, n_iter=20, model=None):

    """LOAD THE MODEL,SET UP THE PIPELINE AND TRAIN THE ENTITY RECOGNIZER"""

    if model is not None:

        nlp=spacy.load(model) #LOAD EXISTING SPACY MODEL

        print("LOADED MODEL '%S'" %model)

    else:

        nlp = spacy.blank("en") #CREATE BLANK LANGUAGE CLASS

        print("CREATED BLANK 'en' MODEL ")

        

        # THE PIPELINE EXECUTION

        # CREATE THE BUILT-IN PIPELINE COMPONENTS AND THEM TO THE PIPELINE

        # NLP.CREATE_PIPE WORKS FOR BUILT-INS THAT ARE REGISTERED IN THE SPACY

        

        if "ner" not in nlp.pipe_names:

            ner = nlp.create_pipe("ner")

            nlp.add_pipe(ner,last=True)

            

             # OTHERWISE, GET IT SO WE CAN ADD LABELS

                

        else:

            ner = nlp.get_pipe("ner")

            

        # ADD LABELS 

        for _, annotations in display_data:

                for ent in annotations.get("entities"):

                    ner.add_label(ent[2])

                    # GET NAMES OF OTHER PIPES TO DISABLE THEM DURING TRAINING

        

        pipe_exceptions = ["ner","trf_wordpiecer","trf_tok2vec"]

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

        

        with nlp.disable_pipes(*other_pipes): # TRAINING OF ONLY NER

            

             # RESET AND INTIALIZE THE WEIGHTS RANDOML - BUT ONLY IF WE'RE

            # TRAINING A MODEL

            

            if model is None:

                nlp.begin_training()

            else:

                nlp.resume_training()

            

            for itn in trange(n_iter):

                random.shuffle(train_data)

                losses={}

                # BATCH UP THE EXAMPLE USING SPACY'S MNIBATCH

                batches = minibatch(train_data,size=compounding(4.0,1000.0,1.001))

                #PRINT(BATCHES)

                for batch in batches:

                    texts , annotations = zip(*batch)

                    nlp.update(

                        texts, #BATCH OF TEXTS

                        annotations, # BATCH OF ANNOTATIONS

                        drop = 0.5,  # DROPOUT - MAKE IT HARDER TO MEMORISE DATA

                         losses = losses,

                )

            print("losses", losses)

        save_model(output_dir, nlp, 'st_ner')

        

 
def get_model_path(sentiment):

    model_out_path = None 

    if sentiment == 'positive':

        model_out_path = 'models/model_pos'

    elif sentiment == 'negative':

        model_out_path = 'models/model_neg'

    return model_out_path
def save_model(output_dir,nlp,new_model_name):

    if output_dir is not None:

        if not os.path.exists(output_dir):

            os.makedirs(output_dir)

        nlp.meta["name"] = new_model_name

        nlp.to_disk(output_dir)

        print("SAVED MODEL TO",output_dir)
model_path = '/kaggle/working/models/'

model_path_pos = model_path + 'model_pos'

model_path_neg = model_path + 'model_neg'

from IPython.core.display import HTML

def multi_table(table_list):

    ''' ACCEPS A LIST OF IPYTABLE OBJECTS AND RETURNS A TABLE WHICH CONTAINS EACH IPYTABLE IN A CELL

    '''

    return HTML(

        '<TABLE><TR STYLE="BACKGROUND-COLOR:WHITE;">' + 

        ''.join(['<TD>' + table._repr_html_() + '</TD>' for table in table_list]) +

        '</TR></TABLE>'

    )
multi_table([data.head(10),submission.head(10)])
display = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
display['Num_words_text'] = display['text'].apply(lambda x:len(str(x).split())) #Number Of words in main Text in train set
display = display[display['Num_words_text']>=3]
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
def get_model_out_path(sentiment):

    '''

    Returns Model output path

    '''

    model_out_path = None

    if sentiment == 'positive':

        model_out_path = 'models/model_pos'

    elif sentiment == 'negative':

        model_out_path = 'models/model_neg'

    return model_out_path
def get_training_data(sentiment):

    '''

    Returns Trainong data in the format needed to train spacy NER

    '''

    train_data = []

    for index, row in display.iterrows():

        if row.sentiment == sentiment:

            selected_text = row.selected_text

            text = row.text

            start = text.find(selected_text)

            end = start + len(selected_text)

            train_data.append((text, {"entities": [[start, end, 'selected_text']]}))

    return train_data
sentiment = 'positive'



train_data = get_training_data(sentiment)

model_path = get_model_out_path(sentiment)

# For DEmo Purposes I have taken 3 iterations you can train the model as you want

train(train_data, model_path, n_iter=3, model=None)
sentiment = 'negative'



train_data = get_training_data(sentiment)

model_path = get_model_out_path(sentiment)



train(train_data, model_path, n_iter=3, model=None)
model_path = '/kaggle/working/models/'

model_path_pos = model_path + 'model_pos'

model_path_neg = model_path + 'model_neg'
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
selected_text=[]

if model_path is not None:

    print("LOADING MODELS  FROM ", model_path)

    model_pos = spacy.load(model_path_pos)

    model_neg = spacy.load(model_path_neg)

    for index,row in data.iterrows():

        text = row.text.lower()

        if row.sentiment == 'neutral':

            selected_text.append(text)

        elif row.sentiment == 'positive':

            selected_text.append(predict(text,model_pos))

        else:

            selected_text.append(predict(text,model_neg))       
assert len(data.text) == len(selected_text)

submission['selected_text'] = selected_text

submission.to_csv('submission.csv',index=False)
from IPython.core.display import HTML

def multi_table(table_list):

    ''' ACCEPS A LIST OF IPYTABLE OBJECTS AND RETURNS A TABLE WHICH CONTAINS EACH IPYTABLE IN A CELL

    '''

    return HTML(

        '<TABLE><TR STYLE="BACKGROUND-COLOR:WHITE;">' + 

        ''.join(['<TD>' + table._repr_html_() + '</TD>' for table in table_list]) +

        '</TR></TABLE>'

    )
multi_table([data.head(10),submission.head(10)])
fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words and Letters')

fig.show()