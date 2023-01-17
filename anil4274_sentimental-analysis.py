def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))





Sentence_1 = 'Today is Friday'

Sentence_2 = 'Tomorrow is Saturday'

Sentence_3 = 'Day After Tomorrow is Sunday'



print(jaccard(Sentence_1,Sentence_2))

print(jaccard(Sentence_1,Sentence_3))

print(jaccard(Sentence_2,Sentence_3))
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'



#!pip install chart_studio

#!pip install textstat



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# text processing libraries

import re #regular expression

import string #strings manipulation

import nltk #natural language toolkit

from nltk.corpus import stopwords #stopwords

from tqdm import tqdm

import spacy

from spacy.util import compounding

from spacy.util import minibatch





# Visualisation libraries

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import iplot

from collections import Counter

from string import *

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



# sklearn 

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



# File system manangement

import os



# Pytorch

import torch



#Transformers

from transformers import BertTokenizer



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')



import random
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
print(train.shape)

print(test.shape)
train.info()
test.info()
train.head()
train.describe()
train['sentiment'].value_counts()
train['NoOfSelectedTextWords'] = train['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text

train['NoOfTextWords'] = train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main text

train['DiffOfTextWordsToSelectedTextWords'] = train['NoOfTextWords'] - train['NoOfSelectedTextWords'] #Difference in Number of words text and Selected Text
train.head()
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
train['text'] = train['text'].apply(lambda x:clean_text(x))

train['selected_text'] = train['selected_text'].apply(lambda x:clean_text(x))
train.head()
train['temp_list'] = train['selected_text'].apply(lambda x:str(x).split())
def remove_stopword(x):

    return [y for y in x if y not in stopwords.words('english')]

train['temp_list'] = train['temp_list'].apply(lambda x:remove_stopword(x))
top = Counter([item for sublist in train['temp_list'] for item in sublist])

df = pd.DataFrame(top.most_common(20))

df = df.iloc[1:,:]

df.columns = ['commonwords','count']

df.style.background_gradient(cmap='Purples')
def text_preprocessing(text):

    """

    Parsing the text and removing stop words.



    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(tokenized_text)

    return combined_text
positive_text = train[train['sentiment'] == 'positive']['selected_text']

negative_text = train[train['sentiment'] == 'negative']['selected_text']

neutral_text = train[train['sentiment'] == 'neutral']['selected_text']
positive_text_clean = positive_text.apply(lambda x: text_preprocessing(x))

negative_text_clean = negative_text.apply(lambda x: text_preprocessing(x))

neutral_text_clean = neutral_text.apply(lambda x: text_preprocessing(x))
train['sentiment'].value_counts().iplot(kind='bar',yTitle='Percentage', 

                                                      linecolor='black', 

                                                      opacity=0.7,

                                                      color='red',

                                                      theme='pearl',

                                                      bargap=0.6,

                                                      gridcolor='white',

                                                      title='Distribution of Sentiment column from the train dataset')
test['sentiment'].value_counts().iplot(kind='bar',yTitle='Percentage', 

                                                      linecolor='black', 

                                                      opacity=0.7,

                                                      color='green',

                                                      theme='pearl',

                                                      bargap=0.6,

                                                      gridcolor='white',

                                                      title='Distribution  of Sentiment column from the test dataset')
from palettable.colorbrewer.qualitative import Pastel1_7

plt.figure(figsize=(16,10))

my_circle=plt.Circle((0,0), 0.7, color='white')

plt.rcParams['text.color'] = 'black'

plt.pie(df['count'], labels=df['commonwords'], colors=Pastel1_7.hex_colors)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title('commonwords')

plt.show()
from wordcloud import WordCloud

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[30, 15])

# Positive sentiment visualizing

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(positive_text_clean))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Positive text',fontsize=40);



# Negative sentiment visualizing

wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(negative_text_clean))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Negative text',fontsize=40);



# Neutral sentiment visualizing

wordcloud3 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(neutral_text_clean))

ax3.imshow(wordcloud3)

ax3.axis('off')

ax3.set_title('Neutral text',fontsize=40);
from plotly import graph_objs as go



fig = go.Figure(go.Funnelarea(

    text =train['sentiment'].value_counts().index,

    values = train['sentiment'].value_counts().values,

    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}

    ))

fig.show()
import plotly.express as px

fig = px.violin(train, y="NoOfSelectedTextWords", color="sentiment",

                violinmode='overlay', # draw violins on top of each other

                # default violinmode is 'group' as in example above

                hover_data=train)

fig.show()
import seaborn as sns

plt.figure(figsize=(12,6))

p1=sns.kdeplot(train['NoOfSelectedTextWords'], shade=True, color="r").set_title('Kernel Distribution of Number Of words')

p1=sns.kdeplot(train['NoOfTextWords'], shade=True, color="b")
import plotly.express as px

fig = px.treemap(df, path=['commonwords'], values='count',title='Tree of Most Common Words')

fig.show()
fig = px.bar(df, x="count", y="commonwords", title='Commmon Words in Text', orientation='h', width=700, height=700, color='commonwords')

fig.show()
sns.jointplot(x=train['NoOfTextWords'], y=train['NoOfSelectedTextWords'], kind="kde")
import plotly.express as px

#df = px.data.iris()

fig = px.scatter(train, x="NoOfTextWords", y="NoOfSelectedTextWords", color="sentiment",

                 size='DiffOfTextWordsToSelectedTextWords', hover_data=train)

fig.show()
import plotly.express as px

fig = px.histogram(train, x="NoOfTextWords", y="NoOfSelectedTextWords", color="sentiment", marginal="rug",

                   hover_data=train)

fig.show()
from nltk import FreqDist



fdist=FreqDist()



for word in train['selected_text'].values:

    fdist[word.lower()]+=1

fdist_top20=fdist.most_common(20)

df = pd.DataFrame(fdist_top20, columns =['commonwords', 'count']) 
df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

df_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
df_train['Num_words_text'] = df_train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main Text in train set
df_train = df_train[df_train['Num_words_text']>=3]
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

    for index, row in df_train.iterrows():

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

#print(train_data)

# For Demo Purposes I have taken 3 iterations you can train the model as you want

train(train_data, model_path, n_iter=3, model=None)
sentiment = 'negative'



train_data = get_training_data(sentiment)

model_path = get_model_out_path(sentiment)



train(train_data, model_path, n_iter=3, model=None)
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

MODELS_BASE_PATH = '../working/models/'



if MODELS_BASE_PATH is not None:

    print("Loading Models  from ", MODELS_BASE_PATH)

    model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')

    model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')

        

    for index, row in df_test.iterrows():

        text = row.text

        output_str = ""

        if row.sentiment == 'neutral' or len(text.split()) <= 2:

            selected_texts.append(text)

        elif row.sentiment == 'positive':

            selected_texts.append(predict_entities(text, model_pos))

        else:

            selected_texts.append(predict_entities(text, model_neg))

        

df_test['selected_text'] = selected_texts
df_submission['selected_text'] = df_test['selected_text']

df_submission.to_csv("submission.csv", index=False)

display(df_submission.head(10))