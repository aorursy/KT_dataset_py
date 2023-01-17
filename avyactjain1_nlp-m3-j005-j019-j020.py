import re

import string

import numpy as np 

import random

import pandas as pd 

import nltk

nltk.download('stopwords')

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
! git clone https://github.com/avyactjain/nlpM3.git
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
print(train.shape)

print(test.shape)
train.dropna(inplace=True)

train.info()
train.head()
train.describe()
temp = train.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)

temp.style.background_gradient(cmap='Greens')
fig = go.Figure(go.Funnelarea(

    text =temp.sentiment,

    values = temp.text,

    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}

    ))

fig.show()
def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
train['text'] = train['text'].apply(lambda x:clean_text(x))

train['selected_text'] = train['selected_text'].apply(lambda x:clean_text(x))
train['temp_list'] = train['selected_text'].apply(lambda x:str(x).split())

top = Counter([item for sublist in train['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')
def remove_stopword(x):

    return [y for y in x if y not in stopwords.words('english')]

train['temp_list'] = train['temp_list'].apply(lambda x:remove_stopword(x))
fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 

             width=700, height=700,color='Common_words')

fig.show()
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), color = 'white',

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'u', "im"}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color=color,

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=1980, 

                    height=1020,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  
plot_wordcloud(train[train['sentiment'] == 'neutral']['text'],color='white',max_font_size=100,title_size=100,title="WordCloud of Neutral Tweets")
plot_wordcloud(train[train['sentiment'] == 'positive']['text'],color='white',max_font_size=100,title_size=100,title="WordCloud of Positive Tweets")
plot_wordcloud(train[train['sentiment'] == 'negative']['text'],color='white',max_font_size=100,title_size=100,title="WordCloud of Negative Tweets")
df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
df_train['Num_words_text'] = df_train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main Text in train set
df_train = df_train[df_train['Num_words_text']>=3]
def save_model(output_dir, nlp, new_model_name):



    output_dir = '/kaggle/working/working/' + output_dir

    if output_dir is not None:        

        if not os.path.exists(output_dir):

            os.makedirs(output_dir)

        nlp.meta["name"] = new_model_name

        nlp.to_disk(output_dir)

        print("Saved model to", output_dir)





def train(train_data, output_dir, n_iter=20, model=None):



    if model is not None:

        nlp = spacy.load(output_dir)  # load existing spaCy model

        print("Loaded model '%s'" % model)

    else:

        nlp = spacy.blank("en")  # create blank Language class

        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:

        ner = nlp.create_pipe("ner")

        nlp.add_pipe(ner, last=True)

    else:

        ner = nlp.get_pipe("ner")

    

    for _, annotations in train_data:

        for ent in annotations.get("entities"):

            ner.add_label(ent[2])



    # get names of other pipes to disable them during training

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):  # only train NER

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



    model_out_path = None

    if sentiment == 'positive':

        model_out_path = 'models/model_pos'

    elif sentiment == 'negative':

        model_out_path = 'models/model_neg'

    return model_out_path





def get_training_data(sentiment):



    train_data = []

    for index, row in df_train.iterrows():

        if row.sentiment == sentiment:

            selected_text = row.selected_text

            text = row.text

            start = text.find(selected_text)

            end = start + len(selected_text)

            train_data.append((text, {"entities": [[start, end, 'selected_text']]}))

    return train_data



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
sentiment = 'positive'



train_data = get_training_data(sentiment)

model_path = get_model_out_path(sentiment)



train(train_data, model_path, n_iter=3, model=None)
sentiment = 'negative'



train_data = get_training_data(sentiment)

model_path = get_model_out_path(sentiment)



train(train_data, model_path, n_iter=3, model=None)
selected_texts = []

MODELS_BASE_PATH = '/kaggle/working/nlpM3/saved_models/'

# MODELS_BASE_PATH = '/kaggle/working/working/models/'



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

        

df_test['selected_text(output)'] = selected_texts
df_test.head()
df_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

df_submission['selected_text'] = df_test['selected_text(output)']

display(df_submission.head(10))

df_submission.to_csv("submission.csv", index=False)
