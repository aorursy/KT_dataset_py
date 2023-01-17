# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import re

import string

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from collections import Counter

import spacy

from tqdm import trange

import random

from spacy.util import compounding,minibatch

from sklearn.feature_extraction.text import CountVectorizer

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from plotly.subplots import make_subplots

import plotly.graph_objects as go



stop = stopwords.words('english')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
print('Train Shape:',train.shape)

print('Test Shape:',test.shape)

train.head()
print('Sentiment of text : {} \nOur training text :\n{}\nSelected text which we need to predict:\n{}'.format(train['sentiment'][1],train['text'][1],train['selected_text'][1]))
train.isnull().sum()
train.dropna(inplace=True)
train.sentiment.describe()
# distribution of tweets by sentiment in the training set

temp=train.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)

temp.style.background_gradient(cmap='Blues')
fig=make_subplots(1,2,subplot_titles=('Train set','Test set'))

x=train.sentiment.value_counts()

fig.add_trace(go.Bar(x=x.index,y=x.values,marker_color=['#3368d4','#32ad61','#f24e4e'],name='train'),row=1,col=1)

x=test.sentiment.value_counts()

fig.add_trace(go.Bar(x=x.index,y=x.values,marker_color=['#3368d4','#32ad61','#f24e4e'],name='test'),row=1,col=2)
def Jaccard_similarity(str1,str2):

    a = set(str1.lower().split())

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c))/(len(a)+len(b)-len(c))
str1 = 'My name is Kevin'

str2 = 'Myself Kevin'

jaccard_score = Jaccard_similarity(str1,str2)

print('Jaccard Score :',jaccard_score)
def Jaccard_similarity(df):

    a = set(df['text'].lower().split())

    b = set(df['selected_text'].lower().split())

    c = a.intersection(b)

    return float(len(c))/(len(a)+len(b)-len(c))
train['jaccard_score'] = train.apply(Jaccard_similarity,axis=1)
train['no_words_st'] = train.selected_text.apply(lambda x: len(str(x).split()))

train['no_words_t'] = train.text.apply(lambda x: len(str(x).split()))

train['diff_words']  = train['no_words_t'] - train['no_words_st']                                                
train.head()
#Distribution of Length b/w selected_text and text'

plt.hist(train['no_words_st'],bins=20,label='selected_text')

plt.hist(train['no_words_t'],bins=20,label='text')

plt.title('Distribution of Length b/w selected_text and text')

plt.legend()

plt.show()
plt.figure(figsize=(8,6))

sns.kdeplot(train['no_words_st'],shade=True,color='b')

sns.kdeplot(train['no_words_t'],shade=True,color='r')

plt.title('Distribution of Length')

plt.show()
plt.figure(figsize=(8,6))

sns.kdeplot(train[train['sentiment']=='positive']['diff_words'],shade=True,color='b',label='diff_words_pos')

sns.kdeplot(train[train['sentiment']=='negative']['diff_words'],shade=True,color='r',label='diff_words_neg')

plt.title('Distribution of Differnce in length of Positive words & Negative Words')

plt.show()
plt.figure(figsize=(8,6))

sns.kdeplot(train[train['sentiment']=='positive']['jaccard_score'],shade=True,color='b',label='Jaccard_score_pos')

sns.kdeplot(train[train['sentiment']=='negative']['jaccard_score'],shade=True,color='r',label='Jaccard_score_neg')

plt.title('Distribution of Jaccard Score of Positive words , Negative Words & Neutral Words')

plt.show()
train[train['sentiment']=='neutral']['jaccard_score'].describe()
plt.figure(figsize=(12,6))

sns.boxplot(train[train['sentiment']=='neutral']['jaccard_score'])

plt.show()
plt.plot(train[train['sentiment']=='neutral']['jaccard_score'],'r+')

plt.show()
def clean_text(text):



    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
def clean_text1(text):



    # tokenize text and remove puncutation

    text = [word.strip(string.punctuation) for word in text.split(" ")]

    # remove words that contain numbers

    text = [word for word in text if not any(c.isdigit() for c in word)]

    # remove stop words

    text = [x for x in text if x not in stop]

    # remove empty tokens

    text = [t for t in text if len(t) > 0]

    # remove words with only one letter

    text = [t for t in text if len(t) > 1]

    # join all

    text = " ".join(text)

    return(text)
train['text'] = train['text'].apply(str).apply(lambda x: clean_text(x))

train['selected_text'] = train.selected_text.apply(str).apply(lambda x: clean_text(x))
train['cleaned_text'] = train['text'].apply(lambda x: clean_text1(x))

train['cleaned_selected_text'] = train.selected_text.apply(lambda x: clean_text1(x))
train.head(3)
word_token = word_tokenize("".join(train['cleaned_selected_text']))

print(word_token[:50])
most_comman_token_15 = Counter(word_token).most_common(15)

most_comman_token_15_df = pd.DataFrame(most_comman_token_15)

most_comman_token_15_df.columns = ['Word','Count']

most_comman_token_15_df.style.background_gradient(cmap='Blues')
def plot_wordcloud(text,mask=None,max_words=400,max_font_size=100,figure_size=(24.0,16.0),title=None,title_size=40,image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords={'u',"im"}

    stopwords=stopwords.union(more_stopwords)

    

    wordcloud = WordCloud(background_color='white',

                         stopwords = stopwords,max_words=max_words,

                         max_font_size=max_font_size,random_state=42,mask=mask)

    

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors),interpolation="bilinear");

        plt.title(title,fontdict={'size':title_size,

                                  'verticalalignment':'bottom'})

    else:

            plt.imshow(wordcloud);

            plt.title(title,fontdict={'size':title_size,'color':'red',

                                     'verticalalignment':'bottom'})

            plt.axis('off');

    plt.tight_layout()  

    

d = '../input/imagetc/'
positive_sentiment = train[train['sentiment']=='positive']

negative_sentiment = train[train['sentiment']=='negative']

neutral_sentiment = train[train['sentiment']=='neutral']
plt.figure(figsize=(8,6))

sns.kdeplot(neutral_sentiment['no_words_st'],shade=True,color='b',label='neu_no_words_st')

sns.kdeplot(neutral_sentiment['no_words_t'],shade=True,color='r',label='neu_no_words_t')

plt.title('Distribution of Number of words in selected text & text in neutral dataframe')

plt.show()
word_token_pos = word_tokenize("".join(positive_sentiment['cleaned_selected_text']))

print(word_token_pos[:50])
most_comman_token_15_pos = Counter(word_token_pos).most_common(15)

most_comman_token_15_pos_df = pd.DataFrame(most_comman_token_15_pos)

most_comman_token_15_pos_df.columns = ['Word','Count']

most_comman_token_15_pos_df.style.background_gradient(cmap='Blues')
twitter_mask=np.array(Image.open(d+'twitter.png'))

plot_wordcloud(positive_sentiment.text,mask=twitter_mask,max_font_size=80,title_size=30,title="WordCloud for Positive tweets")
word_token_neg = word_tokenize("".join(negative_sentiment['cleaned_selected_text']))

print(word_token_neg[:50])
most_comman_token_15_neg = Counter(word_token_neg).most_common(15)

most_comman_token_15_neg_df = pd.DataFrame(most_comman_token_15_neg)

most_comman_token_15_neg_df.columns = ['Word','Count']

most_comman_token_15_neg_df.style.background_gradient(cmap='Reds')
twitter_mask=np.array(Image.open(d+'twitter.png'))

plot_wordcloud(negative_sentiment.text,mask=twitter_mask,max_font_size=80,title_size=30,title="WordCloud for Negative tweets")
word_token_neu = word_tokenize("".join(neutral_sentiment['cleaned_selected_text']))

print(word_token_neu[:50])
most_comman_token_15_neu = Counter(word_token_neu).most_common(15)

most_comman_token_15_neu_df = pd.DataFrame(most_comman_token_15_neu)

most_comman_token_15_neu_df.columns = ['Word','Count']

most_comman_token_15_neu_df.style.background_gradient(cmap='Greens')
twitter_mask=np.array(Image.open(d+'twitter.png'))

plot_wordcloud(neutral_sentiment.text,mask=twitter_mask,max_font_size=80,title_size=30,title="WordCloud for Neutral tweets")
def get_top_n_words(corpus,n_grams=None):

    vec = CountVectorizer(ngram_range=(n_grams,n_grams)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    

    sum_of_words = bag_of_words.sum(axis=0)

    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    word_freq = sorted(word_freq, key = lambda x: x[1], reverse=True)

    return word_freq[:15]
top_n_bigrams = get_top_n_words(train['text'].dropna(),2)

x,y = map(list,zip(*top_n_bigrams))

plt.figure(figsize=(9,7))

sns.barplot(x=y,y=x)

plt.show()
top_n_bigrams = get_top_n_words(train['selected_text'].dropna(),2)

x,y = map(list,zip(*top_n_bigrams))

plt.figure(figsize=(9,7))

sns.barplot(x=y,y=x)

plt.show()
top_n_trigrams = get_top_n_words(train['text'].dropna(),3)

x,y = map(list,zip(*top_n_trigrams))

plt.figure(figsize=(9,7))

sns.barplot(x=y,y=x)

plt.show()
top_n_trigrams = get_top_n_words(train['selected_text'].dropna(),3)

x,y = map(list,zip(*top_n_trigrams))

plt.figure(figsize=(9,7))

sns.barplot(x=y,y=x)

plt.show()
top_n_trigrams_pos = get_top_n_words(positive_sentiment['text'].dropna(),3)

x,y = map(list,zip(*top_n_trigrams_pos))

plt.figure(figsize=(9,7))

sns.barplot(x=y,y=x)

plt.show()
top_n_trigrams_neg = get_top_n_words(negative_sentiment['text'].dropna(),3)

x,y = map(list,zip(*top_n_trigrams_neg))

plt.figure(figsize=(9,7))

sns.barplot(x=y,y=x)

plt.show()
top_n_trigrams_neu = get_top_n_words(neutral_sentiment['text'].dropna(),3)

x,y = map(list,zip(*top_n_trigrams_neu))

plt.figure(figsize=(9,7))

sns.barplot(x=y,y=x)

plt.show()
data_copy = train.copy()

data_train = data_copy[data_copy['no_words_t']>=3]
data_train.head()
def get_training_data(sentiment):

    train_data=[]

    

    '''

    Returns Training data in the format needed to train spacy NER

    '''

    for index,row in data_train.iterrows():

        if row.sentiment == sentiment:

            selected_text = row.cleaned_selected_text

            text = row.text

            start = text.find(selected_text)

            end = start + len(selected_text)

            train_data.append((text, {"entities": [[start,end,'selected_text']]}))

    return train_data
def training(train_data, output_dir, n_iter=20, model=None):

    """Load the model,set up the pipeline and train the entity recognizer"""

    if model is not None:

        nlp=sapcy.load(model) #load existing spaCy model

        print("Loaded model '%s'" %model)

    else:

        nlp = spacy.blank("en") #create blank Language class

        print("Created blank 'en' model ")

        

        # The pipeline execution

        # Create the built-in pipeline components and them to the pipeline

        # nlp.create_pipe works for built-ins that are registered in the spacy

        

        if "ner" not in nlp.pipe_names:

            ner = nlp.create_pipe("ner")

            nlp.add_pipe(ner,last=True)

            

        # otherwise, get it so we can add labels

        

        else:

            ner = nlp.get_pipe("ner")

            

        # add labels 

        for _, annotations in train_data:

                for ent in annotations.get("entities"):

                    ner.add_label(ent[2])

        

        # get names of other pipes to disable them during training

        

        pipe_exceptions = ["ner","trf_wordpiecer","trf_tok2vec"]

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

        

        with nlp.disable_pipes(*other_pipes): # training of only NER

            

            # reset and intialize the weights randoml - but only if we're

            # training a model

            

            if model is None:

                nlp.begin_training()

            else:

                nlp.resume_training()

            

            for itn in trange(n_iter):

                random.shuffle(train_data)

                losses={}

                

                # batch up the example using spaCy's mnibatch

                batches = minibatch(train_data,size=compounding(4.0,1000.0,1.001))

                #print(batches)

                for batch in batches:

                    texts , annotations = zip(*batch)

                    nlp.update(

                        texts, #batch of texts

                        annotations, # batch of annotations

                        drop = 0.5,  # dropout - make it harder to memorise data

                        losses = losses,

                )

            print("Losses", losses)

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

        print("Saved model to",output_dir)
sentiment ='positive'

train_data = get_training_data(sentiment)

model_path = get_model_path(sentiment)

training(train_data,model_path,n_iter=3,model=None)
sentiment ='negative'

train_data = get_training_data(sentiment)

model_path = get_model_path(sentiment)

training(train_data,model_path,n_iter=3,model=None)
MODEL_PATH = '/kaggle/working/models/'

MODEL_PATH_POS = MODEL_PATH + 'model_pos'

MODEL_PATH_NEG = MODEL_PATH + 'model_neg'
def predict(text,model):

    docx = model(text)

    ent_arr=[]

    for ent in docx.ents:

        #print(ent.text)

        start = text.find(ent.text)

        end = start + len(ent.text)

        entity_arr = [start,end,ent.label_]

        if entity_arr not in ent_arr:

            ent_arr.append(entity_arr)

    selected_text = text[ent_arr[0][0]:ent_arr[0][1]] if len(ent_arr)>0 else text

    return selected_text
selected_text=[]

if MODEL_PATH is not None:

    print("Loading Models  from ", MODEL_PATH)

    model_pos = spacy.load(MODEL_PATH_POS)

    model_neg = spacy.load(MODEL_PATH_NEG)

    for index,row in test.iterrows():

        text = row.text.lower()

        if row.sentiment == 'neutral':

            selected_text.append(text)

        elif row.sentiment == 'positive':

            selected_text.append(predict(text,model_pos))

        else:

            selected_text.append(predict(text,model_neg))       
assert len(test.text) == len(selected_text)

submission['selected_text'] = selected_text

submission.to_csv('submission.csv',index=False)
from IPython.core.display import HTML



def multi_table(table_list):

    ''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell

    '''

    return HTML(

        '<table><tr style="background-color:white;">' + 

        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +

        '</tr></table>'

    )
multi_table([test.head(10),submission.head(10)])