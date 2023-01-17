# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

!pip install chart_studio

!pip install textstat



# Visualisation Library

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#visualisation libraries

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import chart_studio.plotly as py

import plotly.figure_factory as ff

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')





#ml

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#standard libraries

import emoji

import re

import string



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_new=pd.read_json('/kaggle/input/indonesiandata/emotion_id_tweets.json',lines=True)

df_new.head()
ex_label=[]

for i in range(df_new['annotation'].shape[0]):

    if df_new['annotation'][i]['labels']==[]:

        ex_label.append('no_emotion')

    else:

        ex_label.append(df_new.annotation[i]['labels'][0])
df=pd.DataFrame()

df['content']=df_new['content']

df['emotion']=ex_label

df.head()
df.describe()
df['len']=df['content'].astype(str).apply(len)

df.head()
df=df.drop_duplicates(subset='content',keep='first')

df.shape
def missing_value_of_data(data):

    total=data.isnull().sum().sort_values(ascending=False)

    percentage=round(total/data.shape[0]*100,2)

    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

missing_value_of_data(df)

def count_values_in_column(data,feature):

    total=data.loc[:,feature].value_counts(dropna=False)

    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)

    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

count_values_in_column(df,'emotion')

name_of_emotion=df.emotion.unique()

value_of_emotion=list(df.emotion.value_counts())



fig=go.Figure([go.Bar(x=list(name_of_emotion), y=value_of_emotion)])

fig.show()
!pip install translators
import translators as ts
def ngrams_top(corpus,ngram_range,n=None):

    """

    List the top n words in a vocabulary according to occurrence in a text corpus.

    """

    vec = CountVectorizer(stop_words = 'english',ngram_range=ngram_range).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    total_list=words_freq[:n]

    df=pd.DataFrame(total_list,columns=['text','count'])

    

    english_ngram=[] #translating indonesian to english

    for i  in range(10):

        english_ngram.append(ts.google(df['text'][i], 'auto', 'en'))

      

 

    # Plotting Grams and their english conversion   

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(

        go.Bar(x=df['count'][::-1],

                    y=df['text'][::-1],

                   name='Indonesian',

                    marker_color='rgb(55, 83, 109)',

                    orientation='h'

                    ),

        row=1, col=1

    )

    fig.add_trace(

        go.Bar(x=df['count'][::-1],

                    y=english_ngram[::-1],

                   name='English',

                    marker_color='rgb(123, 67, 199)',

                    orientation='h'

                    ),

        row=1, col=2

    )

    fig.update_layout(height=600, width=2000, title_text=str(ngram_range[0])+" grams for Indonesian/English")

    fig.show()
ngrams_top(df['content'],(1,1),n=10)

ngrams_top(df['content'],(2,2),n=10)

ngrams_top(df['content'],(3,3),n=10)
joy=df[df['emotion']=='joy']

trust=df[df['emotion']=='trust']

fear=df[df['emotion']=='fear']

anger=df[df['emotion']=='anger']

anticipation=df[df['emotion']=='anticipation']

disgust=df[df['emotion']=='disgust']

sadness=df[df['emotion']=='sadness']

surprise=df[df['emotion']=='surprise']
joy['len'].iplot(

    kind='hist',

    bins=100,

    xTitle='text length',

    linecolor='black',

    color='red',

    yTitle='count',

    title='Joy Text Length Distribution')

trust['len'].iplot(

    kind='hist',

    bins=100,

    xTitle='text length',

    linecolor='black',

    color='green',

    yTitle='count',

    title='Trust Text Length Distribution')



fear['len'].iplot(

    kind='hist',

    bins=100,

    xTitle='text length',

    linecolor='black',

    color='pink',

    yTitle='count',

    title='Fear Text Length Distribution')



anger['len'].iplot(

    kind='hist',

    bins=100,

    xTitle='text length',

    linecolor='black',

    color='orange',

    yTitle='count',

    title='Anger Text Length Distribution')



anticipation['len'].iplot(

    kind='hist',

    bins=100,

    xTitle='text length',

    linecolor='black',

    yTitle='count',

    title='Anticipation Text Length Distribution')

disgust['len'].iplot(

    kind='hist',

    bins=100,

    xTitle='text length',

    linecolor='black',

    

    yTitle='count',

    title='Disgust Text Length Distribution')

sadness['len'].iplot(

    kind='hist',

    bins=100,

    xTitle='text length',

    linecolor='black',

    color='gold',

    yTitle='count',

    title='Sadness Text Length Distribution')

surprise['len'].iplot(

    kind='hist',

    bins=100,

    xTitle='text length',

    linecolor='black',

    color='blue',

    yTitle='count',

    title='Surprise Text Length Distribution')
def ngrams_tops(corpus,ngram_range,emotions,n=None):

    """

    List the top n words in a vocabulary according to occurrence in a text corpus.

    """

    vec = CountVectorizer(stop_words = 'english',ngram_range=ngram_range).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    total_list=words_freq[:n]

    df=pd.DataFrame(total_list,columns=['text','count'])

    

    english_ngram=[] #translating indonesian to english

    for i  in range(10):

        english_ngram.append(ts.google(df['text'][i], 'auto', 'en'))

      

 

    # Plotting Grams and their english conversion   

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(

        go.Bar(x=df['count'][::-1],

                    y=df['text'][::-1],

                   name='Indonesian',

                    marker_color='rgb(55, 83, 109)',

                    orientation='h'

                    ),

        row=1, col=1

    )

    fig.add_trace(

        go.Bar(x=df['count'][::-1],

                    y=english_ngram[::-1],

                   name='English',

                    marker_color='rgb(123, 67, 199)',

                    orientation='h'

                    ),

        row=1, col=2

    )

    fig.update_layout(height=600, width=2000,title_text="Most Used words in "+str(emotions).capitalize())

    fig.show()



    

def most_used(emotions):

    emotion=df.loc[df['emotion']==str(emotions)]



    _1gram_emotion=ngrams_tops(emotion['content'],(1,1),emotions,n=10)

    

    return _1gram_emotion

   

        
most_used('joy')
most_used('fear')
most_used('trust')
most_used('surprise')
most_used('anger')
most_used('anticipation')
most_used('disgust')
most_used('sadness')
def find_emoji(text):

    emo_text=emoji.demojize(text)

    line=re.findall(r'\:(.*?)\:',emo_text)

    return line

sentence="I love ⚽ very much 😁"

find_emoji(sentence)



# Emoji cheat sheet - https://www.webfx.com/tools/emoji-cheat-sheet/

# Uniceode for all emoji : https://unicode.org/emoji/charts/full-emoji-list.html
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



sentence="Its all about \U0001F600 face"

print(sentence)

remove_emoji(sentence)
df['content']=df['content'].apply(lambda x: remove_emoji(x))
def rep(text):

    grp = text.group(0)

    if len(grp) > 1:

        return grp[0:1] # can change the value here on repetition

    return grp

   

def unique_char(rep,sentence):

    convert = re.sub(r'(\w)\1+', rep, sentence) 

    return convert



sentence="heyyy this is loooong textttt"

unique_char(rep,sentence)

df['content']=df['content'].apply(lambda x : unique_char(rep,x))