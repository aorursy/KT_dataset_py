#importing necessary libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from plotly import graph_objs as go

import plotly.figure_factory as ff

import re

import string

from collections import Counter

import plotly.express as px



import nltk

from nltk.corpus import stopwords

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



from tqdm import tqdm

import os

import nltk

import spacy

import random

from spacy.util import compounding

from spacy.util import minibatch



import warnings

warnings.filterwarnings("ignore")



# Importing and Loading the data into data frame

train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sample_sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
train_df
test_df
##checking shape of dataframe

print(train_df.shape)

print(test_df.shape)
train_df.info()
train_df.describe()
#Identifying Missing Values in Column

train_df.isnull().sum()
#dropping missing value

train_df.dropna(inplace=True)
#Identifying Missing Values in Column

train_df.isnull().sum()
#Identifying Missing Values in Column

test_df.isnull().sum()
#Count of texts in each category of sentiments

temp = train_df.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)

temp.style.background_gradient(cmap='Greens')
plt.figure(figsize=(10,5))

sns.countplot(x='sentiment',data=train_df)
#funnel chart for visualization

fig = go.Figure(go.Funnelarea(

    text =temp.sentiment,

    values = temp.text,

    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}

    ))

fig.show()
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
jaccard_out=[]



for ind,row in train_df.iterrows():

    str1 = row.text

    str2 = row.selected_text



    jaccard_score = jaccard(str1,str2)

    jaccard_out.append([str1,str2,jaccard_score])

# merging the above jaccard_score with orginal dataset

jaccard = pd.DataFrame(jaccard_out,columns=["text","selected_text","jaccard_score"])

train_df = train_df.merge(jaccard,how='outer')

train_df
#creating a column for Difference In Number Of words of Selected_text and Text





#Number Of words in whole text

train_df['Count_text'] = train_df['text'].apply(lambda x:len(str(x).split()))



#Number Of words in Selected Text

train_df['Count_ST'] = train_df['selected_text'].apply(lambda x:len(str(x).split()))

#Difference in Number of words text and Selected Text

train_df['diff_count_words'] = train_df['Count_text'] - train_df['Count_ST'] 

train_df
hist_data = [train_df['Count_ST'],train_df['Count_text']]



group_labels = ['Selected_Text', 'Text']



# Create distplot with custom bin_size

fig = ff.create_distplot(hist_data, group_labels,show_curve=False)

fig.update_layout(title_text='Distribution of Number Of words in Selected_Text & Text')

fig.update_layout(

    autosize=False,

    width=900,

    height=600,

  

)

fig.show()
#Distribution of Number Of words in Selected_Text & Text using Kernel Distribution

#the tweets having number of words greater than 25 are very less and thus the number of words distribution plot is right skewed

plt.figure(figsize=(10,5))

p1=sns.kdeplot(train_df['Count_ST'], shade=True, color="r").set_title('Kernel Distribution of Number Of words')

p1=sns.kdeplot(train_df['Count_text'], shade=True, color="b")
# This plot will show the comparision of the negative(red) and postive(blue) setiments based on Difference between Number Of words of "text" and "selected text"

#jaccard_score is 1 if there is no difference

plt.figure(figsize=(12,6))

p1=sns.kdeplot(train_df[train_df['sentiment']=='positive']['diff_count_words'], shade=True, color="b").set_title('Kernel Distribution of Difference in Number Of words')

p2=sns.kdeplot(train_df[train_df['sentiment']=='negative']['diff_count_words'], shade=True, color="r")

plt.legend(labels=['diff_count_words_positive','diff_count_words_negative'])

#p3=sns.kdeplot(train_df[train_df['sentiment']=='neutral']['diff_count_words'], shade=True, color="g") #RuntimeError: Selected KDE bandwidth is 0. Cannot estimate density.
#neutral KDE bandwidth is 0. Cannot estimate density.

#therefore plotting using distplot

plt.figure(figsize=(10,5))

sns.distplot(train_df[train_df['sentiment']=='neutral']['diff_count_words'],kde=False)
#performing the above operation for jaccard_score

#jaccard_score is more for more number of matching words

plt.figure(figsize=(10,5))

p1=sns.kdeplot(train_df[train_df['sentiment']=='positive']['jaccard_score'], shade=True, color="b").set_title('KDE of Jaccard Scores across different Sentiments')

p2=sns.kdeplot(train_df[train_df['sentiment']=='negative']['jaccard_score'], shade=True, color="r")

plt.legend(labels=['positive','negative'])
#for neutral

plt.figure(figsize=(10,5))

sns.distplot(train_df[train_df['sentiment']=='neutral']['jaccard_score'],kde=False)
# find those clusters

df = train_df[train_df['Count_text']<=2]

df

#jaccord_score mean for each sentiment 

df.groupby('sentiment').mean()['jaccard_score']
#having a view of positive sentiments

df[df['sentiment']=='positive']
#cleaning the corpus

#Make text lowercase, remove text in square brackets,remove links,remove punctuation and remove words containing numbers.

def clean_text(text):

   

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
train_df['text'] = train_df['text'].apply(lambda x:clean_text(x))

train_df['selected_text'] = train_df['selected_text'].apply(lambda x:clean_text(x))
train_df.head()
#Most Common words in our Target-Selected Text

train_df['temp_list'] = train_df['selected_text'].apply(lambda x:str(x).split())

top = Counter([item for sublist in train_df['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')
fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 

             width=700, height=700,color='Common_words')

fig.show()
#While we cleaned our dataset we didnt remove the stop words and hence we can see the most coomon word is 'to' .

def remove_stopword(x):

    return [y for y in x if y not in stopwords.words('english')]

train_df['temp_list'] = train_df['temp_list'].apply(lambda x:remove_stopword(x))
top = Counter([item for sublist in train_df['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp = temp.iloc[1:,:]

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Purples')
fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')

fig.show()
#most common word in text

train_df['temp_list1'] = train_df['text'].apply(lambda x:str(x).split()) #List of words in every row for text

train_df['temp_list1'] = train_df['temp_list1'].apply(lambda x:remove_stopword(x)) #Removing Stopwords
top = Counter([item for sublist in train_df['temp_list1'] for item in sublist])

temp = pd.DataFrame(top.most_common(25))

temp = temp.iloc[1:,:] # removed first common word  I'm  and took data from second row

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')
#SO we can see the Most common words in Selected text and Text are almost the same,which was obvious

fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Text', orientation='h', 

             width=700, height=700,color='Common_words')

fig.show()
#most common word sentiment wise

Positive_sent = train_df[train_df['sentiment']=='positive']

Negative_sent = train_df[train_df['sentiment']=='negative']

Neutral_sent = train_df[train_df['sentiment']=='neutral']
#MosT common positive words

top = Counter([item for sublist in Positive_sent['temp_list'] for item in sublist])

temp_positive = pd.DataFrame(top.most_common(20))

temp_positive.columns = ['Common_words','count']

temp_positive.style.background_gradient(cmap='Greens')
fig = px.bar(temp_positive, x="count", y="Common_words", title='Most Commmon Positive Words', orientation='h', 

             width=700, height=700,color='Common_words')

fig.show()
#MosT common negative words

top = Counter([item for sublist in Negative_sent['temp_list'] for item in sublist])

temp_negative = pd.DataFrame(top.most_common(20))

temp_negative = temp_negative.iloc[1:,:]

temp_negative.columns = ['Common_words','count']

temp_negative.style.background_gradient(cmap='Reds')
fig = px.treemap(temp_negative, path=['Common_words'], values='count',title='Tree Of Most Common Negative Words')

fig.show()
#MosT common Neutral words

top = Counter([item for sublist in Neutral_sent['temp_list'] for item in sublist])

temp_neutral = pd.DataFrame(top.most_common(20))

temp_neutral = temp_neutral.loc[1:,:]

temp_neutral.columns = ['Common_words','count']

temp_neutral.style.background_gradient(cmap='Reds')
fig = px.bar(temp_neutral, x="count", y="Common_words", title='Most Commmon Neutral Words', orientation='h', 

             width=700, height=700,color='Common_words')

fig.show()
fig = px.treemap(temp_neutral, path=['Common_words'], values='count',title='Tree Of Most Common Neutral Words')

fig.show()
#We can see words like get,go,dont,got,u,cant,lol,like are common in all three segments . 

#That's interesting because words like dont and cant are more of negative nature and words like lol are more of positive nature.

#Does this mean our data is incorrectly labelled , we will have more insights on this after N-gram analysis

#It will be interesting to see the word unique to different sentiments



#unique word in each segment

raw_text = [word for word_list in train_df['temp_list1'] for word in word_list]
def words_unique(sentiment,numwords,raw_words):

    '''

    Input:

        segment - Segment category (ex. 'Neutral');

        numwords - how many specific words do you want to see in the final result; 

        raw_words - list  for item in train_data[train_data.segments == segments]['temp_list1']:

    Output: 

        dataframe giving information about the name of the specific ingredient and how many times it occurs in the chosen cuisine (in descending order based on their counts)..



    '''

    allother = []

    for item in train_df[train_df.sentiment != sentiment]['temp_list1']:

        for word in item:

            allother .append(word)

    allother  = list(set(allother ))

    

    specificnonly = [x for x in raw_text if x not in allother]

    

    mycounter = Counter()

    

    for item in train_df[train_df.sentiment == sentiment]['temp_list1']:

        for word in item:

            mycounter[word] += 1

    keep = list(specificnonly)

    

    for word in list(mycounter):

        if word not in keep:

            del mycounter[word]

    

    Unique_words = pd.DataFrame(mycounter.most_common(numwords), columns = ['words','count'])

    

    return Unique_words
#positive tweet

Unique_Positive= words_unique('positive', 20, raw_text)

print("The top 20 unique words in Positive Tweets are:")

Unique_Positive.style.background_gradient(cmap='Greens')
fig = px.treemap(Unique_Positive, path=['words'], values='count',title='Tree Of Unique Positive Words')

fig.show()
from palettable.colorbrewer.qualitative import Pastel1_7

plt.figure(figsize=(16,10))

my_circle=plt.Circle((0,0), 0.7, color='white')

plt.pie(Unique_Positive['count'], labels=Unique_Positive.words, colors=Pastel1_7.hex_colors)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title('DoNut Plot Of Unique Positive Words')

plt.show()
Unique_Negative= words_unique('negative', 10, raw_text)

print("The top 10 unique words in Negative Tweets are:")

Unique_Negative.style.background_gradient(cmap='Reds')
from palettable.colorbrewer.qualitative import Pastel1_7

plt.figure(figsize=(16,10))

my_circle=plt.Circle((0,0), 0.7, color='white')

plt.rcParams['text.color'] = 'black'

plt.pie(Unique_Negative['count'], labels=Unique_Negative.words, colors=Pastel1_7.hex_colors)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title('DoNut Plot Of Unique Negative Words')

plt.show()
Unique_Neutral= words_unique('neutral', 10, raw_text)

print("The top 10 unique words in Neutral Tweets are:")

Unique_Neutral.style.background_gradient(cmap='Oranges')
from palettable.colorbrewer.qualitative import Pastel1_7

plt.figure(figsize=(16,10))

my_circle=plt.Circle((0,0), 0.7, color='white')

plt.pie(Unique_Neutral['count'], labels=Unique_Neutral.words, colors=Pastel1_7.hex_colors)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title('DoNut Plot Of Unique Neutral Words')

plt.show()
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

    for index, row in df_train.iterrows():

        if row.sentiment == sentiment:

            selected_text = row.selected_text

            text = row.text

            start = text.find(selected_text)

            end = start + len(selected_text)

            train_data.append((text, {"entities": [[start, end, 'selected_text']]}))

    return train_data
#training model for positive and negative tewwts

sentiment = 'positive'



train_data = get_training_data(sentiment)

model_path = get_model_out_path(sentiment)

# For DEmo Purposes I have taken 3 iterations you can train the model as you want

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

MODELS_BASE_PATH = '../input/models/models/'

#kaggle/input/models/models/



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