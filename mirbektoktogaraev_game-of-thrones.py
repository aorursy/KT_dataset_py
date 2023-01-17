import numpy as np 

import pandas as pd

import plotly.express as px ## Visualization

import plotly.graph_objects as go ## Visualization

import matplotlib.pyplot as plt ## Visualization

import plotly as py ## Visualization

from wordcloud import WordCloud, STOPWORDS ## To create word clouds from script

import os

%config IPCompleter.greedy=True

import nltk

from nltk.corpus import stopwords

from  nltk.stem import SnowballStemmer

import re

import gensim

from gensim.models import word2vec

from sklearn.manifold import TSNE



# WORD2VEC 

W2V_SIZE = 300

W2V_WINDOW = 7

W2V_EPOCH = 32

W2V_MIN_COUNT = 10



os.chdir("../input/game-of-thrones-script-all-seasons/")
#read dataset

df = pd.read_csv('Game_of_Thrones_Script.csv')
df.head()
df.info()
#Change date format

df.loc[:,'Release Date'] = pd.to_datetime(df['Release Date'])



df['Year'] = df['Release Date'].dt.year

df['Month'] = df['Release Date'].dt.month

month_mapper = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',

               7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

df.loc[:,'Month'] = df['Month'].map(month_mapper)
#Some preprocessing

stop_words = stopwords.words("english") #A stop.word is a commonly used word (“the”, “a”, “an”, “in”)

stemmer = SnowballStemmer("english") #A stemming algorithm reduces the words “chocolates”, “choco” to the root word,“chocolate”



# TEXT CLENAING

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):

    # Remove link,user and special characters

    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()

    tokens = []

    for token in text.split():

        if token not in stop_words:

            if stem:

                tokens.append(stemmer.stem(token))

            else:

                tokens.append(token)

    return " ".join(tokens)

#Apply function

df.Sentence = df.Sentence.apply(lambda x: preprocess(x))
#Function to count words

def counter(strng):

    counter = 0

    wordlist = strng.split()

    for word in wordlist:

        counter +=1

    return counter 

#We will create a new columns to check most talkative character

df['count_words'] = df.Sentence.apply(lambda x: counter(x))
#We will drop some occasional characters.

characters_drop = ['man', 'women', 'boy','girl', 'old man',]

df =df[-df['Name'].isin(characters_drop)] 
#Total dialogues by Seasons

temp = df['Season'].value_counts().reset_index()

temp.columns=['Season', 'Counts']

temp.sort_values(by='Season', inplace=True)

fig = px.bar(temp, 'Season', 'Counts')

fig.update_layout(

    autosize=False,

    width=1000,

    height=600,

    title={

        

        'text': "Total dialougue counts in season.",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    bargap=0.2, # gap between bars of adjacent location coordinates

    bargroupgap=0.1 # gap between bars of the same location coordinates

)

fig.show()
plt.rcParams["figure.figsize"] = (15,10)

temp = df.groupby(['Season','Episode'])['count_words'].sum().unstack().plot(kind='bar', fill = 'count_words',stacked=True)

plt.title("Number of words by Episodes in all Seasons", fontsize=20)

plt.show()
#Most frequent 20 words

from collections import Counter

words = Counter(" ".join(df["Sentence"]).split()).most_common(20)



names, values = zip(*words)

fig = px.bar(x=names, y=values, labels={'x':'words', 'y':'count'})

fig.update_layout(

    autosize=False,

    width=1000,

    height=600,

    title={

        

        'text': "Most frequently used 20 words through all Seasons",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    bargap=0.2, # gap between bars of adjacent location coordinates

    bargroupgap=0.1 # gap between bars of the same location coordinates

)

fig.show()
#Most common words in GOT using wordCloud

wordcloud = WordCloud(width=1600, height=800, min_font_size=10, background_color ='#add8e6').generate(

    ' '.join(i for i in df['Sentence']))

plt.figure(figsize = (12, 12), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title("Most frequently used words through all Seasons", fontsize=20)

plt.show()
#Most common 20 words of tyrion_lannister

tyrion_lannister = df[df['Name']=='tyrion lannister']

words = Counter(" ".join(tyrion_lannister["Sentence"]).split()).most_common(20)



names, values = zip(*words)

fig = px.bar(x=names, y=values, labels={'x':'words', 'y':'count'})

fig.update_layout(

    autosize=False,

    width=1000,

    height=600,

    title={

        

        'text': "Most frequently used 20 words of Tyrion Lannister",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    bargap=0.2, # gap between bars of adjacent location coordinates

    bargroupgap=0.1 # gap between bars of the same location coordinates

)

fig.show()
#Most common words of tyrion_lannister using wordCloud



wordcloud = WordCloud(width=1600, height=800, min_font_size=10, background_color ='#add8e6').generate(

    ' '.join(i for i in tyrion_lannister['Sentence']))

plt.figure(figsize = (12, 12), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)

plt.title("Most frequently used words of Tyrion Lannister", fontsize=20)

plt.show()
#Most common 20 words of daenerys_targaryen

daenerys_targaryen = df[df['Name']=='daenerys targaryen']

words = Counter(" ".join(daenerys_targaryen["Sentence"]).split()).most_common(20)



names, values = zip(*words)

fig = px.bar(x=names, y=values, labels={'x':'words', 'y':'count'})

fig.update_layout(

    autosize=False,

    width=1000,

    height=600,

    title={

        

        'text': "Most frequently used 20 words of Daenerys Targaryen",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    bargap=0.2, # gap between bars of adjacent location coordinates

    bargroupgap=0.1 # gap between bars of the same location coordinates

)

fig.show()
#Most common words of daenerys_targaryen using wordCloud

daenerys_targaryen = df[df['Name']=='daenerys targaryen']

wordcloud = WordCloud(width=1600, height=800, min_font_size=10, background_color ='#add8e6').generate(

    ' '.join(i for i in daenerys_targaryen['Sentence']))

plt.figure(figsize = (12, 12), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.title("Most frequently used words of Daenerys Targaryen", fontsize=20)

plt.tight_layout(pad = 0) 

plt.show() 
#"20 characters with most dialogues"

temp = df['Name'].value_counts().reset_index()

temp.columns=['Character', 'No of Dialouges']

fig = px.bar(temp.head(20), 'Character', 'No of Dialouges')

fig.update_layout(

    autosize=False,

    width=1000,

    height=600,

    title={

        

        'text': "20 characters that took part in the most dialogues",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    bargap=0.2, # gap between bars of adjacent location coordinates

    bargroupgap=0.1 # gap between bars of the same location coordinates

)

fig.show()
#20 character with most words in dialogues

plt.xticks(rotation ="horizontal")

p = df.groupby("Name")['count_words'].sum().sort_values(ascending=False)[:20].plot(kind='bar',figsize=(30,12), fontsize = 20)

plt.title("Characters by number of words", fontsize=20)

plt.xticks(rotation=-90)

plt.show()
#I want to find the most important words for each of the main charatcers.

#daenerys_targaryen using tf-idf

from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer()

x = tfidf.fit_transform(daenerys_targaryen.Sentence)



feature_array = np.array(tfidf.get_feature_names())

tfidf_sorting = np.argsort(x.toarray()).flatten()[::-1]



n = 20

top_n = feature_array[tfidf_sorting][:n]



text = ' '.join(top_n)



# Create a cloud image:

wordcloud = WordCloud(width=1600, height=800,min_font_size=10, background_color ='#add8e6').generate(text)



# Display the generated image:

plt.figure(figsize = (12, 12), facecolor = None) 

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.title("Most important 20 words of Daenerys Targaryen", fontsize=20)

plt.tight_layout(pad = 0) 

plt.show()
#Tyrion_lannister important words.

#tyrion_lannister using tf-idf



tfidf = TfidfVectorizer()

x = tfidf.fit_transform(tyrion_lannister.Sentence)



feature_array = np.array(tfidf.get_feature_names())

tfidf_sorting = np.argsort(x.toarray()).flatten()[::-1]



n = 20

top_n = feature_array[tfidf_sorting][:n]



text = ' '.join(top_n)



# Create a cloud image:

wordcloud = WordCloud(width=1600, height=800,min_font_size=10, background_color ='#add8e6').generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.title("Most important 20 words of Tyrion Lannister", fontsize=20)



plt.axis("off")

plt.show()
#Jon Snow important words.

jon_snow = df[df['Name']=='jon snow']



tfidf = TfidfVectorizer()

x = tfidf.fit_transform(jon_snow.Sentence)



feature_array = np.array(tfidf.get_feature_names())

tfidf_sorting = np.argsort(x.toarray()).flatten()[::-1]



n = 30

top_n = feature_array[tfidf_sorting][:n]

text = ' '.join(top_n)



# Create a cloud image:

wordcloud = WordCloud(width=1600, height=800,min_font_size=10, background_color ='#add8e6').generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.title("Most important 20 words of Jon Snow", fontsize=20)

plt.axis("off")

plt.show()
#Cersei Lannister important words.

cersei_lannister = df[df['Name']=='cersei lannister']



tfidf = TfidfVectorizer()

x = tfidf.fit_transform(cersei_lannister.Sentence)



feature_array = np.array(tfidf.get_feature_names())

tfidf_sorting = np.argsort(x.toarray()).flatten()[::-1]



n = 30

top_n = feature_array[tfidf_sorting][:n]

text = ' '.join(top_n)



# Create a cloud image:

wordcloud = WordCloud(width=1600, height=800,min_font_size=10, background_color ='#add8e6').generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.title("Most important 20 words of Cersei Lannister", fontsize=20)

plt.axis("off")

plt.show()
#Build a corpus for the word2vec model

def build_corpus(data):

    "Creates a list of lists containing words from each sentence"

    corpus = []

    for sentence in data:

        word_list = sentence.split(" ")

        corpus.append(word_list)    

           

    return corpus

#cersei_lannister Words TSNE



corpus = build_corpus(cersei_lannister.Sentence) 

model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 

                                            window=W2V_WINDOW, 

                                            min_count=W2V_MIN_COUNT, 

                                            workers=4)

model.build_vocab(corpus)



# define the function to compute the dimensionality reduction

# and then produce the biplot

def tsne_plot(model):

    "Creates a TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(18, 18)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

        plt.title("Cersei Lannister Words TSNE", fontsize=20)



    plt.show()

    

# call the function on our dataset

tsne_plot(model)
#jon_snow Words TSNE



corpus = build_corpus(jon_snow.Sentence) 

model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 

                                            window=W2V_WINDOW, 

                                            min_count=W2V_MIN_COUNT, 

                                            workers=4)

model.build_vocab(corpus)



# define the function to compute the dimensionality reduction

# and then produce the biplot

def tsne_plot(model):

    "Creates a TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(18, 18)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

        plt.title("Jon Snow Words TSNE", fontsize=20)



    plt.show()

    

# call the function on our dataset

tsne_plot(model)
#daenerys_targaryen Words TSNE



corpus = build_corpus(daenerys_targaryen.Sentence) 

model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 

                                            window=W2V_WINDOW, 

                                            min_count=W2V_MIN_COUNT, 

                                            workers=4)

model.build_vocab(corpus)



# define the function to compute the dimensionality reduction

# and then produce the biplot

def tsne_plot(model):

    "Creates a TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(18, 18)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

        plt.title("Daenerys Targaryen Words TSNE", fontsize=20)



    plt.show()

    

# call the function on our dataset

tsne_plot(model)
#tyrion_lannister Words TSNE



corpus = build_corpus(tyrion_lannister.Sentence)   

model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 

                                            window=W2V_WINDOW, 

                                            min_count=W2V_MIN_COUNT, 

                                            workers=4)

model.build_vocab(corpus)



# define the function to compute the dimensionality reduction

# and then produce the biplot

def tsne_plot(model):

    "Creates a TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(18, 18)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

        plt.title("Tyrion Lannister Words TSNE", fontsize=20)



    plt.show()

    

# call the function on our dataset

tsne_plot(model)
#We will create new dataframe to do bar chart race.

df_new = pd.DataFrame(index=np.arange(30),columns=["Name"])



#Get 30 main character names from our big dataset 

temp = df['Name'].value_counts().reset_index()[:30]

temp.columns=['Character', 'No of Dialouges']

names = temp.Character

#Asign column with sorted names

for i in range(len(names)):

    df_new.loc[i,["Name"]] = temp.Character[i]

    

df_new.sort_values('Name', inplace=True)

df_new = df_new.reset_index(drop=True)

del temp



#We will create 2 lists to iterate trough loop

seasons = ['Season 1', 'Season 2', 'Season 3', 'Season 4', 

           'Season 5', 'Season 6', 'Season 7', 'Season 8']

episodes = ['Episode 1','Episode 2','Episode 3','Episode 4',

            'Episode 5','Episode 6','Episode 7','Episode 8','Episode 9','Episode 10']



temp_df = df.groupby(['Season', 'Episode', 'Name'])['count_words'].sum().reset_index()

temp_df = temp_df[temp_df.Name.isin(names)]

#We will get words sums for our characters from each episode and all Seasons and join to new_df

for season in seasons:

    for episode in episodes:

            tempor = temp_df[(temp_df['Season']==season) & (temp_df['Episode']== episode)]

            tempor = tempor.drop(columns=['Season', 'Episode'])

            tempor = tempor.rename(columns={"count_words": season + " " + episode})

            df_new = df_new.merge(tempor, how='left', on='Name')



#We should delete some columns since some Seasons has less Episodes.

col_delete = ['Season 7 Episode 8', 'Season 7 Episode 9', 'Season 7 Episode 10',

              'Season 8 Episode 7','Season 8 Episode 8', 'Season 8 Episode 9', 

              'Season 8 Episode 10' ]



df_new = df_new.drop(columns=col_delete)

#Fill NAN with 0

df_new = df_new.fillna(0)



#Now we should change each number of word to cumulative value n = n-1 + n

for i in df_new.index:

    zero = 0

    for x in range(len(df_new.columns)-1):

        df_new.iloc[i,x+1] = zero + df_new.iloc[i,x+1]

        zero = df_new.iloc[i,x+1]



#Now we will save our df to try on flourish studio

#df_new.to_csv('GOT_with_count_words.csv')
import IPython

url = "https://flo.uri.sh/visualisation/1937774/embed"

iframe = '<iframe src=' + url + ' width=950 height=600></iframe>'

IPython.display.HTML(iframe)