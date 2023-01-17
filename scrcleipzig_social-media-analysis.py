import pandas as pd 

import re

import unicodedata

import csv

import os

import nltk

import warnings



from wordcloud import WordCloud

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.stem.snowball import SnowballStemmer



from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



#Topic Modelling

from collections import Counter

from collections import OrderedDict #Ordena alfabeticamente

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation



#Sentiment Analysis

!pip install PyICU

!pip install pycld2

!pip install Morfessor

import polyglot

from polyglot.text import Text, Word

from polyglot.downloader import downloader

downloader.download("embeddings2.pt")

downloader.download("sentiment2.pt")

downloader.download("morph2.pt")





warnings.filterwarnings("ignore")

%matplotlib inline
BBFacebook1 = pd.read_csv("../input/BBFacebook1.csv")

BBFacebook2 = pd.read_csv("../input/BBFacebook2.csv")
BBFacebook1.head(5)
def concat(dataframe_array):

    data = pd.concat(dataframe_array)

    return data
dataset = [BBFacebook1,BBFacebook2]

data = concat(dataset)
#Checking the current dataset size

len(data)
# Tokenizer

def tokenize(text):

    return text.split()



# Stopwords

nltk_stopwords = set(stopwords.words('portuguese'))



def remove_stop_words(text, stopwords):

    for sw in stopwords:

        text = re.sub(r'\b%s\b' % sw, "", text)

        

    return text



# Special Characters

def remove_others(text):

    new_text = text.replace('"','')

    new_text = new_text.replace('(','')

    new_text = new_text.replace(')','')

    new_text = new_text.replace("'","")

    new_text = new_text.replace('%','')

    new_text = re.sub(r"u+h","", new_text)

    new_text = re.sub(r"o+h","", new_text)

    new_text = re.sub(r"a+h","", new_text)

    new_text = re.sub(r"ah+","", new_text)

    return new_text



# Lower Case Transformation

def lowercase(text):

    text = text.lower()

    return text



# Punctuation Removal

def remove_punctuation(text):  

    # re.sub(replace_expression, replace_string, target)

    new_text = re.sub(r"\.|,|;|:|-|!|\?|´|`|^|'", "", text)

    new_text = re.sub(r"[A-Z0-9]+|\.|\!|\,",'', new_text)

    new_text = re.sub(r"\$|\@|\(|\)|\&|\¨|\_|\=",'', new_text)

    new_text = re.sub(r"\\|\||\/|\>|\<|\[|\]|\{|\}",'', new_text)

    new_text = re.sub(r"\n|\t|\r",'', new_text)

    new_text = re.sub(r"\`|\´|\^|\%|\;|\:|\§|\ª|\º|\₢|\°",'', new_text)

    return new_text



# Accent Removal

def remove_accentuation(text):

    nfkd_form = unicodedata.normalize('NFKD', text)

    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])



# Numeric Character Removal

def remove_numbers(text):

    text = str(text)

    new_text = re.sub(r"[0-9]+", "", text)

    return new_text



# Stemming

def stemming(text):

    stemmer = SnowballStemmer("portuguese")

    word_list = text.split()

    result = []

    for w in word_list:

        result.append(stemmer.stem(w))

    result = " ".join(result)

    return result



#Emoji Removal

def remove_emojis(text):

    return text.encode('ascii', 'ignore').decode('ascii')





#Main Preprocessing Function

remove_emojis

def preprocessing(dataframe, fieldName, config):

    texts = []

    for tx in dataframe[fieldName].tolist():

        if config["lowercase"] == True:

            text = lowercase(tx)



        if config["remove_accentuation"] == True:

            text = remove_accentuation(text)



        if config["remove_punctuation"] == True:

            text = remove_punctuation(text)



        if config["remove_others"] == True:

            text = remove_others(text)



        if config["remove_numbers"] == True:

            text = remove_numbers(text)



        if config["remove_stopwords"] == True:

            text = remove_stop_words(text, stopwords=nltk_stopwords)



        if config["stemming"] == True:

            text = stemming(text)

            

        if config["remove_emojis"] == True:

            text = remove_emojis(text)

            

        texts.append(text)

        

    return texts
config = {  

            "lowercase": True, \

            "remove_accentuation": True, \

            "remove_punctuation": True, \

            "remove_others": True, \

            "remove_numbers": True, \

            "remove_stopwords": True,\

            "stemming":False,\

            "remove_emojis": True,\

         }
comments_pp = pd.DataFrame({'comments': preprocessing(data, fieldName='Comment', config=config)})
comments_pp.head(5)
def transformData(data, fieldName, my_tokenizer, weight):

    

    if weight == "TP":

        vectorizer = CountVectorizer(tokenizer=my_tokenizer, binary=True)

        X = vectorizer.fit_transform(data[fieldName])

    

    elif weight == "TF":

        vectorizer = CountVectorizer(tokenizer=my_tokenizer)

        X = vectorizer.fit_transform(data[fieldName])

        

    elif weight == "TFIDF":

        vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)

        X = vectorizer.fit_transform(data[fieldName])



    return (vectorizer, X)
# Calls the function responsible for vectorizing the data set

vectorizer, X = transformData(comments_pp, fieldName="comments", my_tokenizer=word_tokenize, weight="TFIDF")
feature_names = vectorizer.get_feature_names()

# Transforms our collection of text from the dataset to an Ordered Dictionary, like items like <index, text>.

text_collection = OrderedDict([(index, text) for index, text in enumerate(comments_pp["comments"])])



corpus_index = [n for n in text_collection]

# Creates a DataFrame from a Document-Term Matrix returned by the vectorizer.

bow = pd.DataFrame(X.todense(), index=corpus_index, columns=feature_names)
#Function and parameters that generate topic modeling

def LDA(bow,topics,n_top_words):

    feature_names = list(bow)

    model = LatentDirichletAllocation(n_components=topics, max_iter=10,

                                learning_method = 'online',

                                 learning_decay=0.9,

                                random_state = 4)

    model.fit(bow)

    

    for index, topic in enumerate(model.components_):

        message = "\nTopic #{}:".format(index)

        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])

        print(message)

        print("="*70)
LDA(bow,5,10)
data = comments_pp.comments.tolist()



# Polarity Extraction

polarity=[]



for dat in data:

    text = Text(dat)

    text.language = "pt"

    try:

        polarity.append(text.polarity)

    except:

        polarity.append(0)



# Creating a DataFrame containing extracted polarities

d = {'text': data, 'polarity': polarity}

dfSenti = pd.DataFrame(data=d)



# Sentiment Classification (Negative, Neutral, Positive)

dataLabel = dfSenti.polarity.tolist()



negative = 0

neutral  = 0

positive = 0

        

for p in dataLabel:

    if (p < 0 and p >= -1):

        negative = negative + 1

    if (p > 0 and p <= 1):

        positive = positive + 1

    if (p == 0):

        neutral = neutral + 1



print ("Negative Polarity: ", negative)

print ("Neutral Polarity..: ", neutral)

print ("Positive Polarity: ", positive)
plt.figure(figsize=(8, 5))

plt.axis('equal')

plt.tight_layout(True)



plt.pie([negative, neutral, positive], startangle=303, labels=['Negative', 'Neutral', 'Positive'], 

        explode=(0.05, 0.05, 0.05), autopct='%1.1f%%', shadow=True)
def wordcloud(dataframe,fieldname):

    

    comment_words = ' '

    

    text = " ".join(comments for comments in dataframe[fieldname])

    

    # split the value 

    tokens = text.split() 



    for words in tokens: 

        comment_words = comment_words + words + ' '





    wordcloud = WordCloud(width = 5000, height = 4000, 

                    background_color ='white', 

                    stopwords = nltk_stopwords, 

                    min_font_size = 10).generate(comment_words) 



    # plot the WordCloud image                        

    plt.figure(figsize = (12, 12), facecolor = 'k', edgecolor = 'k' ) 

    plt.imshow(wordcloud) 

    plt.axis("off") 

    plt.tight_layout(pad = 0) 



    plt.show() 
wordcloud(comments_pp,'comments')