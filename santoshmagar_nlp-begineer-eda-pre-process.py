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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# lets install extra liberary

! pip install autocorrect
""" ---- NLP text cleaning ----"""

import spacy

nlp = spacy.load("en_core_web_sm")

# import en_core_web_sm

# nlp = en_core_web_sm.load()

import re

import string

import unidecode

# from pycontractions import Contractions

from autocorrect import Speller

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
messages=pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding="latin-1")

messages.head()
messages.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1, inplace=True)

messages.columns = ["category", "text"]

print(messages.shape)

messages.head()
# # let's add some columns feature

def get_avg_word_len(x):

    words=x.split()

    word_len=0

    for word in words:

        word_len=word_len+len(word)

    return word_len/len(words)



messages['msg_len'] = messages['text'].apply(len)

messages['word_count'] = messages['text'].apply(lambda x :  len(x.split())  )

messages['avg_word_len'] = messages['text'].apply(lambda x :  get_avg_word_len(x)  )

messages['class']=messages['category'].apply(lambda x: 1 if x=='ham' else 0) # let's give ham=1 and spam=0 class

# # messages['class']=np.where(messages['category']=='spam',0,1)
messages.head()
stop_words = set(stopwords.words("english"))

spell_check = Speller(lang='en')
"""remove accented characters, to lower case, Remove numerical"""

def text_process_1(text):

    # remove accented characters from text, e.g. café 

    text = unidecode.unidecode(text)

    # change to lower case

    text = text.lower()

    # remove tags

    text=re.sub('<[^<]+?>','', text)

    # Remove numerical like  1996, 6 ,6df

    text=''.join(c for c in text if not c.isdigit())

    # return

    return text





""" token & lemmatizer -- using spaCy liberary """ 

""" spacy lematizer  also Expand Contractions words"""

def text_process_2(text):

    text=nlp(text)

    text=[token.lemma_ if token.lemma_ != "-PRON-" else token.lower_ for token in text ]

    # return

    return ' '.join(text) #as we are joining the list value so need a ' ' sinle space between them 





""" Remove stopword & punctuation & single character"""

def text_process_3(text):

    # Check characters to see if they are in punctuation then remove them

    text=''.join([char for char in text if char not in string.punctuation])

    # Remove stopword and single character 

    text = [word for word in word_tokenize(text) if word not in stop_words and len(word )>1]

    # return

    return ' '.join(text) #as we are joining the list value so need a ' ' sinle space between them 





"""autocorrect"""

def text_process_4(text):

    # spell check autocorrect

    text=[spell_check(w) for w in text.split() ]

    # Again

    # Remove stopword and single character if generated

    text = [word for word in text if word not in stop_words and len(word )>1]

    # return

    return ' '.join(text) #as we are joining the list value so need a ' ' sinle space between them 





"""Detect number in word if present and remove Eg: five, three  """

""" using spacy """

def text_process_5(text):

    text = nlp(text)

    text = [token.text for token in text if token.pos_ != 'NUM'  ]

    #text = [w2n.word_to_num(token.text) if token.pos_ == 'NUM' else token.text for token in text]

    return ' '.join(text)



msg=" ...18u..  âñ don't <h1>HELLO!!</h1> the??/ him he functions fna is a great  going go 66s ain’t wif ac acc early Available otw fiev hundrade "

msg1=text_process_1(msg)

msg2=text_process_2(msg1)

msg3=text_process_3(msg2)

msg4=text_process_4(msg3)

msg5=text_process_5(msg4)





print(msg1)

print(msg2)

print(msg3)

print(msg4)

print(msg5)
# text_process_1 : remove accented characters, to lower case, Remove numerical

messages["text"] = messages["text"].apply(text_process_1)

# text_process_2 : token & lemmatizer & Expand Contractions words

messages["text"] = messages["text"].apply(text_process_2)
# text_process_3 : Remove stopword & punctuation & single character

messages["text"] = messages["text"].apply(text_process_3)
# text_process_4 : autocorrect words

messages["text"] = messages["text"].apply(text_process_4)
# text_process_5 : Detect number in word if present and remove Eg: five, three

messages["text"] = messages["text"].apply(text_process_5)
messages.head()
messages.isnull().sum()
messages["category"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), 

                                     autopct = '%1.1f%%', shadow = True)

plt.ylabel("Spam vs Ham")

plt.legend(["Ham", "Spam"])

plt.show()
# lets see on bar graph

messages["category"].value_counts().plot(kind = 'bar')

plt.show()
plt.figure(figsize=(10,5))

plt.hist(messages['msg_len'], bins=40)

plt.show()
f, ax = plt.subplots(1, 2, figsize = (20, 6))



sns.distplot(messages[messages["category"] == "spam"]["msg_len"], bins = 20, ax = ax[0])

ax[0].set_xlabel("Spam Message Word Length")



sns.distplot(messages[messages["category"] == "ham"]["msg_len"], bins = 20, ax = ax[1])

ax[1].set_xlabel("Ham Message Word Length")



plt.show()
plt.figure(figsize=(10,5))

plt.hist(messages['word_count'], bins=40)

plt.title("word count on messgaes")

plt.show()
f, ax = plt.subplots(1, 2, figsize = (20, 6))



sns.distplot(messages[messages["category"] == "spam"]["word_count"], bins = 20, ax = ax[0])

ax[0].set_xlabel("Spam Message Word count")



sns.distplot(messages[messages["category"] == "ham"]["word_count"], bins = 20, ax = ax[1])

ax[1].set_xlabel("Ham Message Word count")



plt.show()
# function to draw the wordCloud from the text msg-paragraph 

def show_word_cloud(Msg):

  text=' '

  for words in Msg:

    text+=" "+words



  #word cloud

  wordcloud = WordCloud(width=600, 

                        height=400,

                        background_color = 'black'

                        ).generate(text.lower())

  plt.figure( figsize=(10,8),

             facecolor='k')

  plt.imshow(wordcloud, interpolation = 'bilinear')

  plt.axis("off")

  plt.tight_layout(pad=0)

  plt.show()

  del text
show_word_cloud(messages['text'].values)
# function to return the top 10 common words with freq

def feature_bow(msg):

    cv=CountVectorizer()

    bow=cv.fit_transform(msg)

    features_df=pd.DataFrame(bow.toarray(), columns=cv.get_feature_names())

    words = cv.get_feature_names()

    feature_df = pd.DataFrame(

        data =list(zip(words, features_df[words].sum())),

        columns = ['feature','freq']

        )

    #sort the df according to freq

    feature_df.sort_values(by='freq',ascending=False, inplace=True)

    feature_df.reset_index(drop=True, inplace=True)

    # most occuring 10 words

    return feature_df.head(10)

    
feature_freq=feature_bow(messages['text'])

feature_freq
plt.figure(figsize=(12,5))

sns.barplot(x='feature',y='freq',data=feature_freq)

plt.title("Top 10 feature words and frequency from whole dataset")

plt.show()
#  Lets study individual Spam/ham words

spam_messages = messages[messages["category"] == "spam"]["text"]

ham_messages = messages[messages["category"] == "ham"]["text"]

print(f"spam len:{len(spam_messages)}")

print(f"ham len:{len(ham_messages)}")

print(f"spam+ham: {len(spam_messages)+len(ham_messages)}")

print(f"total len:{messages.shape[0]}")
show_word_cloud(spam_messages)
feature_freq=feature_bow(spam_messages)

feature_freq

plt.figure(figsize=(12,5))

sns.barplot(x='feature',y='freq',data=feature_freq)

plt.title("Top 10 feature words and frequency from spam_messages")

plt.show()

    
show_word_cloud(ham_messages)
feature_freq=feature_bow(ham_messages)

feature_freq

plt.figure(figsize=(12,5))

sns.barplot(x='feature',y='freq',data=feature_freq)

plt.title("Top 10 feature words and frequency from ham_messages")

plt.show()

    