# !pip install pyspellchecker
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#For plotting

import matplotlib.pyplot as plt

import seaborn as sns



# text processing libraries

import re

import string

import nltk

from nltk.corpus import stopwords

# from spellchecker import SpellChecker



# sklearn 

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test_data = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
print("The train data has {} rows and {} columns".format(train_data.shape[0],train_data.shape[1]))

print("The test data has {} rows and {} columns".format(test_data.shape[0],test_data.shape[1]))
train_data.isna().sum()
test_data.isna().sum()
train_data.dropna(inplace = True)
train_data.head()
test_data.head()
#There are in total 27K training rows

train_data.sentiment.value_counts()
#There are in total 3.5K test rows

test_data.sentiment.value_counts()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))



pd.value_counts(train_data['sentiment']).plot(kind = 'pie', ax=ax1,autopct='%1.1f%%')

pd.value_counts(test_data['sentiment']).plot(kind ='pie', ax=ax2,autopct='%1.1f%%')

ax1.set_title("Train Data")

ax2.set_title("Test Data")

plt.suptitle("Percentage Distribution of sentiments across train and test data",fontweight = "bold")

plt.show()
#Text Length

train_data["text_len"] = train_data["text"].astype(str).apply(len)

test_data["text_len"] = test_data["text"].astype(str).apply(len)

train_data["selected_text_len"] = train_data["selected_text"].astype(str).apply(len)



#Word Count (before preprocessing)

train_data["text_wc"] = train_data["text"].apply(lambda x: len(str(x).split()))

test_data["text_wc"] = test_data["text"].apply(lambda x: len(str(x).split()))

train_data["selected_text_wc"] = train_data["selected_text"].apply(lambda x: len(str(x).split()))
train_data.head()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

sns.kdeplot(train_data.text_len, shade=True, ax= ax1)

sns.kdeplot(test_data.text_len, shade=True, ax = ax1)

sns.kdeplot(train_data.text_wc, shade=True, ax= ax2)

sns.kdeplot(test_data.text_wc, shade=True, ax = ax2)

ax1.set_title("Text Length")

ax2.set_title("Text Word Count")

ax1.legend(['Train Data','Test Data'])

ax2.legend(['Train Data','Test Data'])

plt.suptitle("Comparison of text lengths and word counts of the train and test data ",fontweight = "bold")

plt.show()
fig, (ax) = plt.subplots(3, 2, figsize=(10, 10))

sns.kdeplot(train_data[train_data.sentiment=="neutral"].text_len, shade=True, ax= ax[0][0])

sns.kdeplot(test_data[test_data.sentiment=="neutral"].text_len, shade=True, ax = ax[0][0])

sns.kdeplot(train_data[train_data.sentiment=="neutral"].text_wc, shade=True, ax= ax[0][1])

sns.kdeplot(test_data[test_data.sentiment=="neutral"].text_wc, shade=True, ax = ax[0][1])

ax[0][0].set_title("Neutral Text Length")

ax[0][1].set_title("Neutral Text Word Count")

ax[0][0].legend(['Train Data','Test Data'])

ax[0][1].legend(['Train Data','Test Data'])



sns.kdeplot(train_data[train_data.sentiment=="positive"].text_len, shade=True, ax= ax[1][0])

sns.kdeplot(test_data[test_data.sentiment=="positive"].text_len, shade=True, ax = ax[1][0])

sns.kdeplot(train_data[train_data.sentiment=="positive"].text_wc, shade=True, ax= ax[1][1])

sns.kdeplot(test_data[test_data.sentiment=="positive"].text_wc, shade=True, ax = ax[1][1])

ax[1][0].set_title("Positive Text Length")

ax[1][1].set_title("Positive Text Word Count")

ax[1][0].legend(['Train Data','Test Data'])

ax[1][1].legend(['Train Data','Test Data'])



sns.kdeplot(train_data[train_data.sentiment=="negative"].text_len, shade=True, ax= ax[2][0])

sns.kdeplot(test_data[test_data.sentiment=="negative"].text_len, shade=True, ax = ax[2][0])

sns.kdeplot(train_data[train_data.sentiment=="negative"].text_wc, shade=True, ax= ax[2][1])

sns.kdeplot(test_data[test_data.sentiment=="negative"].text_wc, shade=True, ax = ax[2][1])

ax[2][0].set_title("Negative Text Length")

ax[2][1].set_title("Negative Text Word Count")

ax[2][0].legend(['Train Data','Test Data'])

ax[2][1].legend(['Train Data','Test Data'])



plt.suptitle("Comparison of text lengths and word counts of the train and test data for each sentiment",fontweight = "bold")

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

sns.kdeplot(train_data[train_data.sentiment=="positive"].text_len, shade=True,color = "g", ax= ax1)

sns.kdeplot(train_data[train_data.sentiment=="negative"].text_len, shade=True,color = "r", ax= ax1)

sns.kdeplot(train_data[train_data.sentiment=="neutral"].text_len, shade=True,color = "y", ax= ax1)

sns.kdeplot(train_data[train_data.sentiment=="positive"].selected_text_len, shade=True,color = "g", ax= ax2)

sns.kdeplot(train_data[train_data.sentiment=="negative"].selected_text_len, shade=True,color = "r", ax= ax2)

sns.kdeplot(train_data[train_data.sentiment=="neutral"].selected_text_len, shade=True, color = "y", ax= ax2)

ax1.set_title("Text lengths")

ax2.set_title("Selected text lengths")

ax1.legend(['positive','negative','neutral'])

ax2.legend(['positive','negative','neutral'])

plt.suptitle("Comparison of length of different sentiments for actual text and selected text",fontweight = "bold")

plt.show()
fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(16, 4))

sns.kdeplot(train_data[train_data.sentiment=="positive"].text_len, shade=True,color = "g", ax= ax1)

sns.kdeplot(train_data[train_data.sentiment=="positive"].selected_text_len, shade=True,color = "b", ax= ax1)



sns.kdeplot(train_data[train_data.sentiment=="negative"].text_len, shade=True,color = "r", ax= ax2)

sns.kdeplot(train_data[train_data.sentiment=="negative"].selected_text_len, shade=True,color = "m", ax= ax2)



sns.kdeplot(train_data[train_data.sentiment=="neutral"].text_len, shade=True,color = "y", ax= ax3)

sns.kdeplot(train_data[train_data.sentiment=="neutral"].selected_text_len, shade=True, color = "c", ax= ax3)



ax1.set_title("positive")

ax2.set_title("negative")

ax3.set_title("neutral")

plt.suptitle("Comparison of text lengths and selected text lengths across different sentiments",fontweight = "bold")
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
train_data['jaccard_actual_selected'] = train_data.apply(lambda row:jaccard(row["text"],row["selected_text"]),axis=1)
train_data.head()
train_data.groupby('sentiment')['jaccard_actual_selected'].mean()
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,3))

sns.distplot(train_data.loc[train_data['sentiment']=="negative","jaccard_actual_selected"],kde = False,ax=ax1)

sns.distplot(train_data.loc[train_data['sentiment']=="neutral","jaccard_actual_selected"],kde = False,ax=ax2)

sns.distplot(train_data.loc[train_data['sentiment']=="positive","jaccard_actual_selected"],kde = False,ax=ax3)

ax1.set_title("Negative")

ax2.set_title("Neutral")

ax3.set_title("Positive")
# spell = SpellChecker()

# def correct_spellings(text):

#     corrected_text = []

#     misspelled_words = spell.unknown(text.split())

#     for word in text.split():

#         if word in misspelled_words:

#             corrected_text.append(spell.correction(word))

#         else:

#             corrected_text.append(word)

#     return " ".join(corrected_text)

        

# text = "corect me plese"

# correct_spellings(text)
# text preprocessing helper functions



def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text





def text_preprocessing(text):

    """

    Cleaning and parsing the text.



    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(tokenized_text)

    return combined_text
# Applying the cleaning function to both test and training datasets

train_data['text_clean'] = train_data['text'].apply(str).apply(lambda x: text_preprocessing(x))

test_data['text_clean'] = test_data['text'].apply(str).apply(lambda x: text_preprocessing(x))
train_data.head()
test_data.head()
test_data.head()
#source of code : https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d

def get_top_n_gram(corpus,ngram_range,n):

    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
pos_unigrams = get_top_n_gram(train_data.loc[train_data['sentiment']=="positive",'text_clean'],(1,1),20)

neg_unigrams = get_top_n_gram(train_data.loc[train_data['sentiment']=="negative",'text_clean'],(1,1),20)

neutral_unigrams = get_top_n_gram(train_data.loc[train_data['sentiment']=="neutral",'text_clean'],(1,1),20)



pos_bigrams = get_top_n_gram(train_data.loc[train_data['sentiment']=="positive",'text_clean'],(2,2),20)

neg_bigrams = get_top_n_gram(train_data.loc[train_data['sentiment']=="negative",'text_clean'],(2,2),20)

neutral_bigrams = get_top_n_gram(train_data.loc[train_data['sentiment']=="neutral",'text_clean'],(2,2),20)



pos_trigrams = get_top_n_gram(train_data.loc[train_data['sentiment']=="positive",'text_clean'],(3,3),20)

neg_trigrams = get_top_n_gram(train_data.loc[train_data['sentiment']=="negative",'text_clean'],(3,3),20)

neutral_trigrams = get_top_n_gram(train_data.loc[train_data['sentiment']=="neutral",'text_clean'],(3,3),20)
df_pos_unigram = pd.DataFrame(pos_unigrams,columns = ["word","count"])

df_neg_unigram = pd.DataFrame(neg_unigrams,columns = ["word","count"])

df_neutral_unigram = pd.DataFrame(neutral_unigrams,columns = ["word","count"])



df_pos_bigrams = pd.DataFrame(pos_bigrams,columns = ["words","count"])

df_neg_bigrams = pd.DataFrame(neg_bigrams,columns = ["words","count"])

df_neutral_bigrams = pd.DataFrame(neutral_bigrams,columns = ["words","count"])



df_pos_trigrams = pd.DataFrame(pos_trigrams,columns = ["words","count"])

df_neg_trigrams = pd.DataFrame(neg_trigrams,columns = ["words","count"])

df_neutral_trigrams = pd.DataFrame(neutral_trigrams,columns = ["words","count"])
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

sns.barplot(x="count",y = "word",data = df_neutral_unigram,ax=ax1,color = "yellow")

sns.barplot(x="count",y = "words",data = df_neutral_bigrams,ax=ax2,color = "gold")

sns.barplot(x="count",y = "words",data = df_neutral_trigrams,ax=ax3,color = "goldenrod")

ax1.set_title("Uni-gram")

ax2.set_title("Bi-gram")

ax3.set_title("Tri-gram")

plt.suptitle("Most preferred N-grams in Neutral tweets",fontweight = "bold")

plt.tight_layout()

plt.subplots_adjust(top=0.85)
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

sns.barplot(x="count",y = "word",data = df_neg_unigram,ax=ax1,color = "tomato")

sns.barplot(x="count",y = "words",data = df_neg_bigrams,ax=ax2,color = "red")

sns.barplot(x="count",y = "words",data = df_neg_trigrams,ax=ax3,color = "maroon")

ax1.set_title("Uni-gram")

ax2.set_title("Bi-gram")

ax3.set_title("Tri-gram")

plt.suptitle("Most preferred N-grams in Negative tweets",fontweight = "bold")

plt.tight_layout()

plt.subplots_adjust(top=0.85)
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

sns.barplot(x="count",y = "word",data = df_pos_unigram,ax=ax1,color = "lime")

sns.barplot(x="count",y = "words",data = df_pos_bigrams,ax=ax2,color = "green")

sns.barplot(x="count",y = "words",data = df_pos_trigrams,ax=ax3,color = "darkgreen")

ax1.set_title("Uni-gram")

ax2.set_title("Bi-gram")

ax3.set_title("Tri-gram")

plt.suptitle("Most preferred N-grams in Positive tweets",fontweight = "bold")

plt.tight_layout()

plt.subplots_adjust(top=0.85)