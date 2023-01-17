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
from sklearn.feature_extraction.text import CountVectorizer
import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/ import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch
import string

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize']=10,6
plt.rcParams['axes.grid']=True
plt.gray()

use_cuda = True
pd.set_option('display.max_columns', None)
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv") 
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv") 
sample_submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv") 
train.head()
test.head()
test.shape
sample_submission.head()
# Checking the shape of train and test data
print(train.shape)
print(test.shape)
# Checking Missing value in the training set
print(train.isnull().sum())
# Checking Missing Value in the testing set
print(test.isnull().sum())
# Création d'une fonction permettant de calculer le total de valeurs manquantes, le pourcentage et le type de 
 ## chaque colonne 
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))
missing_data(train)
missing_data(test)
percent_sentiment = train.groupby('sentiment').count()
percent_sentiment['percent'] = 100*(percent_sentiment['text']/train['sentiment'].count())
percent_sentiment.reset_index(level=0, inplace=True)
percent_sentiment

# Droping the row with missing values
train.dropna(axis = 0, how ='any',inplace=True)
# Positive tweet
print("Positive Tweet example :",train[train['sentiment']=='positive']['text'].values[0])
#negative_text
print("Negative Tweet example :",train[train['sentiment']=='negative']['text'].values[0])
#neutral_text
print("Neutral tweet example  :",train[train['sentiment']=='neutral']['text'].values[0])
# Distribution of the Sentiment Column
train['sentiment'].value_counts()
# Train data 
sns.countplot(x=train['sentiment'],data=train)
plt.show()
train['sentiment'].value_counts(normalize=True)
f,ax=plt.subplots(1,2,figsize=(13,5))
train['sentiment'].value_counts().plot.pie(explode=[0,0.05,0.5],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('sentiment')
ax[0].set_ylabel('')
sns.countplot('sentiment',data=train,ax=ax[1])
ax[1].set_title('sentiment')
plt.show()
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk import wordnet, pos_tag
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords, wordnet as wn
import re
import string

#Cleaning data

def clean_str(chaine):
    chaine = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", chaine)     
    chaine = re.sub(r"\'s", " \'s", chaine) 
    chaine = re.sub(r"\'ve", " \'ve", chaine) 
    chaine = re.sub(r"n\'t", " n\'t", chaine) 
    chaine = re.sub(r"\'re", " \'re", chaine) 
    chaine = re.sub(r"\'d", " \'d", chaine) 
    chaine = re.sub(r"\'ll", " \'ll", chaine) 
    chaine = re.sub(r",", " , ", chaine) 
    chaine = re.sub(r"!", " ! ", chaine) 
    chaine = re.sub(r"\(", " \( ", chaine) 
    chaine = re.sub(r"\)", " \) ", chaine) 
    chaine = re.sub(r"\?", " \? ", chaine) 
    chaine = re.sub(r"\s{2,}", " ", chaine)
    chaine = chaine.lower() #convert all text in lower case
    chaine = chaine.replace(' +', ' ') # Remove double space
    chaine = chaine.strip() # Remove trailing space at the beginning or end
    chaine = chaine.replace('[^a-zA-Z]', ' ' )# Everything not a alphabet character replaced with a space
    #words =  [word for word in chaine.split() if word not in [i for i in string.punctuation]] #Remove punctuations
    words =  [word for word in chaine.split() if word.isalpha()] #droping numbers and punctuations
    return ' '.join(words)

#Tokenization and punctuation removing and stopwords
def tokeniZ_stopWords(chaine):
    chaine = word_tokenize(chaine)
    list_stopWords = set(stopwords.words('english'))
    words = [word for word in chaine if word not in list_stopWords]
    return words

#Stemming 
ps = PorterStemmer()
sb = SnowballStemmer('english')

#Lemmatization
def lemat_words(tokens_list):
    from collections import defaultdict
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lemma_function = WordNetLemmatizer()
    return [lemma_function.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(tokens_list)]
    #for token, tag in pos_tag(tokens_list):
     #   lemma = lemma_function.lemmatize(token, tag_map[tag[0]])

# Define Ngrams function
def get_ngrams(text, n ):
    n_grams = ngrams(word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]
#Cleaning the train data 
train['text_clean'] = train['text'].apply(clean_str)

#Tokenizing and stopwords removing
train['tokeniZ_stopWords_text'] = train['text_clean'].apply(tokeniZ_stopWords)
#Words Stemming
train['stemming_text'] = [[ps.stem(word) for word in words] for words in train['tokeniZ_stopWords_text'] ]
train['stemming_text_for_tfidf'] = [' '.join(words) for words in train['stemming_text']] 

#Words lemmatization
train['lemmatize_text'] = train['tokeniZ_stopWords_text'].apply(lemat_words)
train['lemmatize_text_for_tfidf'] = [' '.join(x) for x in train['lemmatize_text'] ]

#Calcul longueur des commentaires
train['text_lenght'] = train['text'].apply(len)

#Calcul du nombre de ponctuation par question
from string import punctuation
train['number_punctuation_text'] = train['text'].apply(
    lambda doc: len([word for word in str(doc) if word in punctuation])) 

#Number of unique words in the text
train['number_of_Unique_words_text'] = train['text_clean'].apply([lambda x : len(set(str(x).split()))])

#Number of stopwords in the text
list_stopWords = set(stopwords.words('english'))
train['number_of_StopWords_text'] = train['text_clean'].apply(
    lambda x : len([w for w in x.lower().split() if w in list_stopWords ]))


#Number of upper case words
train['number_of_uppercase_text'] = train['text'].apply(
    lambda x : len([w for w in x.split() if w.isupper()]))


#Average length of words in the text (whithout stop words)
train['average_of_wordsLength_text'] = train['text_clean'].apply(
    lambda x : np.mean([len(w) for w in x.split()]))

#Number of words in the text
train['number_of_words_text'] = train['text_clean'].apply([lambda x : len(str(x).split())])

#Cleaning the train data 
train['selected_text_clean'] = train['selected_text'].apply(clean_str)

#Tokenizing and stopwords removing
train['tokeniZ_stopWords_text'] = train['selected_text_clean'].apply(tokeniZ_stopWords)

#Words Stemming
train['stemming_selected_text'] = [[ps.stem(word) for word in words] for words in train['tokeniZ_stopWords_text'] ]
train['stemming_selected_text_for_tfidf'] = [' '.join(words) for words in train['stemming_selected_text']] 

#Words lemmatization
train['lemmatize_selected_text'] = train['tokeniZ_stopWords_text'].apply(lemat_words)
train['lemmatize_selected_text_for_tfidf'] = [' '.join(x) for x in train['lemmatize_selected_text'] ]


#Calcul longueur des commentaires
train['selected_text_lenght'] = train['selected_text'].apply(len)

#Calcul du nombre de ponctuation par question
from string import punctuation
train['number_punctuation_selected_text'] = train['selected_text'].apply(
    lambda doc: len([word for word in str(doc) if word in punctuation])) 

#Number of unique words in the text
train['number_of_Unique_words_selected_text'] = train['selected_text_clean'].apply([lambda x : len(set(str(x).split()))])

#Number of stopwords in the text
list_stopWords = set(stopwords.words('english'))
train['number_of_StopWords_selected_text'] = train['selected_text_clean'].apply(
    lambda x : len([w for w in x.lower().split() if w in list_stopWords ]))


#Number of upper case words
train['number_of_uppercase_selected_text'] = train['selected_text'].apply(
    lambda x : len([w for w in x.split() if w.isupper()]))


#Average length of words in the text (whithout stop words)
train['average_of_wordsLength_selected_text'] = train['selected_text_clean'].apply(
    lambda x : np.mean([len(w) for w in x.split()]))

#Number of words in the text
train['number_of_words_selected_text'] = train['selected_text_clean'].apply([lambda x : len(str(x).split())])


# Let's create three separate dataframes for positive, neutral and negative sentiments. 
#This will help in analyzing the text statistics separately for separate polarities.

positive = train[train['sentiment']=='positive']
negative = train[train['sentiment']=='negative']
neutral = train[train['sentiment']=='neutral']
# Sentence length analysis

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(positive['text_lenght'],bins=50,color='g')
plt.title('Positive Text Length Distribution')
plt.xlabel('text_lenght')
plt.ylabel('count')


plt.subplot(1, 3, 2)
plt.hist(negative['text_lenght'],bins=50,color='r')
plt.title('Negative Text Length Distribution')
plt.xlabel('text_lenght')
plt.ylabel('count')


plt.subplot(1, 3, 3)
plt.hist(neutral['text_lenght'],bins=50,color='y')
plt.title('Neutral Text Length Distribution')
plt.xlabel('text_lenght')
plt.ylabel('count')
plt.show()
#source of code : https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
#Distribution of top unigrams
positive_unigrams = get_top_n_words(positive['text_clean'],20)
negative_unigrams = get_top_n_words(negative['text_clean'],20)
neutral_unigrams = get_top_n_words(neutral['text_clean'],20)

df1 = pd.DataFrame(positive_unigrams, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')
plt.ylabel('Count')
plt.title('Top 20 unigrams in positve text')
plt.show()

df2 = pd.DataFrame(negative_unigrams, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='red')
plt.title('Top 20 unigram in Negative text')
plt.show()

df3 = pd.DataFrame(neutral_unigrams, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='yellow')
plt.title('Top 20 unigram in Neutral text')
plt.show()
def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
#Distribution of top Bigrams
positive_bigrams = get_top_n_gram(positive['text_clean'],(2,2),20)
negative_bigrams = get_top_n_gram(negative['text_clean'],(2,2),20)
neutral_bigrams = get_top_n_gram(neutral['text_clean'],(2,2),20)

df1 = pd.DataFrame(positive_bigrams, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')
plt.ylabel('Count')
plt.title('Top 20 Bigrams in positve text')
plt.show()

df2 = pd.DataFrame(negative_bigrams, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='red')
plt.title('Top 20 Bigram in Negative text')
plt.show()

df3 = pd.DataFrame(neutral_bigrams, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='yellow')
plt.title('Top 20 Bigram in Neutral text')
plt.show()
# Finding top trigram
positive_trigrams = get_top_n_gram(positive['text_clean'],(3,3),20)
negative_trigrams = get_top_n_gram(negative['text_clean'],(3,3),20)
neutral_trigrams = get_top_n_gram(neutral['text_clean'],(3,3),20)

df1 = pd.DataFrame(positive_trigrams, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')
plt.ylabel('Count')
plt.title('Top 20 trigrams in positve text')
plt.show()

df2 = pd.DataFrame(negative_trigrams, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='red')
plt.title('Top 20 trigram in Negative text')
plt.show()

df3 = pd.DataFrame(neutral_trigrams, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='yellow')
plt.title('Top 20 trigram in Neutral text')
plt.show()


#  Exploring the selected_text column

positive_text = train[train['sentiment'] == 'positive']['selected_text']
negative_text = train[train['sentiment'] == 'negative']['selected_text']
neutral_text = train[train['sentiment'] == 'neutral']['selected_text']
# Positive text
print("Positive Text example :",positive_text.values[0])
#negative_text
print("Negative Tweet example :",negative_text.values[0])
#neutral_text
print("Neutral tweet example  :",neutral_text.values[0])
# Preprocess Selected_text

positive_text_clean = positive_text.apply(clean_str)
negative_text_clean = negative_text.apply(clean_str)
neutral_text_clean = neutral_text.apply(clean_str)
#source of code : https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
top_words_in_positive_text = get_top_n_words(positive_text_clean)
top_words_in_negative_text = get_top_n_words(negative_text_clean)
top_words_in_neutral_text = get_top_n_words(neutral_text_clean)

p1 = [x[0] for x in top_words_in_positive_text[:20]]
p2 = [x[1] for x in top_words_in_positive_text[:20]]


n1 = [x[0] for x in top_words_in_negative_text[:20]]
n2 = [x[1] for x in top_words_in_negative_text[:20]]


nu1 = [x[0] for x in top_words_in_neutral_text[:20]]
nu2 = [x[1] for x in top_words_in_neutral_text[:20]]
# Top positive word
sns.barplot(x=p1,y=p2,color = 'green')
plt.xticks(rotation=45)
plt.title('Top 20 Positive Word')
plt.show()

sns.barplot(x=n1,y=n2,color='red')
plt.xticks(rotation=45)
plt.title('Top 20 Negative Word')
plt.show()

sns.barplot(x=nu1,y=nu2,color='yellow')
plt.xticks(rotation=45)
plt.title('Top 20 Neutral Word')
plt.show()
#Wordclouds
# Wordclouds to see which words contribute to which type of polarity.

from wordcloud import WordCloud
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[30, 15])
wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(positive_text_clean))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Positive text',fontsize=40);

wordcloud2 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(negative_text_clean))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Negative text',fontsize=40);

wordcloud3 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(neutral_text_clean))
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Neutral text',fontsize=40)
# https://www.kaggle.com/ekhtiar/unintended-eda-with-tutorial-notes
def generate_word_cloud(df_data, text_col):
    # convert stop words to sets as required by the wordcloud library
    stop_words = set(stopwords.words("english"))
    
    data_neutral = " ".join(df_data.loc[df_data["sentiment"]=="neutral", text_col].map(lambda x: str(x).lower()))
    data_positive = " ".join(df_data.loc[df_data["sentiment"]=="positive", text_col].map(lambda x: str(x).lower()))
    data_negative = " ".join(df_data.loc[df_data["sentiment"]=="negative", text_col].map(lambda x: str(x).lower()))

    wc_neutral = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(data_neutral)
    wc_positive = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(data_positive)
    wc_negative = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(data_negative)

    # draw the two wordclouds side by side using subplot
    fig, ax = plt.subplots(1, 3, figsize=(20, 20))
    ax[0].set_title("Neutral Wordcloud" , fontsize=10)
    ax[0].imshow(wc_neutral, interpolation="bilinear")
    ax[0].axis("off")
    
    ax[1].set_title("Positive Wordcloud", fontsize=10)
    ax[1].imshow(wc_positive, interpolation="bilinear")
    ax[1].axis("off")
    
    ax[2].set_title("Negative Wordcloud", fontsize=10)
    ax[2].imshow(wc_negative, interpolation="bilinear")
    ax[2].axis("off")
    plt.show()
    
    return [wc_neutral, wc_positive, wc_negative]
train_text_wc = generate_word_cloud(train, "text")

train_sel_text_wc = generate_word_cloud(train, "selected_text")
train_text_wc = generate_word_cloud(train, "text")
missing_data(train)
train['Target'] = train['sentiment'].apply(lambda x: 2 if x == 'positive' else 1 if x == 'neutral' else 0)
train.head()
percent_target = train.groupby('Target').count()
percent_target['percent'] = 100*(percent_target['text']/train['Target'].count())
percent_target.reset_index(level=0, inplace=True)
percent_target
train.head(2)
train.tail()
train[['text_lenght', 'number_punctuation_text', 'number_of_words_text',
       'number_of_Unique_words_text', 'number_of_StopWords_text', 'number_of_uppercase_text',
       'average_of_wordsLength_text']].sample(5)
list_var=['text_lenght', 'number_punctuation_text', 'number_of_words_text',
       'number_of_Unique_words_text', 'number_of_StopWords_text', 'number_of_uppercase_text','average_of_wordsLength_text']
def var_hist_global(df,X='Target',Y=list_var, Title='Features Engineering - Histograms', KDE=False):
    fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6),(ax6,ax7)) = plt.subplots(4, 2 ,figsize=(14,16))#, sharey=True )
    aX = [ax1, ax2,ax3,ax4,ax5,ax6,ax6,ax7]
    for i in range(len(list_var)):   
        sns.distplot( df[list_var[i]][df[X]== 1 ].dropna(), label="Neutral" , ax= aX[i], kde= KDE , color = 'red')           
        sns.distplot( df[list_var[i]][df[X]== 0 ].dropna(), label="Negative", ax= aX[i], kde= KDE , color = "olive")
        sns.distplot( df[list_var[i]][df[X]== 2 ].dropna(), label="Positive", ax= aX[i], kde= KDE , color = "black")
    plt.legend()
    plt.title(Title)
    #plt.show()
    plt.savefig("Features_Engineering_Histograms")
    
var_hist_global(df=train,X='Target',Y=list_var, KDE=True)
# Calculate number of obs per group & median to position labels
list_var = ['text_lenght', 'number_of_Unique_words_text', 'number_of_StopWords_text']
def violin_boxplott(df,X='Target',Y=list_var, Title='Features Engineering - Box plot'): 
    fig, (ax1, ax2 ,ax3) = plt.subplots(1,3 ,figsize=(14,8))#, sharey=True )
    medians = train.groupby(['Target'])['text_lenght', 'number_of_Unique_words_text', 'number_of_StopWords_text'].median().values
 
    sns.boxplot( y=list_var[0],  x=X , data = df, ax= ax1 , palette=['olive','red'])
    sns.boxplot( y=list_var[1],  x=X , data = df, ax= ax2 , palette=['olive','red'])
    sns.boxplot( y=list_var[2],  x=X , data = df, ax= ax3 , palette=['olive','red'])
    #plt.title(Title)
    plt.savefig("Features_Engineering_Boxplot")
violin_boxplott(df=train)



#Word2Vec with preprocessiong questions (without stopwords) 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

d2v_training_data = []
for i, doc in enumerate(train['stemming_selected_text']):
    d2v_training_data.append(TaggedDocument(words=doc,tags=[i]))

# ========== learning doc embeddings with doc2vec ==========

# PV stands for 'Paragraph Vector'
# PV-DBOW (distributed bag-of-words) dm=0

d2v = Doc2Vec(d2v_training_data, vector_size=300, window=10, alpha=0.1, min_alpha=1e-4, dm=0, negative=1, epochs=10, min_count=2, workers=4)
d2v_vecs = np.zeros((len(train['stemming_selected_text']), 300))
for i in range(len(train['stemming_selected_text'])):
    d2v_vecs[i,:] = d2v.docvecs[i]
#Word2Vec with lemmatize words
d2v_training_data = []
for i, doc in enumerate(train['lemmatize_selected_text']):
    d2v_training_data.append(TaggedDocument(words=doc,tags=[i]))

# ========== learning doc embeddings with doc2vec ==========

# PV stands for 'Paragraph Vector'
# PV-DBOW (distributed bag-of-words) dm=0

d2v = Doc2Vec(d2v_training_data, vector_size=200, window=5, alpha=0.1, min_alpha=1e-4, 
              dm=0, negative=1, epochs=10, min_count=2, workers=4)
d2v_vecs_bigram = np.zeros((len(train['lemmatize_selected_text']), 200))
for i in range(len(train['lemmatize_selected_text'])):
    d2v_vecs_bigram[i,:] = d2v.docvecs[i]

test.head()
missing_data(test)
# Test data 
sns.countplot(x=test['sentiment'],data=train)
plt.show()
percent_sentiment = test.groupby('sentiment').count()
percent_sentiment['percent'] = 100*(percent_sentiment['text']/train['sentiment'].count())
percent_sentiment.reset_index(level=0, inplace=True)
percent_sentiment
# Positive tweet
print("Positive Tweet example :",test[test['sentiment']=='positive']['text'].values[0])
#negative_text
print("Negative Tweet example :",test[test['sentiment']=='negative']['text'].values[0])
#neutral_text
print("Neutral tweet example  :",test[test['sentiment']=='neutral']['text'].values[0])
# Distribution of the Sentiment Column
test['sentiment'].value_counts()
#Cleaning the train data 
test['text_clean_test'] = test['text'].apply(clean_str)

#Tokenizing and stopwords removing
test['tokeniZ_stopWords_text_test'] = test['text_clean_test'].apply(tokeniZ_stopWords)
#Words Stemming
test['stemming_text_test'] = [[ps.stem(word) for word in words] for words in test['tokeniZ_stopWords_text_test'] ]
test['stemming_text_for_tfidf_test'] = [' '.join(words) for words in test['stemming_text_test']] 

#Words lemmatization
test['lemmatize_text_test'] = test['tokeniZ_stopWords_text_test'].apply(lemat_words)
test['lemmatize_text_for_tfidf_test'] = [' '.join(x) for x in test['lemmatize_text_test'] ]

#Calcul longueur des commentaires
test['text_lenght_test'] = test['text'].apply(len)

#Calcul du nombre de ponctuation par question
from string import punctuation
test['number_punctuation_text_test'] = test['text'].apply(
    lambda doc: len([word for word in str(doc) if word in punctuation])) 

#Number of unique words in the text
test['number_of_Unique_words_text_test'] = test['text_clean_test'].apply([lambda x : len(set(str(x).split()))])
test
#Number of stopwords in the text
list_stopWords = set(stopwords.words('english'))
test['number_of_StopWords_text_test'] = test['text_clean_test'].apply(
    lambda x : len([w for w in x.lower().split() if w in list_stopWords ]))


#Number of upper case words
test['number_of_uppercase_text_test'] = test['text'].apply(
    lambda x : len([w for w in x.split() if w.isupper()]))


#Average length of words in the text (whithout stop words)
test['average_of_wordsLength_text_test'] = test['text_clean_test'].apply(
    lambda x : np.mean([len(w) for w in x.split()]))

#Number of words in the text
test['number_of_words_text_test'] = test['text_clean_test'].apply([lambda x : len(str(x).split())])
# Let's create three separate dataframes for positive, neutral and negative sentiments. 
#This will help in analyzing the text statistics separately for separate polarities.

positive_test = test[test['sentiment']=='positive']
negative_test = test[test['sentiment']=='negative']
neutral_test = test[test['sentiment']=='neutral']
# Sentence length analysis

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(positive_test['text_lenght_test'],bins=50,color='g')
plt.title('Positive Text Length Distribution test data')
plt.xlabel('text_lenght_test')
plt.ylabel('count')


plt.subplot(1, 3, 2)
plt.hist(negative_test['text_lenght_test'],bins=50,color='r')
plt.title('Negative Text Length Distribution  test data')
plt.xlabel('text_lenght_test')
plt.ylabel('count')


plt.subplot(1, 3, 3)
plt.hist(neutral_test['text_lenght_test'],bins=50,color='y')
plt.title('Neutral Text Length Distribution  test data')
plt.xlabel('text_lenght_test')
plt.ylabel('count')
plt.show()
#Distribution of top unigrams
positive_test_unigrams = get_top_n_words(positive_test['text_clean_test'],20)
negative_test_unigrams = get_top_n_words(negative_test['text_clean_test'],20)
neutral_test_unigrams = get_top_n_words(neutral_test['text_clean_test'],20)

df1 = pd.DataFrame(positive_test_unigrams, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')
plt.ylabel('Count')
plt.title('Top 20 unigrams in positve text')
plt.show()

df2 = pd.DataFrame(negative_test_unigrams, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='red')
plt.title('Top 20 unigram in Negative text')
plt.show()

df3 = pd.DataFrame(neutral_test_unigrams, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='yellow')
plt.title('Top 20 unigram in Neutral text')
plt.show()
#  Exploring the selected_text column

positive_text_test = test[test['sentiment'] == 'positive']['text']
negative_text_test = test[test['sentiment'] == 'negative']['text']
neutral_text_test = test[test['sentiment'] == 'neutral']['text']
# Preprocess Selected_text

positive_text_clean_test = positive_text_test.apply(clean_str)
negative_text_clean_test = negative_text_test.apply(clean_str)
neutral_text_clean_test = neutral_text_test.apply(clean_str)
top_words_in_positive_text_test = get_top_n_words(positive_text_clean_test)
top_words_in_negative_text_test = get_top_n_words(negative_text_clean_test)
top_words_in_neutral_text_test = get_top_n_words(neutral_text_clean_test)

p_test1 = [x[0] for x in top_words_in_positive_text_test[:20]]
p_test2 = [x[1] for x in top_words_in_positive_text_test[:20]]


n_test1 = [x[0] for x in top_words_in_negative_text_test[:20]]
n_test2 = [x[1] for x in top_words_in_negative_text_test[:20]]


nu_test1 = [x[0] for x in top_words_in_neutral_text_test[:20]]
nu_test2 = [x[1] for x in top_words_in_neutral_text_test[:20]]
# Top positive word
sns.barplot(x=p_test1,y=p_test2,color = 'green')
plt.xticks(rotation=45)
plt.title('Top 20 Positive Word')
plt.show()

sns.barplot(x=n_test1,y=n_test2,color='red')
plt.xticks(rotation=45)
plt.title('Top 20 Negative Word')
plt.show()

sns.barplot(x=nu_test1,y=nu_test2,color='yellow')
plt.xticks(rotation=45)
plt.title('Top 20 Neutral Word')
plt.show()
#Wordclouds
# Wordclouds to see which words contribute to which type of polarity.

from wordcloud import WordCloud
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[30, 15])
wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(positive_text_clean_test))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Positive text',fontsize=40);

wordcloud2 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(negative_text_clean_test))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Negative text',fontsize=40);

wordcloud3 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(neutral_text_clean_test))
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Neutral text',fontsize=40)
test_text_wc = generate_word_cloud(test, "text")


test.head()
test['Target'] = test['sentiment'].apply(lambda x: 2 if x == 'positive' else 1 if x == 'neutral' else 0)
test.head()
test[['text_lenght_test', 'number_punctuation_text_test', 'number_of_words_text_test',
       'number_of_Unique_words_text_test', 'number_of_StopWords_text_test', 'number_of_uppercase_text_test',
       'average_of_wordsLength_text_test']].sample(5)
list_var=['text_lenght_test', 'number_punctuation_text_test', 'number_of_words_text_test',
       'number_of_Unique_words_text_test', 'number_of_StopWords_text_test', 'number_of_uppercase_text_test',
       'average_of_wordsLength_text_test']

    
var_hist_global(df=test,X='Target',Y=list_var, KDE=True)


test.columns
X_train =  train[['text_clean', 'stemming_text_for_tfidf', 'lemmatize_text_for_tfidf','tokeniZ_stopWords_text', 'stemming_text', 'lemmatize_text',
                             'text_lenght', 'number_punctuation_text', 'number_of_StopWords_text', 'number_of_Unique_words_text', 'number_of_uppercase_text','average_of_wordsLength_text']]
    



y_train = train['Target']

X_test = test[['text_clean_test','stemming_text_for_tfidf_test', 'lemmatize_text_for_tfidf_test', 'stemming_text_test', 'lemmatize_text_test', 'tokeniZ_stopWords_text_test', 
               'text_lenght_test', 'number_punctuation_text_test','number_of_Unique_words_text_test', 'number_of_StopWords_text_test', 
              'number_of_uppercase_text_test','average_of_wordsLength_text_test']]

y_test = test['Target']

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(  ngram_range=(1,1), 
                                     analyzer='word',
                                     stop_words='english', 
                                     lowercase=True, 
                                     max_df=0.9, # remove too frequent words
                                     min_df=10, # remove too rare words
                                     max_features = None, # max words in vocabulary, will keep most frequent words
                                     binary=False #If True, all non zero counts are set to 1. This is useful for discrete probabilistic models that model binary events rather than integer counts.
                                  )



#Stemmed questions vectorzation
X_text_tfidf_vectorizer_train = tfidf_vectorizer.fit_transform(X_train['stemming_text_for_tfidf'])
X_text_tfidf_vectorizer_test = tfidf_vectorizer.transform(X_test['stemming_text_for_tfidf_test'])

#Lemmentized questions vectorization
X_text_tfidf_Lem_vect_train = tfidf_vectorizer.fit_transform(X_train['lemmatize_text_for_tfidf'])
X_test_tfidf_Lem_vect_test = tfidf_vectorizer.transform(X_test['lemmatize_text_for_tfidf_test'])


#bigram text vectorization
bigram_vectorizer = TfidfVectorizer(  ngram_range=(1,2), 
                                     analyzer='word',
                                     stop_words='english', 
                                     lowercase=True, 
                                     max_df=0.9, # remove too frequent words
                                     min_df=10, # remove too rare words
                                     max_features = None, # max words in vocabulary, will keep most frequent words
                                     binary=False #If True, all non zero counts are set to 1. This is useful for discrete probabilistic models that model binary events rather than integer counts.
                                  )
X_text_bigram_vectorizer_train = bigram_vectorizer.fit_transform(X_train['stemming_text_for_tfidf'])
X_text_bigram_vectorizer_test = bigram_vectorizer.transform(X_test['lemmatize_text_for_tfidf_test'])


#T3gram questions vectorization
t3gram_vectorizer = TfidfVectorizer(  ngram_range=(1,4), 
                                     analyzer='word',
                                     stop_words='english', 
                                     lowercase=True, 
                                     max_df=0.9, # remove too frequent words
                                     min_df=10, # remove too rare words
                                     max_features = None, # max words in vocabulary, will keep most frequent words
                                     binary=False #If True, all non zero counts are set to 1. This is useful for discrete probabilistic models that model binary events rather than integer counts.
                                  )
X_text_t3gram_vectorizer_train = t3gram_vectorizer.fit_transform(X_train['stemming_text_for_tfidf'])
X_text_t3gram_vectorizer_test  = t3gram_vectorizer.transform(X_test['stemming_text_for_tfidf_test'])

#Range single word to t3gram text vectorization
st3gram_vectorizer = TfidfVectorizer(  ngram_range=(1,3), 
                                     analyzer='word',
                                     stop_words='english', 
                                     lowercase=True, 
                                     max_df=0.9, # remove too frequent words
                                     min_df=10, # remove too rare words
                                     max_features = None, # max words in vocabulary, will keep most frequent words
                                     binary=False #If True, all non zero counts are set to 1. This is useful for discrete probabilistic models that model binary events rather than integer counts.
                                  )
X_text_Singt3gram_vectorizer_train = st3gram_vectorizer.fit_transform(X_train['stemming_text_for_tfidf'])
X_text_Singt3gram_vectorizer_test  = st3gram_vectorizer.transform(X_test['stemming_text_for_tfidf_test'])

selected_text_tfidf_vectorizer_train = tfidf_vectorizer.fit_transform(train['stemming_selected_text_for_tfidf'])

selected_text_tfidf_Lem_vect_train = tfidf_vectorizer.fit_transform(train['lemmatize_selected_text_for_tfidf'])

selected_text_bigram_vectorizer_train = bigram_vectorizer.fit_transform(train['stemming_selected_text_for_tfidf'])

selected_text_t3gram_vectorizer_train = t3gram_vectorizer.fit_transform(train['stemming_selected_text_for_tfidf'])
selected_text_Singt3gram_vectorizer_train = st3gram_vectorizer.fit_transform(train['stemming_selected_text_for_tfidf'])


#Word2Vec with preprocessiong text (without stopwords) 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

d2v_training_data = []
for i, doc in enumerate(X_train['stemming_text']):
    d2v_training_data.append(TaggedDocument(words=doc,tags=[i]))

# ========== learning doc embeddings with doc2vec ==========

# PV stands for 'Paragraph Vector'
# PV-DBOW (distributed bag-of-words) dm=0

d2v = Doc2Vec(d2v_training_data, vector_size=300, window=10, alpha=0.1, min_alpha=1e-4, dm=0, negative=1, epochs=10, min_count=2, workers=4)
d2v_vecs = np.zeros((len(X_train['stemming_text']), 300))
for i in range(len(X_train['stemming_text'])):
    d2v_vecs[i,:] = d2v.docvecs[i]
    
d2v_test = np.zeros((len(X_test['stemming_text_test']), 300))
for i in range(len(X_test['stemming_text_test'])):
    d2v_test[i,:] = d2v.infer_vector(X_test['stemming_text_test'].iloc[i])
    

#Word2Vec with lemmatize words
d2v_training_data = []
for i, doc in enumerate(X_train['lemmatize_text']):
    d2v_training_data.append(TaggedDocument(words=doc,tags=[i]))

# ========== learning doc embeddings with doc2vec ==========

# PV stands for 'Paragraph Vector'
# PV-DBOW (distributed bag-of-words) dm=0

d2v = Doc2Vec(d2v_training_data, vector_size=200, window=5, alpha=0.1, min_alpha=1e-4, 
              dm=0, negative=1, epochs=10, min_count=2, workers=4)
d2v_vecs_bigram = np.zeros((len(X_train['lemmatize_text']), 200))
for i in range(len(X_train['lemmatize_text'])):
    d2v_vecs_bigram[i,:] = d2v.docvecs[i]
    
d2v_test_bigram = np.zeros((len(X_test['lemmatize_text_test']), 200))
for i in range(len(X_test['lemmatize_text_test'])):
    d2v_test_bigram[i,:] = d2v.infer_vector(X_test['lemmatize_text_test'].iloc[i])
    

X_train
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectPercentile
from sklearn.pipeline import Pipeline

features = SelectKBest(mutual_info_classif,k=2).fit(X_train[['text_lenght', 'number_punctuation_text','number_of_Unique_words_text', 'number_of_StopWords_text', 
                                                             'number_of_uppercase_text','average_of_wordsLength_text']].fillna(0),y_train)
independance_test = np.zeros((6,2))
for idx,i in enumerate(['text_lenght', 'number_punctuation_text', 'number_of_Unique_words_text', 'number_of_StopWords_text', 'number_of_uppercase_text',
                        'average_of_wordsLength_text']):
    #independance_test[idx,0]= features.pvalues_[idx]
    independance_test[idx,1]= features.scores_[idx]
    #print (i,features.pvalues_[idx],features.scores_[idx])
    #print('%s  %s'%(i,features.scores_[idx]))

    
    
list_var=['text_lenght', 'number_punctuation_text','number_of_Unique_words_text', 'number_of_StopWords_text', 'number_of_uppercase_text',
          'average_of_wordsLength_text']
independance_df = pd.DataFrame({'Variables': list_var, 'p_values': independance_test[:,0], 'MI': independance_test[:,1]},index=None)
independance_df
plt.figure(figsize=(12, 10))
_ = sns.heatmap(train[['text_lenght', 'number_punctuation_text','number_of_Unique_words_text', 'number_of_StopWords_text', 'number_of_uppercase_text','average_of_wordsLength_text']].corr()
                ,cmap="YlGnBu", annot=True, fmt=".2f")
plt.savefig("Correlation Matrice")
plt.show()
#!pip install hpelm
from sklearn import svm 

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, KFold
random_state = 42
kf = KFold(n_splits=2,random_state=random_state)
n_iter= 50
from sklearn.neural_network import MLPClassifier
Model_final_MLPClassifier = MLPClassifier(random_state=random_state).fit(X_text_tfidf_vectorizer_train,y_train)
## Predictions 

Predictions = Model_final_MLPClassifier.predict(X_text_tfidf_vectorizer_test)
sample_submission
sample_submission['selected_text'] = Predictions
sample_submission['selected_text'] = sample_submission['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
sample_submission['selected_text'] = sample_submission['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
sample_submission['selected_text'] = sample_submission['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head()
sentiment_list = { 0: 'negative', 
                  2 : 'positive', 
                  1: 'neutral'}
#sentiment_list

sample_submission['Sentiment_pred'] = sample_submission['Sentiment_preds'].map(sentiment_list)
sample_submission.head()
sample_submission['text2'] = sample_submission["text"].apply(lambda x: x.split())
sample_submission
text2 = sample_submission['text2']
text2
text2 = [l[-int(Predictions.tolist()[ind]):] for ind, l in enumerate(text2)]
text2[:5]
sample_submission['text22'] = text2
sample_submission.head()
sample_submission['selected_text'] = sample_submission["text22"].apply(lambda x: " ".join(x))
sample_submission
submission   = sample_submission[['textID', "selected_text"]]
submission
submission.to_csv('submission.csv', index=False)

