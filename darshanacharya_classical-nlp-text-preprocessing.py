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
import numpy as np 
import pandas as pd

df = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df.head()
df['review'].duplicated().sum()
df.drop_duplicates(subset = 'review' , keep = False , inplace = True)
df.shape
df['review'].describe()
import re
def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

import string
def remove_punctuation(text):
    s = string.punctuation  # Sample string
    rem_punc = text.translate(str.maketrans('', '', s))
    return rem_punc

def remove_url(text):
    rem_url = re.sub(r'https?://\S+', '', text)
    return rem_url
    
def denoise_text(text):
    text = clean_html(text)
    text = remove_punctuation(text)
    text = remove_url(text)
    return text

df['review'] = df['review'].apply(denoise_text)
print(df['review'].head())
def parse_sentence(sentence):
    sentence = sentence.lower()
    return sentence

df['review'] = df["review"].map(parse_sentence)
print(df['review'].head())

import nltk
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))

def remove_stopwords(text):   
    stopwords_removed = " ".join([word for word in text.split() if word not in stopwords])
    return stopwords_removed

df['review'] = df['review'].apply(remove_stopwords)
df['review'].head()
from nltk.stem import PorterStemmer 

def simple_stemmer(text): 
    ps = nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text 

df['review'] = df['review'].apply(simple_stemmer)
df['review'].head()
from nltk.stem import WordNetLemmatizer

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return lemmatized_text

df['review'] = df['review'].apply(lemmatize)

#Example
text = WordNetLemmatizer().lemmatize('eating', 'v')
print(text)
import spacy
nlp = spacy.load('en_core_web_sm')
text = 'Tesla, Inc. is an American electric vehicle and clean energy company based in  California. The company specializes in electric vehicle manufacturing.Elon Musk is the current CEO. The company values today at $300 billion'
doc = nlp(text)
for ent in doc.ents:
      print(ent.text, ent.label_)

from spacy import displacy 
displacy.render(doc, style='ent')
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

TEXT =  'Tesla, Inc. is an American electric vehicle and clean energy company based in  California. The company specializes in electric vehicle manufacturing.Elon Musk is the current CEO. The company values today at $300 billion'
#Apply word tokenization and part-of-speech tagging to the sentence.
def preprocess(sent):
    tokenized = nltk.word_tokenize(sent)
    tagged = nltk.pos_tag(tokenized)
    return tagged
#Sentence filtered with Word Tokenization

result = preprocess(TEXT)
print("POS_Tags for Sentence")
result
#Chunking Pattern 
pattern = 'NP: {<DT>?<JJ>*<NN>}'
#create a chunk parser and test it on our sentence.
cp = nltk.RegexpParser(pattern)
cs = cp.parse(result)
print(cs)
#split the dataset 

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2, random_state=42, shuffle=True)
print(train.shape, test.shape)
from sklearn.feature_extraction.text import CountVectorizer  

count_vect = CountVectorizer(max_features=5000)
bow_data = count_vect.fit_transform(train['review'])
print(bow_data[1])
count_vect = CountVectorizer(ngram_range=(1,2))
Bigram_data = count_vect.fit_transform(train['review'])
print(Bigram_data[1])
from sklearn.feature_extraction.text import TfidfVectorizer 
tf_idf = TfidfVectorizer(max_features=5000)
tf_data = tf_idf.fit_transform(train['review'])
print(tf_data)
from gensim.models import Word2Vec

splitted = []
w2v_data =  train['review']

for row in w2v_data: 
    splitted.append([word for word in row.split()])  #splitting words
    
train_w2v = Word2Vec(splitted,min_count=5,size=50, workers=4)
avg_data = []

for row in splitted:
    vec = np.zeros(50)
    count = 0
    for word in row:
        try:
            vec += train_w2v[word]
            count += 1
        except:
            pass
    avg_data.append(vec/count)
    
print(avg_data[1])
train['review']
data = train['review']
tf_idf = TfidfVectorizer(max_features=5000)
tf_idf_data = tf_idf.fit_transform(data)
tf_w_data = []
tf_idf_data = tf_idf_data.toarray() # converting to non-sparse array

i = 0
for row in splitted:
    vec = [0 for i in range(50)]
    
    temp_tfidf = []
    for val in tf_idf_data[i]:
        if val != 0:         
            temp_tfidf.append(val)
    
    count = 0
    tf_idf_sum = 0
    for word in row:
        try:
            count += 1
            tf_idf_sum = tf_idf_sum + temp_tfidf[count-1]
            vec += (temp_tfidf[count-1] * train_w2v[word])
        except:
            pass
    vec = (float)(1/tf_idf_sum) * vec
    tf_w_data.append(vec)
    i = i + 1

print(tf_w_data[1])

    
i = 0
for row in splitted:
    vec = [0 for i in range(50)]
    
    temp_tfidf = []
    for val in tf_idf_data[i]:
        if val != 0:
            temp_tfidf.append(val)
    
    count = 0
    tf_idf_sum = 0
    for word in row:
        try:
            count += 1
            tf_idf_sum = tf_idf_sum + temp_tfidf[count-1]
            vec += (temp_tfidf[count-1] * train_w2v[word])
        except:
            pass
    vec = (float)(1/tf_idf_sum) * vec
    tf_w_data.append(vec)
    i = i + 1

print(tf_w_data[1])
