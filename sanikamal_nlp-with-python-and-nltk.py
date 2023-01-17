import numpy as np

import pandas as pd

import nltk

# nltk.download()

dir(nltk)
raw_data = open('../input/sms-spam-collection-dataset/spam.csv',encoding='ISO-8859-1').read()

raw_data[0:500]

parsed_data = raw_data.replace('\t','\n').split('\n')

parsed_data[0:10]
label_list = parsed_data[0::2]

msg_list = parsed_data[1::2]

print(label_list[0:5])

print(msg_list[0:5])
print(len(label_list))

print(len(msg_list))



print(label_list[-3:])



combined_df = pd.DataFrame({

    'label': label_list[:-1],

    'sms': msg_list

})





combined_df.head()
dataset = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', sep=",",encoding='ISO-8859-1')

dataset.head()
dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)

dataset.info()
dataset.columns=['label','sms']

dataset.head()
dataset.tail()
print(f'Inpute data has {len(dataset)} rows, {len(dataset.columns)} columns')
print(f'ham = {len(dataset[dataset["label"] == "ham"])}')

print(f'spam = {len(dataset[dataset["label"] == "spam"])}')


print(f"Numbers of missing label = {dataset['label'].isnull().sum()}")

print(f"Numbers of missing msg = {dataset['sms'].isnull().sum()}")
import string

string.punctuation
def remove_punctuation(txt):

    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])

    return txt_nopunct
dataset['msg_clean'] = dataset['sms'].apply(lambda x: remove_punctuation(x))

dataset.head()
import re



def tokenize(txt):

    tokens = re.split('\W+', txt)

    return tokens





dataset['msg_clean_tokenized'] = dataset['msg_clean'].apply(lambda x: tokenize(x.lower()))



dataset.head()
stopwords = nltk.corpus.stopwords.words('english')

stopwords[0:10]
def remove_stopwords(txt_tokenized):

    txt_clean = [word for word in txt_tokenized if word not in stopwords]

    return txt_clean



dataset['msg_no_sw'] = dataset['msg_clean_tokenized'].apply(lambda x: remove_stopwords(x))

dataset.head()
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()

# dir(porter_stemmer)
print(porter_stemmer.stem('programer'))

print(porter_stemmer.stem('programming'))

print(porter_stemmer.stem('program'))
print(porter_stemmer.stem('run'))

print(porter_stemmer.stem('running'))
def stemming(tokenized_text):

    text = [porter_stemmer.stem(word) for word in tokenized_text]

    return text
dataset['msg_stemmed'] = dataset['msg_no_sw'].apply(lambda x: stemming(x))

dataset.head()
# WordNet lexical database for lemmatization

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()



print(wordnet_lemmatizer.lemmatize('goose'))

print(wordnet_lemmatizer.lemmatize('geese'))
def lemmatization(token_txt):

    text = [wordnet_lemmatizer.lemmatize(word) for word in token_txt]

    return text
dataset['msg_lemmatized'] = dataset['msg_no_sw'].apply(lambda x : lemmatization(x))

dataset.head()
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()



corpus = ["This is a sentence is",

         "This is another sentence",

         "third document is here"]





X = cv.fit(corpus)

print(X.vocabulary_)

print(cv.get_feature_names())
X = cv.transform(corpus)

#X = cv.fit_transform(corpus)

print(X.shape)

print(X)

print(X.toarray())



df = pd.DataFrame(X.toarray(), columns = cv.get_feature_names())

print(df)
def clean_text(txt):

    txt = "".join([c for c in txt if c not in string.punctuation])

    tokens = re.split('\W+', txt)

    txt = [porter_stemmer.stem(word) for word in tokens if word not in stopwords]

    return txt



cv1 = CountVectorizer(analyzer=clean_text)



X = cv1.fit_transform(dataset['sms'])

print(X.shape)

print(cv1.get_feature_names())
data_sample = dataset[0:10]

cv2 = CountVectorizer(analyzer=clean_text)



X = cv2.fit_transform(data_sample['sms'])

print(X.shape)



df = pd.DataFrame(X.toarray(), columns=cv2.get_feature_names())

df.head(10)
def clean_text(txt):

    txt = "".join([c for c in txt if c not in string.punctuation])

    tokens = re.split('\W+', txt)

    txt = " ".join([porter_stemmer.stem(word) for word in tokens if word not in stopwords])

    return txt



dataset['sms_clean'] = dataset['sms'].apply(lambda x: clean_text(x))

dataset.head()
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(2,3))



corpus = ["This is a sentence is",

         "This is another sentence",

         "third document is here"]



#X = cv.fit(corpus)

#print(X.vocabulary_)

#print(cv.get_feature_names())



#X = cv.transform(corpus)

X = cv.fit_transform(corpus)

print(X.shape)

#print(X)

#print(X.toarray())



df = pd.DataFrame(X.toarray(), columns = cv.get_feature_names())

print(df)
cv1 = CountVectorizer(ngram_range=(2,2))



X = cv1.fit_transform(dataset['sms_clean'])

print(X.shape)
print(cv1.get_feature_names())

data_sample = dataset[0:10]

cv2 = CountVectorizer(ngram_range=(2,2))



X = cv2.fit_transform(data_sample['sms_clean'])

print(X.shape)
df = pd.DataFrame(X.toarray(), columns=cv2.get_feature_names())

df.head(10)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()



corpus = ["This is a sentence is",

         "This is another sentence",

         "third document is here"]





X = tfidf_vect.fit(corpus)

print(X.vocabulary_)

print(tfidf_vect.get_feature_names())



X = tfidf_vect.transform(corpus)

#X = cv.fit_transform(corpus)

print(X.shape)

print(X)

print(X.toarray())



df = pd.DataFrame(X.toarray(), columns = tfidf_vect.get_feature_names())

print(df)
data_sample = dataset[0:10]

tfidf2 = TfidfVectorizer(analyzer=clean_text)



X = tfidf2.fit_transform(data_sample['sms'])

print(X.shape)
df = pd.DataFrame(X.toarray(), columns=tfidf2.get_feature_names())

df.head(10)