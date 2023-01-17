import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#Reading the data



data = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')

data.head()
#Removing the columns that are not needed



data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

data = data.rename(columns={"v1":"label", "v2":"body_text"})
data.describe()
data.groupby("label").describe()
#The shape of the dataset



print("Input data has {} rows and {} columns".format(len(data), len(data.columns)))
#How many spam/ham are there



print("Out of the total {} rows, {} are spam, {} are ham".format(len(data),

                                                       len(data[data['label']=='spam']),

                                                       len(data[data['label']=='ham'])))
data.info()
#How much missing data is there



print("Number of null in label: {}".format(data['label'].isnull().sum()))

print("Number of null in text: {}".format(data['body_text'].isnull().sum()))
import string

string.punctuation
def remove_punct_num(text):

    text_nopunct= "".join([char for char in text if char not in string.punctuation])

    text_nonum=''.join([i for i in text_nopunct if not i.isdigit()])

    return text_nonum



data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punct_num(x))



data.head()
import re
def tokenize(text):

    tokens = re.split('\W+', text)

    return tokens



data['body_text_clean'] = data['body_text_clean'].apply(lambda x: tokenize(x.lower()))



data.head()
stopword = nltk.corpus.stopwords.words('english')
stopword[0:100:10]
def remove_stopwords(tokenized_list):

    text = [word for word in tokenized_list if word not in stopword]

    return text



data['body_text_clean'] = data['body_text_clean'].apply(lambda x: remove_stopwords(x))



data.head()
ps = nltk.PorterStemmer()

ps
print(ps.stem('grows'))

print(ps.stem('growing'))

print(ps.stem('grow'))
print(ps.stem('run'))

print(ps.stem('running'))

print(ps.stem('runner'))
print(ps.stem("fast"))

print(ps.stem("fasting"))

print(ps.stem("fastest"))
#Stemming our data



def stemming(input_text):

    text = [ps.stem(word) for word in input_text]

    return text



data['body_text_stemmed'] = data['body_text_clean'].apply(lambda x: stemming(x))



data.head(10)
wn = nltk.WordNetLemmatizer()

wn
print(ps.stem('meanness'))

print(ps.stem('meaning'))
print(wn.lemmatize('meanness'))

print(wn.lemmatize('meaning'))
print(ps.stem('thinking'))

print(ps.stem('thinker'))
print(wn.lemmatize('thinking'))

print(wn.lemmatize('thinker'))
def lemmatizing(input_text):

    text = [wn.lemmatize(word) for word in input_text]

    return text



data['body_text_lemmatized'] = data['body_text_clean'].apply(lambda x: lemmatizing(x))



data.head(10)
data_vector= data[["label","body_text_lemmatized"]]
data_vector.head()
len(data_vector)
for i in range(0,5572):

    st=data_vector["body_text_lemmatized"][i]

    new_st=" ".join(st)

    data_vector["body_text_lemmatized"][i]=new_st
data_vector.head()
from sklearn.feature_extraction.text import CountVectorizer



count_vect = CountVectorizer()



X_counts = count_vect.fit_transform(data_vector["body_text_lemmatized"])
print(X_counts.shape)
print(count_vect.get_feature_names())
X_counts_df = pd.DataFrame(X_counts.toarray())

X_counts_df
ngram_vect = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X_counts2 = ngram_vect.fit_transform(data_vector["body_text_lemmatized"])
print(X_counts2.shape)
print(ngram_vect.get_feature_names())
X_counts_df2 = pd.DataFrame(X_counts2.toarray())

X_counts_df2
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vect = TfidfVectorizer()
X_tfidf = tfidf_vect.fit_transform(data_vector["body_text_lemmatized"])
print(X_tfidf.shape)
print(tfidf_vect.get_feature_names())
idf_df=pd.DataFrame(X_tfidf.toarray())

idf_df