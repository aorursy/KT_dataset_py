### This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from collections import  Counter

plt.style.use('ggplot')

stop=set(stopwords.words('english'))

import re

from nltk.tokenize import word_tokenize

import gensim

import string

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from sklearn.model_selection import train_test_split
tweet= pd.read_csv('../input/nlp-getting-started/train.csv')

test= pd.read_csv('../input/nlp-getting-started/test.csv')
tweet[tweet['target']==1]['text'].values[0]
tweet.head(10)
test.head(10)
tweet.shape
x= tweet.target.value_counts()
print(x)
sns.barplot(x.index,x)
y= tweet[tweet['target']==1]['text'].str.split()

print(y)
word_len=tweet[tweet['target']==1]['text'].str.split().str.len()

print(word_len)
plt.hist(word_len)
def create_corpus(target):

    corpus= []

    

    for l in y:

        for q in l:

            corpus.append(q)

    return(corpus)
corpus= create_corpus(1)



dic= defaultdict(int)



for i in corpus:

    if i in stop:

        dic[i]= dic[i]+1

top = sorted(dic.items(), key= lambda x: x[1], reverse= True)[:10]

m,n =zip(*top)

plt.bar(m,n)
df= pd.concat([tweet, test])

df.shape
def sterilization(data):

    

    data = re.sub('https?://\S+|www\.\S+', '', data)

    data = re.sub('<.*?>', '', data)

    emoj = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    emoj.sub(r'', data)

    data = data.lower()

    data = data.translate(str.maketrans('','', string.punctuation))

    data = re.sub(r'\[.*?\]', '', data)

    data = re.sub(r'\w*\d\w*','', data)

    



    return data

    

    
df['text']=df['text'].apply(lambda x : sterilization(x))

df.head(10)
from nltk.tokenize import RegexpTokenizer



tokenizer = RegexpTokenizer(r'\w+')



df["tokens"] = df["text"].apply(tokenizer.tokenize)

df.head()
word2vec_path = "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin"

word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
def filtered(text):

    words= [w for w in text if w not in stop]

    

    return words
df['tokens']= df['tokens'].apply(lambda x: filtered(x))

df.head()
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):

    if len(tokens_list)<1:

        return np.zeros(k)

    if generate_missing:

        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]

    else:

        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]

   

    length = len(vectorized)

    summed = np.sum(vectorized, axis=0)

    averaged = np.divide(summed, length)

    return averaged



def get_word2vec_embeddings(vectors, df, generate_missing=False):

    embeddings = df['tokens'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))

    return list(embeddings)
tweet.shape
X_train= df[:7613]

y_train= X_train['target']
X_test= df[7613:]
embeddings = get_word2vec_embeddings(word2vec, X_train)
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, y_train, test_size= 0.2, random_state=40)
from sklearn.linear_model import LogisticRegression

clf_w2v = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 

                         multi_class='multinomial', random_state=40)

clf_w2v.fit(X_train_word2vec, y_train_word2vec)

y_predicted_word2vec = clf_w2v.predict(X_test_word2vec)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report



def get_metrics(y_test, y_predicted):  

    # true positives / (true positives+false positives)

    precision = precision_score(y_test, y_predicted, pos_label=None,

                                    average='weighted')             

    # true positives / (true positives + false negatives)

    recall = recall_score(y_test, y_predicted, pos_label=None,

                              average='weighted')

    

    # harmonic mean of precision and recall

    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    

    # true positives + true negatives/ total

    accuracy = accuracy_score(y_test, y_predicted)

    return accuracy, precision, recall, f1



accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test_word2vec, y_predicted_word2vec)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec, 

                                                                       recall_word2vec, f1_word2vec))
test_embeds= get_word2vec_embeddings(word2vec, X_test)
from xgboost import XGBClassifier

from sklearn.metrics import f1_score

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
clf = XGBClassifier(colsample_bytree=0.7, learning_rate= 0.03, max_depth= 10,

                    min_child_weight=11, missing= -999, n_estimators= 1200,

                    nthread= 4, objective='binary:logistic', seed=1337, silent=1, subsample=0.8)
XTa= np.array(X_train_word2vec)

yTa= np.array(y_train_word2vec)
clf.fit(XTa,yTa)
y_predicted_word2vec = clf.predict(X_test_word2vec)
y_predicted_word2vec
accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test_word2vec, y_predicted_word2vec)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec, 

                                                                       recall_word2vec, f1_word2vec))
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")



submission["target"] = clf.predict(test_embeds)



submission.to_csv("submission.csv", index=False)
from IPython.display import FileLink

FileLink(r'submission.csv')