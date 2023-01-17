

import numpy as np # linear algebra

import pandas as pd # data processing,



from sklearn.feature_extraction import text



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string

from keras import layers, models, optimizers

import nltk



from nltk.corpus import stopwords

import os

import seaborn as sns

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from bokeh.io import show, output_file

import warnings 

%matplotlib inline

warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.corpus import stopwords

stop = stopwords.words('english')

from bokeh.palettes import Spectral4









!ls ../input

data = '../input/corpus.txt'

from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string

from keras.preprocessing import text, sequence



from keras import layers, models, optimizers
data = open('../input/corpus.txt').read()

labels, texts = [], []
for i, line in enumerate(data.split("\n")):

  content = line.split()

  labels.append(content[0:])

  texts.append(' '.join(content[1:]))
trainDF = pd.DataFrame()

trainDF['text'] = texts

trainDF['label'] = labels
trainDF['texts'] = trainDF['label'].map(lambda text: len(text))
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(trainDF['text'])
vec = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

vec.fit((trainDF['text']))
print([w for w in sorted(vec.vocabulary_.keys())])
import pandas as pd



jk=pd.DataFrame(vec.transform(['text']).toarray(), columns=sorted(vec.vocabulary_.keys()))
print(jk)

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(binary=True)
data2 = np.array(data)
#instantiate CountVectorizer()

docs=["data"]

from sklearn.feature_extraction.text import CountVectorizer



cv=CountVectorizer()

 

# this steps generates word counts for the words in your docs

word_count_vector=cv.fit_transform(docs)
word_count_vector.shape

from sklearn.feature_extraction.text import TfidfTransformer



tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

tfidf_transformer.fit(word_count_vector)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["lo"])

 

# sort ascending

df_idf.sort_values(by=['lo'])


labels, texts = [], []
for i, line in enumerate(data2.split("\n")):

    content = line.split()

    labels.append(content[0])

    texts.append(" ".join(content[1:]))
# load the dataset







# create a dataframe using texts and lables

trainDF = pandas.DataFrame()

trainDF['text'] = texts

trainDF['label'] = labels
data.cont('labels')['texts'].count()

trainDF.describe()

trainDF['texts'] = trainDF['label'].map(lambda text: len(text))

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['texts'], trainDF['label'])



train_x_index = train_x.index.tolist()

valid_x_index = valid_x.index.tolist()

train_y_index = train_y.index.tolist()

valid_y_index = valid_y.index.tolist()

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_x_index)

valid_y = encoder.fit_transform(valid_x_index)

train_x = encoder.fit_transform(train_y_index)

valid_x = encoder.fit_transform(valid_y_index)

print(train_y.shape)

print(valid_y.shape)

print(train_y[:])



print(max(train_x_index), len(train_x_index))

print(max(valid_x_index), len(valid_x_index))

print('train_x index:', sorted(train_x_index)[:15], sorted(train_x_index)[-5:])

print('valid_x index:', sorted(valid_x_index)[:15], sorted(valid_x_index)[-5:])

#train_x.head()

print(train_x)





print(train_x[:2], '\n')





print(1, train_x[:1], '\n')





print(train_x[:3], '\n')

count_vect = text.CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')



count_vect.fit(trainDF['text'])

vocab = count_vect.vocabulary_

reverse_vocab = { idx: word for word, idx in vocab.items() }
