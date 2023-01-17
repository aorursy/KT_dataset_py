

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
messages = pd.read_csv('../input/spam.csv', delimiter = ',', encoding='latin-1')

messages.head()
messages.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

messages.info()
sns.countplot(messages.v1)

plt.xlabel('Label')

plt.title('Number of ham and spam messages')
messages.head()
messages.describe()
messages.groupby('v1').describe()
messages['length'] = messages['v2'].apply(len)
messages.head()
messages['length'].plot.hist(bins=50)
messages['length'].describe()
messages.hist(column='length', by='v1', bins=60, figsize=(12,4));
import string

string.punctuation
from nltk.corpus import stopwords

stopwords.words("english")[100:110]
mess = 'Sample Message! Notice: it has punctuation.'

nopunc = [c for c in mess if c not in string.punctuation]
nopunc = "".join(nopunc)

nopunc
nopunc.split()
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess
def text_process(mess):

    '''

    1.remove punctuaton

    2.remove stop words

    3. return list of clean text words

    '''

    nopunc = [char for char in mess if char not in string.punctuation]

    nopunc="".join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    
messages.head()
messages = messages.rename(columns={"v1": "label", "v2": "message"})
messages['spam'] = messages['label'].map({'spam': 1, 'ham': 0}).astype(int)

messages.head(10)
messages['message'].head(5).apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
print(len(bow_transformer.vocabulary_))
# grab the 4th message



mess4 = messages['message'][3]

print(mess4)
bow4 = bow_transformer.transform([mess4])
print(bow4)
print(bow4.shape)
bow_transformer.get_feature_names()[3996]
bow_transformer.get_feature_names()[9445]
# apply to whole dataframe

messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix', messages_bow.shape)
messages_bow.nnz  # non zero occurance
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4) # weight values for each of the word 
messages_tfidf = tfidf_transformer.transform(messages_bow)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(messages_tfidf, messages['label'])
model.predict(tfidf4)
messages['label'][3]
all_pred = model.predict(messages_tfidf)
all_pred
from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3, random_state=42)
from sklearn.pipeline import Pipeline
pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=text_process)),

    ('tfidf', TfidfTransformer()),

    ('classifier', RandomForestClassifier())

])
pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)
from sklearn.metrics import classification_report
print(classification_report(label_test, predictions))