import pandas as pd
import numpy as np
import re
import sklearn
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
df.info()
df.head()
df.groupby('Category').describe()
df['length'] = df['Message'].apply(len)
df.head()
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
df['length'].plot(bins=50, kind='hist') 
df.length.describe()
df.hist(column='length', by='Category', bins=20,figsize=(12,4))
from nltk.corpus import stopwords
import string
stopwords.words('english')[0:10] # Show some stop words
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    lemma = nlp.WordNetLemmatizer()
    nopunc = [ lemma.lemmatize(word) for word in nopunc]
df['Message'].head(5).apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(df['Message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))
messages_bow = bow_transformer.transform(df['Message'])
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state = 42)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))
msg_train.head()
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
nn=MLPClassifier(random_state=1)
pipeline_nn = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MLPClassifier()),  # train on TF-IDF vectors w/ SVM
])
pipeline_nn.fit(msg_train,label_train)
predictions_nn = pipeline_nn.predict(msg_test)
predictions_nn
print(classification_report(predictions_nn,label_test))
