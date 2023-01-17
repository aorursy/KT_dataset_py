import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data=pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')
data.head()
data=data.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
data.columns=['type','message']
data.head()
data.groupby('type').describe()
data['length'] = data['message'].apply(len)

data.head()
data.length.describe()
sns.set_style('darkgrid')

data['length'].plot(bins=50, kind='hist') 
import string 

from nltk.corpus import stopwords
def clean_text(message):

        no_punc=[char for char in message if char not in string.punctuation]

        no_punc=''.join(no_punc)

        return[word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
data.head()
data['message'].head(5).apply(clean_text)
from sklearn.model_selection import train_test_split
msg_train,msg_test,type_train,type_test=train_test_split(data['message'],data['type'],test_size=.2)
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

pipeline = Pipeline([

    ('bag_of_words', CountVectorizer(analyzer=clean_text)),  # strings to token integer counts

    ('tf-idf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),

])
pipeline.fit(msg_train,type_train)
predictions = pipeline.predict(msg_test)
from sklearn.metrics import classification_report,confusion_matrix
print('classification matrix is',classification_report(predictions,type_test))

print('confusion matrix is',confusion_matrix(predictions,type_test))