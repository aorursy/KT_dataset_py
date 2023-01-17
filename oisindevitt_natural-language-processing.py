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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
messages = [line.rstrip()for line in open(r'../input/smsspamcollection/SMSSpamCollection')]
print(len(messages)) #total amount of messages
messages[56]
for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')
messages[0] #below you can see \t, we are going to remove this by using pandas
messages = pd.read_csv(r'../input/smsspamcollection/SMSSpamCollection',sep='\t',names=['label','message'])
messages
messages.head() #we can see our two columns, note we seperated on tab remember
messages.describe() 
messages.groupby('label').describe()
messages['length'] = messages['message'].apply(len)
messages.head()
messages['length'].plot.hist(bins=150)
messages.describe()
messages[messages['length'] == 910] #so wrap the above to see it
messages[messages['length'] == 910]['message'].iloc[0] #then add the column and iloc[0] to see the whole thing
messages.hist(column= 'length', by='label',bins=60,figsize=(12,4))
import string
string.punctuation
mess = 'Sample Message!, Notice: It has no punctuation'
nopunctuation = [i for i in mess if i not in string.punctuation]
nopunctuation
nopunctuation = ''.join(nopunctuation) #now we have it in a usable format
nopunctuation
from nltk.corpus import stopwords 
stopwords.words('english')
clean_mess = [i for i in nopunctuation.split() if i.lower() not in stopwords.words('english')]
clean_mess
def text_process(mess):
    
    """
    1.  remove punctuation (punc)
    2.  remove stopwords (common words)
    3. return list of clean text words
    """
    
    nopunc = [i for i in mess if i not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return [i for i in nopunc.split() if i.lower() not in stopwords.words('english')]
messages['message'].head().apply(text_process) #you can see common words removed
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message']) #bow is bag of words
print(len(bow_transformer.vocabulary_)) #this just prints the total number of vocabulary words
mess4 = messages['message'][3]
mess4
bow4 = bow_transformer.transform([mess4])
print(bow4)
bow_transformer.get_feature_names()[6222] #you can use this to find out what a feature (word) is
messages_bow = bow_transformer.transform(messages['message']) #we are transforming the entire coulmn this time
print('Shape of Sparse Matrix:',messages_bow.shape)
messages_bow.nnz #these are the non zero occurances
sparsity = (100.0 * messages_bow.nnz/(messages_bow.shape[0] * messages_bow.shape[0]))
print('sparsity: {}'.format(sparsity))
from sklearn.feature_extraction.text import TfidfTransformer
TfidfTransformer = TfidfTransformer().fit(messages_bow)
tfidf4 = TfidfTransformer.transform(bow4)
print(tfidf4)
TfidfTransformer.idf_[bow_transformer.vocabulary_['university']]
messages_tfidf = TfidfTransformer.transform(messages_bow)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label']) #assign variable and fit model on transfomer and data  
spam_detect_model.predict(tfidf4)[0] #single use
pred_all = spam_detect_model.predict(messages_tfidf) #whole dataframe
pred_all
from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=0.3)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
pipeline = Pipeline([
        ('bow',CountVectorizer(analyzer=text_process)),
        ('tfidf',TfidfTransformer()),
        ('classifier',MultinomialNB())
    ])
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))