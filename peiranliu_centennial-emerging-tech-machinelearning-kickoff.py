# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import nltk

messages=[line.rstrip() for line in open('../input/CleanDataSetForEmerTech')]
print(len(messages))
for mess_no,message in enumerate(messages[:10]):

    print(mess_no,message)

    print('\n')
import pandas as pd

messages=pd.read_csv('../input/CleanDataSetForEmerTech',sep='\t',names=['label','message'])
messages.head()
messages.describe()
messages.groupby('label').describe()
messages['length']=messages['message'].apply(len)
messages.head()
import matplotlib.pyplot as plt

import seaborn as sns #using seaborn do data visualization

%matplotlib inline
messages['length'].plot.hist(bins=50) # see distribution
messages['length'].describe()
messages[messages['length']==910]['message'].iloc[0]
messages.hist(column='length',by='label',bins=60,figsize=(12,4))
import string  #ready for clean the English Punctuation Data

from nltk.corpus import stopwords
def text_process(mess):   #create a function to clean your data by those step

    """

    remove punc

    remove stop words

    return list of clean text words

    """

    

    nopunc=[char for char in mess if char not in string.punctuation]

    

    nopunc=''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
messages.head()
messages['message'].head(5).apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer=CountVectorizer(analyzer=text_process).fit(messages['message'])
mess4 = messages['message'][3]

print(mess4)
bow4 = bow_transformer.transform([mess4])

print(bow4)
print(bow4.shape)
bow_transformer.get_feature_names()[4068]
messages_bow=bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix:', messages_bow.shape)
messages_bow.nnz
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

print('sparsity: {}'.format((sparsity)))   #show how many zero in your actual matrix
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer= TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)   #TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).*

                #*TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).*

#**TF: Term Frequency**，它度量一个术语在文档中出现的频率。由于每个文档的长度都不同，

# 所以在长文档中出现的术语可能比短文档中出现的次数要多得多。因此，术语频率通常除以文档长度(即。作为标准化的一种方法:

# *TF(t) = (t项在文档中出现的次数)/(文档中出现的总次数).*

# **IDF:逆文档频率**，它度量一个术语的重要性。在计算TF时，所有项都被认为是同等重要的。然而，我们知道，某些术语，如“is”、“of”和“that”，

# 可能出现很多次，但并不重要。因此，我们需要通过计算下面的方法，在减少频繁项的同时，增加罕见项的数量:

# *IDF(t) = log_e(文档总数/包含t项的文档数量).*

messages_tfidf=tfidf_transformer.transform(messages_bow)
from sklearn.naive_bayes import MultinomialNB
noserious_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])
noserious_detect_model.predict(tfidf4)[0]
messages['label'][3]
all_pred = noserious_detect_model.predict(messages_tfidf)
all_pred
#start training our model for detecting

#from sklearn.cross_validation import train_test_split   out of the current training version

from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=0.3)
from sklearn.pipeline import Pipeline
pipeline = Pipeline([                # create artificial intelligence Pipe line 

    ('bow',CountVectorizer(analyzer=text_process)),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB())

    

])
pipeline.fit(msg_train,label_train)  #fitting the train data
predictions = pipeline.predict(msg_test)
from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))   # get prediction result