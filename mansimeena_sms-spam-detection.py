import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
import numpy as np # linear algebra
import pandas as pd # data processing

#importing visualising libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
%matplotlib inline

#NLP
import nltk
from nltk.corpus import stopwords

#Data Cleaning
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#Training Model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
msgs = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
msgs.head()
#Dropping unnecessary columns
msgs.drop(msgs.columns[[2, 3, 4]], axis = 1, inplace = True)
#Renaming columns
msgs.rename(columns = {'v1': 'label', 'v2': 'message'},inplace=True)
msgs.head()
msgs.describe()
msgs.info()
msgs.groupby('label').describe()
Category_count=np.array(msgs['label'].value_counts())
labels=sorted(msgs['label'].unique())
fig = go.Figure(data=[go.Pie(labels=labels, values=Category_count, hole=.3)])
fig.show()
msgs['length'] = msgs['message'].apply(len)
msgs.head()
fig = px.histogram(msgs, x="length",color="label")
fig.show()
msgs.hist(column='length', by='label',bins=50, figsize=(10,4))
msgs.length.describe()
msgs[msgs['length']==910]['message'].iloc[0]
#Forming function for msgs
#Removing punctuations and stopwords
def text_process(mess):
  nopunc=[char for char in mess if char not in string.punctuation]
  nopunc=''.join(nopunc)
  return [word for word in nopunc.split() if word.lower() not in stopwords.words('english') ]
#making lists of tokens(lemmas)
msgs['message'].apply(text_process)
#converting text doc to a matrix of token counts using scikit countvectorizer 
bow_transformer = CountVectorizer(analyzer=text_process).fit(msgs['message'])

# Print total number of vocab words
print (len(bow_transformer.vocabulary_))
#calculating sparsity
msgs_bow=bow_transformer.transform(msgs['message'])
print ('Shape of Sparse Matrix:{}',format(msgs_bow.shape))
print ('Amount of Non-Zero occurences:{}',format(msgs_bow.nnz))
print ('sparsity: %.2f%%' % (100.0 * msgs_bow.nnz / (msgs_bow.shape[0] * msgs_bow.shape[1])))
print(msgs_bow)
tfidf_transformer = TfidfTransformer().fit(msgs_bow)
msgs_tfidf = tfidf_transformer.transform(msgs_bow)
print (msgs_tfidf.shape)
print(msgs_tfidf)
spam_detect_model=MultinomialNB().fit(msgs_tfidf, msgs['label'])
#test
all_predictions=spam_detect_model.predict(msgs_tfidf)
print(all_predictions)
from sklearn.metrics import classification_report
print(classification_report(msgs['label'],all_predictions))