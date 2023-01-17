import nltk
import pandas as pd
#nltk.download_shell()
#messages=[line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
messages = pd.read_csv('../input/spam.csv',encoding='latin-1')
messages.head(2)
messages.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace = True)
import pandas as pd
column_names=['label','msg']
messages.columns = column_names
messages.head()
messages.groupby('label').describe()
messages['length']=messages['msg'].apply(len)
messages.head()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.figure(figsize=(10,4))
messages['length'].plot.hist(bins=200)
messages['length'].describe()
messages[messages['length']==910]['msg'].iloc[0]
messages.hist(column='length',by='label',bins=100,figsize=(10,4))

import string
from nltk.corpus import stopwords
stopwords.words('english')
def text_process(mess):
    '''
    1.remove punc
    2.remove stop words
    3.return clean text
    '''
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
messages.head()
messages['msg'].head().apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer=CountVectorizer(analyzer=text_process).fit(messages['msg'])
print(len(bow_transformer.vocabulary_))
mess4=messages['msg'][3]
mess4
bow_transformer
bow4=bow_transformer.transform([mess4])
print(bow4)
bow_transformer.get_feature_names()[4551]
type(bow_transformer)
messages_bow=bow_transformer.transform(messages['msg'])
print('shape of sparse matrix:',messages_bow.shape)
messages_bow.nnz
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(messages_bow)
tfidf4=tfidf_transformer.transform(bow4)
print(tfidf4)
tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]
messages_tfidf = tfidf_transformer.transform(messages_bow)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])
spam_detect_model.predict(tfidf4)
all_pred=spam_detect_model.predict(messages_tfidf)
all_pred
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test=train_test_split(messages['msg'],messages['label'],test_size=0.3)
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))
