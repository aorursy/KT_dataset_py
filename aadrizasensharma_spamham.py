import numpy as np
import pandas as pd 
import os
print(os.listdir("../input"))
message = [line.rstrip() for line in open('../input/smsspamcollection/SMSSpamCollection')]
print(len(message))
for message_no,message in enumerate(message[:10]):
    print(message_no,message)
    print('\n')
import pandas as pd
message=pd.read_csv('../input/smsspamcollection/SMSSpamCollection',sep='\t',names=["labels","message"])
message.head()
message.describe()
message.groupby('labels').describe()
message['length']=message['message'].apply(len)
message.head()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
message['length'].plot(bins=50,kind='hist')
message.length.describe()
message[message['length']==910]['message'].iloc[0]
import string
mess = 'my sample message!...'
nopunc=[char for char in mess if char not in string.punctuation]
nopunc=''.join(nopunc)
print(nopunc)
from nltk.corpus import stopwords
stopwords.words('english')[0:10]
nopunc.split()
clean_mess=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess
def text_process(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
message.head()
message['message'].head(5).apply(text_process)
message.head()
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(message['message'])
print(len(bow_transformer.vocabulary_))
message4=message['message'][3]
print(message4)
bow4=bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)
print(bow_transformer.get_feature_names()[4068])
print(bow_transformer.get_feature_names()[9554])
messages_bow = bow_transformer.transform(message['message'])
print('Shape of Sparse Matrix: ',messages_bow.shape)
print('Amount of non-zero occurences:',messages_bow.nnz)
from PIL import Image
sparsity =(100.0 * messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1]))
print('sparsity:{}'.format(round(sparsity)))
img_array = np.array(Image.open('../input/explanation/Capture.JPG'))
plt.figure(figsize=(16,10))
plt.imshow(img_array)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])
messages_tfidf=tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf,message['labels'])
print('predicted:',spam_detect_model.predict(tfidf4)[0])
print('expected:',message.labels[3])
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(message['labels'],all_predictions))
print(confusion_matrix(message['labels'],all_predictions))
from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test = train_test_split(message['message'],message['labels'],test_size=0.2)
print(len(msg_train),len(msg_test),len(label_train),len(label_test))
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
   ( 'bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB()),
])
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))