import nltk
messages = [line.rstrip() for line in open(r"../input/SMSSpamCollection")]
print(len(messages))
for mess_no,message in enumerate(messages[:10]):

    print(mess_no,message)
messages[0]
import pandas as pd

message = pd.read_csv(r"../input/SMSSpamCollection",sep = '\t',names = ['label','message'])
message.head()
message.describe()
#Organizing Data

message.groupby('label').describe()
message['length'] =  message['message'].apply(len)
message.head()
import matplotlib.pyplot as plt

import seaborn as sns
message['length'].plot.hist(bins=50)

#Bimodal Behaviour
message['length'].describe()
message[message['length'] == 910]['message'].iloc[0]

#Finding the longest message
message.hist(column = 'length', by = 'label',bins = 60,figsize=(12,8))

#Spam Messages tend to have more characters
#Converting into Vectors

#Remove StopWords

#Return a list of words

#Remove Punctuations
import string
mess = 'Sample Message! Notice: It has a punctuation.'
string.punctuation
no_punc = [mess for mess in mess if mess not in string.punctuation]
no_punc
from nltk.corpus import stopwords
#StopWords

stopwords.words('english')
bacttostring  = ''.join(no_punc)
bacttostring
Clean_Mess = [word for word in bacttostring.split() if word not in stopwords.words('english')]
Clean_Mess
def text_process(mess):

    '''

    1: Removie Punctuation

    2: Remove StopWords

    3: Return List of clean Text

    '''

    nopunc = [char for char in mess if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word not in stopwords.words('english')]
message.head()
message['message'][:5].apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer = text_process).fit(message['message'])
print(len(bow_transformer.vocabulary_))
mess4 = message['message'][3]
print(mess4)
bow4 = bow_transformer.transform([mess4])
print(bow4)
print(bow4.shape)
bow_transformer.get_feature_names()[4221]
bow_transformer.get_feature_names()[9746]
message_bow = bow_transformer.transform(message['message'])
Sparsity = (100.0* message_bow.nnz/(message_bow.shape[1]*message_bow.shape[0]))
print("Sparsity:{}".format(Sparsity))
from sklearn.feature_extraction.text import TfidfTransformer
tfdif_transformer = TfidfTransformer().fit(message_bow)
tfidf4 = tfdif_transformer.transform(bow4)
print(tfidf4)
tfdif_transformer.idf_[bow_transformer.vocabulary_['university']]
messages_tfidf = tfdif_transformer.transform(message_bow)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf,message['label'])
spam_detect_model.predict(tfidf4)[0]
all_pred = spam_detect_model.predict(messages_tfidf)
all_pred
from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test = train_test_split(message['message'],message['label'],test_size = .30)
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('bow',CountVectorizer(analyzer=text_process)),

                     ('tfidf',TfidfTransformer()),

                     ('classifier',MultinomialNB())

                    ])
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([('bow',CountVectorizer(analyzer=text_process)),

                     ('tfidf',TfidfTransformer()),

                     ('classifier',RandomForestClassifier())

                    ])
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
### Classification Report



from sklearn.metrics import classification_report



print(classification_report(label_test,predictions))