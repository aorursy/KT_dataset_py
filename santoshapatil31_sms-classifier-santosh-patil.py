import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import string
#importing sms data set

sms = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')

sms.head()



sms = sms.rename(columns={"v1":"label", "v2":"message"})

sms.head()

sms.describe()

sms.groupby('label').describe()

sms = sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
sms.head()
sms.describe()

sms.groupby('label').describe()
sms.drop_duplicates(inplace=True)

sms.groupby('label').describe()
countspam=0

countham=0

unlabelled=0

for x in sms['label']:

    if x=='spam': 

        countspam=countspam+1

    elif x=='ham':

        countham=countham+1

    else:

        unlabeled=unlabeled+1

print("Total number of spam messages",countspam)        

print("Total number of ham messages",countham)  

print("Total number of unlabelled messages",unlabelled)  
sms['length'] = sms['message'].apply(len)

sms.head();

plt.title('Histogram distribution of the spam and not spam messages')

sms['length'].hist(bins=60,figsize=(12,5))

plt.xlim(-40,900)

plt.xlabel('length')

plt.ylabel('frequency')

plt.show()







sms['length'].hist(by=sms['label'],bins=60,figsize=(12,5))

plt.xlim(-40,900);

plt.xlabel('length')

plt.ylabel('frequency')

plt.show()
# find number of words in the message

def word_counter(msg):

    no_words = sum([i.isalpha() for i in msg.split()])

    return no_words
sms['word count']=sms['message'].apply(word_counter)





plt.title('Histogram distribution of number of words in both spam and not spam messages')

sms['length'].hist(bins=100,figsize=(12,5))

plt.xlabel('No. of words')

plt.ylabel('frequency')

plt.show()







sms['word count'].hist(by=sms['label'],bins=50,figsize=(12,5))

plt.xlabel('No. of words')

plt.ylabel('frequency')

plt.show()
sms['labels'] = sms['label'].map({'ham': 0, 'spam': 1})

sms.head()
# function to delete punctuations by python string 

print('List of supported punctuations',string.punctuation)

def delete_punctuation(msg):

    new_msg=''.join([p for p in msg if p not in string.punctuation])

    return new_msg

sms['del_p_message']=sms['message'].apply(delete_punctuation)



sms['del_p_length'] = sms['del_p_message'].apply(len)

sms.head()
plt.title('Histogram distribution of total characters spam and not spam messages after deleting punctuations')

sms['del_p_length'].hist(bins=250,figsize=(12,5))

plt.xlim(-40,900)

plt.xlabel('length')

plt.ylabel('frequency')

plt.show()





sms['del_p_length'].hist(by=sms['label'],bins=200,figsize=(12,5))

plt.xlim(-40,900);

plt.xlabel('length')

plt.ylabel('frequency')

plt.show()





sms['words after removing punct.']=sms['del_p_message'].apply(word_counter)





plt.title('Histogram distribution of number of words in both spam and not spam messages after removing punctuation')

sms['words after removing punct.'].hist(bins=100,figsize=(12,5))

plt.xlabel('No. of words')

plt.ylabel('frequency')

plt.show()







sms['words after removing punct.'].hist(by=sms['label'],bins=50,figsize=(12,5))

plt.xlabel('No. of words')

plt.ylabel('frequency')

plt.show()



from nltk.corpus import stopwords

stopwords.words("english")



# definie function to delete stop words

def delete_stopwords(msgp):

    msg_no_stopwords=' '.join([x for x in msgp.split() if x.lower() not in stopwords.words("english")])

    

    return msg_no_stopwords



sms['del_SW_message']=sms['del_p_message'].apply(delete_stopwords)



sms['del_SW_length']=sms['del_SW_message'].apply(len)

sms['del_SW_message'].head()

sms.head()
plt.title('Histogram distribution of the spam and not spam messages after deleting Stop words')

sms['del_SW_length'].hist(bins=50,figsize=(12,5))

plt.xlim(-40,900)

plt.xlabel('length')

plt.ylabel('frequency')

plt.show()





sms['del_SW_length'].hist(by=sms['label'],bins=60,figsize=(12,5))

plt.xlim(-40,400);

plt.xlabel('length')

plt.ylabel('frequency')

plt.show()



sms['del_SW_length'].hist(bins=150,alpha=0.8,figsize=(12,5),label='no stopwords')

sms['del_p_length'].hist(bins=150,alpha=0.5,figsize=(12,5),label='punctuation length')

sms['length'].hist(bins=150,alpha=0.2,figsize=(12,5),label='true length')

plt.legend(loc='upper right')

plt.show()

sms['words after removing SW']=sms['del_SW_message'].apply(word_counter)





plt.title('Histogram distribution of number of words in both spam and not spam messages after removing STOPWORDS')

sms['words after removing SW'].hist(bins=100,figsize=(12,5))

plt.xlabel('No. of words')

plt.ylabel('frequency')

plt.show()







sms['words after removing SW'].hist(by=sms['label'],bins=50,figsize=(12,5))

plt.xlabel('No. of words')

plt.ylabel('frequency')

plt.show()



#pictorially represent spam and not spam words using WordCloud

from wordcloud import WordCloud

spam_words = ' '.join(list(sms[sms['labels'] == 1]['message']))

spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)

plt.figure(figsize = (10, 8), facecolor = 'k')

plt.imshow(spam_wc)

plt.axis('off')

plt.tight_layout(pad = 0)

plt.show()



notspam_words = ' '.join(list(sms[sms['labels'] == 0]['message']))

notspam_wc = WordCloud(width = 512,height = 512).generate(notspam_words)

plt.figure(figsize = (10, 8), facecolor = 'k')

plt.imshow(notspam_wc)

plt.axis('off')

plt.tight_layout(pad = 0)

plt.show()

from nltk.stem import SnowballStemmer 

stemmer= SnowballStemmer("english")

def stemming(text):

    text = text.split()

    words = ""

    for i in text:

            stemmer = SnowballStemmer("english")

            words =words + (stemmer.stem(i))+" "

    return words
sms['stemmed_message']=sms['del_SW_message'].apply(stemming)

sms['stem_message_length']= sms['stemmed_message'].apply(len)

sms['stemmed_word_length']= sms['stemmed_message'].apply(word_counter)
sms.head()
plt.title('Histogram distribution of the spam and not spam messages after Stemming')

sms['stem_message_length'].hist(bins=150,figsize=(12,5))

plt.xlim(-40,700)

plt.xlabel('length')

plt.ylabel('frequency')

plt.show()





sms['stem_message_length'].hist(by=sms['label'],bins=150,figsize=(12,5))

plt.xlim(-40,700);

plt.xlabel('length')

plt.ylabel('frequency')

plt.show()





sms['length'].hist(bins=150,alpha=0.4,figsize=(12,5),label='true length msg')

sms['del_SW_length'].hist(bins=150,alpha=0.5,figsize=(12,5),label='no stopwords msg')

sms['stem_message_length'].hist(bins=150,alpha=0.6,figsize=(12,5),label='stem msg length')

plt.legend(loc='upper right')

plt.show()
plt.title('Histogram distribution of number of words in both spam and not spam messages after stemming')

sms['stemmed_word_length'].hist(bins=100,figsize=(12,5))

plt.xlabel('No. of words')

plt.ylabel('frequency')

plt.show()







sms['stemmed_word_length'].hist(by=sms['label'],bins=50,figsize=(12,5))

plt.xlabel('No. of words')

plt.ylabel('frequency')

plt.show()

from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()

count_v=count_vector.fit_transform(sms['stemmed_message'])

from sklearn.feature_extraction.text import TfidfTransformer



sms_tfidf = TfidfTransformer().fit_transform(count_v)
#splitting data into training and test data

from sklearn.model_selection import train_test_split



xtrain, xtest, ytrain, ytest=train_test_split(sms_tfidf,sms['labels'], random_state=1)

print('Training messages size',xtrain.shape[0])

print('Testing messages size',xtest.shape[0])
#training data using naive bayes from sklearn



from sklearn.naive_bayes import MultinomialNB

naive_bayes=MultinomialNB()

naive_bayes.fit(xtrain,ytrain)
#testing data and making predictions

prediction=naive_bayes.predict(xtest)
#calculate metrics of the algorithm

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

print('accuracy score ',accuracy_score(ytest,prediction))

from sklearn.preprocessing import MinMaxScaler

xxtrain=xtrain.A

xxtest=xtest.A

scaler = MinMaxScaler()

scaled_xtrain = scaler.fit_transform(xxtrain)

scaled_xtest  = scaler.transform(xxtest)
naive_bayes_with_scalling = MultinomialNB().fit(scaled_xtrain, ytrain)

pred_NB_scaled = naive_bayes_with_scalling.predict(scaled_xtest)

accuracy_NB_scalling = accuracy_score(ytest, pred_NB_scaled)

print('accuracy score with scalling ',accuracy_NB_scalling)
from scipy.sparse import  hstack

sms_stack_len = hstack((sms_tfidf ,np.array(sms['stemmed_word_length'])[:,None])).A



xl_train, xl_test, yl_train, yl_test = train_test_split(sms_stack_len,sms['labels'], random_state=1)
naive_bayes_withlength=MultinomialNB()

naive_bayes_withlength.fit(xl_train,yl_train)

pred_naive_bayes_withlength=naive_bayes_withlength.predict(xl_test)

print('accuracy score with word count feature without scalling ',accuracy_score(yl_test,pred_naive_bayes_withlength))
X2_tfidf_train = xl_train

X2_tfidf_test  =  xl_test



scaler = MinMaxScaler()

X2_tfidf_train = scaler.fit_transform(X2_tfidf_train)

X2_tfidf_test  = scaler.transform(X2_tfidf_test)



naive_bayes_with_sc_len = MultinomialNB().fit(X2_tfidf_train, yl_train)

pred_NB_sc_len = naive_bayes_with_sc_len.predict(X2_tfidf_test)

accuracy_NB_sc_len = accuracy_score(yl_test, pred_NB_sc_len)

print('accuracy score with word count feature with scalling',accuracy_NB_sc_len)