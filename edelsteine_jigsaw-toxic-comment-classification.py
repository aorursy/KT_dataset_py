

#!pip install textblob

from textblob import TextBlob

import pandas as pd

import numpy as np

import re

import string

import nltk



#nltk.download('wordnet')



#nltk.download('stopwords')

from nltk.corpus import stopwords 

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet as wn

from nltk import word_tokenize, pos_tag

from collections import defaultdict



#!pip install langdetect







from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt 

import pickle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data_path='/kaggle/input/jigsaw-toxic-comment/train.csv'





train = pd.read_csv(train_data_path,encoding='utf-8')

test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment/test.csv',encoding='utf-8')
import matplotlib.pyplot as plt

%matplotlib inline



train.hist(bins=50,figsize=(20,15))

plt.show()
train_data=train

train_data
train_data.drop(['id'],axis=1,inplace=True)

train_data


test_data=test

print(test_data)

def clean_data(all_comment):

  clean_comment=[]

  for input_text in all_comment:

    #removing URL

    input_text=re.sub(r'http\S+','',input_text)

    #removing digits

    input_text=re.sub(r'\d*','',input_text)

    #removing punctuation

    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~\,.'

    input_text=''.join(ch for ch in input_text if ch not in set(punctuation))

    #converting in lower case

    input_text=input_text.lower()

    # removing whitespace, newline'''

    input_text=re.sub(r'\n',' ',input_text)

    input_text=re.sub(r'\t',' ',input_text)

    input_text=' '.join(input_text.split())

    clean_comment.append(input_text)

  return clean_comment


def spell_checker(input_text):

    for x in input_text:

        text=TextBlob(x)

        txt=pd.Series(text.correct())

        input_text.append(txt)

    return input_text
def contradiction(all_comment):

  comment=[]

  for all_str in all_comment:

        all_str = re.sub(r"\*'r", ' are ', all_str)

        all_str = re.sub(r"\*'m ", ' am ', all_str)

        all_str = re.sub(r' u ', ' you ', all_str)

        all_str = re.sub(r" *'s ", ' is ', all_str)

        all_str = re.sub(r' b ', ' be ', all_str)

        all_str = re.sub(r' hv ', ' have ', all_str)

        all_str = re.sub(r' bt ', ' but ', all_str)

        all_str = re.sub(r' ur ', ' your ', all_str)

        all_str = re.sub(r' n ', ' and ', all_str)

        all_str = re.sub(r" *n't " , ' not ', all_str)

        all_str = re.sub(r' bro ', ' brother ', all_str)

        all_str = re.sub(r' it(z)+ ', ' it\'s ', all_str)

        all_str = re.sub(r' btw ', ' by the way ', all_str)

        comment.append(all_str)

  return comment





stop_words = set(stopwords.words('english'))

lem = WordNetLemmatizer() 





def _remove_noise(input_text):

    print(input_text)

    comment=[]

    word=[]

    stop_free_text=[]

    

    for i in range(len(input_text)):

            word_tokens = word_tokenize(input_text[i])

            #print('list and tokens\n',input_text[i],'\n',word_tokens)

            for w in word_tokens: 

              if w not in stop_words:

                    #print('not stop word\n',w)

                    lemma=lem.lemmatize(w, pos="v")

                    word.append(lemma)

                    #print('lemmitized word:\n',word)

                    sentence=' '.join(word)

                    #print('sentence\n',sentence)

            comment.append(sentence)

            word.clear()

           

    

    return comment 



def detect_language(all_comment):

    

    eng_comm=[]

    from langdetect import detect

    for x in all_comment:

      try:

       check=detect(x)

       if check=='en':

         eng_comm.append(x)

       else:

         eng_comm.append('')

      except:

        eng_comm.append('')

         

    return eng_comm



    


train_data['comment_text']=detect_language(train_data['comment_text'])

print(train_data['comment_text'])

train_data['comment_text'].replace('', np.nan, inplace=True)

train_data.dropna(subset=['comment_text'], inplace=True)

train_data.reset_index(inplace=True)

print('remove all other language\ns',train_data)

train_data['comment_text']=contradiction(train_data['comment_text'])

print('Contradiction\n',train_data['comment_text'])

train_data['comment_text']=clean_data(train_data['comment_text'])

print('removed punctuation\n',train_data['comment_text'])

train_data['comment_text']=spell_checker(train_data['comment_text'])

print('Corrected spelling\n',train_data['comment_text'])

train_data['comment_text']=_remove_noise(train_data['comment_text'])

print('removed stop words\n',train_data['comment_text'])



test_data['comment_text']=detect_language(test_data['comment_text'])

print(test_data['comment_text'])

test_data['comment_text'].replace('', np.nan, inplace=True)

test_data.dropna(subset=['comment_text'], inplace=True)

test_data.reset_index(inplace=True)

print('remove all other language\ns',test_data)

test_data['comment_text']=contradiction(test_data['comment_text'])

print('Contradiction\n',test_data['comment_text'])

test_data['comment_text']=clean_data(test_data['comment_text'])

print('removed punctuation\n',test_data['comment_text'])

test_data['comment_text']=spell_checker(test_data['comment_text'])

print('Corrected spelling\n',test_data['comment_text'])

test_data['comment_text']=_remove_noise(test_data['comment_text'])

print('removed stop words\n',test_data['comment_text'])



train_label=train_data.drop(['comment_text','index',],axis=1)
print(train_data['toxic'].value_counts())

print(train_data['severe_toxic'].value_counts())

print(train_data['obscene'].value_counts())

print(train_data['threat'].value_counts())

print(train_data['insult'].value_counts())

print(train_data['identity_hate'].value_counts())

from sklearn.model_selection import train_test_split

from sklearn.utils import resample

#split data into test and training sets

X_train, X_test, y_train, y_test = train_test_split( train_data['comment_text'], train_label, test_size=0.20, random_state=42)

#combine them back for resampling

train_info = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes

negative = train_info[train_info.threat==0]

positive = train_info[train_info.threat==1]

# upsample minority

pos_upsampled = resample(positive,replace=True, # sample with replacement

                         n_samples=len(negative), # match number in majority class

                         random_state=27) # reproducible results

# combine majority and upsampled minority

upsampled = pd.concat([negative, pos_upsampled])



negative = upsampled[upsampled.severe_toxic==0]

positive = upsampled[upsampled.severe_toxic==1]



pos_upsampled = resample(positive,replace=True, # sample with replacement

                         n_samples=len(negative), # match number in majority class

                         random_state=27) # reproducible results

upsampled = pd.concat([negative, pos_upsampled])

# check new class counts

upsampled.threat.value_counts()



print(upsampled['toxic'].value_counts())

print(upsampled['severe_toxic'].value_counts())

print(upsampled['obscene'].value_counts())

print(upsampled['threat'].value_counts())

print(upsampled['insult'].value_counts())

print(upsampled['identity_hate'].value_counts())

upsampled
unsampled['identity_hate'].value_counts()
upsampled['identity_hate'].value_counts()

test_comm


#upsampled.reset_index(inplace=True)

train_label=unsampled.drop(['comment_text'],axis=1)

train_label

from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer()



train_comment=vectorizer.fit_transform(unsampled['comment_text'])



valid_comment=vectorizer.transform(X_test)



#

test_comment=vectorizer.transform(test_data['comment_text'])
test_comment.shape
train_comment.shape
def model(X_train,y_train,valid_comment,y_test,test_comment):

    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()

    clf.fit(X_train,y_train)

    prediction1=clf.predict(valid_comment)

    prediction2=clf.predict(test_comment)

    accuracy=clf.score(valid_comment,y_test)

    print('accuracy',accuracy)

    

    return prediction1,prediction2


toxic_prediction,toxic_test_pred=model(train_comment,train_label['toxic'],valid_comment,y_test['toxic'],test_comment)





severe_toxic_prediction,severe_toxic_test_pred=model(train_comment,train_label['severe_toxic'],valid_comment,y_test['severe_toxic'],test_comment)





obscene_prediction,obsence_test_pred=model(train_comment,train_label['obscene'],valid_comment,y_test['obscene'],test_comment)



threat_prediction,threat_test_pred=model(train_comment,train_label['threat'],valid_comment,y_test['threat'],test_comment)



insult_prediction,insult_test_pred=model(train_comment,train_label['insult'],valid_comment,y_test['insult'],test_comment)





identity_hate_prediction,identity_hate_test_pred=model(train_comment,train_label['identity_hate'],valid_comment,y_test['identity_hate'],test_comment)

print(y_test['severe_toxic'].value_counts())
from sklearn.metrics import roc_auc_score

auc=[]

auc.append(roc_auc_score(y_test['toxic'], toxic_prediction))

auc.append(roc_auc_score(y_test['severe_toxic'], severe_toxic_prediction))

auc.append(roc_auc_score(y_test['obscene'], obscene_prediction))

auc.append(roc_auc_score(y_test['threat'], threat_prediction))

auc.append(roc_auc_score(y_test['insult'], insult_prediction))

auc.append(roc_auc_score(y_test['identity_hate'], identity_hate_prediction))

 

print('average AUC', sum(auc)/len(auc))
output = pd.DataFrame({'PassengerId': test_data.id, 'toxic':toxic_test_pred,	

                      'severe_toxic':severe_toxic_test_pred,'obscene':obsence_test_pred,

                       'threat':threat_test_pred,	'insult':insult_test_pred,'identity_hate':identity_hate_test_pred})

output.to_csv('my_submission.csv', index=False)

final=pd.read_csv('my_submission.csv')

final