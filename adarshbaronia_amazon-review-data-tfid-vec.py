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
import bz2
import re

train =  '/kaggle/input/amazonreviews/train.ft.txt.bz2'
test = '/kaggle/input/amazonreviews/test.ft.txt.bz2'

train_file=bz2.BZ2File(train)
test_file=bz2.BZ2File(test)
train_file_lines = train_file.readlines()
test_file_lines=test_file.readlines()
train_file

train_file_lines = [x.decode('utf-8') for x in train_file_lines]
test_file_lines = [x.decode('utf-8') for x in test_file_lines]
train_file_lines[:2]
[0 if x.split(' ')[0]=='__label__1' else 1 for x in train_file_lines[:10]]
[x.split(' ',1)[1][:-1].lower() for x in train_file_lines[:2]]
train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]

for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d','0',train_sentences[i])
    
test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file_lines]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file_lines]

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d','0',test_sentences[i])
                                                       
for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])
        
for i in range(len(test_sentences)):
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import nltk
from nltk.corpus import stopwords
#from nltk.classify import SklearnClassifier

#from wordcloud import WordCloud,STOPWORDS
#import matplotlib.pyplot as plt
#%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
train_data={'Sentence': train_sentences, 'Labels': train_labels}
trained_data=pd.DataFrame(train_data)
test_data={'Sentence': test_sentences, 'Labels': test_labels}
tested_data=pd.DataFrame(test_data)
train, test=trained_data.head(900), tested_data.head(100)
sents=[]
alll=[]
stopwords_set=set(stopwords.words('english'))

def word_clean(train):
    for _,row in train.iterrows():
        words_filtered=[e.lower() for e in row.Sentence.split() if len(e)>=3]
        words_filtered=[word for word in words_filtered if 'http' not in word
                       and not word.startswith('@')
                       and not word.startswith('#')
                       and word !='RT']
        words_filtered=[word for word in words_filtered if not word in stopwords_set]
        sents.append((words_filtered, row.Labels))
        alll.extend(words_filtered)
cleaned_train=word_clean(train)
def get_word_features(wordlist):
    wordlists=nltk.FreqDist(wordlist)
    features=list(wordlists.keys())
    return features

w_features=get_word_features(alll)
w_features[:5]
def extract_features(doc):
    doc_u=set(doc)
    features={}
    for word in w_features:
        features['contains(%s)' %word]=(word in doc_u)
    return features
training_set=nltk.classify.apply_features(extract_features, sents)

classifier=nltk.NaiveBayesClassifier.train(training_set)

train_pos=train[train['Labels']==1]['Sentence']
train_neg=train[train['Labels']==0]['Sentence']
test_pos=test[test['Labels']==1]['Sentence']
test_neg=test[test['Labels']==0]['Sentence']

train_pos[1]
test_neg.head()
neg_cnt=0
pos_cnt=0
for obj in test_neg:
    res=classifier.classify(extract_features(obj.split()))
    if (res==0):
        neg_cnt+=1
        
for obj in test_pos:
    res=classifier.classify(extract_features(obj.split()))
    if (res==1):
        pos_cnt+=1

print('[Negative]: %s/%s' %(len(test_neg), neg_cnt))
print('[Positive]: %s/%s' %(len(test_pos), pos_cnt))
acc=((neg_cnt+pos_cnt)/(len(test_neg)+len(test_pos)))*100
acc
classifier.show_most_informative_features(5)

contractions= {'you’ve': 'you have', 'you’re': 'you are', 'you’ll’ve': 'you shall have', 'you’ll': 'you will', 'you’d’ve': 'you would have', 'you’d': 'you would', 'y’all’ve': 'you all have', 'y’all’re': 'you all are', 'y’all’d’ve': 'you all would have', 'y’all’d': 'you all would', 'y’all': 'you all', 'wouldn’t’ve': 'would not have', 'wouldn’t': 'would not', 'would’ve': 'would have', 'won’t’ve': 'will not have', 'won’t': 'will not', 'will’ve': 'will have', 'why’ve': 'why have', 'why’s': 'why is', 'who’ve': 'who have', 'who’s': 'who is', 'who’ll’ve': 'who will have', 'who’ll': 'who will', 'where’ve': 'where have', 'where’s': 'where is', 'where’d': 'where did', 'when’ve': 'when have', 'when’s': 'when is', 'what’ve': 'what have', 'what’s': 'what is', 'what’re': 'what are', 'what’ll’ve': 'what will have', 'what’ll': 'what will', 'weren’t': 'were not', 'we’ve': 'we have', 'we’re': 'we are', 'we’ll’ve': 'we will have', 'we’ll': 'we will', 'we’d’ve': 'we would have', 'we’d': 'we would', 'wasn’t': 'was not', 'to’ve': 'to have', 'they’ve': 'they have', 'they’re': 'they are', 'they’ll’ve': 'they will have', 'they’ll': 'they will', 'they’d’ve': 'they would have', 'they’d': 'they would', 'there’s': 'there is', 'there’d’ve': 'there would have', 'there’d': 'there would', 'that’s': 'that is', 'that’d’ve': 'that would have', 'that’d': 'that would', 'so’s': 'so is', 'so’ve': 'so have', 'shouldn’t’ve': 'should not have', 'shouldn’t': 'should not', 'should’ve': 'should have', 'she’s': 'she is', 'she’ll’ve': 'she will have', 'she’ll': 'she will', 'she’d’ve': 'she would have', 'she’d': 'she would', 'shan’t’ve': 'shall not have', 'sha’n’t': 'shall not', 'shan’t': 'shall not', 'oughtn’t’ve': 'ought not have', 'oughtn’t': 'ought not', 'o’clock': 'of the clock', 'needn’t’ve': 'need not have', 'needn’t': 'need not', 'mustn’t’ve': 'must not have', 'mustn’t': 'must not', 'must’ve': 'must have', 'mightn’t’ve': 'might not have', 'mightn’t': 'might not', 'might’ve': 'might have', 'mayn’t': 'may not', 'ma’am': 'madam', 'let’s': 'let us', 'it’s': 'it is', 'it’ll’ve': 'it will have', 'it’ll': 'it will', 'it’d’ve': 'it would have', 'it’d': 'it would', 'isn’t': 'is not', 'I’ve': 'I have', 'I’m': 'I am', 'I’ll’ve': 'I will have', 'I’ll': 'I will', 'I’d’ve': 'I would have', 'I’d': 'I would', 'how’s': 'how is', 'how’ll': 'how will', 'how’d’y': 'how do you', 'how’re': 'how are', 'how’d': 'how did', 'he’s': 'he is', 'he’ll’ve': 'he will have', 'he’ll': 'he will', 'he’d’ve': 'he would have', 'he’d': 'he would', 'haven’t': 'have not', 'hasn’t': 'has not', 'hadn’t’ve': 'had not have', 'hadn’t': 'had not', 'don’t': 'do not', 'doesn’t': 'does not', 'didn’t': 'did not', 'couldn’t’ve': 'could not have', 'couldn’t': 'could not', 'could’ve': 'could have', '’cause': 'because', 'can’t’ve': 'can not have', 'can’t': 'can not', 'aren’t': 'are not', 'ain’t': 'are not', 'dec.': 'december', 'nov.': 'november', 'oct.': 'october', 'sep.': 'september', 'aug.': 'august', 'jul.': 'july', 'jun.': 'june', 'apr.': 'april', 'mar.': 'march', 'feb.': 'february', 'jan.': 'january', "you've": 'you have', "you're": 'you are', "you'll've": 'you shall have', "you'll": 'you will', "you'd've": 'you would have', "you'd": 'you would', "y'all've": 'you all have', "y'all're": 'you all are', "y'all'd've": 'you all would have', "y'all'd": 'you all would', "y'all": 'you all', "wouldn't've": 'would not have', "wouldn't": 'would not', "would've": 'would have', "won't've": 'will not have', "won't": 'will not', "will've": 'will have', "why've": 'why have', "why's": 'why is', "who've": 'who have', "who's": 'who is', "who'll've": 'who will have', "who'll": 'who will', "where've": 'where have', "where's": 'where is', "where'd": 'where did', "when've": 'when have', "when's": 'when is', "what've": 'what have', "what's": 'what is', "what're": 'what are', "what'll've": 'what will have', "what'll": 'what will', "weren't": 'were not', "we've": 'we have', "we're": 'we are', "we'll've": 'we will have', "we'll": 'we will', "we'd've": 'we would have', "we'd": 'we would', "wasn't": 'was not', "to've": 'to have', "they've": 'they have', "they're": 'they are', "they'll've": 'they will have', "they'll": 'they will', "they'd've": 'they would have', "they'd": 'they would', "there's": 'there is', "there'd've": 'there would have', "there'd": 'there would', "that's": 'that is', "that'd've": 'that would have', "that'd": 'that would', "so's": 'so is', "so've": 'so have', "shouldn't've": 'should not have', "shouldn't": 'should not', "should've": 'should have', "she's": 'she is', "she'll've": 'she will have', "she'll": 'she will', "she'd've": 'she would have', "she'd": 'she would', "shan't've": 'shall not have', "sha'n't": 'shall not', "shan't": 'shall not', "oughtn't've": 'ought not have', "oughtn't": 'ought not', "o'clock": 'of the clock', "needn't've": 'need not have', "needn't": 'need not', "mustn't've": 'must not have', "mustn't": 'must not', "must've": 'must have', "mightn't've": 'might not have', "mightn't": 'might not', "might've": 'might have', "mayn't": 'may not', "ma'am": 'madam', "let's": 'let us', "it's": 'it is', "it'll've": 'it will have', "it'll": 'it will', "it'd've": 'it would have', "it'd": 'it would', "isn't": 'is not', "I've": 'I have', "I'm": 'I am', "I'll've": 'I will have', "I'll": 'I will', "I'd've": 'I would have', "I'd": 'I would', "how's": 'how is', "how'll": 'how will', "how'd'y": 'how do you', "how're": 'how are', "how'd": 'how did', "he's": 'he is', "he'll've": 'he will have', "he'll": 'he will', "he'd've": 'he would have', "he'd": 'he would', "hasn't": 'has not',"haven't": 'have not', "hadn't've": 'had not have', "hadn't": 'had not', "don't": 'do not', "doesn't": 'does not', "didn't": 'did not', "couldn't've": 'could not have', "couldn't": 'could not', "could've": 'could have', "'cause": 'because', "can't've": 'can not have', "can't": 'can not', "aren't": 'are not',"ain't": 'are not', "aren't": 'are not'}


import nltk
import re
import numpy as np
stopwords=nltk.corpus.stopwords.words('english')
stopwords.remove('no')
stopwords.remove('not')
stopwords.remove('but')

def clean_data(text):
    text1=text.split()
    for i, j in enumerate(text1):
        if j in contractions.keys():
            text1[i]=contractions[j]
        else:
            text1[i]
    text=" ".join(text1)
    text=str(text).lower()
    text=text.strip()
    text=re.sub(r'[^a-z\s]','',text,re.I|re.A)
    token=nltk.word_tokenize(text)
    tokens=[word for word in token if word not in stopwords]
    text=' '.join(tokens)
    
    return text

train_dt=pd.DataFrame()
train_dt['Sentence']=train['Sentence'].apply(lambda x:clean_data(x))
train_dt['Labels']=train['Labels']
train_dt.head(10)

test_dt=pd.DataFrame()
test_dt['Sentence']=test['Sentence'].apply(lambda x:clean_data(x))
test_dt['Labels']=test['Labels']
test_dt.head(10)
import textblob as tb
train_sentiment_obj=train_dt['Sentence'].apply(lambda x:tb.TextBlob(x).sentiment)
train_sentiment=pd.DataFrame()
train_sentiment['Subjectivity']=[word[0] for word in train_sentiment_obj]
train_sentiment['Polarity'] =[word[1] for word in train_sentiment_obj]

test_sentiment=pd.DataFrame()
test_sentiment_obj=test_dt['Sentence'].apply(lambda x:tb.TextBlob(x).sentiment)
test_sentiment['Subjectivity']=[word[0] for word in test_sentiment_obj]
test_sentiment['Polarity'] =[word[1] for word in test_sentiment_obj]
test_sentiment.head(2)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(stop_words='english')
train_cv=cv.fit_transform(train_dt.Sentence)
train_cv=pd.DataFrame(train_cv.toarray(), columns=cv.get_feature_names())
train_cv.index=train_dt.index
train_cv[:2]
test_cv=cv.transform(test_dt.Sentence)
test_cv=pd.DataFrame(test_cv.toarray(), columns=cv.get_feature_names())
test_cv.index=test_dt.index
test_cv.head(2)
                     
from sklearn.linear_model import LogisticRegression
train_combined=pd.concat([train_cv,train_sentiment], axis=1)
test_combined=pd.concat([test_cv, test_sentiment], axis=1)
train_combined.head()
LR=LogisticRegression()
LR.fit(train_combined,train['Labels'])
predict=LR.predict(test_combined)
predict
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
print('Confusion Matrix:\n',confusion_matrix(test['Labels'],predict))
print('\n')
print('Accuracy Score:\n',accuracy_score(test['Labels'], predict))
print('\nClassification Score: \n', classification_report(test['Labels'], predict))
from sklearn.naive_bayes import BernoulliNB
RF=BernoulliNB()
RF.fit(train_combined, train['Labels'])
predict_rf_1= RF.predict(test_combined)

print('Confusion Matrix: \n', confusion_matrix(test['Labels'], predict_rf_1))
print('\nAccuracy Score: ', accuracy_score(test['Labels'], predict_rf_1))
print('\nClassification Score: \n', classification_report(test['Labels'], predict_rf_1))
import numpy as np
normalize_corpus = np.vectorize(clean_data)
norm_corpus = normalize_corpus(list(train['Sentence']))
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(ngram_range=(1,2),min_df=2)
tfidf_matrix = tf.fit_transform(norm_corpus)
train_tfid = pd.DataFrame(tfidf_matrix.toarray(),columns=tf.get_feature_names())
test_tfid_matrix=tf.transform(test['Sentence'])
test_tfid = pd.DataFrame(test_tfid_matrix.toarray(), columns=tf.get_feature_names())
test_tfid.shape
#assign index of train & test to respective tfid matrix
train_tfid.index=train.index
test_tfid.index = test.index
#Combined_Data
train_combined_tf = pd.concat([train_sentiment, train_tfid], axis=1)
test_combined_tf = pd.concat([test_sentiment, test_tfid], axis=1)
#model fit amd predict using Logistic Regression

LR.fit(train_combined_tf,train['Labels'] )
predict_tf = LR.predict(test_combined_tf)
from sklearn.metrics import classification_report

print('Confusion Matrix: \n', confusion_matrix(test['Labels'], predict_tf))
print('\nAccuracy Score: ', accuracy_score(test['Labels'], predict_tf))
print('\nClassification Score: \n', classification_report(test['Labels'], predict_tf))



from sklearn.naive_bayes import BernoulliNB
RF=BernoulliNB()
RF.fit(train_combined_tf, train['Labels'])
predict_rf= RF.predict(test_combined_tf)

print('Confusion Matrix: \n', confusion_matrix(test['Labels'], predict_rf))
print('\nAccuracy Score: ', accuracy_score(test['Labels'], predict_rf))
print('\nClassification Score: \n', classification_report(test['Labels'], predict_rf))