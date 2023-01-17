import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
test = pd.read_csv('../input/test.csv')

print ('Shape of train ',train.shape)
print ('Shape of test ',test.shape)
print ('Shape of sample_submission ',sample_submission.shape)
print ('Taking a look at Sincere Questions')
train.loc[train['target'] == 0].sample(5)['question_text']
print ('Taking a look at Insincere Questions')
train.loc[train['target'] == 1].sample(5)['question_text']
samp = train.sample(1)
sentence = samp.iloc[0]['question_text']
print (sentence)
import re
sentence = re.sub(r'\d+','',sentence)
print ('Sentence After removing numbers\n',sentence)
import string
sentence = sentence.translate(sentence.maketrans("","",string.punctuation))
print ('Sentence After Removing Punctuations\n',sentence)
import nltk
stop_words = set(nltk.corpus.stopwords.words('english'))
words_in_sentence = list(set(sentence.split(' ')) - stop_words)
print (words_in_sentence)
from nltk.stem import PorterStemmer
stemmer= PorterStemmer()
for i,word in enumerate(words_in_sentence):
    words_in_sentence[i] = stemmer.stem(word)
print (words_in_sentence)    
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
words = []
for i,word in enumerate(words_in_sentence):
    words_in_sentence[i] = lemmatizer.lemmatize(word)
print (words_in_sentence)
from sklearn.model_selection import train_test_split
train, test = train_test_split(train, test_size=0.2)
word_count = {}
word_count_sincere = {}
word_count_insincere = {}
sincere  = 0
insincere = 0 

import re
import string
import nltk
stop_words = set(nltk.corpus.stopwords.words('english'))
from nltk.stem import PorterStemmer
stemmer= PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

#for i in range(0,len(train.shape[0])):
#row_count = 1000
row_count = train.shape[0]
for row in range(0,row_count):
    insincere += train.iloc[row]['target']
    sincere += (1 - train.iloc[row]['target'])
    sentence = train.iloc[row]['question_text']
    sentence = re.sub(r'\d+','',sentence)
    sentence = sentence.translate(sentence.maketrans("","",string.punctuation))
    words_in_sentence = list(set(sentence.split(' ')) - stop_words)
    for index,word in enumerate(words_in_sentence):
        word = stemmer.stem(word)
        words_in_sentence[index] = lemmatizer.lemmatize(word)
    for word in words_in_sentence:
        if train.iloc[row]['target'] == 0:   #Sincere Words
            if word in word_count_sincere.keys():
                word_count_sincere[word]+=1
            else:
                word_count_sincere[word] = 1
        elif train.iloc[row]['target'] == 1:    #Insincere Words
            if word in word_count_insincere.keys():
                word_count_insincere[word]+=1
            else:
                word_count_insincere[word] = 1
        if word in word_count.keys():             #For all words. I use this to compute probability.
            word_count[word]+=1
        else:
            word_count[word]=1
            
print ('Done')
word_probability = {}
total_words = 0
for i in word_count:
    total_words += word_count[i]
for i in word_count:
    word_probability[i] = word_count[i] / total_words
print ('Total words ',len(word_probability))
print ('Minimum probability ',min (word_probability.values()))
threshold_p = 0.0001
for i in list(word_probability):
    if word_probability[i] < threshold_p:
        del word_probability[i]
        if i in list(word_count_sincere):   #list(dict) return it;s key elements
            del word_count_sincere[i]
        if i in list(word_count_insincere):  
            del word_count_insincere[i]
print ('Total words ',len(word_probability))            
total_sincere_words = sum(word_count_sincere.values())
cp_sincere = {}  #Conditional Probability
for i in list(word_count_sincere):
    cp_sincere[i] = word_count_sincere[i] / total_sincere_words

total_insincere_words = sum(word_count_insincere.values())
cp_insincere = {}  #Conditional Probability
for i in list(word_count_insincere):
    cp_insincere[i] = word_count_insincere[i] / total_insincere_words


#for i in range(0,len(train.shape[0])):
#row_count = 1000
row_count = test.shape[0]

p_insincere = insincere / (sincere + insincere)
p_sincere = sincere / (sincere + insincere)
accuracy = 0

for row in range(0,row_count):
    sentence = test.iloc[row]['question_text']
    target = test.iloc[row]['target']
    sentence = re.sub(r'\d+','',sentence)
    sentence = sentence.translate(sentence.maketrans("","",string.punctuation))
    words_in_sentence = list(set(sentence.split(' ')) - stop_words)
    for index,word in enumerate(words_in_sentence):
        word = stemmer.stem(word)
        words_in_sentence[index] = lemmatizer.lemmatize(word)
    insincere_term = p_insincere
    sincere_term = p_sincere
    
    sincere_M = len(cp_sincere.keys())
    insincere_M = len(cp_insincere.keys())
    for word in words_in_sentence:
        if word not in cp_insincere.keys():
            insincere_M +=1
     #       print (word)
        if word not in cp_sincere.keys():
            sincere_M += 1
#        print (word)
         
    for word in words_in_sentence:
        if word in cp_insincere.keys():
            insincere_term *= (cp_insincere[word] + (1/insincere_M))
        else:
            insincere_term *= (1/insincere_M)
        if word in cp_sincere.keys():
            sincere_term *= (cp_sincere[word] + (1/sincere_M))
        else:
            sincere_term *= (1/sincere_M)
        
    if insincere_term/(insincere_term + sincere_term) > 0.5:
        response = 1
    else:
        response = 0
    if target == response:
        accuracy += 1
    
print ('Accuracy is ',accuracy/row_count*100)