# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





#loading the data set : 

train_df = pd.read_csv("../input/nlp-getting-started/train.csv")

test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

print(train_df.shape)

train_df.head(5)
# in this notebook the EDA will not be performed.. instead we will work with word2vec model. 

# EDA can be found in https://www.kaggle.com/rinisett/nlp-twitter-real-or-not-getting-started
# text pre-processing : 



import re 

import string

import nltk

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer



def cleaning_process(text1) :

     text = text1

     text = re.sub(r"http\S+", "", text) # removing url

     nopunc = [char for char in text if char not in string.punctuation] # removing punctuation

     text = ''.join(nopunc)

     text = ''.join([i for i in text if not i.isdigit()]) # removing digits

     text = text.lower()

     #print(text)

     tokens = nltk.word_tokenize(text)

     tokens = [w for w in tokens if w not in stopwords.words('english')]

     return tokens



train_df['cleaned_text'] = train_df['text'].apply(lambda x : cleaning_process(x))

print(train_df['text'][30:40])

train_df['cleaned_text'] = train_df['cleaned_text'].apply(lambda x : ' '.join(x))

print(train_df['cleaned_text'][30:40])

train_df.head(4)
#Word2Vec



import gensim

import sys

#from gensim.models import Word2Vec



rev_lines = (train_df['cleaned_text'].apply(lambda x : x.split())).values.tolist()

#rev_lines = train_df['cleaned_text'].apply(lambda x : x.split())

print(len(rev_lines))

#print(rev_lines)

print( " ----- ",type(rev_lines))



model = gensim.models.Word2Vec(sentences = rev_lines, size = 100, window = 2, workers = 3, min_count = 1, iter = 10)

words = list(model.wv.vocab)

print("vocab size", len(words))

print(model.wv.most_similar(positive = 'dead'))

#print(model['bomb'])

print(type(model))



#word_vectors = model.wv

#print(word_vectors)



# finding out word vectors

word_vectors = model[model.wv.vocab]



print(word_vectors)

#print(type(word_vectors))

print(len(words))

print(word_vectors.shape)
#sent_len = len(train_df['cleaned_text'][9].split())

#sent_list = train_df['cleaned_text'][9].split()

#print(sent_list)

# print(sent_len)   

# for x in sent_list :

#  sent_wv = model['x']

#  print(x, sent_wv) 

#  print(sent_wv.shape)    

#  print(type(sent_wv))

# avg_wv = np.mean(sent_wv)

# print(sent_len)

# print(type(sent_wv))

# print(avg_wv)



# trying to find out single word-vec for each sentence by averaging out the word-vec in the sentence.



def w2v_sent(text1) :

    sent_len = len(text1.split())

    sent_list = text1.split()

    #print("shape of wv",model[sent_list[0]].shape)

    sent_wv = np.asarray([model[sent_list[0]]])

    #print(sent_list[0], sent_wv)

    for x in range (1, sent_len) :

       # print("vector of x", x, model[x])

        #sent_wv = model[x]

        sent_wv = np.concatenate((sent_wv, [model[sent_list[x]]]), axis=0)

    #print(text1, " --- ", sent_wv)

    #avg_wv = np.mean(sent_wv, axis = 0)

    avg_wv = gensim.matutils.unitvec(np.array(sent_wv).mean(axis=0)).astype(np.float32)

    #print("Avg of sent", avg_wv)

    print("Shape of avg wv ", avg_wv.shape)

    return avg_wv



sent1_wv = np.asarray([w2v_sent(train_df['cleaned_text'][0])])

for x in range(1,len(train_df['cleaned_text'])) :

    sent1_wv = np.concatenate((sent1_wv,[w2v_sent(train_df['cleaned_text'][x])]),axis=0)

    

#print(type(total_sent_wv))

print(sent1_wv.shape)

# https://medium.com/@zafaralibagh6/a-simple-word2vec-tutorial-61e64e38a6a1 

# https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456

# https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-11-cnn-word2vec-41f5e28eda74





#https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

# https://stackoverflow.com/questions/24169238/dealing-with-negative-values-in-sklearn-multinomialnb



from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import ComplementNB

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier 

from sklearn.metrics import f1_score

import pandas as pd

df_ML = pd.DataFrame(columns=['Technique', 'Accuracy'])


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(sent1_wv, train_df['target'], test_size=0.3, random_state = 141)



print(type(msg_train), type(label_train))



print(msg_train.shape, label_train.shape)

print(msg_test.shape, label_test.shape)





#model_MNB = KNeighborsClassifier(n_neighbors=2)

#model_MNB = SVC()

#model_MNB = LogisticRegression()

model_MNB = MultinomialNB()

#model_MNB = GaussianNB()

#model_MNB = RandomForestClassifier(n_estimators = 200)

#model_MNB = KMeans(n_clusters=4)

#model_MNB = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)

#model_MNB = ComplementNB()







scalar = MinMaxScaler()

msg_train = scalar.fit_transform(msg_train)

msg_test = scalar.fit_transform(msg_test)



model_MNB.fit(msg_train, label_train) # training the mode



predictions = model_MNB.predict(msg_test) # predicting on the validation set

print(type(predictions))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score





#print(" Multinomial NB with TFIDF")

print(confusion_matrix(label_test,predictions))

print(classification_report(label_test,predictions))

print(accuracy_score(label_test, predictions))



value1 = "Multinomial MB"

value2 = accuracy_score(label_test, predictions)

df_ML = df_ML.append({'Technique': value1, 'Accuracy': value2},  ignore_index=True)


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(sent1_wv, train_df['target'], test_size=0.3, random_state = 141)



print(type(msg_train), type(label_train))



print(msg_train.shape, label_train.shape)

print(msg_test.shape, label_test.shape)





#model_MNB = KNeighborsClassifier(n_neighbors=2)

#model_MNB = SVC()

model_LR = LogisticRegression()

#model_MNB = MultinomialNB()

#model_MNB = GaussianNB()

#model_MNB = RandomForestClassifier(n_estimators = 200)

#model_MNB = KMeans(n_clusters=4)

#model_MNB = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)

#model_MNB = ComplementNB()







scalar = MinMaxScaler()

#msg_train = scalar.fit_transform(msg_train)

#msg_test = scalar.fit_transform(msg_test)



model_LR.fit(msg_train, label_train) # training the mode



predictions = model_LR.predict(msg_test) # predicting on the validation set

print(type(predictions))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score





#print(" Multinomial NB with TFIDF")

print(confusion_matrix(label_test,predictions))

print(classification_report(label_test,predictions))

print(accuracy_score(label_test, predictions))



value1 = "Logistic Regression"

value2 = accuracy_score(label_test, predictions)

df_ML = df_ML.append({'Technique': value1, 'Accuracy': value2},  ignore_index=True)
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(sent1_wv, train_df['target'], test_size=0.3, random_state = 141)



print(type(msg_train), type(label_train))



print(msg_train.shape, label_train.shape)

print(msg_test.shape, label_test.shape)





#model_MNB = KNeighborsClassifier(n_neighbors=2)

#model_MNB = SVC()

#model_MNB = LogisticRegression()

#model_MNB = MultinomialNB()

#model_MNB = GaussianNB()

model_RF = RandomForestClassifier(n_estimators = 200)

#model_MNB = KMeans(n_clusters=4)

#model_MNB = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)

#model_MNB = ComplementNB()







scalar = MinMaxScaler()

#msg_train = scalar.fit_transform(msg_train)

#msg_test = scalar.fit_transform(msg_test)



model_RF.fit(msg_train, label_train) # training the mode



predictions = model_RF.predict(msg_test) # predicting on the validation set

print(type(predictions))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score





#print(" Multinomial NB with TFIDF")

print(confusion_matrix(label_test,predictions))

print(classification_report(label_test,predictions))

print(accuracy_score(label_test, predictions))



value1 = "Random Forest Estimator"

value2 = accuracy_score(label_test, predictions)

df_ML = df_ML.append({'Technique': value1, 'Accuracy': value2},  ignore_index=True)
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(sent1_wv, train_df['target'], test_size=0.3, random_state = 141)



print(type(msg_train), type(label_train))



print(msg_train.shape, label_train.shape)

print(msg_test.shape, label_test.shape)





#model_MNB = KNeighborsClassifier(n_neighbors=2)

#model_MNB = SVC()

#model_MNB = LogisticRegression()

#model_MNB = MultinomialNB()

#model_MNB = GaussianNB()

#model_MNB = RandomForestClassifier(n_estimators = 200)

#model_MNB = KMeans(n_clusters=4)

model_SGDC = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)

#model_MNB = ComplementNB()







scalar = MinMaxScaler()

msg_train = scalar.fit_transform(msg_train)

msg_test = scalar.fit_transform(msg_test)



model_SGDC.fit(msg_train, label_train) # training the mode



predictions = model_SGDC.predict(msg_test) # predicting on the validation set

print(type(predictions))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score





#print(" Multinomial NB with TFIDF")

print(confusion_matrix(label_test,predictions))

print(classification_report(label_test,predictions))

print(accuracy_score(label_test, predictions))



value1 = "SGD Classifier"

value2 = accuracy_score(label_test, predictions)

df_ML = df_ML.append({'Technique': value1, 'Accuracy': value2},  ignore_index=True)
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(sent1_wv, train_df['target'], test_size=0.3, random_state = 141)



print(type(msg_train), type(label_train))



print(msg_train.shape, label_train.shape)

print(msg_test.shape, label_test.shape)





#model_MNB = KNeighborsClassifier(n_neighbors=2)

model_SVC = SVC()

#model_MNB = LogisticRegression()

#model_MNB = MultinomialNB()

#model_MNB = GaussianNB()

#model_MNB = RandomForestClassifier(n_estimators = 200)

#model_MNB = KMeans(n_clusters=4)

#model_MNB = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)

#model_MNB = ComplementNB()







scalar = MinMaxScaler()

#msg_train = scalar.fit_transform(msg_train)

#msg_test = scalar.fit_transform(msg_test)



model_SVC.fit(msg_train, label_train) # training the mode



predictions = model_SVC.predict(msg_test) # predicting on the validation set

print(type(predictions))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score





#print(" Multinomial NB with TFIDF")

print(confusion_matrix(label_test,predictions))

print(classification_report(label_test,predictions))

print(accuracy_score(label_test, predictions))



value1 = "SVC"

value2 = accuracy_score(label_test, predictions)

df_ML = df_ML.append({'Technique': value1, 'Accuracy': value2},  ignore_index=True)
df_ML
# In this study we see that word2vec gives best result with random forest .. The comparison is given in above table.
# for test data

print(test_df.shape)

print(train_df.shape)



# https://github.com/RaRe-Technologies/movie-plots-by-genre/blob/master/ipynb_with_output/Document%20classification%20with%20word%20embeddings%20tutorial%20-%20with%20output.ipynb



# predicting for the test data



test_df['cleaned_text'] = test_df['text'].apply(lambda x : cleaning_process(x))

test_df['cleaned_text'] = test_df['cleaned_text'].apply(lambda x : ' '.join(x))



# we will use the vocabulary created by the training data and import the word vectors of test data from the available model

# so we directly use sent_wv to find out the wv and averaging out for each sentences of test data 





sent1_wv_test = np.asarray([w2v_sent(test_df['cleaned_text'][0])])

for x in range(1,len(test_df['cleaned_text'])) :

    sent1_wv_test = np.concatenate((sent1_wv_test,[w2v_sent(test_df['cleaned_text'][x])]),axis=0)

    

#print(type(total_sent_wv))

print(sent1_wv_test.shape)







#test_df.head(30)



# we see that the word geese is not there in the vocabulary.. 



# So, let is try to find the number of words in test data that is not there in training data



# preparing a list of all words in test data

test_data_voc_list = []    

for x in range(0, len(test_df['cleaned_text'])) :

    sent = test_df['cleaned_text'][x]

    sent_token = sent.split()

    for i in range(0, len(sent_token)):

        test_data_voc_list.append(sent_token[i])

    #print(sent)

    #print(sent_token)

    

#print(test_data_voc_list)

#print(len(test_data_voc_list))







# finding how many new words are there in test data

count = 0 

new_word_list = []

for word in range (0, len(test_data_voc_list)) :

    if not test_data_voc_list[word] in model.wv.vocab : 

        #print(test_data_voc_list[word])

        count = count + 1

        new_word_list.append(test_data_voc_list[word])

        



print(new_word_list)

print(count)

print(len(test_data_voc_list))

print(len(new_word_list))

print(len(new_word_list) * 100/len(test_data_voc_list))





 # one way to solve this problem is to replace the new word with an empty string, or it is better to use the pretrained vocabulary like word2vec google data or glove. 
# we r removing the words from the test data which are not there in the model vocab.. 



"""

test_df['cleaned_text'][1:15]

#test_df['text'][1:15]

sent = test_df['cleaned_text'][281]

token = sent.split()

print(sent)

print(token)



for i in range(0, len(token)) :

    if token[i] in new_word_list:

        print(token[i])

resultwords  = [word for word in token if word.lower() not in new_word_list]

result = ' '.join(resultwords)

print(result)

"""





def word_not_in_vocab(text1) :

    sent_len = len(text1.split())

    sent_list = text1.split()

    #print(text1)

    #print(sent_list)   

    resultwords  = [word for word in sent_list if word.lower() not in new_word_list]

    result = ' '.join(resultwords)

    #print(result)

    return result   

    

    

#sent_1 = word_not_in_vocab(test_df['cleaned_text'][2])

#print(sent_1)





test_df['cleaned_removed_text'] = test_df['cleaned_text'].apply(lambda x : word_not_in_vocab(x))

#print(test_df['cleaned_text'], test_df['cleaned_removed_text'])

test_df.head(100)
#empty_str = test_df['cleaned_text'][13]

#print(empty_str)

#print(len(empty_str))



# need to find out how many strings are empty 

count = 0

for i in range(0, len(test_df['cleaned_text'])) :

    sent = test_df['cleaned_text'][i]

    #print(sent)

    if(len(sent) == 0):

        count = count+1

        print(count, test_df['keyword'][i], test_df['text'][i])
# so.  it is a single line with empty string.. we need to fill this otherwise while calculating the word_vectors code will crash ....

# from this row it seems that it does not concern with any diaster... so may be we can manually put some value to this string

# so let us replace this string with a very common word .. 



train_df.head(10)



train_data_voc_list = []    

for x in range(0, len(train_df['cleaned_text'])) :

    sent = train_df['cleaned_text'][x]

    sent_token = sent.split()

    for i in range(0, len(sent_token)):

        train_data_voc_list.append(sent_token[i])



#print(train_data_voc_list)



        

from collections import Counter 

Counter = Counter(train_data_voc_list) 

  

# most_common() produces k frequently encountered 

# input values and their respective counts. 

most_occur = Counter.most_common(10) 

  

print(most_occur) 
# So, these are the most common words we have... I choose the word 'get' and 



for x in range(1,len(test_df['cleaned_removed_text'])) :

    #print(x)

    #print("sent = {}".format(test_df['text'][x]))

    #print("cleaned sent = {}".format(test_df['cleaned_text'][x]))

    if ((len(test_df['cleaned_text'][x]) == 0) or (len(test_df['cleaned_removed_text'][x]) == 0)) :

        print("sent = {}".format(test_df['text'][x]))

        print(x, "cleaned sent = {}".format(test_df['cleaned_text'][x]))

        test_df['cleaned_text'][x] = test_df['text'][x]

        test_df['cleaned_removed_text'][x] = 'get'

        print(x, "cleaned sent = {}".format(test_df['cleaned_text'][x]))

        



print(test_df['cleaned_removed_text'][13])        




# Now we try to find out the word_vectors :         

print("len ={}".format(len(test_df['cleaned_removed_text'])))

sent1_wv_test = np.asarray([w2v_sent(test_df['cleaned_removed_text'][0])])

for x in range(1,len(test_df['cleaned_removed_text'])) :

    print(x)

    print("sent = {}".format(test_df['text'][x]))

    print("cleaned sent = {}".format(test_df['cleaned_removed_text'][x]))

    sent1_wv_test = np.concatenate((sent1_wv_test,[w2v_sent(test_df['cleaned_removed_text'][x])]),axis=0)

    

#print(type(total_sent_wv))

print(sent1_wv_test.shape) 

        
# now we can go for predictions.... 



#model_MNB = RandomForestClassifier(n_estimators = 200)

predictions_test = model_RF.predict(sent1_wv_test) # predicting on the validation set

print(predictions_test)

test_df['target'] = predictions_test

test_df[1:100]
sub_df = test_df[['id', 'target']]

sub_df.head(4)
sub_df.to_csv('submission.csv',index=False)