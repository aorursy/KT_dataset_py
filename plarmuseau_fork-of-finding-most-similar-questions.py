import numpy as np

import pandas as pd

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



# timing function

import time   

start = time.clock() #_________________ measure efficiency timing



input_folder='../input/'

train = pd.read_csv(input_folder + 'train.csv',encoding='utf8')[:10000]

test  = pd.read_csv(input_folder + 'test.csv',encoding='utf8')[:50000]



# lege opvullen

train.fillna(value='leeg',inplace=True)

test.fillna(value='leeg',inplace=True)



print("Original data: trainQ: {}, testQ: {}".format(train.shape, test.shape) )

end = time.clock()

print('open:',end-start)
def cleantxt(x):   

    # Pad punctuation with spaces on both sides

    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:

        x = x.replace(char, ' ' + char + ' ')

    return x



train['question1']=train['question1'].map(cleantxt)

train['question2']=train['question2'].map(cleantxt)

test['question1']=test['question1'].map(cleantxt)

test['question2']=test['question2'].map(cleantxt)



train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist())

test_qs = pd.Series(test['question1'].tolist() + test['question2'].tolist())

all_qs = train_qs.append(test_qs)



count_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2) )

count_vectorizer.fit(all_qs)  #Learn vocabulary and idf, return document freq list.

print('lengt dictionary',len(count_vectorizer.vocabulary_))

freq_term_matrix = count_vectorizer.transform(all_qs)

tfidf = TfidfTransformer(norm="l2")

tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)



end = time.clock()

print('clean and make freq word dict:',end-start)
def sort_coo(m):

    tuples = zip(m.row, m.col, m.data)

    return sorted(tuples, key=lambda x: (x[0], x[2]),reverse=True)[:40]



all_qs=pd.DataFrame(all_qs)

print ('some samples')

print ('------------')



submit=[]

for xi in range(int(len(train_qs)+len(test_qs)/2),len(all_qs)):

    pos2=len(train)

    corr= tf_idf_matrix[xi:xi+1].dot(tf_idf_matrix[xi-len(test):xi-len(test)+1].T)     

    if (xi/500)==round(xi/500,0):

        print(corr,all_qs.iloc[xi],all_qs.iloc[xi-len(test)])



    submit.append(corr.todense().item())

submit=pd.DataFrame(submit)  

submit.to_csv('simpletfidf.csv')

end = time.clock()

print('submission:',end-start)