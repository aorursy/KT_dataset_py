import numpy as np

import pandas as pd

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



# timing function

import time   

start = time.clock() #_________________ measure efficiency timing



input_folder='../input/'

train = pd.read_csv(input_folder + 'train.csv',encoding='utf8')

test  = pd.read_csv(input_folder + 'test.csv',encoding='utf8')



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



for xi in range(int(len(train_qs)+len(test_qs)/2),len(all_qs),100000):

    print ('----------------------------------------------------------')

    print('Q2:',all_qs.iloc[xi],' Q1:',all_qs.iloc[xi-len(test)])

    A=tf_idf_matrix[xi:xi+1].dot(tf_idf_matrix.T)

    Ac=A.tocoo()

    At=sort_coo(Ac)

    Q12Corr=tf_idf_matrix[xi:xi+1].dot(tf_idf_matrix[xi-len(test):xi-len(test)+1].T)

    print('Correlation test Q2 with Q1___',Q12Corr)

    print ('-----------------------------------------------------------')

    for yi in range(0,len(At)):



            if At[yi][1]>len(train_qs):

                typevr='test___'

            else:

                typevr='train___'

            if At[yi][2]==int( xi-len(test_qs)/2 ):

                print('TestPairQuestion__________',At[yi][2],all_qs.iloc[At[yi][1]])

            elif At[yi][2]>0.55: 

                print(typevr,'corr',At[yi][2],all_qs.iloc[At[yi][1]])