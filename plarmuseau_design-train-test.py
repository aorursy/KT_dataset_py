import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

que = pd.read_csv('../input/questions.csv')

ans = pd.read_csv('../input/answers.csv')

pro = pd.read_csv('../input/professionals.csv')
que.shape,ans.shape,pro.shape
que_ans = que.merge(right=ans, how='inner', left_on='questions_id', right_on='answers_question_id')

qa_pro = que_ans.merge(right=pro, left_on='answers_author_id', right_on='professionals_id')

qa_pro.head()

qa_pro=qa_pro.sort_values('answers_date_added')

train=qa_pro[:37500]

test=qa_pro[37500:]

train.shape,test.shape

import re



def ngrams(string, n=3):

    string = re.sub(r'[,-./]|\sBD',r'', string)

    ngrams = zip(*[string[i:] for i in range(n)])

    return [''.join(ngram) for ngram in ngrams]



print('All 3-grams in "McDonalds":')

ngrams('McDonalds')



import numpy as np

from scipy.sparse import csr_matrix

!pip install sparse_dot_topn

import sparse_dot_topn.sparse_dot_topn as ct



def awesome_cossim_top(A, B, ntop, lower_bound=0):

    # force A and B as a CSR matrix.

    # If they have already been CSR, there is no overhead

    A = A.tocsr()

    B = B.tocsr()

    M, _ = A.shape

    _, N = B.shape

 

    idx_dtype = np.int32

 

    nnz_max = M*ntop

 

    indptr = np.zeros(M+1, dtype=idx_dtype)

    indices = np.zeros(nnz_max, dtype=idx_dtype)

    data = np.zeros(nnz_max, dtype=A.dtype)



    ct.sparse_dot_topn(

        M, N, np.asarray(A.indptr, dtype=idx_dtype),

        np.asarray(A.indices, dtype=idx_dtype),

        A.data,

        np.asarray(B.indptr, dtype=idx_dtype),

        np.asarray(B.indices, dtype=idx_dtype),

        B.data,

        ntop,

        lower_bound,

        indptr, indices, data)



    return csr_matrix((data,indices,indptr),shape=(M,N))





def get_matches_df(sparse_matrix, name_vector, top=100):

    non_zeros = sparse_matrix.nonzero()

    

    sparserows = non_zeros[0]

    sparsecols = non_zeros[1]

    

    if top:

        nr_matches = top

    else:

        nr_matches = sparsecols.size

    

    left_row = np.empty([nr_matches], dtype=object)

    left_side = np.empty([nr_matches], dtype=object)

    right_row =np.empty([nr_matches], dtype=object)

    right_side = np.empty([nr_matches], dtype=object)

    similairity = np.zeros(nr_matches)

    

    for index in range(0, nr_matches):

        #print(index,sparserows[index],name_vector[sparserows[index]])

        left_row[index] =sparserows[index]

        right_row[index] =sparsecols[index]

        left_side[index] = name_vector[sparserows[index]]

        right_side[index] = name_vector[sparsecols[index]]

        similairity[index] = sparse_matrix.data[index]

    

    return pd.DataFrame({'left_nr':left_row,

                        'left_side': left_side,

                         'right_nr':right_row,

                          'right_side': right_side,

                           'similarity': similairity})

from sklearn.feature_extraction.text import TfidfVectorizer



Txt=train.questions_title



vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)

tf_idf_matrix = vectorizer.fit_transform(Txt)

tf_idf_matrix
import time

t1 = time.time()

matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10, 0.75)

t = time.time()-t1

print("SELFTIMED:", t)
print( matches[:10] )
matches_df = get_matches_df(matches, Txt.values, top=100000)

matches_df = matches_df[matches_df['similarity'] < 0.99999] # Remove all exact matches

matches_df