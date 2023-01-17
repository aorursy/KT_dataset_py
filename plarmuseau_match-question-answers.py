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

pro_title=qa_pro.groupby('professionals_id')['questions_title'].apply(lambda x: "%a" % ','.join(x))
pro_title=pd.DataFrame( pro_title)

pro_title
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





def get_matches_df(sparse_matrix, name_vectorQ,name_vectorR, top=100):

    non_zeros = sparse_matrix.nonzero()

    

    sparserows = non_zeros[0]

    sparsecols = non_zeros[1]

    

    if top<sparsecols.size:

        nr_matches = top

    else:

        nr_matchesQ = sparserows.size

        nr_matchesR =sparsecols.size

    

    left_row = np.empty([nr_matchesQ], dtype=object)

    left_side = np.empty([nr_matchesQ], dtype=object)

    right_row =np.empty([nr_matchesQ], dtype=object)

    right_side = np.empty([nr_matchesQ], dtype=object)

    similairity = np.zeros(nr_matchesQ)

    

    for index in range(0, nr_matchesQ):

        #print(index,sparserows[index],name_vector[sparserows[index]])

        left_row[index] =sparserows[index]

        right_row[index] =sparsecols[index]

        left_side[index] = name_vectorQ[sparserows[index]]

        right_side[index] = name_vectorR[sparsecols[index]]

        similairity[index] = sparse_matrix.data[index]

    

    return pd.DataFrame({'left_nr':left_row,

                        'left_side': left_side,

                         'right_nr':right_row,

                          'right_side': right_side,

                           'similarity': similairity})

from sklearn.feature_extraction.text import TfidfVectorizer







vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)

tfidf_Q = vectorizer.fit_transform(qa_pro.questions_title)

tfidf_R =vectorizer.transform(pro_title.questions_title)
import time

t1 = time.time()

matches = awesome_cossim_top(tfidf_Q, tfidf_R.transpose(), 10, 0.5)

t = time.time()-t1

print("SELFTIMED:", t)
print( matches[:10] ),matches.shape
matches_df = get_matches_df(matches, qa_pro.questions_title.values,pro_title.questions_title.values ,top=1000000)

matches_df = matches_df[matches_df['similarity'] < 0.99999] # Remove all exact matches

matches_df
matches_df['left_nr']=pd.to_numeric(matches_df['left_nr'].values)

matches_df.info()
ma_pro=matches_df.merge( qa_pro,left_on='left_nr',right_index=True)
ma_pro['right_nr']=pd.to_numeric(ma_pro['right_nr'].values)

ma_pro.merge(pro_title.reset_index(),left_on='right_nr',right_index=True)
ma_pro.merge(pro_title.reset_index(),left_on='right_nr',right_index=True)[['left_side','professionals_id_x','professionals_id_y']]
matches = awesome_cossim_top(tfidf_R, tfidf_Q.transpose(), 30, 0.5)



matches_df = get_matches_df(matches, pro_title.questions_title.values,qa_pro.questions_title.values ,top=1000000)

matches_df = matches_df[matches_df['similarity'] < 0.99999] # Remove all exact matches

matches_df