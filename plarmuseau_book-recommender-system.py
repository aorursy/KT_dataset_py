import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

ratings=pd.read_csv('../input/ratings.csv')

books=pd.read_csv('../input/books.csv')

print(ratings.head())

books['elo']=1200.0

print(books.head())

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm





from scipy.sparse import coo_matrix, csr_matrix

def read_data(filename):

    """ Reads in the last.fm dataset, and returns a tuple of a pandas dataframe

    and a sparse matrix of song/user/playcount """

    # read in triples of user/song/playcount from the input dataset

    data = pd.read_csv(filename,

                             usecols=[0,1,2],        #[36, 11, 10] vrk_pat_primkey,prd_atc_primkey,vdp_aantal

                             names=['song', 'user','plays'],skiprows=1) #[:1000000]   # user = patient, or prescriptionnr song=atc



    data=data.dropna(axis=0, how='any')  #drop nan

    data['plays']=data['plays']+1

    print(data.head())

    # map each song and user to a unique numeric value

    data['user'] = data['user'].astype("category")

    data['song'] = data['song'].astype("category")



    # create a sparse matrix of all the users/plays

    plays = coo_matrix((data['plays'].astype(float),

                       (data['song'].cat.codes.copy(),

                        data['user'].cat.codes.copy())))

    data['song_nr']=data['song'].cat.codes.copy()

    return data, plays,data.groupby(['song_nr','song']).plays.sum(),data['user'].cat.codes.copy()



data,matrix,songsd,user=read_data('../input/ratings.csv')

data.head()
from sklearn.preprocessing import normalize





def cosine(plays):

    normalized = normalize(plays)

    return normalized.dot(normalized.T)





def bhattacharya(plays):

    plays.data = np.sqrt(plays.data)

    return cosine(plays)





def ochiai(plays):

    plays = csr_matrix(plays)

    plays.data = np.ones(len(plays.data))

    return cosine(plays)





def bm25_weight(data, K1=1.2, B=0.8):

    """ Weighs each row of the matrix data by BM25 weighting """

    # calculate idf per term (user)

    N = float(data.shape[0])

    idf = np.log(N / (1 + np.bincount(data.col)))



    # calculate length_norm per document (artist)

    row_sums = np.squeeze(np.asarray(data.sum(1)))

    average_length = row_sums.sum() / N

    length_norm = (1.0 - B) + B * row_sums / average_length



    # weight matrix rows by bm25

    ret = coo_matrix(data)

    ret.data = ret.data * (K1 + 1.0) / (K1 * length_norm[ret.row] + ret.data) * idf[ret.col]

    return ret





def bm25(plays):

    plays = bm25_weight(plays)

    return plays.dot(plays.T)



def get_largest(row, N=10):

    if N >= row.nnz:

        best = zip(row.data, row.indices)

    else:

        ind = np.argpartition(row.data, -N)[-N:]

        best = zip(row.data[ind], row.indices[ind])

    return sorted(best, reverse=True)





def calculate_similar_artists(similarity, artists, artistid):

    neighbours = similarity[artistid]

    top = get_largest(neighbours)

    return [(artists[other], score, i) for i, (score, other) in enumerate(top)]





songsd = dict(enumerate(data['song'].cat.categories))

user_count = data.groupby('song').size()

to_generate = sorted(list(songsd), key=lambda x: -user_count[x])



similarity = bm25(matrix)
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)

Xr=svd.fit_transform(bm25(matrix))  

print(svd.explained_variance_ratio_)  

print(svd.explained_variance_ratio_.sum())
from sklearn.metrics.pairwise import cosine_similarity

Udf=pd.DataFrame(cosine_similarity(Xr))
booknr=4536

print(Udf[booknr].sort_values(ascending=False)[:10])

books[books['id'].isin( Udf[booknr].sort_values(ascending=False)[:10].index )]