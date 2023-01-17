import mlxtend
import sklearn
import pandas as pd
import numpy as np
import gensim
import implicit
import surprise
from mlxtend.preprocessing import TransactionEncoder

data = np.array([
    ['우유', '기저귀', '쥬스'],
    ['양상추', '기저귀', '맥주'],
    ['우유', '양상추', '기저귀', '맥주'],
    ['양상추', '맥주']
])
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)
df
%%time
from mlxtend.frequent_patterns import apriori

apriori(df, min_support=0.5, use_colnames=True)
data = np.array([
    ['우유', '기저귀', '쥬스'],
    ['양상추', '기저귀', '맥주'],
    ['우유', '양상추', '기저귀', '맥주'],
    ['양상추', '맥주']
])
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)
df
%%time
from mlxtend.frequent_patterns import fpgrowth

fpgrowth(df, min_support=0.5, use_colnames=True)
docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
] 
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
countvect = vect.fit_transform(docs)
countvect_df = pd.DataFrame(countvect.toarray(), columns = sorted(vect.vocabulary_))
countvect_df.index = ['문서1', '문서2', '문서3', '문서4']
countvect_df
from sklearn.feature_extraction.text import TfidfVectorizer

tfidv = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None).fit(docs)
tfidv_df = pd.DataFrame(tfidv.transform(docs).toarray(), columns = sorted(tfidv.vocabulary_))
tfidv_df
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(tfidv_df, tfidv_df)
from gensim.models import Word2Vec

docs = [
  'you say goodbye and I say hello .'
]

sentences = [list(sentence.split(' ')) for sentence in docs]
sentences
model = Word2Vec(size=3, window=1, min_count=1, sg=1)
model.build_vocab(sentences)
model.wv.most_similar("say")
import surprise
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise import Reader, Dataset, SVD, SVDpp, NMF, KNNBaseline
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k')
df = pd.DataFrame(data.raw_ratings, columns=["user", "item", "rate", "id"])
df = df.astype(np.float32)

del df["id"]
df.head(10)
%%time 
reader = Reader(rating_scale=(1, 5))
knndata = Dataset.load_from_df(df[['user', 'item', 'rate']], reader)

sim_options = {'name': 'cosine'}
knn = surprise.KNNBasic(sim_options=sim_options, k=20)
score = cross_validate(knn, knndata, measures=['RMSE'], cv=5, verbose=True)
%%time 
user = 196

score_dict = {}
for sim in knn.get_neighbors(user, k=20):
    df_ = df[df['user'] == sim]
    for item, rate in zip(df_['item'].values, df_['rate'].values):
        if item not in df[df['user'] == user]['item'].values:
            try:
                score_dict[item] += rate
            except:
                score_dict[item] = rate
# 상위 10개의 영화만 추천 
dict(sorted(score_dict.items(), key = lambda x: -x[1])[0:10]).keys()
import tensorflow as tf
import numpy as np
from tqdm import tqdm_notebook as tqdm

import numpy as np

# Base code : https://yamalab.tistory.com/92
class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        """
        :param R: rating matrix
        :param k: latent parameter
        :param learning_rate: alpha on weight update
        :param reg_param: beta on weight update
        :param epochs: training epochs
        :param verbose: print status
        """

        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose


    def fit(self):
        """
        training Matrix Factorization : Update matrix latent weight and bias

        참고: self._b에 대한 설명
        - global bias: input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용
        - 정규화 기능. 최종 rating에 음수가 들어가는 것 대신 latent feature에 음수가 포함되도록 해줌.

        :return: training_process
        """

        # init latent features
        self._P = np.random.normal(size=(self._num_users, self._k))
        self._Q = np.random.normal(size=(self._num_items, self._k))

        # init biases
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):
            # rating이 존재하는 index를 기준으로 training
            xi, yi = self._R.nonzero()
            for i, j in zip(xi, yi):
                self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))


    def cost(self):
        """
        compute root mean square error
        :return: rmse cost
        """

        # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
        # 참고: http://codepractice.tistory.com/90
        xi, yi = self._R.nonzero()
        # predicted = self.get_complete_matrix()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - self.get_prediction(x, y), 2)
        return np.sqrt(cost/len(xi))


    def gradient(self, error, i, j):
        """
        gradient of latent feature for GD

        :param error: rating - prediction error
        :param i: user index
        :param j: item index
        :return: gradient of latent feature tuple
        """

        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        return dp, dq


    def gradient_descent(self, i, j, rating):
        """
        graident descent function

        :param i: user index of matrix
        :param j: item index of matrix
        :param rating: rating of (i,j)
        """

        # get error
        prediction = self.get_prediction(i, j)
        error = rating - prediction

        # update biases
        self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

        # update latent feature
        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq


    def get_prediction(self, i, j):
        """
        get predicted rating: user_i, item_j
        :return: prediction of r_ij
        """
        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)


    def get_complete_matrix(self):
        """
        computer complete matrix PXQ + P.bias + Q.bias + global bias

        - PXQ 행렬에 b_P[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
        - b_Q[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
        - b를 더하는 것은 각 element마다 bias를 더해주는 것

        - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

        :return: complete matrix R^
        """
        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)



# run example
if __name__ == "__main__":
    # rating matrix - User X Item : (7 X 5)
    R = np.array([
        [1, 0, 0, 1, 3],
        [2, 0, 3, 1, 1],
        [1, 2, 0, 5, 0],
        [1, 0, 0, 4, 4],
        [2, 1, 5, 4, 0],
        [5, 1, 5, 4, 0],
        [0, 0, 0, 1, 0],
    ])

    # P, Q is (7 X k), (k X 5) matrix
    
%%time
factorizer = MatrixFactorization(R, k=3, learning_rate=0.01, reg_param=0.01, epochs=100, verbose=True)
factorizer.fit()
factorizer.get_complete_matrix()
from implicit.evaluation import *
from implicit.als import AlternatingLeastSquares as ALS
# Implicit data
# 예시를 위해서 rate의 값을 1로 변경해주었습니다. 
df['rate'] = 1
user2idx = {}
for i, l in enumerate(df['user'].unique()):
    user2idx[l] = i
    
movie2idx = {}
for i, l in enumerate(df['item'].unique()):
    movie2idx[l] = i
idx2user = {i: user for user, i in user2idx.items()}
idx2movie = {i: item for item, i in movie2idx.items()}
useridx = df['useridx'] = df['user'].apply(lambda x: user2idx[x]).values
movieidx = df['movieidx'] = df['item'].apply(lambda x: movie2idx[x]).values
rating = df['rate'].values
import scipy

purchase_sparse = scipy.sparse.csr_matrix((rating, (useridx, movieidx)), shape=(len(set(useridx)), len(set(movieidx))))
als_model = ALS(factors=20, regularization=0.08, iterations = 20)
als_model.fit(purchase_sparse.T)
als_model.recommend(0, purchase_sparse, N=150)[0:10]