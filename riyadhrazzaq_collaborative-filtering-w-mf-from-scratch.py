import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from IPython import display

import matplotlib.pyplot as plt
# data

itemcol = [w.strip().replace(" ","_") for w in "itemid | movie_title | release_date | video release date |IMDb URL | unknown | Action | Adventure | Animation |Children's | Comedy | Crime | Documentary | Drama | Fantasy |Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |Thriller | War | Western |".split('|')]

itemcol.remove('')

data = pd.read_csv("../input/movielens-100k-dataset/ml-100k/u.data",delimiter='\t',

                   names=['userid','itemid','rating','timestamp'])

user = pd.read_csv("../input/movielens-100k-dataset/ml-100k/u.user",delimiter='|',names=['userid','age','gender','occu','zip'])

item = pd.read_csv("../input/movielens-100k-dataset/ml-100k/u.item",delimiter='|',

                   names=itemcol,encoding='latin_1')
data.head()
print(data.shape)

print(user.shape)

print(item.shape)
data.rating.describe()
n_item = item.shape[0] 

n_user = user.shape[0]

print('user id starts from %d and item id starts from %d'%(user.userid.describe()['min'],

                                                          item.itemid.describe()['min']) )
# indexing userid, movieid from 0

item['itemid'] = item['itemid'].apply(lambda x: x - 1)

user['userid'] = user['userid'].apply(lambda x: x - 1)

data['itemid'] = data['itemid'].apply(lambda x: x - 1)

data['userid'] = data['userid'].apply(lambda x: x - 1)
# Genre aggregate

def genre(series):

    genres = series.index[6:-2]

    

    text = []

    for i in genres:

        if series[i] == 1:

            text.append(i)

    return ", ".join(text)

item['genre'] = item.apply(genre,axis=1)
def split(data):

    """

    Splits 100k data between train and test set, and builts corresponding rating matrix, A for each set

    """

    n_user = data.userid.nunique()

    n_item = data.itemid.nunique()

    train,test = train_test_split(data,test_size=0.3)

    def fun1(x):

        A_train[x[0],x[1]] = x[2]

        return x

    def fun2(x):

        A_test[x[0],x[1]] = x[2]

        return x

    A_train = np.zeros((n_user,n_item))

    A_test = np.zeros((n_user,n_item))

    train.apply(fun1, axis=1)

    test.apply(fun2,axis=1)

    return A_train, A_test
class MF:

    """

    Matrix factorization class. 

    """

    def __init__(self,lmbda = 0.01, learning_rate=0.001,max_iteration=10000,rank=10,verbose=True,gap=None):

        """

        params

        ------

        lmbda: float. Regularizer parameter.

        

        learning_rate: float. Step size or learning rate of SGD

        

        max_iteration: int. 

        

        rank: int. Embedding dimension of the U,V matrix where A = U.V

        

        verbose: bool. Whether to print iteration log or not.

        

        gap: bool. 

            Gap between each iteration log when verbose is true. Default value is 10th factor of max_iteration.

        """

        self.lmbda = lmbda

        self.lr = learning_rate

        self.max_iteration = max_iteration

        self.rank = rank

        self.verb = verbose

        self.gap = gap

        self.U = None

        self.V = None

        self.gap = (max_iteration / 10) if gap is None else gap

        

    def mse(self,truth, pred):

        """Returns the mse of nonzero errors"""

        pred = pred[truth.nonzero()].flatten()

        truth = truth[truth.nonzero()].flatten()

        return mean_squared_error(truth, pred)



    def graph(self,testset=False):

        """

        Training and test graph with other meta data.

        """

        fig, ax = plt.subplots(facecolor='white',figsize=(10,5))

        train = [w[0] for w in self.history]

        test = [w[1] for w in self.history]

        x = list(range(0,self.max_iteration+2,int(self.gap)))

        ax.plot(x,train,color='red',label='Train MSE')

        if testset==True:

            ax.plot(x,test,color='green',label='Test MSE')

        ax.legend()

        ax.grid(True)

        ax.set_xlabel("Iteration")

        ax.set_ylabel("MSE")

        caption = f'lmbda: {lmb} lr: {self.lr} iteration: {self.max_iteration}'

        plt.title(caption)

        plt.show()



    def predict(self,query_embedding,type='neighbour',name='Aladdin',measure='cosine'):

        """

        params

        ------

        query_embedding: 1D array. 

            Query's embeddding vector. For example if we want to find similar movies like Aladdin,

            query_embedding will be Aladdin's vector from V. 

        V: array-like. 2d. Item embedding.

        type: {similar, suggest}. 

            Not in use now. for future functionality.

        name: str. Movie name.

        measure: {dot,cosine}

            similarity measure for query and V. 



        returns

        -------

        sim_vector: similarity vector between query_embedding and V.

        """

        

        u = query_embedding

        V = self.V

        if measure == 'cosine':

            V = V / np.linalg.norm(V, axis=1, keepdims=True)

            u = u / np.linalg.norm(u)

        sim_vector = u.dot(V.T)

        return sim_vector



    

    def SGD(self,A,rated_rows,rated_cols,A_test=None):

        """

        Stochastic Gradient Descent. 

        

        params

        ------

        A: 2D array. shape(n_user,n_item)

            Training rating matrix. 

        

        rated_rows: 1D array.

            Observed indices rows from A. Meaning i where A_{i,j} > 0.

        

        rated_cols: 1D array.

            Observed indices' column from A. Meaning j where A_{i,j} > 0.

            

        A_test: Test A.

            *optional.*

            

        returns

        -------

        none

        """

        print("Master Yoda has started teaching...")

        self.history= []

        for itr in range(self.max_iteration):

            # choosing an observed user,item combination

            u = np.random.choice(rated_rows)

            i = np.random.choice(rated_cols)

            #forward pass

            error = A[u,i] - np.dot(self.U[u], self.V[i])  # check this line alone

    #         cost = error**2 + lmbda * (np.linalg.norm(self.U[u])**2 + np.linalg.norm(self.V[i])**2)        

            # backward pass

            tmp = self.U[u]

            self.U[u] = self.U[u] + self.lr * (error * self.V[i] - self.lmbda * self.U[u])

            self.V[i] = self.V[i] + self.lr * (error * tmp - self.lmbda * self.V[i])

            

            if (itr % self.gap) == 0 or itr == self.max_iteration - 1:

                A_hat = np.dot(self.U,self.V.T)

                train_mse = self.mse(A,A_hat)

                test_mse = -1

                if isinstance(A_test,np.ndarray):

                    test_mse = self.mse(A_test,A_hat)

                self.history.append((train_mse,test_mse))

                if self.verb==True:

                    print("iteration %d, TrainMSE: %.2f TestMSE: %.2f"%

                          (itr,train_mse,test_mse))

    

    def fit(self,A,A_test=None):

        """

        Fit the U,V to A.

        """

        rated_rows,rated_cols = A.nonzero()

        n_user = A.shape[0]

        n_item = A.shape[1]

        if self.U is None:

            self.U = np.random.rand(n_user,self.rank)

            self.V = np.random.rand(n_item,self.rank)

        # used in verbose mode

        self.SGD(A,rated_rows,rated_cols,A_test)
A_train,A_test = split(data)
lmb = 0.1

lr = 0.001

mx_itr = 60000

gap = mx_itr / 10

view = True



model = MF(lmb,learning_rate=lr,max_iteration=mx_itr,rank=30,verbose=True)

model.fit(A_train,A_test)

model.graph(testset=True)
def get_movie_suggestion(model,name='Aladdin'):

    # might return multiple movies

    movieids =  item[item['movie_title'].str.contains(name)].index.values

    if len(movieids) == 0:

        print('No movie found by that name. Remember, searching is case-sensitive')

        return

    print('Found ',len(movieids),'searching by: ',item.loc[movieids[0],'movie_title'])

    query = model.V[movieids[0]] # a single movie embedding

    sim_vector = model.predict(query)

    item['similarity'] = sim_vector

    top5 = item.sort_values(['similarity'],axis=0,inplace=False,ascending=False)[

        ['itemid','movie_title','genre','similarity','IMDb_URL']

    ].head()

#     top5 = top5.append(item.loc[movieids[0],['itemid','movie_title','genre','similarity']])

    display.display(top5)
get_movie_suggestion(model,name='GoldenEye')
# sample 

F = np.array([[0,4,0,5],

             [0,5,0,2],

             [0,0,2,4],

             [0,5,1,2]])

kid_model = MF(0.01,learning_rate=0.01,max_iteration=500,rank=2,verbose=True)

kid_model.fit(F)

# kid_model.graph()

newF = np.round(np.dot(kid_model.U,kid_model.V.T))

newF