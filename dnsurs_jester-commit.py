import pandas as pd
from sklearn.model_selection import train_test_split

# import surprise
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Reader

from surprise import KNNBasic

from surprise import evaluate
from surprise import CoClustering
from surprise.prediction_algorithms.matrix_factorization import SVD, SVDpp, NMF
from surprise.similarities import cosine
#http://home.ifi.uio.no/paalh/students/MariusLorstadSolvangSteffenSand.pdf
jester_ratings = pd.read_csv('../input/jester_ratings.dat', sep='\s+', header=None)
jester_ratings.describe()
jester_ratings.columns = ["User ID", "Item ID", "Rating"]
jester_ratings
reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(-10, 10))
#load_from_df() to load a dataset from a pandas DataFrame
jester = Dataset.load_from_df(jester_ratings, reader=reader)
ids = {8,104}
j8n104 = jester_ratings.loc[jester_ratings['Item ID'].isin(ids)]
j8n104.sort_values(by=['Item ID'])
j8n104.describe()
j8n104
sim_options = {'name': 'cosine', 'user_based':False, 'item_based':True}
fullset = jester.build_full_trainset()
fullset

#build an algo and train it
algo = KNNBasic(sim_options = sim_options)
algo.fit(fullset)

j8n104.head()
uid = str(4) #raw user id
iid = str(8) #raw item id
#predict for specific user and item
pred = algo.predict(uid, iid, r_ui=-4.906, verbose=True)


#sample the random trainset and testset, with test set being 25% of the ratings
trainset, testset = train_test_split(jester, test_size=.25)
trainset
#https://surprise.readthedocs.io/en/stable/getting_started.html#getting-started
# SVD algorithm.
algo = SVD()

#Train algo on trainset
algo.fit(trainset)

#predict ratings for testset
predictions = algo.test(testset)

#compute RMSE
accuracy.rmse(predictions)

# CoClustering algo
algo = CoClustering()

algo.fit(trainset)

predictions = algo.test(testset)

accuracy.rmse(predictions)
sim_options = {'name': 'cosine',
               'user_based': False  #item-based CF
               }

algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

predictions = algo.test(testset)

accuracy.rmse(predictions)
