#Please run this code in Google Co-Lab



!wget https://www.dropbox.com/s/llg91ednbv6gcu5/ratings.csv


!pip3 install scikit-surprise
import pandas as pd

import numpy as np



from surprise import Reader, Dataset, SVD



from surprise.accuracy import rmse, mae

from surprise.model_selection import cross_validate
df = pd.read_csv('ratings.csv')

df.head()
print('Shape of the dataframe', df.shape)

print('Contains:',df.shape[0],'rows')

print('Contains:',df.shape[1],'columns')
df.isnull().sum()
# Let's drop the timestamp column because we are not gonna be using this column



df.drop('timestamp', inplace=True, axis = 1)
df.head()
print('Number of Unique Movies:', df['movieId'].nunique())

print('Number of Unique Users:', df['userId'].nunique())

import plotly.express as px
import matplotlib.pyplot as plt



plt.figure(figsize=(12,8))

fig = px.histogram(df, x= df['rating'])

fig.show()
filter_movies = df['movieId'].value_counts() > 3

filter_movies = filter_movies[filter_movies].index.tolist()
filter_movies[0:5]
filter_users = df['userId'].value_counts() > 3

filter_users = filter_users[filter_users].index.tolist()
filter_users[0:5]
print('Original Shape:',df.shape)

df = df[(df['movieId'].isin(filter_movies)) & (df['userId'].isin(filter_users))]

print('New Shape:', df.shape)
cols = ['userId', 'movieId', 'rating']



reader = Reader(rating_scale=(0.5, 5))

data = Dataset.load_from_df(df[cols], reader)



trainset = data.build_full_trainset()

antitest = trainset.build_anti_testset()

#Creating the Model

algo = SVD(n_epochs=25, verbose= True)
#Training the Model



cross_validate(algo, data, measures=['RMSE', 'MAE'], cv = 5, verbose=True)

predictions = algo.test(antitest)

predictions[0]
from collections import defaultdict

def get_top_n(predictions, n):



  top_n = defaultdict(list)

  for uid, iid, _, est, _ in predictions:

    top_n[uid].append((iid, est))



  for uid, user_ratings in top_n.items():

    user_ratings.sort(key = lambda x: x[1], reverse = True)

    top_n[uid] = user_ratings[ :n]



  return top_n

  pass



top_n = get_top_n(predictions, n=3)
for uid, user_ratings in top_n.items():

  print(uid, [iid for (iid, rating) in user_ratings])