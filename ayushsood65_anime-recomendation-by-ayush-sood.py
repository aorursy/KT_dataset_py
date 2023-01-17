# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset=pd.read_csv("../input/anime-recommendations-database/anime.csv")
dataset
dataset.isnull().sum()
dataset=dataset.dropna()

dataset
dataset.shape
dataset.info()
rating_df = pd.read_csv('../input/anime-recommendations-database/rating.csv')

rating_df
rating_df.shape
df_merge=pd.merge(dataset, rating_df, on = 'anime_id')
df_merge.shape
df_merge.rating_x.min()
df_merge = df_merge[df_merge.rating_y != -1]
df_merge.shape
sample = df_merge.sample(frac=.25)

sample.shape
sample.dtypes
sample['rating_x'] = sample['rating_x'].astype(int)
sample.dtypes
ratings_x = sample['rating_x'].value_counts() #continuous

ratings_y = sample['rating_y'].value_counts() #discrete

print(ratings_x)

print(ratings_y)
sample.rating_x = sample.rating_x.apply(round) 
sample.head()
sample.shape
import matplotlib.pyplot as plt

ratings_sorted = sorted(list(zip(ratings_y.index, ratings_y)))

plt.bar([r[0] for r in ratings_sorted], [r[1] for r in ratings_sorted], color='blue')

plt.xlabel("Rating")

plt.ylabel("# of Ratings")

plt.title("Distribution of Ratings")

plt.show()
import seaborn as sns

fig = plt.figure(figsize=(12

                          ,5))

sns.countplot(sample['type'], palette='gist_rainbow')

plt.title("Most Viewed Anime on Type", fontsize=20)

plt.xlabel("Types", fontsize=20)

plt.ylabel("Number of Views with Reviews", fontsize = 20)

plt.grid()

plt.show()
from surprise import Reader

from surprise import Dataset

data = sample[['user_id', 'anime_id', 'rating_x']] #may need to do rating_x rounded and then use rating_y

reader = Reader(line_format='user item rating', sep='')
anime = Dataset.load_from_df(data, reader)
#train_test_split

from surprise.model_selection import train_test_split

from surprise.model_selection import GridSearchCV

from surprise.model_selection import cross_validate

from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNBaseline

from surprise.prediction_algorithms import knns

from surprise.prediction_algorithms import SVD

from surprise.similarities import cosine, msd, pearson

from surprise import accuracy

trainset, testset = train_test_split(anime, test_size=.2)
#INSTANTIATE the SVD and fit only the train set

svd = SVD()

svd.fit(trainset)
predictions = svd.test(testset) 

accuracy.rmse(predictions)
#