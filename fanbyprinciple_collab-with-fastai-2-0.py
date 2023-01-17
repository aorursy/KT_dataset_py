import os

os.listdir("../input/movielens-100k-dataset/ml-100k")
!pip install -Uqq fastbook
import fastbook


from fastbook import *
from fastai.collab import *
from fastai.tabular.all import *

path = '../input/movielens-100k-dataset/ml-100k'
path = Path(path)
movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie', 'title'),header=None)
movies.head()
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user','movie','rating','timestamp'])
ratings.head()
ratings = ratings.merge(movies)
ratings.head()
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()
# fastai has a function get_emb_sz that returns recommended sizes for embedding matrices for your data
embs = get_emb_sz(dls)
embs
learn = collab_learner(dls, use_nn=True, y_range=(0, 5.5), layers=[100,50])
learn.fit_one_cycle(5, 5e-3, wd=0.1)
learn.show_results()