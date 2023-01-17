# Download and extract the data (only for Linux and Mac)

!rm -rf ml-100k ml-100k.zip

!wget -q http://files.grouplens.org/datasets/movielens/ml-100k.zip

!unzip -q ml-100k.zip

!ls ml-100k
import pandas as pd

from fastai.collab import CollabDataBunch, collab_learner
cols = ['User ID','Movie ID','Rating','Timestamp']

ratings_df = pd.read_csv('ml-100k/u.data', delimiter='\t', 

                         header=None, names=cols)

ratings_df.sample(5)
data = CollabDataBunch.from_df(ratings_df, valid_pct=0.1)

data.show_batch()
learn = collab_learner(data, n_factors=40, y_range=[0,5.5], wd=.1)
learn.lr_find()

learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(5, 0.01)
(users, items), ratings = next(iter(data.valid_dl))

preds = learn.model(users, items)

print('Real\tPred\tDifference')

for p in list(zip(ratings, preds))[:16]:

    print('{}\t{:.1f}\t{:.1f}'.format(p[0],p[1],p[1]-p[0]))
!pip install jovian --upgrade -q
import jovian
jovian.commit()