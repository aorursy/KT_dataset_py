from fastai.collab import *

from fastai.tabular import *

import seaborn as sns
ratings = pd.read_csv('/kaggle/input/songsDataset.csv')

ratings.head()
ratings.columns = ['userID', 'songID', 'rating']
len(ratings)
ratings['rating'].value_counts()
sns.countplot(ratings['rating'])
data = CollabDataBunch.from_df(ratings, seed=42, valid_pct=0.2)
data.show_batch()
y_range = [0.5,5.5]
learn = collab_learner(data, n_factors=50, y_range=y_range, wd=1e-1)
learn.lr_find()

learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(5, 5e-3)
learn.save('/kaggle/working/colab-50')
learn.model
# uncomment these lines of code to try various sizes for embeddings



# for factor in [5,10,20,30]:

#     print("results for n_factors = " + str(factor))

#     learn = collab_learner(data, n_factors=factor, y_range=y_range, wd=1e-1)

#     learn.fit_one_cycle(5, 5e-3)