import numpy as np

import pandas as pd



from fastai.tabular import *

from fastai.collab import *



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Загружаем датасеты

data_reduced = pd.read_csv('/kaggle/input/recommendationsv4/train.csv')

data_reduced = data_reduced[['userid','itemid','rating']]

data_reduced.columns = ['user_id','item_id','rating']

# Удалим дубликаты из тренировочного датасета

data_reduced.drop_duplicates(inplace = True)



test_data = pd.read_csv('/kaggle/input/recommendationsv4/test.csv')

test_data = test_data[['userid', 'itemid']]

test_data.columns = ['user_id','item_id']



submission = pd.read_csv('/kaggle/input/recommendationsv4/sample_submission.csv')



data_collab = CollabDataBunch.from_df(

    data_reduced,

    seed=42,

    user_name='user_id',

    item_name='item_id',

    rating_name='rating',

    )

data_collab.show_batch()
learn = collab_learner(data_collab, n_factors=50, y_range=(0, 1), wd=1e-2)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, 1e-2)
learn.save("trained_model", return_path=True)
data_collab = CollabDataBunch.from_df(data_reduced, test=test_data, seed=42, valid_pct=0.2, user_name='user_id', item_name='item_id', rating_name='rating')

learn = collab_learner(data_collab, n_factors=50, y_range=(0, 1), wd=1e-2)
learn_loaded = learn.load(Path('trained_model'))
preds, y = learn_loaded.get_preds(DatasetType.Test)
submission['rating']= preds

submission.to_csv('submission_fastai.csv', index=False)
submission['rating'].mean()