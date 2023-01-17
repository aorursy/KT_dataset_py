!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

!pip install fastai==2.0.9
import random, os

import numpy as np

import torch

from fastai.vision.all import *
def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything()

path = Path('/kaggle/input/lego-minifigures-classification/')

lego_data = pd.read_csv(path/'index.csv', index_col=0);lego_data.head()

lego_metadata = pd.read_csv(path/'metadata.csv', index_col=0); lego_metadata.head()

df_lego = pd.merge(lego_data, lego_metadata[['class_id', 'minifigure_name']], on='class_id')

df_lego['labels'] = df_lego['minifigure_name'].apply(lambda x: x.lower())

df_lego['is_valid'] = df_lego['train-valid'].apply(lambda x:x=='train')

df_lego['fname'] = df_lego['path']; df_lego.head()
data = ImageDataLoaders.from_df(df_lego, path, valid_pct=0.10,

                                   item_tfms=Resize(412),

                                   bs=10, num_workers=4, valid_col='is_valid', label_col="labels")
data.show_batch()

data
learn = cnn_learner(data, resnet152, metrics=[error_rate, accuracy], model_dir = Path('../kaggle/working'),path = Path("."))
learn.lr_find()
lr1 = 1e-3

lr2 = 1e-1

learn.fit_one_cycle(100,slice(lr1,lr2),cbs=EarlyStoppingCallback(patience=2))
# lr1 = 1e-3

lr = 1e-1

learn.fit_one_cycle(200,slice(lr),cbs=EarlyStoppingCallback(patience=2))
learn.unfreeze()

learn.lr_find()

learn.fit_one_cycle(100,slice(1e-1),cbs=EarlyStoppingCallback(patience=3))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
learn.export()

learn.model_dir = "/kaggle/working"

learn.save("stage-1")