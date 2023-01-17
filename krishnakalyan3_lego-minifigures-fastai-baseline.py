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
dls = ImageDataLoaders.from_df(df_lego, path, valid_pct=0.10,

                                   item_tfms=Resize(412),

                                   bs=10, num_workers=4, valid_col='is_valid', label_col="labels")
dls.show_batch()
print(dls.vocab); print(dls.c)
learn = cnn_learner(dls, resnet34, metrics=[error_rate, accuracy], model_dir="/tmp/model/").to_fp16()
learn.lr_find()
learn.fit_one_cycle(50, lr_max=1e-2, cbs=EarlyStoppingCallback(patience=1))
learn.unfreeze()
learn.fit_one_cycle(10, lr_max=1e-4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(5,5), dpi=60)
preds, _ = learn.get_preds(); preds.shape