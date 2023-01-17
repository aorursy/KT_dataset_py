!pip install git+https://github.com/fastai/fastcore > /dev/null

!pip install git+https://github.com/fastai/fastai2 > /dev/null

!pip install efficientnet-pytorch
%load_ext autoreload

%autoreload 2



import os

import pandas as pd

import sys

from collections import Counter

from pathlib import Path



from tqdm.notebook import tqdm

from torchvision.models import densenet121

from sklearn.metrics import roc_auc_score

from torch.utils.data.sampler import WeightedRandomSampler



from fastai2.basics import *

from fastai2.callback.all import *

from fastai2.vision.all import *





def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
from efficientnet_pytorch import EfficientNet
from fastai2.vision.all import *
DATA_PATH = Path('../input/plant-pathology-2020-fgvc7')

IMG_PATH = DATA_PATH / 'images'

LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']



IMG_SIZE = 284

SEED = 2020

BS = 8



ARCH = "efficientnet-b0"
seed_everything(SEED)
train_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
_, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(10, 3))

for ax, column in zip(axes, LABEL_COLS):

    train_df[column].value_counts().plot.bar(title=column, ax=ax)

plt.show()
hs, ws = [], []

for _, row in tqdm(train_df.iterrows(), total=len(train_df)):

    img = Image.open(IMG_PATH/(row.image_id+'.jpg'))

    h, w = img.size

    hs.append(h)

    ws.append(w)
_, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(10, 3))

for ax, column, vals in zip(axes, ['heights', 'widths'], [hs, ws]):

    ax.hist(vals, bins=100)

    ax.set_title(f'{column} hist')



plt.show()
Counter(hs), Counter(ws)
def get_label(row):

    for k, v in row[LABEL_COLS].items():

        if v == 1:

            return k



train_df['label'] = train_df.apply(get_label, axis=1)
datablock = (

            DataBlock(

                blocks=(ImageBlock, CategoryBlock(vocab=LABEL_COLS)),

                getters=[

                    ColReader('image_id', pref=IMG_PATH, suff='.jpg'),

                    ColReader('label')

                ],

                splitter=RandomSplitter(valid_pct=0.15, seed=SEED),

                item_tfms=Resize(IMG_SIZE*2),

                batch_tfms=aug_transforms(size=IMG_SIZE*2, max_rotate=30., min_scale=0.75, flip_vert=True, do_flip=True),

            )

)
dls = datablock.dataloaders(source=train_df, bs=BS)
dls.c
dls.show_batch()
def comp_metric(preds, targs, labels=range(len(LABEL_COLS))):

    # One-hot encode targets

    targs = np.eye(4)[targs]

    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])



def healthy_roc_auc(*args):

    return comp_metric(*args, labels=[0])



def multiple_diseases_roc_auc(*args):

    return comp_metric(*args, labels=[1])



def rust_roc_auc(*args):

    return comp_metric(*args, labels=[2])



def scab_roc_auc(*args):

    return comp_metric(*args, labels=[3])
def get_learner(dls):

    

    model = EfficientNet.from_pretrained("efficientnet-b7", advprop=True)

    model._fc = nn.Linear(2560, 4)# the last layer... B7



    learn = Learner(

        dls, model,

        loss_func=LabelSmoothingCrossEntropy(),

        metrics=[

            AccumMetric(healthy_roc_auc, flatten=False),

            AccumMetric(multiple_diseases_roc_auc, flatten=False),

            AccumMetric(rust_roc_auc, flatten=False),

            AccumMetric(scab_roc_auc, flatten=False),

            AccumMetric(comp_metric, flatten=False)]

        ).to_fp16()

    return learn
!mkdir '/kaggle/working/models'
learn = get_learner(dls)

learn.model_dir = '/kaggle/working/models'
learn.lr_find()
learn.fit_one_cycle(3)
learn.save('stage1')
learn.fit_one_cycle(6, slice(1e-5, 1e-4))
learn.save('stage2')
learn.recorder.plot_loss()
learn.export()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15, 10))
interp.plot_confusion_matrix(normalize=True, figsize=(6, 6))