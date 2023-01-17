from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
import pandas as pd

from fastai import *
from fastai.vision import *
!ls ../input
path = Path('../input/')
path_test = Path('../input/test')
path_train = Path('../input/train')
df = pd.read_csv(path/'train.csv')#.sample(frac=0.05)
df.head()
df.Id.value_counts().head()
(df.Id == 'new_whale').mean()
(df.Id.value_counts() == 1).mean()
df.Id.nunique()
df.shape
fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}
SZ = 224
BS = 64
NUM_WORKERS = 0
SEED=0
#tfms = get_transforms(do_flip=False)
#data = ImageDataBunch.from_df(path_train, df, ds_tfms=tfms, size=150,num_workers=0)
data = (
    ImageItemList
        .from_folder('../input/train')
        .random_split_by_pct(seed=SEED)
        .label_from_func(lambda path: fn2label[path.name])
        .add_test(ImageItemList.from_folder('../input/test'))
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='../input/train')
)
data.show_batch(rows=3)
name = f'res50-{SZ}'
!git clone https://github.com/radekosmulski/whale
import sys
 # Add directory holding utility functions to path to allow importing utility funcitons
#sys.path.insert(0, '/kaggle/working/protein-atlas-fastai')
sys.path.append('/kaggle/working/whale')
from whale.utils import map5
MODEL_PATH = "/tmp/model/"
learn = create_cnn(data, models.resnet50, metrics=[accuracy, map5], model_dir=MODEL_PATH)
learn.fit_one_cycle(2)
learn.recorder.plot_losses()
learn.save(f'{name}-stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
max_lr = 1e-4
lrs = [max_lr/100, max_lr/10, max_lr]
learn.fit_one_cycle(5, lrs)
learn.save(f'{name}-stage-2')
learn.recorder.plot_losses()
preds, _ = learn.get_preds(DatasetType.Test)
from whale.utils import *
def create_submission(preds, data, name, classes=None):
    if not classes: classes = data.classes
    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})
    sub['Id'] = top_5_pred_labels(preds, classes)
    sub.to_csv(f'{name}.csv', index=False) # compression='gzip'
create_submission(preds, learn.data, name)
pd.read_csv(f'{name}.csv').head()
!rm -rf /kaggle/working/whale
#!kaggle competitions submit -c humpback-whale-identification -f {name}.csv.gz -m "{name}"
#!kaggle competitions submit -c humpback-whale-identification -f {name}.csv -m "{name}"
