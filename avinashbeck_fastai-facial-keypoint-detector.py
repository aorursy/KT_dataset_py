import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path
!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!pip install -Uqq fastbook     # FastAI v2
import fastbook
fastbook.setup_book()
from fastbook import *
import torch
torch.cuda.is_available()
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(42)
path = Path("../input/facial-keypoints-68-dataset")
path.ls()
train_df = pd.read_csv(path/"training_frames_keypoints.csv")
train_df.rename(columns = {"Unnamed: 0": "fnames"}, inplace=True)
print(train_df.shape)
train_df.head()
test_df = pd.read_csv(path/"test_frames_keypoints.csv")
test_df.rename(columns = {"Unnamed: 0": "fnames"}, inplace=True)
print(test_df.shape)
test_df.head()
fpath = "../input/facial-keypoints-68-dataset/"
train_df['fnames'] = train_df['fnames'].apply(lambda x: fpath +"training/" + x )
train_df.head()
test_df['fnames'] = test_df['fnames'].apply(lambda x: fpath +"test/" + x )
test_df.head()
row = train_df.iloc[0,:]
img = Image.open(row[0])
img
pts = row[1:]
n = 2
pts = [pts[i: i+n]  for i in range(0, len(pts), n)]   # Convert the list to sublist with size of 2 for X and Y coordindates
pnt_img = TensorImage(img)
pnts = np.array(pts, dtype = 'f')
tfm = Transform(TensorPoint.create)
tpnts = tfm(pnts)
ctx = pnt_img.show(figsize=(5,5), cmap='Greys')
tpnts.show(ctx=ctx);
# Helper function for the DataBlock

def get_x(r): return r["fnames"]


def get_y(r):
    pts = r[1:]
    n = 2
    pts = [pts[i: i+n]  for i in range(0, len(pts), n)]   # Convert the list to sublist with size of 2
    pnts = np.array(pts, dtype = 'f')
    return pnts
faces = DataBlock(blocks = (ImageBlock, PointBlock),
                 get_x = get_x,
                 get_y = get_y,
                 splitter = RandomSplitter(valid_pct = 0.2, seed= 42),
                 item_tfms = Resize(512, 512),
                 batch_tfms = aug_transforms(size = (512,512)))

faces.summary(train_df, bs = 8, show_batch=True)    # Check whether datablock is working correctly
dls = faces.dataloaders(train_df)
dls.show_batch(max_n = 9, figsize=(8,6))
xb,yb = dls.one_batch()
xb.shape,yb.shape
dls.loss_func                # This looks correct
# from torchvision.models import densenet121
ARCH = resnet18     # densenet121, resnet34
learn = cnn_learner(dls, ARCH, y_range = (-1,1))
learn.lr_find()
lr = 2e-2
learn.fine_tune(15, lr)     # Train more epochs
learn.show_results(ds_idx=1, nrows=3, figsize=(6,8))
lr = 2e-2
learn.fine_tune(15, lr)     # Train more epochs
learn.show_results(ds_idx=1, nrows=3, figsize=(6,8))
# Save the model

learn.save("stage-1")
learn = learn.load("stage-1")
learn.unfreeze()
learn.fit_one_cycle(15, slice(1e-4, 1e-3))    # go more epochs
learn.fit_one_cycle(15, slice(1e-5, 1e-4))    
learn.show_results(ds_idx=1, nrows=3, figsize=(6,8))
learn.save("stage-2")
learn.export("/kaggle/working/models/export.pkl")
test_dl = dls.test_dl(test_df)
test_preds, _ = learn.get_preds(dl=test_dl)
test_preds.shape
