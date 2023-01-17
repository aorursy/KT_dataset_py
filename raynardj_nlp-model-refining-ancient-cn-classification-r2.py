# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from pathlib import Path
DATA =  Path("/kaggle/input/zhwikisource-title-draft/cntext_rule_sep.csv")
df = pd.read_csv(DATA)
df.sample(10)
df = df.sample(frac = 1.).reset_index().drop("index",axis=1)
df[~df.isy].sample(10)
from transformers import BertTokenizer,BertModel
tok = BertTokenizer.from_pretrained("bert-base-chinese")
tok.encode_plus("最高人民法院关于处理自首和立功具体应用法律若干问题的解释"),
tok.encode_plus("欽定八旗通志 (四庫全書本)/卷021")
bert = BertModel.from_pretrained("bert-base-chinese")
import torch
test_toks = tok.encode_plus("欽定八旗通志 (四庫全書本)/卷021")["input_ids"]
test_x = torch.LongTensor(test_toks)[None,:]
with torch.no_grad():
    y_1,y_2 = bert(test_x)
y_1.shape
y_2.shape
CUDA = torch.cuda.is_available()
if CUDA:
    bert.cuda()
from torch import nn
class newTop(nn.Module):
    def __init__(self,in_ = 768,hs = 768):
        super().__init__()
        self.ff = nn.Sequential(*[
            nn.Linear(in_,hs,bias = False),
            nn.BatchNorm1d(hs),
            nn.ReLU(),
            nn.Linear(hs,1)
        ])
        
    def forward(self,x):
        return self.ff(x)
from torch.utils.data.dataset import Dataset

class TextDS(Dataset):
    def __init__(self,df):
        self.df = df
        self.pre = list(df.preview)
        self.y = list(df.isy)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        txt = self.pre[idx]
        ids = tok.encode(txt,max_length=100,pad_to_max_length=True)
        return np.array(ids),np.array([1 if self.y[idx] else 0,])
!pip install forgebox
from forgebox.ftorch.train import Trainer
from forgebox.ftorch.callbacks import stat
t = Trainer(TextDS(df),val_dataset=TextDS(df),shuffle=False,
        batch_size = 32, print_on = 5, 
        callbacks=[stat],val_callbacks=[stat],using_gpu = CUDA)
model = newTop()
if CUDA:
    model.cuda()
t.opt["adm_top"] = torch.optim.Adam(model.parameters())
# The loos function that will reduce down the loss to a single scalar/ mini-batch
lossf = nn.BCEWithLogitsLoss()
# The loss function that will keep the dimension on mini-batch
loss_e = nn.BCEWithLogitsLoss(reduce = False)
from forgebox.ftorch.metrics import metric4_bi
class Looper:
    def __init__(self):
        self.record = []
        
    def __call__(self,*args):
        for arg in args:
            self.record.append(arg)
@t.step_train
def action(batch):
    if batch.i == 0: batch.l = Looper()
    batch.opt.zero_all()
    x,y = batch.data
    x = x.long()
    y = y.float()

    with torch.no_grad():
        _,vec = bert(x)
    y_ = model(vec)
    loss = lossf(y_,y)
    
    loss.backward()
    batch.opt.step_all()
    acc,rec,prec,f1 = metric4_bi(y_,y)
    return {"loss":loss.item(),
            "f1":f1.item(),
            "acc":acc.item(),
            "rec":rec.item(),
            "prec":prec.item()}

@t.step_val
def val_action(batch):
    if batch.i == 0: 
        batch.l = Looper()
        bert.eval()
        model.eval()
    with torch.no_grad():
        x,y = batch.data
        x = x.long()
        y = y.float()

        _,vec = bert(x)
        y_ = model(vec.detach())
        loss = loss_e(y_,y)
    
    batch.l(*loss.cpu().numpy())
    return {"loss":loss.mean().item(),}
t.train(1)
torch.save(model.state_dict(),"top_layer.pth")
t.epoch = 0
t.val_track[0] = list()
t.val_gen = iter(t.val_data)

val_t = t.progress(t.val_len)
for i in val_t:
    t.val_iteration(i, val_t)

for v_cb_func in t.val_callbacks:
    v_cb_func(record=t.val_track[0] )
df["loss"] = t.l.record
df["loss"] = df.loss.apply(lambda x:x[0])
df.sort_values(by = "loss",ascending = False)
df.sort_values(by = "loss",ascending = False).reset_index().drop("index",axis=1).to_csv("cntext_loss.csv",index = False)
