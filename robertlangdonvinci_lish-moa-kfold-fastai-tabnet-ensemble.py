from fastai.tabular.all import *

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from fastai.callback import *

from tqdm.notebook import tqdm

from ml_stratifiers import MultilabelStratifiedKFold
pd.set_option('display.max_rows', 50)

pd.set_option('display.max_columns', 50)
path = Path('../input/lish-moa')
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

sample_submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
train_features.head()
train_targets_scored.head()
cols = train_targets_scored.columns.tolist()[1:]
train_features.sig_id.nunique()
train_features.cp_type.value_counts()
train_features.cp_time.value_counts()
train_features.cp_dose.value_counts()
train_targets_scored.sum()[1:].sort_values().head(10)
train_targets_scored.sum()[1:].sort_values().tail(50)
train_targets_scored.sum()[1:].sort_values().tail(50).index[-24:]
trn_df = train_features.merge(train_targets_scored,on='sig_id',how='left')
df = trn_df.sample(frac=1.,random_state=2020)

df['kfold'] = -1

kf = MultilabelStratifiedKFold(n_splits=5)

y = df[cols].values

for fold, (t_,v_) in enumerate(kf.split(X=df,y=y)):

    df.loc[v_,'kfold'] = fold
df.to_csv('data_kfold.csv',index=False)
df.head()
sig_ids = test_features[test_features['cp_type'] == 'ctl_vehicle']['sig_id'].values
len(cols)
cat_names = ['cp_type', 'cp_time', 'cp_dose']

cont_names = [c for c in train_features.columns if c not in cat_names and c != 'sig_id']
def get_data(fold):

    

    val_idx = df[df.kfold==fold].index

    dls = TabularDataLoaders.from_df(df, path=path, y_names=cols,

                                        cat_names = cat_names,

                                        cont_names = cont_names,

                                        procs = [Categorify, FillMissing, Normalize],

                                        valid_idx=val_idx,

                                        #y_block=MultiCategoryBlock(encoded=True,vocab=cols),

                                        bs=64)

    return dls

    
class LabelSmoothingCE(Module):

    def __init__(self, eps=0.1, reduction='mean'): self.eps,self.reduction = eps,reduction



    def forward(self, output, target):

        c = output.size()[-1]

        log_preds = F.log_softmax(output, dim=-1)

        if self.reduction=='sum': loss = -log_preds.sum()

        else:

            loss = -log_preds.sum(dim=-1) #We divide by that size at the return line so sum and not mean

            if self.reduction=='mean':  loss = loss.mean()

        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target.long(), reduction=self.reduction)



    def activation(self, out): return F.softmax(out, dim=-1)

    def decodes(self, out):    return out.argmax(dim=-1)
test_sc = []



for i in tqdm(range(5)):

    

    dls = get_data(i) # Data

    

    learn = tabular_learner(dls , y_range=(0,1), layers=[1024, 512, 512, 256], loss_func = BCELossFlat(), model_dir='/kaggle/working/') # Model

    

    name = 'best_model_' + str(i) 

    cb = SaveModelCallback(monitor='valid_loss',fname=name ,mode='min') # Callbacks

    

    lr = 9e-3

    learn.fit_one_cycle(10, slice(lr/(2.6**4),lr),cbs=cb) # Training

    

    learn.load(name) # Load best model

    

    test_dl = learn.dls.test_dl(test_features)

    sub = learn.get_preds(dl=test_dl) # prediction

    test_sc.append(sub[0].numpy())

    

    learn.export('/kaggle/working/'+name+'.pkl') # export model

    

test_sc = np.array(test_sc)
avg_prds = test_sc.mean(axis=0)
submission = sample_submission.copy()

submission[cols] = avg_prds

submission.loc[submission['sig_id'].isin(test_features.loc[test_features['cp_type'] =='ctl_vehicle', 'sig_id']), train_targets_scored.columns[1:]] = 0

submission['atp-sensitive_potassium_channel_antagonist'] = 0

submission['erbb2_inhibitor'] = 0
submission.head()
submission.to_csv('submission_tabular.csv',index=False)
import sys

sys.path.append('../input/pytorch-tabnet')

sys.path.append('../input/fastai-tabnet')
from fastai.basics import *

from pytorch_tabnet import *

from fast_tabnet.core import *
test_sc_tab = []

lr = 9e-3



for i in tqdm(range(5)):

    

    dls = get_data(i) # Data

    emb_szs = get_emb_sz(dls)

    

    model = TabNetModel(emb_szs, len(dls.cont_names), dls.c, n_d=8, n_a=32, n_steps=1); 

    

    opt_func = partial(Adam, wd=0.01, eps=1e-5)

    learn = Learner(dls, model, BCEWithLogitsLossFlat(), opt_func=opt_func, lr=lr, model_dir='/kaggle/working/')

    

    name = 'best_model_tabnet_' + str(i) 

    

    cb = SaveModelCallback(monitor='valid_loss',fname=name ,mode='min') # Callbacks

    

    lr = 9e-3

    learn.fit_one_cycle(30, slice(lr/(2.6**4),lr),cbs=cb) # Training

    

    learn.load(name) # Load best model

    

    test_dl = learn.dls.test_dl(test_features)

    sub = learn.get_preds(dl=test_dl) # prediction

    test_sc_tab.append(sub[0].numpy())

    

    learn.export('/kaggle/working/'+name+'.pkl') # export model

    

test_sc_tab = np.array(test_sc_tab)
avg_prds_tab = test_sc_tab.mean(axis=0)
submission_tab = sample_submission.copy()

submission_tab[cols] = avg_prds_tab

submission_tab.loc[submission_tab['sig_id'].isin(test_features.loc[test_features['cp_type'] =='ctl_vehicle', 'sig_id']), train_targets_scored.columns[1:]] = 0

submission_tab['atp-sensitive_potassium_channel_antagonist'] = 0

submission_tab['erbb2_inhibitor'] = 0
submission_tab.to_csv('submission_tabnet.csv',index=False)
final_prds = np.array((list(avg_prds),list(avg_prds_tab))).mean(axis=0)
fin_wt_prds = avg_prds*(0.6) + avg_prds_tab*(0.4)
submission_fin = sample_submission.copy()

submission_fin[cols] = fin_wt_prds

submission_fin.loc[submission_fin['sig_id'].isin(test_features.loc[test_features['cp_type'] =='ctl_vehicle', 'sig_id']), train_targets_scored.columns[1:]] = 0

submission_fin['atp-sensitive_potassium_channel_antagonist'] = 0

submission_fin['erbb2_inhibitor'] = 0
#submission_fin[cols].clip(1e-8,0.99975,inplace=True)

submission_fin.to_csv('submission.csv',index=False)