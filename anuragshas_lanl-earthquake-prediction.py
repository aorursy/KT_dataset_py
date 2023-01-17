# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tqdm

import glob

from fastai.tabular import *
seed = 2019



# python RNG

random.seed(seed)



# pytorch RNGs

torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True

if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)



# numpy RNG

np.random.seed(seed)
def gen_features(X):

    strain = []

    strain.append(X.mean()) #0

    strain.append(X.std()) #1

    strain.append(X.min()) #2

    strain.append(X.max()) #3

    strain.append(X.kurtosis()) #4

    strain.append(X.skew()) #5

    strain.append(np.quantile(X,0.01)) #6

    strain.append(np.quantile(X,0.05)) #7

#     strain.append(np.quantile(X,0.10)) #8

#     strain.append(np.quantile(X,0.90)) #9

    strain.append(np.quantile(X,0.95)) #10

    strain.append(np.quantile(X,0.99)) #11

    strain.append(np.abs(X).max()) #12

    strain.append(np.abs(X).mean()) #13

    strain.append(np.abs(X).std()) #14

    strain.append(np.square(X).kurtosis()) #15

    strain.append(X.mad()) #16

    return pd.Series(strain)
train = pd.read_csv('../input/train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})



train_df = pd.DataFrame()

y_train = pd.Series()

for df in tqdm.tqdm_notebook(train):

    ch = gen_features(df['acoustic_data'])

    train_df = train_df.append(ch, ignore_index=True)

    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]),ignore_index=True)
train_df['time_to_failure'] = y_train
train_df.head()
X_test = pd.DataFrame()

seg_ids = []

for segs in tqdm.tqdm_notebook(sorted(glob.glob('../input/test/seg_*'))):

    seg_name = segs[segs.rfind('/')+1:segs.rfind('.')]

    sub = pd.read_csv(segs,iterator=True,dtype={'acoustic_data': np.int16})

    for df in sub:

        ch = gen_features(df['acoustic_data'])

        X_test = X_test.append(ch, ignore_index=True)

    seg_ids.append(seg_name)
X_test.head()
procs = [Normalize]

cont_vars = [x for x in range(train_df.shape[1]-1)]
data = (TabularList.from_df(train_df,procs=procs,cont_names=cont_vars)

                .split_by_rand_pct(0.01)

                .label_from_df(cols='time_to_failure')

                .add_test(TabularList.from_df(X_test))

                .databunch())
len(data.train_ds.cont_names)
class SmoothL1LossFlat(nn.SmoothL1Loss):

    def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:

        return super().forward(input.view(-1), target.view(-1))
learn = tabular_learner(data, layers=[1024,512], ps=[0.07,0.7], emb_drop=0.7,metrics=[mean_absolute_error])
learn.loss_func = SmoothL1LossFlat()
learn.model
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, 2e-2, wd=0.2)
test_preds=learn.get_preds(ds_type=DatasetType.Test)
sub_csv = pd.read_csv('../input/sample_submission.csv')
sub_csv.head()
sub_csv['time_to_failure'] = test_preds[0].numpy()
sub_csv.to_csv('submission.csv',index=False)