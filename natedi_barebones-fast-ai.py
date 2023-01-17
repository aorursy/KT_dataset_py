
%matplotlib  inline
%reload_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np

from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)

PATH_WRITE = "/kaggle/working/"


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
ssub = pd.read_csv('../input/sample_submission.csv')

print(f'train: {train.shape}', f'test {test.shape}')
train.head()
test.head()
for col in ['store', 'item']:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
    
train.describe(include='all')
train.isnull().sum()
train2 = train.copy()
test2 = test.copy()

add_datepart(train2, "date", drop=False)
add_datepart(test2, "date", drop=False)
train2.head()
test2.head()
cat_vars = list(train2)
[cat_vars.remove(col) for col in ['sales', 'Elapsed', 'date']]
for v in cat_vars: train2[v] = train2[v].astype('category').cat.as_ordered()
apply_cats(test2, train2)
for v in ['sales', 'Elapsed']:
    train2[v] = train2[v].fillna(0).astype('float32')
    if v in test2:
        test2[v] = test2[v].fillna(0).astype('float32')
train2 = train2.set_index('date')
test2 = test2.set_index('date')

df, y, nas, mapper = proc_df(train2, 'sales', do_scale=True)
yl = np.log(y+1)
test2['sales'] = 0
df_test, _, nas, mapper = proc_df(test2, 'sales', do_scale=True, skip_flds=['id'], mapper=mapper, na_dict=nas)
df_test.info()
df.info()
val_idx = np.flatnonzero((df.index<datetime.datetime(2018,1,1)) & (df.index>=datetime.datetime(2017,10,1)))
def inv_y(a): return np.exp(a) - 1

def smape(y_pred, targ):
    targ = inv_y(targ)
    pred = inv_y(y_pred)
    ape = 2 * np.abs(pred - targ) / (np.abs(pred) + np.abs(targ))
    return ape.mean() 

max_log_y = np.max(yl)
y_range = (0, max_log_y*1.2)
class _ColumnarModelData(ColumnarModelData):
    @classmethod
    def from_data_frames(cls, path, trn_df, val_df, trn_y, val_y, cat_flds, bs, is_reg, test_df=None):
        test_ds = ColumnarDataset.from_data_frame(test_df, cat_flds, None, is_reg) if test_df is not None else None
        return cls(path, ColumnarDataset.from_data_frame(trn_df, cat_flds, trn_y, is_reg),
                    ColumnarDataset.from_data_frame(val_df, cat_flds, val_y, is_reg), bs, test_ds=test_ds)


md = _ColumnarModelData.from_data_frame('.', val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128, test_df=df_test)
cat_sz = [(c, len(train2[c].cat.categories)+1) for c in cat_vars]
cat_sz
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
emb_szs

m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range, 
                   tmp_name=f"{PATH_WRITE}tmp", models_name=f"{PATH_WRITE}models")
lr = 1e-3
m.lr_find()
m.sched.plot(100)
m.fit(lr, 3, metrics=[smape])
m.save('val0')
m.load('val0')
x,y=m.predict_with_targs()
smape(x, y)
pred_test=m.predict(True)
pred_test=np.exp(pred_test) - 1
test2['sales'] = pred_test
test2[['id','sales']].to_csv('predictions0.csv', index=False)
