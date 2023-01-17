!pip install -q comet_ml
from comet_ml import Experiment
from fastai.tabular import *
DATA_DIR='../input/'
def load_df(mode = 'train'):
    df = pd.read_csv(f'{DATA_DIR}{mode}_transaction.csv')
    # (590540, 434)
    dfi = pd.read_csv(f'{DATA_DIR}{mode}_identity.csv')
    df =  pd.merge(df, dfi, how='left', on='TransactionID')
    return df

df = load_df()
cat_names = [
              # Transaction
              'ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
              *[f'card{i}' for i in range(1,7)],
              *[f'M{i}' for i in range(1,10)],
              # Identity
              'DeviceType', 'DeviceInfo', 
               *[f'id_{i}' for i in range(12,39)],
             ]
cont_names = [i for i in df.keys() if i not in cat_names]

# Following SafeBox Kernel
drop_col = ['isFraud', 'TransactionID', 'TransactionDT', 'V300', 'V309', 'V111', 'C3', 'V124', 'V106', 'V125', 'V315', 'V134', 'V102', 'V123', 'V316', 'V113', 'V136', 'V305', 'V110', 'V299', 'V289', 'V286', 'V318', 'V103', 'V304', 'V116', 'V298', 'V284', 'V293', 'V137', 'V295', 'V301', 'V104', 'V311', 'V115', 'V109', 'V119', 'V321', 'V114', 'V133', 'V122', 'V319', 'V105', 'V112', 'V118', 'V117', 'V121', 'V108', 'V135', 'V320', 'V303', 'V297', 'V120']
for c in drop_col:
    if c in cat_names:
        cat_names.remove(c)
    else:
        cont_names.remove(c)

procs = [FillMissing, Categorify, Normalize] # Order matters
path = '';
metrics = [AUROC()]
dep_var = 'isFraud'
label_cls = CategoryList
log = False
y_range = None
kwargs = {'log': log} if log else {}

bs=4096
split_idx=int(0.8*df.shape[0])
# Take one sample of NA for each column
na_idxs = np.unique(np.concatenate([np.where(df[k].isna())[0][0:1] for k in df.keys()]))
train_idxs = np.concatenate([
    np.arange(split_idx),
    na_idxs[np.where(na_idxs>split_idx)],
])
valid_idxs = np.setdiff1d(arange_of(df), train_idxs)
# train_idxs.shape, valid_idxs.shape
data_bunch = TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs) \
    .split_by_idxs(train_idxs, valid_idxs) \
    .label_from_df(cols=dep_var, label_cls=label_cls, **kwargs).databunch(bs=bs*2*2)

len(data_bunch.train_ds), len(data_bunch.valid_ds)
learn = tabular_learner(data_bunch, layers=[200, 100], ps=[0.001, 0.01], emb_drop=0.04, metrics=metrics, y_range=y_range)
# learn.lr_find(stop_div=True, num_it=100)
# learn.recorder.plot()
experiment = Experiment(api_key="ADD_YOUR_KEY-from-comet.ml", project_name="ieee", workspace="username")
with experiment.train():
    learn.fit_one_cycle(1, 1e-2, wd=0.2)
    learn.fit_one_cycle(3, 1e-3, wd=0.2)

# from fastai.callbacks.tracker import EarlyStoppingCallback
# cbs=[EarlyStoppingCallback(learn, monitor='auroc', min_delta=0.001, patience=3, )]
# with experiment.train():
#     learn.fit_one_cycle(30, 1e-4, wd=0.2, callbacks=cbs)
filename='ieee'
!mkdir models
learn.export(f'models/{filename}.pkl')
# Cleanup

del learn
del df
import gc
del data_bunch
gc.collect()
df_test = load_df('test')
df_test[dep_var] = 0
for i in range(1,15):
    df_test[f'C{i}'] = df_test[f'C{i}'].fillna(-999)
# df_test.fillna(-999, inplace=True)
learn = load_learner(
    '',
    f'models/{filename}.pkl', 
    TabularList.from_df(df_test, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs),
    bs=bs
)
with experiment.test():
    test_preds, targs = learn.get_preds(ds_type=DatasetType.Test)
len(np.where(test_preds[:, 1]>0.5)[0])
# Sanity check
assert(len(np.where(test_preds[:, 1]>0.5)[0])>8000)
submission = df_test[['TransactionID']].copy()
submission['isFraud']=test_preds[:, 1]
name = filename
submission.to_csv('submission.csv', index=False, header=True)