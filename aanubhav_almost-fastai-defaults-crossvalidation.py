from fastai import *

from fastai.tabular import *

from fastai.callbacks import EarlyStoppingCallback, ReduceLROnPlateauCallback

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import StratifiedKFold
path = Path('../input/cat-in-the-dat-ii/') 

df = pd.read_csv(path/'train.csv')

df.set_index('id',drop=True,inplace=True)

test_df = pd.read_csv(path/'test.csv')

test_df.set_index('id',drop=True,inplace=True)

sample = pd.read_csv(path/'sample_submission.csv')
df.head(3)
test_df.head(3)
df.bin_0.nunique(), df.bin_1.nunique(), df.bin_2.nunique(), df.bin_3.nunique(), df.bin_4.nunique()
df.nom_0.nunique(), df.nom_1.nunique(), df.nom_2.nunique(), df.nom_3.nunique(), df.nom_4.nunique(), df.nom_5.nunique(), df.nom_6.nunique(), df.nom_7.nunique(), df.nom_8.nunique(), df.nom_9.nunique(), 
df.ord_0.nunique(), df.ord_1.nunique(), df.ord_2.nunique(), df.ord_3.nunique(), df.ord_4.nunique(), df.ord_5.nunique() 
df.columns
dep_var = 'target'

cat_names = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1',

             'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8',

             'nom_9','ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5',

             'day', 'month']

procs = [FillMissing, Categorify, Normalize]
embed = {}

for col in cat_names:

    embed[col] = min(50, df[col].nunique()) 
embed
# test = TabularList.from_df(test_df, path=path, cat_names=cat_names, cont_names=cont_names)#, procs=procs)



test = TabularList.from_df(test_df, path=path, cat_names=cat_names)
num_splits = 10

skf = StratifiedKFold(num_splits, shuffle=True)#, random_state=42)
pred_list = []

for train_idx, valid_idx in skf.split(df.drop(labels='target', axis=1),df['target']):

#     print(train_idx[:10], valid_idx[:10])

#     print(len(train_idx), len(valid_idx))

    data_fold = (TabularList.from_df(df, path=path, cat_names=cat_names, procs=procs)

                 .split_by_idxs(train_idx, valid_idx)

                 .label_from_df(cols=dep_var)

                 .add_test(test)

                 .databunch(bs=1024))

#     print(len(data_fold.train_ds), len(data_fold.valid_ds), len(data_fold.test_ds))

    

    learn = tabular_learner(data_fold, layers=[300, 300], emb_drop=0.04, ps=[0.001, 0.01],metrics=AUROC(), emb_szs = embed,

                            callback_fns=[partial(ReduceLROnPlateauCallback, monitor='auroc', min_delta=0.01, patience=1, min_lr=1e-06, factor=0.10, mode='max'),

                                          partial(EarlyStoppingCallback, monitor='auroc', min_delta=0.001, patience=2, mode='max')])

    learn.loss_func = nn.CrossEntropyLoss()

#     print(learn.model)

    learn.fit(10, 3e-3, wd = 0.2)

    

    pred = learn.get_preds(DatasetType.Test)[0][:, 1]

    pred_list.append(pred.numpy())

    

    
len(pred_list)
pred_list
np_preds = np.array(pred_list)
preds = np.sum(np_preds, axis=0)/num_splits
preds.shape
max(preds), min(preds)
def get_submision(preds):

    sample['target'] = preds

    return sample
submission = get_submision(preds)
submission.head()
submission.to_csv('./sub_10f_commit.csv',index=False)