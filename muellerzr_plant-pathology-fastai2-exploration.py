!pip install fastai2 --quiet
from fastai2.vision.all import *
path = Path('../input/plant-pathology-2020-fgvc7')
train_df = pd.read_csv(path/'train.csv')
train_df.head()
train_df.iloc[0, 1:]
train_df.iloc[0,1:][train_df.iloc[0, 1:]==1].index[0]
%%timeit

_ = train_df.iloc[0,1:][train_df.iloc[0, 1:]==1].index[0]
df_np = train_df.to_numpy()
index2name = {1:'healthy',

      2:'multiple_diseases',

      3:'rust',

      4:'scab'}
df_np[0]
%%timeit

idx = np.where(df_np[1]==1)[0][0]

y = index2name[idx]
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock))
def get_x(fn): print(fn)
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),

                  get_x=get_x)
dblock.summary(df_np)
def get_x(row): return path/Path('images/'+row[0]+'.jpg')
get_x(df_np[0])
PILImage.create(get_x(df_np[0]))
idx = np.where(df_np[1]==1)[0][0]

y = index2name[idx]
def get_y(row):

    idx = np.where(row==1)[0][0]

    return index2name[idx]
get_y(df_np[0])
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),

                  get_x=get_x,

                  get_y=get_y)
splitter = RandomSplitter(valid_pct=0.2, seed=42)
splitter
idxs = list(range(1,11)); idxs
splitter(idxs)
item_tfms = RandomResizedCrop(224)

batch_tfms=[*aug_transforms(size=224, flip_vert=True),

                                   Normalize]
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),

                  get_x=get_x,

                  get_y=get_y,

                  splitter=splitter,

                  item_tfms=item_tfms,

                  batch_tfms=batch_tfms)
dls = dblock.dataloaders(df_np, bs=64)
dls.train.after_batch
norm = dls.train.after_batch.normalize
norm.mean, norm.std
dset = Datasets(items=df_np, tfms=[[get_x, PILImage.create], [get_y, Categorize]])
dset[0]
dl = TfmdDL(dset, after_item=dls.train.after_item,

                     after_batch=[IntToFloatTensor(), Normalize()],

                     bs=64, device='cuda')
dl = TfmdDL(dset, after_item=dls.train.after_item,

                     after_batch=[IntToFloatTensor(), Normalize()],

                     bs=len(dset), device='cuda')
norm = dl.after_batch.normalize
norm.mean, norm.std
norm.mean.flatten()
plant_norm = (norm.mean.flatten(), norm.std.flatten()); plant_norm
item_tfms = RandomResizedCrop(224)

batch_tfms=[*aug_transforms(size=224, flip_vert=True),

                                   Normalize.from_stats(*plant_norm)]
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),

                  get_x=get_x,

                  get_y=get_y,

                  splitter=splitter,

                  item_tfms=item_tfms,

                  batch_tfms=batch_tfms)
dls = dblock.dataloaders(df_np, bs=32)
dls.show_batch()
net = xresnet50(pretrained=False, act_cls=Mish, sa=True, n_out=dls.c)
@delegates(RAdam)

def ranger(p, lr, mom=0.95, wd=0.01, eps=1e-6, **kwargs):

    "Convenience method for `Lookahead` with `RAdam`"

    return Lookahead(RAdam(p, lr=lr, mom=mom, wd=wd, eps=eps, **kwargs))
opt_func = ranger
from sklearn.metrics import roc_auc_score



def roc_auc(preds, targs, labels=range(4)):

    targs = np.eye(4)[targs]

    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])



def healthy_roc_auc(*args):

    return roc_auc(*args, labels=[0])



def multiple_diseases_roc_auc(*args):

    return roc_auc(*args, labels=[1])



def rust_roc_auc(*args):

    return roc_auc(*args, labels=[2])



def scab_roc_auc(*args):

    return roc_auc(*args, labels=[3])
metric = partial(AccumMetric, flatten=False)
metrics=[

            error_rate,

            metric(healthy_roc_auc),

            metric(multiple_diseases_roc_auc),

            metric(rust_roc_auc),

            metric(scab_roc_auc),

            metric(roc_auc)]
learn = Learner(dls, net, opt_func=ranger, loss_func=CrossEntropyLossFlat(),

               metrics=metrics)
learn.lr_find()
learn.fit_flat_cos(5, 1e-3)
from fastai2.test_utils import synth_learner

one_cycle = synth_learner()

one_cycle.fit_one_cycle(1)
one_cycle.recorder.plot_sched()
flat_cos = synth_learner()

flat_cos.fit_flat_cos(1)
flat_cos.recorder.plot_sched()
learn.fit_flat_cos(5, 1e-4)
preds, targs = learn.tta(ds_idx=1, n=4)
error_rate(preds, targs)
from sklearn.model_selection import StratifiedKFold
train_lbls = []

for _, lbl in dset:

    train_lbls.append(lbl)
imgs = df_np[:,0]
def get_data(idx, size=224, bs=64):

    dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),

                       get_x=get_x,

                       get_y=get_y,

                       splitter=IndexSplitter(idx),

                       item_tfms=RandomResizedCrop(size+64),

                       batch_tfms=[*aug_transforms(size=size, flip_vert=True),

                                   Normalize.from_stats(*plant_norm)],

                      )

    return dblock.dataloaders(train_df, bs=bs)
test_df = pd.read_csv(path/'test.csv')
test_np = test_df.to_numpy()
test_dl = learn.dls.test_dl(test_np)
test_dl.show_batch()
import gc
test_preds = []

skf = StratifiedKFold(n_splits=3, shuffle=True)

for _, val_idx in skf.split(imgs, np.array(train_lbls)):

    dls = get_data(val_idx, 128, 32)

    net = xresnet50(pretrained=False, act_cls=Mish, sa=True, n_out=dls.c)

    learn = Learner(dls, net, opt_func=ranger, loss_func=CrossEntropyLossFlat(),

               metrics=metrics)

    learn.fit_flat_cos(5, 4e-3, cbs=EarlyStoppingCallback(monitor='roc_auc'))

    learn.save('initial')

    del learn

    torch.cuda.empty_cache()

    gc.collect()

    

    dls = get_data(val_idx, 256, 16)

    learn = Learner(dls, net, opt_func=ranger, loss_func=CrossEntropyLossFlat(),

               metrics=metrics)

    learn.load('initial');

    learn.fit_flat_cos(10, 1e-3, cbs=EarlyStoppingCallback(monitor='roc_auc'))

    learn.save('stage-1')

    del learn

    torch.cuda.empty_cache()

    gc.collect()

    dls = get_data(val_idx, 448, 8)

    learn = Learner(dls, net, opt_func=ranger, loss_func=CrossEntropyLossFlat(),

               metrics=metrics)

    learn.fit_flat_cos(5, slice(1e-5, 1e-4), cbs=EarlyStoppingCallback(monitor='roc_auc'))

    tst_dl = learn.dls.test_dl(test_df)

    y, _ = learn.tta(dl=tst_dl)

    test_preds.append(y)

    del learn

    torch.cuda.empty_cache()

    gc.collect()
tot = test_preds[0]

for i in test_preds[1:]:

    tot += i
len(test_preds)
tot = tot / 3
subm = pd.read_csv(path/'sample_submission.csv')
subm.iloc[:, 1:] = tot

subm.to_csv("submission.csv", index=False, float_format='%.2f')