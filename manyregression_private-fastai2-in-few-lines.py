# !pip install -q fastai2

!pip install -q git+https://github.com/fastai/fastai2

!pip install -q git+https://github.com/fastai/fastcore
from fastai2.vision.all import *
path = Path("/kaggle/input/plant-pathology-2020-fgvc7")
train_df = pd.read_csv(path/"train.csv")

train_df.head()
train_df.query("image_id == 'Train_5'")
get_image_files(path/"images")[5]
train_df.iloc[0, 1:][train_df.iloc[0, 1:] == 1].index[0]
# LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']
# BS = 128

BS = 8
def get_data(size=224):

    return DataBlock(blocks    = (ImageBlock, CategoryBlock),

                       get_x=ColReader(0, pref=path/"images", suff=".jpg"),

                       get_y=lambda o:o.iloc[1:][o.iloc[1:] == 1].index[0],

                       splitter=RandomSplitter(seed=42),

                       item_tfms=Resize(size),

                       batch_tfms=aug_transforms(flip_vert=True),

                      ).dataloaders(train_df, bs=BS)
dls = get_data((450, 800))
# dblock.summary(train_df)
# dsets = dblock.datasets(train_df)

# dsets.train[0]
# 1/0
# del learn
# import gc

# gc.collect()
# dls = dblock.dataloaders(train_df, bs=BS)

dls.show_batch()
from sklearn.metrics import roc_auc_score



def roc_auc(preds, targs, labels=range(4)):

    # One-hot encode targets

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
# from fastai2.callback.cutmix import CutMix
# loss = partial(CrossEntropyLossFlat, weights=tensor([1,1.5,1,1]))
metric = partial(AccumMetric, flatten=False)



def get_learner(size=224):

    dls = get_data(size)

    return cnn_learner(dls, resnet152, metrics=[

                        error_rate,

                        metric(healthy_roc_auc),

                        metric(multiple_diseases_roc_auc),

                        metric(rust_roc_auc),

                        metric(scab_roc_auc),

                        metric(roc_auc)],

                       ).to_fp16()
learn = get_learner((450, 800))
# learn.lr_find()
lr = 3e-3
# learn.fine_tune(1, lr)
# learn.fit_one_cycle(2, slice(lr/10, lr))
m = "multiple_diseases_roc_auc"

d = 0.005

learn.fine_tune(50, lr, freeze_epochs=1, cbs=[EarlyStoppingCallback(monitor=m, min_delta=d, patience=10),

                                              SaveModelCallback(monitor=m, min_delta=d),

                                              ReduceLROnPlateau(monitor=m, min_delta=d, patience=4)])
# learn.fit_one_cycle(2, slice(lr/1000, lr/100))
# learn.data = get_data(448)
# learn.fit_one_cycle(5, slice(lr/1000, lr/100))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15, 10))
test_df = pd.read_csv(path/"test.csv")

test_df.head()
tst_dl = learn.dls.test_dl(test_df)
preds, _ = learn.get_preds(dl=tst_dl)
subm = pd.read_csv(path/"sample_submission.csv")
subm.iloc[:, 1:] = preds
subm.to_csv("submission.csv", index=False)
pd.read_csv("submission.csv")