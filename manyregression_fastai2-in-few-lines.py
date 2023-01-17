!pip install -q fastai2
from fastai2.vision.all import *
path = Path("/kaggle/input/plant-pathology-2020-fgvc7")
train_df = pd.read_csv(path/"train.csv")

train_df.head()
train_df.query("image_id == 'Train_5'")
get_image_files(path/"images")[5]
train_df.iloc[0, 1:][train_df.iloc[0, 1:] == 1].index[0]
LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']
def get_data(size=224):

    return DataBlock(blocks    = (ImageBlock, CategoryBlock),

                       get_x=ColReader(0, pref=path/"images", suff=".jpg"),

                       get_y=lambda o:o.iloc[1:][o.iloc[1:] == 1].index[0],

                       splitter=RandomSplitter(seed=42),

                       item_tfms=Resize(size),

                       batch_tfms=aug_transforms(flip_vert=True),

                      )
dblock = get_data()
dsets = dblock.datasets(train_df)

dsets.train[0]
BS = (1024 - 256)//8
dls = dblock.dataloaders(train_df, bs=BS)

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
metric = partial(AccumMetric, flatten=False)



learn = cnn_learner(dls, resnet152, metrics=[

            error_rate,

            metric(healthy_roc_auc),

            metric(multiple_diseases_roc_auc),

            metric(rust_roc_auc),

            metric(scab_roc_auc),

            metric(roc_auc)]

        ).to_fp16()
# learn.lr_find()
# 1/0
# del learn
# import gc

# gc.collect()
lr = 3e-3
learn.fine_tune(4, lr)
test_df = pd.read_csv(path/"test.csv")

test_df.head()
tst_dl = learn.dls.test_dl(test_df)
preds, y = learn.get_preds(dl=tst_dl)
preds
subm = pd.read_csv(path/"sample_submission.csv")
subm.iloc[:, 1:] = preds
subm.to_csv("submission.csv", index=False, float_format='%.2f')
pd.read_csv("submission.csv")