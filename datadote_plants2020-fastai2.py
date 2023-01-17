!pip install fastai2 --quiet
from fastai2.vision.all import *
from sklearn.metrics import roc_auc_score
path = Path('../input/plant-pathology-2020-fgvc7')

# df has been split into train/valid (30 classes for each label)
# rest of training labels have been oversampled to balance the classes
#df = pd.read_csv('../input/plant-val120/plants_all_train2429.csv')
df = pd.read_csv('../input/plant-val120/plants_val120_train1974.csv')
#df = pd.read_csv('../input/plant-val120/plants_val120_train1280.csv')
#df = pd.read_csv('../input/plant-val120/plants_val120_train783.csv')

# shuffle data
df = df.sample(frac=1, random_state=42)
df.reset_index(drop=True, inplace=True)
# code below shows distribution of labels
df.drop(columns='image_id').sum()
df.head(5)
# ROC metrics taken from Lex's notebook kernel
def comp_metric(preds, targs, labels=range(4)):
    targs = np.eye(4)[targs]
    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])

def healthy_auc(*args):
    return comp_metric(*args, labels=[0])

def mult_diseases_auc(*args):
    return comp_metric(*args, labels=[1])

def rust_auc(*args):
    return comp_metric(*args, labels=[2])

def scab_auc(*args):
    return comp_metric(*args, labels=[3])
# r = 1 row from dataframe taken
def get_x(r): return path/f"images/{r['image_id']}.jpg"
def get_y(r): return r[r==1].index[0] # want to return label for name

# splitter tells dataloader to split train/valid dataset from dataframe
def splitter(df):
    train = df.index[~df.is_valid].tolist()
    valid = df.index[df.is_valid].tolist()
    return train,valid

item_tfms = RandomResizedCrop(size=1280, min_scale=0.8)

# aug transforms has a lot of things in it.
# Run doc(aug_transforms) for more details
batch_tfms = [*aug_transforms(size=1024,
                              flip_vert=True, 
                              max_rotate=40),
              Normalize.from_stats(*imagenet_stats)]
    
# make data dataloaders
db = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_x = get_x,
    get_y = get_y,
    splitter = splitter,
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
)
    
dls = db.dataloaders(df, bs=8)
# Create resnet50 architecture
learn = cnn_learner(dls,
                    arch=resnet50,
                    pretrained=True,
                    n_out=4,
                    loss_func = LabelSmoothingCrossEntropy(0.1),
                    metrics=[accuracy,
                             AccumMetric(healthy_auc, flatten=False),
                             AccumMetric(mult_diseases_auc, flatten=False),
                             AccumMetric(rust_auc, flatten=False),
                             AccumMetric(scab_auc, flatten=False),
                             AccumMetric(comp_metric, flatten=False)
                            ]).to_fp16()
learn.lr_find() # find a learning rate to use
# trains model with initial coarse epoch, and then 8 fine tune epochs
learn.fine_tune(epochs=8,
                base_lr=0.001, # taken from above plot
                freeze_epochs=1
               )
learn.recorder.plot_sched()
interp = ClassificationInterpretation.from_learner(learn, dl=dls.valid)
interp.plot_confusion_matrix(dpi=60)
interp.plot_top_losses(6)
