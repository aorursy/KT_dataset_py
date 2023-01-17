%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
from fastai.vision import *

from fastai.callbacks import *
import os
print(os.listdir("../input"))
PATH = Path('../input/aptos2019-blindness-detection')
[x for x in PATH.iterdir() if x.is_dir()]
train = pd.read_csv(PATH/'train.csv')

test = pd.read_csv(PATH/'test.csv')

_ = train.hist()
SEED = 20192

def ret_percentage(column):

    return round(column.value_counts(normalize=True) * 100,2)
#Holdout test set from training samples

from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state= SEED)

for train_index, test_index in split.split(train["id_code"], train["diagnosis"]):

    df_train = train.iloc[train_index]

    df_test = train.iloc[test_index]



#print("Old Train Class Percentage Dist:\n", (df_train["diagnosis"]))



   

print("New Train Sample Size", df_train.shape)

print("New Train Class Percentage Dist:\n", ret_percentage(df_train["diagnosis"]))



print("New Test Sample Size",df_test.shape)

print("New Test Class Percentage Dist:\n", ret_percentage(df_test["diagnosis"]))
## Initialize batch processing size

bs = 16  #64

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart

sz = 256 #224 #Image size

n_folds = 2

model_name = "resnet50"

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm



skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state = SEED)

tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.1,max_lighting=0.1,p_lighting=0.5)
tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.1,max_lighting=0.1,p_lighting=0.5)
from sklearn.metrics import cohen_kappa_score

def quadratic_kappa(y_hat, y):

    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')





Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp '../input/resetnet50-weight/resnet50-19c8e357.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'


print(os.listdir("../input"))

#del learn

gc.collect()
kp_score = []

err_rate = []

losses = []

#loss_func = FocalLoss(gamma=1.)

learn = ""

data_fold = ""

predictions = torch.from_numpy(np.zeros((len(df_test))))



for fold, (train_index, val_index) in tqdm(enumerate(skf.split(df_train["id_code"], df_train["diagnosis"]))):

    del learn, data_fold

    gc.collect()

    filename = '/tmp/' + model_name + "fold_" + str(fold)+".pkl"

    print("Fold:", filename)

    print("TRAIN:", train_index, "VALIDATE:", val_index)

    data_fold = (ImageList.from_df(df_train, PATH, folder='train_images', cols="id_code",suffix='.png')

        .split_by_idxs(train_index, val_index)

        .label_from_df(cols='diagnosis')

        .transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') #Data transform

        .databunch(bs=bs)).normalize(imagenet_stats)

   # learn = cnn_learner(data_fold, base_arch=models.resnet50, loss_func = mse, metrics=mse)

    learn = cnn_learner(data_fold, base_arch=models.resnet50, metrics=[accuracy, KappaScore(weights="quadratic")],callback_fns=[BnFreeze,partial(SaveModelCallback, monitor='kappa_score')])

    learn.model_dir = '/tmp/'

    lr = 0.02

    learn.fit_one_cycle(2, slice(lr))

    learn.save('stage-1-rn50')

    learn.unfreeze()

    learn.fit_one_cycle(2, slice(1e-4, lr/5))

    learn.save('stage-2-rn50')

    learn.freeze()

    lr=1e-2/2

    learn.save('stage-1-256-rn50')

    learn.unfreeze()

    learn.fit_one_cycle(2, slice(1e-4, lr/5))

    learn.save('stage-2-256-rn50')

    learn.export(filename)

    loss, err , kp = learn.validate()

    kp_score.append(kp.numpy())

    err_rate.append(err.numpy())

    losses.append(loss)

    learn.data.add_test(ImageList.from_df(df_test ,PATH ,folder='train_images',suffix='.png'))

    preds, _ = learn.TTA(ds_type=DatasetType.Test)

    predictions = predictions + preds.argmax(dim=-1).double()
predictions = torch.round(predictions/n_folds)

df_test['diagnosis_pred'] = pd.Series(predictions.numpy().astype(int), index=df_test.index)

df_test.head()
from scipy.spatial.distance import cosine



print("New Test Set Correlation:", df_test['diagnosis'].corr(df_test['diagnosis_pred']))

print("New Test Set Cosine Similarity:", 1 - cosine(df_test["diagnosis"], df_test["diagnosis_pred"]))

df_test.to_csv('submission.csv',index=False)
df_test.head()