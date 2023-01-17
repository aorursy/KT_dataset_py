import numpy as np

from tqdm import tqdm

import pandas as pd

import random

from pathlib import Path

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

import IPython

import IPython.display

from IPython.display import FileLink

import PIL

import pickle

from sklearn.model_selection import KFold

import torch

import torch.nn as nn

import torch.nn.functional as F

from fastai.core import *

from fastai.basic_data import *

from fastai.basic_train import *

from fastai.torch_core import *

from fastai import *

from fastai.vision import *

from fastai.vision.data import *

from fastai.callbacks import *

import random





import os
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 999

seed_everything(SEED)
def _one_sample_positive_class_precisions(scores, truth):

    """Calculate precisions for each true class for a single sample.



    Args:

      scores: np.array of (num_classes,) giving the individual classifier scores.

      truth: np.array of (num_classes,) bools indicating which classes are true.



    Returns:

      pos_class_indices: np.array of indices of the true classes for this sample.

      pos_class_precisions: np.array of precisions corresponding to each of those

        classes.

    """

    num_classes = scores.shape[0]

    pos_class_indices = np.flatnonzero(truth > 0)

    # Only calculate precisions if there are some true classes.

    if not len(pos_class_indices):

        return pos_class_indices, np.zeros(0)

    # Retrieval list of classes for this sample.

    retrieved_classes = np.argsort(scores)[::-1]

    # class_rankings[top_scoring_class_index] == 0 etc.

    class_rankings = np.zeros(num_classes, dtype=np.int)

    class_rankings[retrieved_classes] = range(num_classes)

    # Which of these is a true label?

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)

    retrieved_class_true[class_rankings[pos_class_indices]] = True

    # Num hits for every truncated retrieval list.

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    # Precision of retrieval list truncated at each hit, in order of pos_labels.

    precision_at_hits = (

            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /

            (1 + class_rankings[pos_class_indices].astype(np.float)))

    return pos_class_indices, precision_at_hits





def calculate_per_class_lwlrap(truth, scores):

    """Calculate label-weighted label-ranking average precision.



    Arguments:

      truth: np.array of (num_samples, num_classes) giving boolean ground-truth

        of presence of that class in that sample.

      scores: np.array of (num_samples, num_classes) giving the classifier-under-

        test's real-valued score for each class for each sample.



    Returns:

      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each

        class.

      weight_per_class: np.array of (num_classes,) giving the prior of each

        class within the truth labels.  Then the overall unbalanced lwlrap is

        simply np.sum(per_class_lwlrap * weight_per_class)

    """

    assert truth.shape == scores.shape

    num_samples, num_classes = scores.shape

    # Space to store a distinct precision value for each class on each sample.

    # Only the classes that are true for each sample will be filled in.

    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))

    for sample_num in range(num_samples):

        pos_class_indices, precision_at_hits = (

            _one_sample_positive_class_precisions(scores[sample_num, :],

                                                  truth[sample_num, :]))

        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (

            precision_at_hits)

    labels_per_class = np.sum(truth > 0, axis=0)

    weight_per_class = labels_per_class / float(np.sum(labels_per_class))

    # Form average of each column, i.e. all the precisions assigned to labels in

    # a particular class.

    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /

                        np.maximum(1, labels_per_class))

    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions

    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)

    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples

    #                = np.sum(per_class_lwlrap * weight_per_class)

    return per_class_lwlrap, weight_per_class



def lwlrap(y_pred,y_true):

    score, weight = calculate_per_class_lwlrap(y_true.cpu().numpy(), y_pred.cpu().numpy())

    lwlrap = (score * weight).sum()

    return torch.from_numpy(np.array(lwlrap))
DATA = Path('../input/freesound-audio-tagging-2019')

PREPROCESSED = Path('../input/fat2019_prep_mels1')

WORK = Path('work')

Path(WORK).mkdir(exist_ok=True, parents=True)



CSV_TRN_CURATED = DATA/'train_curated.csv'

CSV_TRN_NOISY = DATA/'train_noisy.csv'

CSV_TRN_NOISY_BEST50S = PREPROCESSED/'trn_noisy_best50s.csv'

CSV_SUBMISSION = DATA/'sample_submission.csv'



MELS_TRN_CURATED = PREPROCESSED/'mels_train_curated.pkl'

MELS_TRN_NOISY = PREPROCESSED/'mels_train_noisy.pkl'

MELS_TRN_NOISY_BEST50S = PREPROCESSED/'mels_trn_noisy_best50s.pkl'

MELS_TEST = PREPROCESSED/'mels_test.pkl'



trn_curated_df = pd.read_csv(CSV_TRN_CURATED)

trn_noisy_df = pd.read_csv(CSV_TRN_NOISY)

trn_noisy50s_df = pd.read_csv(CSV_TRN_NOISY_BEST50S)

test_df = pd.read_csv(CSV_SUBMISSION)



#df = pd.concat([trn_curated_df, trn_noisy_df], ignore_index=True) # not enough memory

df = pd.concat([trn_curated_df, trn_noisy50s_df], ignore_index=True, sort=True)

test_df = pd.read_csv(CSV_SUBMISSION)



X_train = pickle.load(open(MELS_TRN_CURATED, 'rb')) + pickle.load(open(MELS_TRN_NOISY_BEST50S, 'rb'))

df.sample(10)


CUR_X_FILES, CUR_X = list(df.fname.values), X_train



def _open_fat2019_image(fn, convert_mode, after_open)->Image:

    # open

    idx = CUR_X_FILES.index(fn.split('/')[-1])

    x = PIL.Image.fromarray(CUR_X[idx])

    # crop 1sec

    time_dim, base_dim = x.size

    crop_x = random.randint(0, time_dim - base_dim)

    x = x.crop([crop_x, 0, crop_x+base_dim, base_dim])    

    # standardize

    return Image(pil2tensor(x, np.float32).div_(255))



vision.data.open_image = _open_fat2019_image



def get_data(split=([],[]), with_noisy=False):

    tfms = get_transforms(do_flip=True, max_rotate=0, max_lighting=0.1, max_zoom=0, max_warp=0.)

    src = ImageList.from_df(df if with_noisy else trn_curated_df, path=WORK, cols=['fname'])

    src = src.split_by_idx(split[1]) if len(split[1]) > 0 else src.split_none()

    data = (src.label_from_df(label_delim=',').transform(tfms, size=128).databunch(bs=128).normalize(imagenet_stats))

    return data



classes = get_data().classes
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, 3, 1, 1),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(),

        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(),

        )



        self._init_weights()

        

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight)

                if m.bias is not None:

                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.zeros_(m.bias)

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = F.avg_pool2d(x, 2)

        return x

    

class Classifier(nn.Module):

    def __init__(self, num_classes=1000): # <======== modificaition to comply fast.ai

        super().__init__()

        

        self.conv = nn.Sequential(

            ConvBlock(in_channels=3, out_channels=64),

            ConvBlock(in_channels=64, out_channels=128),

            ConvBlock(in_channels=128, out_channels=256),

            ConvBlock(in_channels=256, out_channels=512),

        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # <======== modificaition to comply fast.ai

        self.fc = nn.Sequential(

            nn.Dropout(0.2),

            nn.Linear(512, 128),

            nn.PReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.1),

            nn.Linear(128, num_classes),

        )



    def forward(self, x):

        x = self.conv(x)

        #x = torch.mean(x, dim=3)   # <======== modificaition to comply fast.ai

        #x, _ = torch.max(x, dim=2) # <======== modificaition to comply fast.ai

        x = self.avgpool(x)         # <======== modificaition to comply fast.ai

        x = self.fc(x)

        return x
class MixUpCallback(LearnerCallback):

    "Callback that creates the mixed-up input and target."

    def __init__(self, learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True):

        super().__init__(learn)

        self.alpha,self.stack_x,self.stack_y = alpha,stack_x,stack_y

    

    def on_train_begin(self, **kwargs):

        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)

        

    def on_batch_begin(self, last_input, last_target, train, **kwargs):

        "Applies mixup to `last_input` and `last_target` if `train`."

        if not train: return

        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))

        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)

        lambd = last_input.new(lambd)

        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)

        x1, y1 = last_input[shuffle], last_target[shuffle]

        if self.stack_x:

            new_input = [last_input, last_input[shuffle], lambd]

        else: 

            new_input = (last_input * lambd.view(lambd.size(0),1,1,1) + x1 * (1-lambd).view(lambd.size(0),1,1,1))

        if self.stack_y:

            new_target = torch.cat([last_target[:,None].float(), y1[:,None].float(), lambd[:,None].float()], 1)

        else:

            if len(last_target.shape) == 2:

                lambd = lambd.unsqueeze(1).float()

            new_target = last_target.float() * lambd + y1.float() * (1-lambd)

        return {'last_input': new_input, 'last_target': new_target}  

    

    def on_train_end(self, **kwargs):

        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()

        



class MixUpLoss(nn.Module):

    "Adapt the loss function `crit` to go with mixup."

    

    def __init__(self, crit, reduction='mean'):

        super().__init__()

        if hasattr(crit, 'reduction'): 

            self.crit = crit

            self.old_red = crit.reduction

            setattr(self.crit, 'reduction', 'none')

        else: 

            self.crit = partial(crit, reduction='none')

            self.old_crit = crit

        self.reduction = reduction

        

    def forward(self, output, target):

        if len(target.size()) == 2:

            loss1, loss2 = self.crit(output,target[:,0].long()), self.crit(output,target[:,1].long())

            d = (loss1 * target[:,2] + loss2 * (1-target[:,2])).mean()

        else:  d = self.crit(output, target)

        if self.reduction == 'mean': return d.mean()

        elif self.reduction == 'sum':            return d.sum()

        return d

    

    def get_old(self):

        if hasattr(self, 'old_crit'):  return self.old_crit

        elif hasattr(self, 'old_red'): 

            setattr(self.crit, 'reduction', self.old_red)

            return self.crit



def mixup(learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True) -> Learner:

    "Add mixup https://arxiv.org/abs/1710.09412 to `learn`."

    learn.callback_fns.append(partial(MixUpCallback, alpha=alpha, stack_x=stack_x, stack_y=stack_y))

    return learn

Learner.mixup = mixup





def _borrowed_model(pretrained=False, **kwargs):

    return Classifier(**kwargs)



def get_learner(data):

    learn = cnn_learner(data, _borrowed_model, pretrained=False, metrics=[lwlrap]).mixup(stack_y=False)

    learn.unfreeze()

    return learn
def train_and_export(learn, fold='', epochs=200, lr=2e-2, fast_start=False):

    # To save the best model

    save_cb = SaveModelCallback(learn, every='improvement', monitor='lwlrap', name=f'best{fold}')

    # Do a quick 1-cycle training to converge faster

    if fast_start: learn.fit_one_cycle(1, lr, callbacks=[save_cb])

    # Fit for given number of epochs

    learn.fit_one_cycle(epochs, lr, callbacks=[save_cb])

    # Export the model definition

    learn.export(file=f'export{fold}.pkl')
def split_kfold(X, n_folds=5, seed=42):

    cv = KFold(n_folds, shuffle=True, random_state=seed)

    return enumerate(cv.split(X))
def run_pipeline(n_folds=5, epochs=200, with_noisy=False, lr=0.02, fast_start=False):

    if n_folds > 0:

        print(f"Training with {n_folds}-fold CV")

        # Train with k-fold

        for fold, split in split_kfold(trn_curated_df, n_folds):

            print(f'\nFold no: {fold}')

            if with_noisy:

                print(f"\tWith noisy data")

                # Train 50% with noisy

                data = get_data(split, with_noisy=True)

                learn = get_learner(data)

                train_and_export(learn, fold, epochs//2, lr, fast_start)

                # Train 50% without noisy

                print(f"\tWithout noisy data")

                data = get_data(split)

                learn.data = data

                train_and_export(learn, fold, epochs//2, lr, fast_start)

            else:

                # Train 100% without noisy

                data = get_data(split)

                learner = get_learner(data)

                train_and_export(learner, fold, epochs, lr, fast_start)

    

    # Also train on entire dataset

    print(f"Training with entire training set")

    if with_noisy:

        # Train 50% with noisy

        print(f"\tWith noisy data")

        data = get_data(with_noisy=True)

        learn = get_learner(data)

        train_and_export(learn, '', epochs//2, lr, fast_start)

        # Train 50% without noisy

        print(f"\tWithout noisy data")

        data = get_data()

        learn.data = data

        train_and_export(learn, '', epochs//2, lr, fast_start)

    else:

        train_and_export(get_learner(get_data()), '', epochs, lr, fast_start)

    

    print("Training complete")

    !ls {WORK}

    !ls {WORK}/models
%%time

# Uncomment for training

# run_pipeline()

run_pipeline(epochs=1)
EXPORTS_DIR=WORK

MODELS_DIR=WORK/'models'



# Fill in this value from created dataset

# EXPORTS_DIR=Path('name-of-folder')

# MODELS_DIR=EXPORTS_DIR
try:

    del X_train

except NameError:

    pass



def _tta_only(learn:Learner, ds_type:DatasetType=DatasetType.Valid, num_pred:int=5) -> Iterator[List[Tensor]]:

    "Computes the outputs for several augmented inputs for TTA"

    dl = learn.dl(ds_type)

    ds = dl.dataset

    old = ds.tfms

    aug_tfms = [o for o in learn.data.train_ds.tfms]

    try:

        pbar = master_bar(range(num_pred))

        for i in pbar:

            ds.tfms = aug_tfms

            yield get_preds(learn.model, dl, pbar=pbar)[0]

    finally: ds.tfms = old



Learner.tta_only = _tta_only



def _TTA(learn:Learner, beta:float=0.15, ds_type:DatasetType=DatasetType.Valid, num_pred:int=5, with_loss:bool=False) -> Tensors:

    "Applies TTA to predict on `ds_type` dataset."

    preds,y = learn.get_preds(ds_type)

    all_preds = list(learn.tta_only(ds_type=ds_type, num_pred=num_pred))

    avg_preds = torch.stack(all_preds).mean(0)

    if beta is None: return preds,avg_preds,y

    else:            

        final_preds = preds*beta + avg_preds*(1-beta)

        if with_loss: 

            with NoneReduceOnCPU(learn.loss_func) as lf: loss = lf(final_preds, y)

            return final_preds, y, loss

        return final_preds, y



Learner.TTA = _TTA
X_test = pickle.load(open(MELS_TEST, 'rb'))

CUR_X_FILES, CUR_X = list(test_df.fname.values), X_test

test = ImageList.from_csv(WORK, Path('..')/CSV_SUBMISSION, folder='test')



def load_and_infer(fold='', best=False, export=True):

    export_path = f'export{fold}.pkl'

    print(f'Inferring with {export_path}')

    learn = load_learner(EXPORTS_DIR, file=export_path, test=test)

    best_path = MODELS_DIR/f'best{fold}.pth'

    if best and os.path.exists(best_path):

        print(f'Loading model {best_path}')

        learn.model.load_state_dict(torch.load(best_path), strict=False)

    preds, _ = learn.TTA(ds_type=DatasetType.Test)

    return torch.sigmoid(preds)



def export_preds(preds, fold=''):

    test_df[classes] = preds

    fname = f'submission{fold}.csv'

    test_df.to_csv(fname, index=False)

    return test_df.head()

def run_inference(n_folds=5, best=False, beta=0.3):

    # Model trained on entire data

    preds=load_and_infer()

    

    # Model train on k-folds

    if n_folds > 0:

        export_preds(preds, '-all')

        kfold_preds = [load_and_infer(fold, best) for fold in range(n_folds)]

        for i, p in enumerate(kfold_preds): 

            export_preds(p, i)

        avg_preds = torch.stack(kfold_preds).mean(0)

        preds = beta*preds + (1-beta)*avg_preds

    

    # Export final submission file

    return export_preds(preds)
# Uncomment for inference

# run_inference()

run_inference(n_folds=2)
!ls