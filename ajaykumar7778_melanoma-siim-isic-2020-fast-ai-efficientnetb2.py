!pip install iterative-stratification

!pip install efficientnet-pytorch
import random

import numpy as np

import pandas as pd

import torch

import PIL.Image as pil

import matplotlib.pyplot as plt



from fastai.vision import *

from efficientnet_pytorch import EfficientNet

from sklearn.model_selection import StratifiedKFold



import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
# !pip uninstall torch torchvision -y

!pip install torch==1.4.0 torchvision==0.5.0
%reload_ext autoreload

%autoreload 2

%matplotlib inline
seed = 42



def random_seed(seed_value):

     

    random.seed(seed_value) 

    np.random.seed(seed_value) 

    torch.manual_seed(seed_value) 

    os.environ['PYTHONHASHSEED'] = str(seed_value)



    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) 

        torch.backends.cudnn.deterministic = True 

        torch.backends.cudnn.benchmark = False



random_seed(seed)
path = '/kaggle/input/siim-isic-melanoma-classification'

path
img_path = '/kaggle/input/melanoma-merged-external-data-512x512-jpeg'

img_path
train_df = pd.read_csv(img_path + '/folds_13062020.csv')

train_df.head()
train_df.shape
test_df = pd.read_csv(path + '/test.csv')

test_df.head()
test_df.shape
sample_df = pd.read_csv(path + '/sample_submission.csv')

sample_df.head()
sample_df.shape
tfms = get_transforms( flip_vert=True, max_rotate=15, max_zoom=1.2, max_lighting=0.3, max_warp=0, p_affine=0, p_lighting=0.8)
class FocalLoss(nn.Module):

    def __init__(self, gamma=2., reduction='mean'):

        super().__init__()

        self.gamma = gamma

        self.reduction = reduction



    def forward(self, inputs, targets):

        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-CE_loss)

        F_loss = ((1 - pt)**self.gamma) * CE_loss

        if self.reduction == 'sum':

            return F_loss.sum()

        elif self.reduction == 'mean':

            return F_loss.mean()
# Settings dashboard 



#########################

# GENERAL



submission_ver = '0002'



#########################

# ARCHITECTURE



# [models.densenet121, models.resnet50, models.resnet152, EfficientNet.from_pretrained('efficientnet-b7', num_classes=4)]

arch = [EfficientNet.from_pretrained('efficientnet-b2', num_classes=2)]



# fc layer size for effnet: b7 -> 2560, b6 -> 2304, b5 -> 2048, b4 -> 1792,  

# ------------------------- b3 -> 1536, b2 -> 1408, b1 -> 1280, b0 -> 1280





#fc_size = 1280



fc_size = 1408 ## B2



# linear layer size for effnet: 1000 / 500

lin_size = 1000



#########################

# FOLDS

n_folds = 5



#########################

# DATA



# [224, 352, 499]

size = [256]



#########################

# TRAIN



bs = 32

stage_1_epochs = 3



lr1       = [1e-1]

lr_eff_1  = [1e-3]



is_stage_2 = False

stage_2_epochs = 4



lr2       = [slice(1e-7, 1e-4)]

lr_eff_2  = [slice(1e-4, 1e-3)]



custom_loss = True

loss_func = FocalLoss()



w_decay = 0.01



#########################

# DEVICE 



use_fp16 = True

num_wkrs = os.cpu_count()



#########################

# OTHER



use_tta = True



submit_after_train = False



oversampling_flag = True

oversampling_size = 5 # 'auto' or number like: 2-10



preds_to_int = False

smooth_preds = False

smooth_alpha = 0.01
num_classes = len(np.unique(train_df['target']))

num_classes
test_df['image_name'] = '512x512-test/512x512-test/' + test_df['image_name'] + '.jpg'
test_data = ImageList.from_df(test_df, img_path)

test_data
labels_df = train_df[['image_id', 'target']].copy()

labels_df.head()
train_df = train_df[['image_id', 'target', 'fold']].copy()

train_df.head()
def k_fold(df, num_fld, seed = seed):

    #df['fold'] = -1

    #strat_kfold = StratifiedKFold(n_splits=num_fld, random_state=seed, shuffle=True)

    #for i, (_, test_index) in enumerate(strat_kfold.split(df.image_name.values, df.target.values)):

    #    df.iloc[test_index, -1] = i

        

    #df['fold'] = df['fold'].astype('int')



    for fold in range(num_fld):

        df.loc[df.fold == fold, f'is_valid_{fold}'] = True

        df.loc[df.fold != fold, f'is_valid_{fold}'] = False
k_fold(train_df, n_folds, seed)
train_df.head(5)
def oversample(fld, df, os_size, num_fld=5):

    # Let's get Fold train data

    train_df_fld = df.loc[df['fold'] != fld]

    valid_df_fld = df.loc[df['fold'] == fld]

    

    # Now let's save as separate df only "multiple_diseases" images for exact fold train data

    train_df_md = train_df_fld.loc[train_df_fld['target'] == 1]

    

    # Oversample to "malignant" class size

    if os_size == 'auto':

        os_size = int(np.floor(train_df_fld.loc[train_df_fld['target'] == 0]['target'].value_counts()[0]/train_df_fld.loc[train_df_fld['target'] == 1]['target'].value_counts()[1]))

    

    train_df_md = train_df_md.append([train_df_md] * (os_size - 1))

    

    # Finally add "multiple_diseases" images to whole data, so this class gets x2 images 

    full_df_fld = pd.concat([train_df_fld, train_df_md, valid_df_fld]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    

    return full_df_fld



for x in range(n_folds):

    fold_df = train_df.copy()

    

    if oversampling_flag:

        fold_df = oversample(x, train_df, oversampling_size)

        

    globals()['src_%s' %x] = (ImageList.from_df(fold_df, img_path + '/512x512-dataset-melanoma', folder='512x512-dataset-melanoma', suffix='.jpg').split_from_df(col = (3 + x)))

    
def get_data(fold, size, bs, padding_mode='reflection'):

    return (globals()['src_%s' %fold].label_from_df(cols='target')

                                        .add_test(test_data)

                                        .transform(tfms, size=size, padding_mode=padding_mode)

                                        .databunch(bs=bs, num_workers = num_wkrs).normalize(imagenet_stats))
def preds_smoothing(encodings , alpha):

    K = encodings.shape[1]

    y_ls = (1 - alpha) * encodings + alpha / K

    return y_ls
def print_metrics(val_preds, val_labels):

    targs, preds = LongTensor([]), Tensor([])  

    

    # val_preds = val_preds[:,1]

    val_preds = F.softmax(val_preds, dim=1)[:,-1]



    preds = torch.cat((preds, val_preds.cpu()))

    targs = torch.cat((targs, val_labels.cpu().long()))



    print('AUCROC = ' + str(auc_roc_score(preds, targs).item()))
gc.collect()
for model in arch:

    

    ############ DEFINING VARS & SETTINGS ############

    if hasattr(model, '__name__'):

        model_name = model.__name__

    else:

        model_name = "EfficientNet"

    

    globals()[model_name + '___val_preds']  = []

    globals()[model_name + '___val_labels'] = []

    globals()[model_name + '___test_preds'] = []

    

    print(f'/////////////////////////////////////////////////////')

    print(f'//////////////// MODEL: {model_name} ////////////////')

    print(f'/////////////////////////////////////////////////////\n')



    for fld in range(n_folds):

        

        print(f'\n//////////////// FOLD {fld} ////////////////\n')

        

        for sz in size:

            

            print(f'-------- Size: {sz} --------\n')

            

            ############ STAGE_1 ############

            

            print("Preparing data & applying settings...\n")

            

            data = get_data(fld, sz, bs)

            

            gc.collect()

            

            if sz == size[0]:

                if model_name != "EfficientNet":

                    learn = cnn_learner(data, model, metrics=[AUROC()], bn_final=True)

                else:

                    model._fc = nn.Sequential(nn.Linear(fc_size, lin_size, bias=True),

                    nn.ReLU(),

                    nn.Dropout(p=0.5),

                    nn.Linear(lin_size, num_classes, bias = True))

                    #

                    learn = Learner(data, model, metrics=[AUROC()])

                    learn = learn.split([learn.model._conv_stem,learn.model._blocks,learn.model._conv_head])

            else:

                learn.data = data



            if custom_loss:

                learn.loss_func = loss_func

            

            if use_fp16:

                learn = learn.to_fp16() 

            

            if model_name != "EfficientNet":

                lr = lr1[size.index(sz)]

            else:

                lr = lr_eff_1[size.index(sz)]

                

            print("Data is ready. Learning - Stage 1...")

            

            #learn.freeze()

            learn.fit_one_cycle(stage_1_epochs, slice(lr), wd=w_decay)

                

            ############ STAGE_2 ############

            

            if is_stage_2:

                

                print("Stage 1 complete. Stage 2...")

                

                if model_name != "EfficientNet":

                    lr = lr2[size.index(sz)]

                else:

                    lr = lr_eff_2[size.index(sz)]

                

                learn.unfreeze()

                learn.fit_one_cycle(stage_2_epochs, lr, wd=w_decay)



            ############ RESULTS ############

            print(f"Final learning is over for size {sz}\n")

            

            val_preds, val_labels = learn.get_preds()

            print_metrics(val_preds, val_labels)



            # learn.save('arch-' + str(model_name) + '_fold-' + str(fld) + '_size-' + str(sz))



            #---------- END OF SIZE ----------

        

        ############ SAVE ############

        

        globals()[model_name + '___val_preds'].append(val_preds)

        globals()[model_name + '___val_labels'].append(val_labels)

        

        if use_tta == False:

            print(f'\nSaving test results for fold {fld}...')

            test_preds, _ = learn.get_preds(DatasetType.Test)

            globals()[model_name + '___test_preds'].append(test_preds[:, 1])

        else:

            print(f'\nSaving test TTA results for fold {fld}...')

            test_preds, _ = learn.TTA(ds_type=DatasetType.Test)

            globals()[model_name + '___test_preds'].append(test_preds[:, 1])

        

        print("Done!")

        

        gc.collect()

        

        #---------- END OF FOLD ----------

    

    print("All folds are trained successfully\n")

    

    print_metrics(torch.cat(globals()[model_name + '___val_preds']), torch.cat(globals()[model_name + '___val_labels']))

    

    print("\nWriting submission file...")

    

    test_df_output = pd.concat([test_df, pd.DataFrame(np.mean(np.stack(globals()[model_name + '___test_preds']), axis=0), columns=['target'])], axis=1)

    sample_df.iloc[:,1:] = test_df_output.iloc[:,5]

    sample_df.to_csv(f'submission_v{submission_ver}.csv', index=False)

    

    print(f'File is ready to submit\n')

    

    if submit_after_train:

        print("Submitting to Kaggle\n")

        !kaggle competitions submit -c siim-isic-melanoma-classification -f 'submission_v{submission_ver}.csv' -m 'Md: {model_name}, Fd: {n_folds}, Bs: {bs}, Sz: {size[0]}, Os: {oversampling_flag}, TTa: {use_tta}'

        print(' \n\n\n\n')

    

    #---------- END OF MODEL ----------