import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys
sys.path.insert(0,"/kaggle/input/our-env/lib")
!pip install fastai==1.0.34
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.optimize as opt
# common
from lib.data.utils import read_dataset_info, save_pred, Oversampling
from lib.constants import *
from lib.data.data_visualization import visualize_samples
# torch
from torchvision.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler
import fastai
import fastai.vision
from fastai.vision.learner import create_cnn, cnn_config, create_body, create_head
from fastai.torch_core import *
from fastai import DatasetType
from fastai.metrics import accuracy_thresh

from lib.torch.dataset import ProteinDataset, get_sample_weights
from lib.torch.losses import FocalLoss
from lib.torch.metrics import f1_macro, fbeta, fit_thresholds
from lib.torch.augmentation import tta, train_transforms
from lib.torch.fit_custom_cycle import *
labeled_dataset_info = read_dataset_info("../input/human-protein-atlas-image-classification/train", "../input/human-protein-atlas-image-classification/train.csv", target_col='Target')
test_dataset_info = read_dataset_info("../input/human-protein-atlas-image-classification/test", "../input/human-protein-atlas-image-classification/sample_submission.csv")
#visualize_samples(labeled_dataset_info)
label_count = np.zeros(len(CLASS_LABEL_DICT))
for info in labeled_dataset_info:
    label_count[info['labels']] += 1
labeled_fractions = torch.cuda.FloatTensor((label_count / len(labeled_dataset_info)).astype(np.float32))
    
image_size = (256, 256)
batch_size = 256
train_stats = ([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])
test_stats = ([0.05913, 0.0454, 0.04066, 0.05928], [0.11734, 0.09503, 0.129, 0.11528])


y = [x['labels'] for x in labeled_dataset_info]
stratifiing_y = [labels[np.argmin(label_count[labels])] for labels in y]
train_indices, validation_indices, _, _  = train_test_split(range(len(y)), y, test_size=0.08, stratify=stratifiing_y, 
                                                            random_state=13)
#train_dataset_info = Oversampling().apply(labeled_dataset_info[train_indices])
train_dataset_info = labeled_dataset_info #[train_indices]

print('Train dataset len = {} / {} (oversampling)'.format(len(train_indices), len(train_dataset_info)))
print('Validation dataset len = {}'.format(len(validation_indices)))
print('Test dataset len = {}'.format(len(test_dataset_info)))

train_dataset = ProteinDataset(train_dataset_info, image_size, train_stats, train_transforms)
validation_dataset = ProteinDataset(labeled_dataset_info[validation_indices], image_size, train_stats)
test_dataset = ProteinDataset(test_dataset_info, image_size, train_stats)

weigths = get_sample_weights(train_dataset_info, label_count)
train_loader = DataLoader(train_dataset, batch_size, sampler=WeightedRandomSampler(weigths, len(train_dataset_info)))
validation_loader = DataLoader(validation_dataset, batch_size)
test_loader = DataLoader(test_dataset, batch_size)

data_bunch = fastai.DataBunch(train_loader, validation_loader, test_loader)
classes_number = 28

def create_model():
    pretrain_model = fastai.vision.models.resnet34
    meta = cnn_config(pretrain_model)
    body = create_body(pretrain_model(True), meta['cut'])

    w = body[0].weight
    body[0] = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
    body[0].weight = torch.nn.Parameter(torch.cat((w, w[:,:1,:,:]),dim=1))

    resnet_body_output = 512
    head = create_head(resnet_body_output*2, nc=classes_number)
    head[3].p = 0.5 # make dropout with p=0.5
    model = nn.Sequential(body, head)
    return model

model = create_model()
layer_groups = split_model_idx(model, [0,41,92,100])
learn = fastai.Learner(data_bunch, model, loss_func=FocalLoss(), metrics=[accuracy_thresh, f1_macro], layer_groups=layer_groups)
learn.freeze()
learn.clip_grad(1.)
lr = 5e-3
weight_history = []
learn.load('../../input/resnet34-optimized-lr-freezedpth/ResNet34_1st_phase')
learn.unfreeze()
lrs=np.array([lr/2,lr/2, lr])
for i in range(2):
    pred_y, true_y = learn.get_preds(DatasetType.Train)
    sample_loss = (pred_y.sigmoid() - true_y).abs().sum(dim=1).numpy()
    weights = get_sample_weights(train_dataset_info, label_count, sample_loss=sample_loss)
    learn.data.train_dl.sampler.weights = torch.tensor(weights)
    
    fit_custom_cycle(learn, 4, lrs/16, div_factor=15, pct_start=0.05)
learn.save('ResNet34_2nd_phase')
pred_y, true_y = learn.get_preds()
class_f1 = fbeta(pred_y, true_y, beta=1, thresh=0.5).numpy()
print(class_f1)
print(np.mean(class_f1))