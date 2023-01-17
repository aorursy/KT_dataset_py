import os
from tqdm import tqdm
from skimage.morphology import label, binary_opening, disk
from fastai import vision, basic_data, layers, metrics
from fastai.callbacks import hooks
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data_root = '../input/airbus-ship-detection/'
path_train = os.path.join(data_root,'train_v2')
path_test = os.path.join(data_root,'test_v2')

img_shape = (768, 768)

# Booleans
USE_SELF_ATTENTION = True
USE_UNET34_AIRBUS = False
USE_FULL_RES_PRED = False
# Get dataframe with label
masks_df = pd.read_csv(os.path.join(data_root, 'train_ship_segmentations_v2.csv'))
masks_df = masks_df[~masks_df['ImageId'].isin(['6384c3e78.jpg'])]  # remove corrupted image
masks_df = masks_df.dropna() # remove images withtout ships
unique_img_ids_df = masks_df.groupby('ImageId').size().reset_index(name='counts')
unique_img_ids_df = unique_img_ids_df.drop(columns=['counts'])
# https://www.kaggle.com/iafoss/unet34-dice-0-87
# https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c
def cuda(x): 
    return x.cuda() if torch.cuda.is_available() else x

def make_one_hot(labels, c=2):
    one_hot = cuda(torch.FloatTensor(labels.size(0), c, labels.size(2), labels.size(3)).zero_())
    target = one_hot.scatter_(1, labels.data, 1)
    target = cuda(Variable(target))
    return target

def dice_loss(input, target):
    """Soft dice loss function.
    https://github.com/pytorch/pytorch/issues/1249"""
    # Input is of shape N,C,H,W
    smooth = 1
    batch_size = input.size(0)
    input = F.softmax(input, dim=1)
    # Since we have only 2 classes transform it to N,H,W and treat as sigmoid
    input = input.view(batch_size, 2, -1)[:, 1, :]
    target = make_one_hot(target).view(batch_size, 2, -1)[:, 1, :]

    inter = torch.sum(input * target) + smooth
    union = torch.sum(input) + torch.sum(target) + smooth

    return -torch.log(2.0 * inter / union)

class FocalLoss(nn.Module):
    """Focal loss function."""
    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # One-hot encode target
        target = target.squeeze(1)
    
        input = input.view(input.size(0),input.size(1),-1) # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)                       # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean()
    
class MixedLoss(nn.Module):
    """Combine two losses and bring them to similar scale."""
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        
    def forward(self, input, target):
        return self.alpha * self.focal(input, target) + dice_loss(input, target)
# https://nbviewer.jupyter.org/github/polakowo/mlprojects/blob/master/airbus-ship-segmentation/ShipDetection.ipynb
def dice(input, target):
    """Dice metric."""
    input = input.argmax(dim=1).float()
    target = target.squeeze(1).float()
    
    smooth = 1.
    numerator = 2. * (input * target).sum()
    denumerator = (input + target).sum()
    return (numerator + smooth) / (denumerator + smooth)

def IoU(input, target):
    """Intersection over Union (IoU) metric."""
    input = input.argmax(dim=1).float()
    target = target.squeeze(1).float()
    
    smooth = 1.
    intersection = (input * target).sum()
    union = (input + target).sum() - intersection
    return (intersection + smooth) / (union + smooth)
def open_mask(fn):
    masks = masks_df[masks_df['ImageId'] == str(os.path.split(fn)[1])]['EncodedPixels'].tolist()
    masks = " ".join(str(x) for x in masks) # convert list to string
    mask_img = vision.image.open_mask_rle(masks, shape=(768, 768))
    return vision.ImageSegment(mask_img.data.T.permute(2,0,1).float())
     
class SegmentationLabelList(vision.ImageList):
    """Our labels are created from encodings in masks_df, no disk I/O operations required."""
    _processor=vision.data.SegmentationProcessor
    def __init__(self, items:basic_data.Iterator, classes:basic_data.Collection=None, **kwargs):
        super().__init__(items, **kwargs)
        self.copy_new.append('classes')
        self.classes, self.loss_func = classes, layers.CrossEntropyFlat(axis=1)

    def open(self, fn): return open_mask(fn)
    def analyze_pred(self, pred, thresh:float=0.5): return pred.argmax(dim=0)[None]
    def reconstruct(self, t:basic_data.Tensor): return vision.ImageSegment(t)

class SegmentationItemList(vision.ImageList):
    "`ItemList` suitable for segmentation tasks."
    _label_cls, _square_show_res = SegmentationLabelList, False
    
def get_data(df, bs=16, img_size=(256, 256)):
    # Do not augment since we have a large dataset anyway, only resize
    tfms = ([vision.transform.crop_pad()], [vision.transform.crop_pad()])
    
    # Build DataBunch
    return (SegmentationItemList.from_df(unique_img_ids_df, path=path_train)
            .split_by_rand_pct(0.2)
            .label_from_func(lambda x: x, classes=[0, 1])
            .transform(tfms, size=img_size, tfm_y=True)
            .add_test(vision.Path(path_test).ls(), tfm_y=False)
            .databunch(path=data_root, bs=bs)
            .normalize(vision.imagenet_stats))
# Build databunch
data = get_data(masks_df)

if USE_UNET34_AIRBUS == True:
    model = torch.load('../input/unet-resnet34-airbus/UNET34Airbus_2cl.pth')
    learner = vision.Learner(data, model, loss_func=MixedLoss(10., 2.), metrics=[dice, IoU])
else:
    model = vision.models.resnet34
    learner = vision.unet_learner(data, model, loss_func=MixedLoss(10., 2.), metrics=[dice, IoU], self_attention=USE_SELF_ATTENTION)
    
learner.model_dir = '/kaggle/working'
data.show_batch()
# Find optimal LR
learner.lr_find()
learner.recorder.plot(suggestion=True)
# Train on 3 epoch
learner.fit_one_cycle(3, max_lr=5e-4)
learner.recorder.plot_losses()
learner.recorder.plot_lr(show_moms=True)
learner.save('unet_3ep')
# Train on 5 more epoch with model unfreezed
learner.unfreeze()
learner.lr_find()
learner.recorder.plot(suggestion=True)
learner.fit_one_cycle(2, max_lr=slice(1e-6, 1e-5))
learner.recorder.plot_losses()
learner.recorder.plot_metrics()
learner.recorder.plot_lr(show_moms=True)
learner.save('unet_5ep')
learner.show_results()
def get_test_masks():
    
    if USE_FULL_RES_PRED == True:
        learner.data = get_data(masks_df, img_size=(768, 768))

    pred_masks = []
    for x, y in tqdm(learner.data.test_ds):
        _, _, output = learner.predict(x) # network output 2x256x256 or 2x768x768
        
        if USE_FULL_RES_PRED == False:
            upsampler = torch.nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False) # 768/256 = factor of 3
            output = upsampler(output.unsqueeze(0)).squeeze(0)  # 2x256x256
        
        probs = F.softmax(output, dim=0)  # 2x256x256 or 2x768x768
        mask_tensor = probs.argmax(dim=0)   # 256x256 or 768x768 (hot tensor)
        
        labels = label(mask_tensor)
        pred_masks.append([vision.image.rle_encode((labels.T)==k) for k in np.unique(labels[labels>0])])
    return pred_masks

def get_test_masks_opening():
    pred_masks = []
    for x, y in tqdm(learner.data.test_ds):
        _, _, output = learner.predict(x) # network output 2x256x256 or 2x768x768

        if USE_FULL_RES_PRED == False:
            upsampler = torch.nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False) # 768/256 = factor of 3
            output = upsampler(output.unsqueeze(0)).squeeze(0)  # 2x256x256

        probs = F.softmax(output, dim=0)  # 2x256x256 or 2x768x768
        mask_tensor = probs.argmax(dim=0)   # 256x256 or 768x768 (hot tensor)

        mask_tensor = binary_opening(mask_tensor, disk(2))
        labels = label(mask_tensor)
        pred_masks.append([vision.image.rle_encode((labels.T)==k) for k in np.unique(labels[labels>0])])
    return pred_masks
    
def create_submission_df(test_masks):
    """Create submission dataframe."""
    test_ids = list(map(lambda x: x.name, learner.data.test_dl.dataset.items))
    img_masks = list(zip(test_ids, test_masks))
    flat_img_masks = [] 
    for img, masks in img_masks:
        if len(masks) > 0:
            for mask in masks:
                flat_img_masks.append([img, mask])
        else:
            flat_img_masks.append([img, None])
    df = pd.DataFrame(flat_img_masks, columns=['ImageId', 'EncodedPixels'])
    return df
test_masks = get_test_masks_opening()
df_submission = create_submission_df(test_masks)
df_submission.to_csv('submission_wo_clf.csv', header=True, index=False)

from IPython.display import FileLink
FileLink('submission_wo_clf.csv')
# Get dataframe with label
clf_df = pd.read_csv(os.path.join('../input', 'clfairbus/clf_256_test_preds.csv'))

for i, row in clf_df.iterrows():
    if row['Label'] == 0:
        df_submission.loc[df_submission['ImageId'] == row['ImageId'], 'EncodedPixels'] = None
df_submission.to_csv('submission_w_clf.csv', header=True, index=False)
FileLink('submission_w_clf.csv')