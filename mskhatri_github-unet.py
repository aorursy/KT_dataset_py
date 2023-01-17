from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious)) # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) +1, Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
%matplotlib inline
%reload_ext autoreload
%autoreload 2
from fastai.conv_learner import *
from fastai.dataset import *
from fastai.models.resnet import vgg_resnet50
from fastai.models.cifar10.senet import *
from skimage.transform import resize
import json
from sklearn.model_selection import train_test_split, StratifiedKFold , KFold
from sklearn.metrics import jaccard_similarity_score
# from pycocotools import mask as cocomask
# from utils import my_eval,intersection_over_union_thresholds,RLenc
# from seg_scripts.utils import *
# from seg_scripts.lovasz_losses import lovasz_hinge
print(torch.__version__)
torch.backends.cudnn.benchmark=True
torch.cuda.is_available()
PATH = Path('/kaggle/input/')
TRN_MASKS = 'train/images/'
TRN_IMG = 'train/images/'
TST_IMG = 'test/images/'
TMP = Path('/tmp/')
MODEL = Path('/tmp/model/')

trn = pd.read_csv(PATH/'train.csv')
dpth = pd.read_csv(PATH/'depths.csv')
def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax
class DepthDataset(Dataset):
    def __init__(self,ds,dpth_dict):
        self.dpth = dpth_dict
        self.ds = ds
        
    def __getitem__(self,i):
        val = self.ds[i]
        return val[0],self.dpth[self.ds.fnames[i].split('/')[2][:-4]],val[1]
    
class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
        
    def get_x(self, i): 
        return open_image(os.path.join(self.path, self.fnames[i]))
    
    def get_y(self, i):
        return open_image(os.path.join(str(self.path), str(self.y[i])))

    def get_c(self): return 0
    
class TestFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform,flip, path):
        self.y=y
        self.flip = flip
        super().__init__(fnames, transform, path)
        
    def get_x(self, i): 
        im = open_image(os.path.join(self.path, self.fnames[i]))
        return np.fliplr(im) if self.flip else im
        
    def get_y(self, i):
        im = open_image(os.path.join(str(self.path), str(self.y[i])))
        return np.fliplr(im) if self.flip else im
    def get_c(self): return 0
x_names = np.array([f'{TRN_IMG}{o.name}' for o in (PATH/TRN_MASKS).iterdir()])
y_names = np.array([f'{TRN_MASKS}{o.name}' for o in (PATH/TRN_MASKS).iterdir()])
tst_x = np.array([f'{TST_IMG}{o.name}' for o in (PATH/TST_IMG).iterdir()])
f_name = [o.split('/')[-1] for o in x_names]

c = dpth.set_index('id')
dpth_dict = c['z'].to_dict()

kf = 5
kfold = KFold(n_splits=kf, shuffle=True, random_state=42)

train_folds = []
val_folds = []
for idxs in kfold.split(f_name):
    train_folds.append([f_name[idx] for idx in idxs[0]])
    val_folds.append([f_name[idx] for idx in idxs[1]])
# train_folds = pickle.load(open('train_folds.pkl',mode='rb'))
# val_folds = pickle.load(open('val_folds.pkl',mode='rb'))
# tst_x = pickle.load(open('tst_x.pkl',mode='rb'))
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()
    
class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))
    
class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,128)
        self.up2 = UnetBlock(128,128,128)
        self.up3 = UnetBlock(128,64,128)
        self.up4 = UnetBlock(128,64,128)
        self.up5 = nn.ConvTranspose2d(128, 1, 2, stride=2)
        
    def forward(self,img,depth):
        x = F.relu(self.rn(img))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x[:,0]
    
    def close(self):
        for sf in self.sfs: sf.remove()


class UnetModel():
    def __init__(self,model,lr_cut,name='unet'):
        self.model,self.name = model,name
        self.lr_cut = lr_cut

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [self.lr_cut]))
        return lgs + [children(self.model)[1:]]
def get_tgs_model():
    f = resnet34
    cut,lr_cut = model_meta[f]
    m_base = get_base(f,cut)
    m = to_gpu(Unet34(m_base))
    models = UnetModel(m,lr_cut)
    learn = ConvLearner(md, models, tmp_name=TMP, models_name=MODEL)
    return learn

def get_base(f,cut):
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)
model = 'simple_resnet34'
bst_acc=[]
use_clr_min=20
use_clr_div=10
aug_tfms = [RandomFlip(tfm_y=TfmType.CLASS)]
model = 'simple_resnet34'
bst_acc=[]
use_clr_min=20
use_clr_div=10
aug_tfms = [RandomFlip(tfm_y=TfmType.CLASS)]

szs = [(256,64)]
for sz,bs in szs:
    print([sz,bs])
    for i in range(kf) :
        print(f'fold_id{i}')
        
        trn_x = np.array([os.path.join(TRN_IMG,o) for o in train_folds[i]])
        trn_y = np.array([os.path.join(TRN_MASKS,o) for o in train_folds[i]])
        val_x = [os.path.join(TRN_IMG,o) for o in val_folds[i]]
        val_y = [os.path.join(TRN_MASKS,o) for o in val_folds[i]]
        
        tfms = tfms_from_model(resnet34, sz=sz, pad=0,crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
        datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms,test=tst_x,path=PATH)
        md = ImageData(PATH, datasets, bs, num_workers=4, classes=None)
        denorm = md.trn_ds.denorm
        md.trn_dl.dataset = DepthDataset(md.trn_ds,dpth_dict)
        md.val_dl.dataset = DepthDataset(md.val_ds,dpth_dict)
        md.test_dl.dataset = DepthDataset(md.test_ds,dpth_dict)
        learn = get_tgs_model() 
        learn.opt_fn = optim.Adam
        learn.metrics=[accuracy_thresh(0.5)]
#         pa = f'{kf}_fold_{model}_{i}'
#         print(pa)

        lr=1e-2
        wd=1e-7
        lrs = np.array([lr/100,lr/10,lr])

        learn.unfreeze()        
        learn.crit = lovasz_hinge
#         learn.load(pa)
#         learn.fit(lrs/2,3, wds=wd, cycle_len=10,use_clr=(20,8))
        learn.fit(lr,1)

               
#         learn.load(pa)        
        #Calcuating mean iou score
        v_targ = md.val_ds.ds[:][1]
        v_preds = np.zeros((len(v_targ),sz,sz))     
        v_pred = learn.predict()
        v_pred = to_np(torch.sigmoid(torch.from_numpy(v_pred)))
        p = ((v_pred)>0.5).astype(np.uint8)
#         bst_acc.append(intersection_over_union_thresholds(v_targ,p))
#         print(bst_acc[-1])
preds = np.zeros(shape = (18000,sz,sz))
for o in [True,False]:
    md.test_dl.dataset = TestFilesDataset(tst_x,tst_x,tfms[1],flip=o,path=PATH)
    md.test_dl.dataset = DepthDataset(md.test_dl.dataset,dpth_dict)
    
    for i in tqdm_notebook(range(kf)):
        pa = f'{kf}_fold_{model}_{i}'
        print(pa)
        learn.load(pa)
        pred = learn.predict(is_test=True)
        pred = to_np(torch.sigmoid(torch.from_numpy(pred)))    
        for im_idx,im in enumerate(pred):
                preds[im_idx] += np.fliplr(im) if o else im
        del pred


p = [cv2.resize(o/10,dsize=(101,101)) for o in preds]
p = [(o>0.5).astype(np.uint8) for o in p]
pred_dict = {id_[11:-4]:RLenc(p[i]) for i,id_ in tqdm_notebook(enumerate(tst_x))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('simple_k_fold_flipped.csv')
plt.imshow(((preds[16]/10)>0.5).astype(np.uint8))













