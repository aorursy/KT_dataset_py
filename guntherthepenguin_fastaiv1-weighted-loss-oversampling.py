import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import os
from sklearn.metrics import f1_score

from fastai import *
from fastai.vision import *

import torch
import torch.nn as nn
import torchvision
import cv2

from tqdm import tqdm
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")

%load_ext autoreload
%autoreload
def open_4_channel(self,fname):
    fname = str(fname)
    # strip extension before adding color
    if fname.endswith('.png'):
        fname = fname[:-4]
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(fname+'_'+color+'.png', flags).astype(np.float32)/255
           for color in colors]

    x = np.stack(img, axis=-1)
    return Image(pil2tensor(x, np.float32).float())
def show_batch_cstm(self, rows:int=5, ds_type:DatasetType=DatasetType.Train, **kwargs)->None:
        "Show a batch of data in `ds_type` on a few `rows`."
        x,y = self.one_batch(ds_type, True, True)
        x=x[:,:3,:,:]
        
        if self.train_ds.x._square_show: rows = rows ** 2
        xs = [self.train_ds.x.reconstruct(grab_idx(x, i, self._batch_first)) for i in range(rows)]
        #TODO: get rid of has_arg if possible
        if has_arg(self.train_ds.y.reconstruct, 'x'):
            ys = [self.train_ds.y.reconstruct(grab_idx(y, i), x=x) for i,x in enumerate(xs)]
        else : ys = [self.train_ds.y.reconstruct(grab_idx(y, i)) for i in range(rows)]
        print(ys)
        self.train_ds.x.show_xys(xs, ys, **kwargs)

path='.'
input_fldr=Path('../input/')
trn_fldr=f'{input_fldr}/train/'
test_fldr=f'{input_fldr}/test/'
trn_lbl=f'{input_fldr}/train.csv'
sample_csv = f'{input_fldr}/sample_submission.csv'
df_trn=pd.read_csv(f'{input_fldr}/train.csv')
df_trn.head()

sz=512
bs=16

targets=[np.array(target.split(' ')).astype(float) for target in df_trn.Target.values]
mapping=MultiLabelBinarizer()
mapping.fit(targets)
targets_mapped=mapping.transform(targets)
X_train, y_train, X_test, y_test = iterative_train_test_split(df_trn.Id[:,None], targets_mapped, test_size = 0.5)
plt.plot(np.sum(y_train,axis=0)-np.sum(y_test,axis=0),label='iterative')
X_train, X_test,y_train , y_test = train_test_split(df_trn.Id[:,None], targets_mapped, test_size = 0.5)
plt.plot(np.sum(y_train,axis=0)-np.sum(y_test,axis=0),label='split')
plt.legend()
p=1.
n=int(p*df_trn.shape[0])
while True:
    if p<1.0:
        idx=np.random.choice(len(df_trn.index),n)
    else:
        idx=range(len(df_trn.index))
    X_train, y_train, X_val, y_val = iterative_train_test_split(df_trn.index[idx,None], targets_mapped[idx,:], test_size = 0.2)
    if (np.sum(targets_mapped[idx,:],axis=0)>0).all() and ((X_train.shape[0]%bs!=1) and (X_val.shape[0]%bs!=1)):
        break
X_train.shape,(X_train.shape[0]%bs),X_val.shape,(X_val.shape[0]%bs)
import numpy as np

name_label_dict = {
    0:   ('Nucleoplasm', 12885),
    1:   ('Nuclear membrane', 1254),
    2:   ('Nucleoli', 3621),
    3:   ('Nucleoli fibrillar center', 1561),
    4:   ('Nuclear speckles', 1858),
    5:   ('Nuclear bodies', 2513),
    6:   ('Endoplasmic reticulum', 1008),   
    7:   ('Golgi apparatus', 2822),
    8:   ('Peroxisomes', 53), 
    9:   ('Endosomes', 45),
    10:  ('Lysosomes', 28),
    11:  ('Intermediate filaments', 1093), 
    12:  ('Actin filaments', 688),
    13:  ('Focal adhesion sites', 537),  
    14:  ('Microtubules', 1066), 
    15:  ('Microtubule ends', 21),
    16:  ('Cytokinetic bridge', 530),
    17:  ('Mitotic spindle', 210),
    18:  ('Microtubule organizing center', 902),
    19:  ('Centrosome', 1482),
    20:  ('Lipid droplets', 172),
    21:  ('Plasma membrane', 3777),
    22:  ('Cell junctions', 802),
    23:  ('Mitochondria', 2965),
    24:  ('Aggresome', 322),
    25:  ('Cytosol', 8228),
    26:  ('Cytoplasmic bodies', 328),   
    27:  ('Rods &amp; rings', 11)
    }

n_labels = 50782

def cls_wts(label_dict, mu=0.5):
    prob_dict, prob_dict_bal = {}, {}
    max_ent_wt = 1/28
    for i in range(28):
        prob_dict[i] = label_dict[i][1]/n_labels
        if prob_dict[i] > max_ent_wt:
            prob_dict_bal[i] = prob_dict[i]-mu*(prob_dict[i] - max_ent_wt)
        else:
            prob_dict_bal[i] = prob_dict[i]+mu*(max_ent_wt - prob_dict[i])            
    return prob_dict, prob_dict_bal
balance_p=.5
prob_d,prob_bal=cls_wts(name_label_dict,balance_p)
alpha=torch.cuda.FloatTensor(1/np.array([prob_bal[i] for i in prob_bal]))
alpha_log=torch.cuda.FloatTensor([-np.log(prob_bal[i])/8 for i in prob_bal])
plt.plot(to_np(alpha)/min(to_np(alpha)),label='Linear scaled weights')
plt.plot(to_np(alpha_log)/min(to_np(alpha_log)),label='Log scaled weights')
plt.legend()

alpha_norm=to_np(alpha)/np.min(to_np(alpha))
plt.plot(alpha_norm,label='Linear scaled oversampling')
alpha_norm[alpha_norm<4]=1
alpha_norm[(alpha_norm>4) & (alpha_norm<6) ]=1.5
alpha_norm[(alpha_norm>6) & (alpha_norm<7) ]=3
alpha_norm[alpha_norm>7]=alpha_norm[alpha_norm>7]
plt.plot(alpha_norm,label='Linear damped oversampling')
#plt.plot(targets_mapped[:n,:].sum(axis=0)/1000)
plt.legend()
over_sample={i:int(item*alpha_norm[i]) for i,item in enumerate(np.unique(np.argmin(-y_train*(to_np(alpha_log)/np.min(to_np(alpha_log))),axis=1),return_counts=True)[1])}

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42,sampling_strategy=over_sample)
X_res, y_res = ros.fit_resample(X_train, np.argmin(-y_train*(to_np(alpha_log)/np.min(to_np(alpha_log))),axis=1))

ImageItemList.open=open_4_channel
ImageDataBunch.show_batch=show_batch_cstm
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=30., max_zoom=1,
                      max_lighting=0.05, max_warp=0.)
test_ids = list(sorted({fname.split('_')[0] for fname in os.listdir(input_fldr/'test')}))
test_fnames = [input_fldr/'test'/test_id for test_id in test_ids]
src = (ImageItemList.from_csv(input_fldr, 'train.csv', folder='train', suffix='.png',num_workers=0)
        .split_by_idxs(X_res[:,0],X_val[:,0])
        .label_from_df(sep=' ',  classes=[str(i) for i in range(28)]))
src.add_test(test_fnames, label='0');
data = (src.transform(tfms, size=sz)
        .databunch(num_workers=0))
stats=data.batch_stats()        
data.normalize(stats)
data.show_batch(rows=3, figsize=(12,9))
RESNET_ENCODERS = {
    34: torchvision.models.resnet34,
    50: torchvision.models.resnet50,
    101: torchvision.models.resnet101,
    152: torchvision.models.resnet152,
}
class Resnet4Channel(nn.Module):
    def __init__(self, encoder_depth=34, pretrained=True, num_classes=28):
        super().__init__()

        encoder = RESNET_ENCODERS[encoder_depth](pretrained=pretrained)
        
        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with zeros
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        w = encoder.conv1.weight
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
        self.conv1.weight = nn.Parameter(torch.cat((w,w[:,:1,:,:]),dim=1))
        
        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        
        self.avgpool = encoder.avgpool
        self.fc = nn.Linear(512 * (1 if encoder_depth==34 else 4), num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    

def resnet34(pretrained):
    return Resnet4Channel(encoder_depth=34)

def resnet50(pretrained):
    return Resnet4Channel(encoder_depth=50)



# copied from https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
def _resnet_split(m): return (m[0][6],m[1])

f1_score = partial(fbeta, thresh=0.35, beta=1)
def _resnet_split(m): return (m[0][6],m[1])
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
learn = create_cnn(
    data,
    resnet34,
    cut=-2,
    split_on=_resnet_split,
    loss_func=FocalLoss(logits=True,alpha=alpha_log),
    path=path,    
    metrics=[f1_score], 
)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1,4e-2)
learn.recorder.plot_losses()
learn.recorder.plot()
learn.unfreeze()
learn.lr_find(num_it=100)
learn.recorder.plot()
lr=1e-3
learn.fit_one_cycle(4, slice(lr/10,lr))

learn.save('4_epochs')
learn.recorder.plot(skip_start=0,skip_end=0)
learn.recorder.plot_losses()

learn.recorder.plot_lr()


pred,y=learn.get_preds()
pred_test,_=learn.get_preds(ds_type=DatasetType.Test)
pred_test_tta,_=learn.TTA(ds_type=DatasetType.Test)
th_t = np.array([0.565,0.39,0.55,0.345,0.33,0.39,0.33,0.45,0.38,0.39,
               0.34,0.42,0.31,0.38,0.49,0.50,0.38,0.43,0.46,0.40,
               0.39,0.505,0.37,0.47,0.41,0.545,0.32,0.1])
from sklearn.metrics import f1_score as f1_sc

def eval_pred(pred,y):
    ths=np.arange(0.001,1,0.01)
    preds_s=F.sigmoid(pred)
    th_val=ths[np.argmax([f1_sc(y,preds_s>th,average='macro') for th in ths])]
    print('F1 macro: ',f1_sc(to_np(y),to_np(preds_s)>th_t,average='macro'))
    print(f'F1 macro (th = {th_val}): ',f1_sc(to_np(y),to_np(preds_s)>th_val,average='macro'))
    plt.plot(f1_sc(to_np(y),to_np(preds_s)>th_t,average=None),label='opt')
    plt.plot(f1_sc(to_np(y),to_np(preds_s)>th_val,average=None),label='valid')
    plt.legend()
ths=np.arange(0.001,1,0.01)
preds_s=F.sigmoid(pred)
th_val=ths[np.argmax([f1_sc(y,preds_s>th,average='macro') for th in ths])]
eval_pred(pred,y)
def save_pred(pred, th=0.5, fname='protein_classification.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line>th)[0]]))
        pred_list.append(s)
        
    sample_df = pd.read_csv(sample_csv)
    sample_list = list(sample_df.Id)
    #fnames_=[fname.split('/')[-1] for fname in learn.data.test_ds.fnames]
    pred_dic = dict((key, value) for (key, value) 
                in zip(test_ids,pred_list))
    pred_list_cor = [pred_dic[id] for id in sample_list]
    df = pd.DataFrame({'Id':sample_list,'Predicted':pred_list_cor})
    df.to_csv(fname, header=True, index=False)
save_pred(to_np(F.sigmoid(pred_test)), th=th_val, fname=f'protein_classification_{np.around(th_val,decimals=2)}.csv')

save_pred(to_np(F.sigmoid(pred_test)), th=th_t, fname='protein_classification_customth.csv')
save_pred(to_np(F.sigmoid(pred_test_tta)), th=th_val, fname=f'protein_classification_{np.around(th_val,decimals=2)}_tta.csv')

save_pred(to_np(F.sigmoid(pred_test_tta)), th=th_t, fname='protein_classification_customth_tta.csv')