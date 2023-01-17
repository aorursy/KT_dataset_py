%reload_ext autoreload

%autoreload 2

%matplotlib inline



import fastai

from fastai.vision import *

from fastai.callbacks import SaveModelCallback

import os

from radam import *

from csvlogger import *

from mish_activation import *

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import cohen_kappa_score,confusion_matrix

import warnings

warnings.filterwarnings("ignore")

CUDA_LAUNCH_BLOCKING=1

fastai.__version__

import sys

sys.path.insert(0,"../input/squeezeexcitationnet/")

from squeeze import *
# !rm -rf ./models
# ##remove this cell if run locally

# !rm -rf ./cache

# !mkdir 'cache'

# !mkdir 'cache/torch'

# !mkdir 'cache/torch/checkpoints'

# #!cp '../input/pytorch-pretrained-models/semi_supervised_resnext50_32x4-ddb3e555.pth' 'cache/torch/checkpoints/'

# !cp '../input/pytorch-se-resnext/se_resnext50_32x4d-a260b3a4.pth' 'cache/torch/checkpoints/'

# torch.hub.DEFAULT_CACHE_DIR = 'cache'
sz = 128

bs = 20

nfolds = 4

SEED = 2020

N = 12 #number of tiles per image

#TRAIN = '../input/panda-16x128x128-tiles-data/train/'

TRAIN = '../input/panda-generating-data2/train_rand2000_256sz_16tiles/'

LABELS = '../input/labels/train.csv'
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)
files = sorted([p[:32] for p in os.listdir(TRAIN)])

len(files)
def get_data():

    import pandas as pd

    #susp = pd.read_csv('../input/suspicious/PANDA_Suspicious_Slides.csv')

    # ['marks', 'No Mask', 'Background only', 'No cancerous tissue but ISUP Grade > 0', 'tiss', 'blank']

    #to_drop = susp.query("reason in ['marks','Background only','tiss','blank']")['image_id']

    df = pd.read_csv(LABELS).set_index('image_id')

    #good_index = list(set(df.index)-set(to_drop))

    files = sorted([p[:32] for p in os.listdir(TRAIN)])

    df = df.loc[files]

    df = df.reset_index()

    splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)

    splits = list(splits.split(df,df.isup_grade))

    folds_splits = np.zeros(len(df)).astype(np.int)

    for i in range(nfolds): 

        folds_splits[splits[i][1]] = i

    df['split'] = folds_splits

    df['gleason_score']=df['gleason_score'].replace('negative','0+0')

    df[['prim_gleason','secon_gleason']] = df.gleason_score.str.split("+",expand=True)

    df[['prim_gleason','secon_gleason']] = df[['prim_gleason','secon_gleason']].astype(np.int64)

    df['prim_gleason']=df['prim_gleason'].replace(3,1)

    df['prim_gleason']=df['prim_gleason'].replace(4,2)

    df['prim_gleason']=df['prim_gleason'].replace(5,3)

    df['secon_gleason']=df['secon_gleason'].replace(3,1)

    df['secon_gleason']=df['secon_gleason'].replace(4,2)

    df['secon_gleason']=df['secon_gleason'].replace(5,3)

    print("****************df shape:",df.shape,"***********************")

    print(">>>>>>>>>Before sampling<<<<<<<<<<<<<")

    for isup in [0,1,2,3,4,5]:

        print("isup grade:",isup,"| n_instances:",df.query('isup_grade=={0}'.format(isup)).shape[0],"| corresponding gleason score:",df[['isup_grade','gleason_score']].query('isup_grade=={0}'.format(isup))['gleason_score'].unique())

        print("----"*20)

    #df.drop([df[df['image_id']=="b0a92a74cb53899311acc30b7405e101"].index[0]],inplace=True)

    #b0a92a74cb53899311acc30b7405e101 is the only image id with gleason 4+3 mapping to isup=2

    #df = pd.concat([df.query('isup_grade==0').iloc[:1200],df.query('isup_grade==1').iloc[:1200],df.query('isup_grade==2 or isup_grade==3 or isup_grade==4 or isup_grade==5')],axis=0)

    df = df.sample(n=5000,random_state=SEED).reset_index(drop=True)#shuffling

    print(">>>>>>>>>After sampling<<<<<<<<<<")

    for isup in [0,1,2,3,4,5]:

        print("isup grade:",isup,"| n_instances:",df.query('isup_grade=={0}'.format(isup)).shape[0],"| corresponding gleason score:",df[['isup_grade','gleason_score']].query('isup_grade=={0}'.format(isup))['gleason_score'].unique())

        print("----"*20)

    return df

df = get_data()

df[['isup_grade','split','prim_gleason','secon_gleason']].hist(bins=50)

df.head()


import seaborn as sns

sns.countplot(df['data_provider'])
#for 2000 images obtained of size 256 and tiles=16

mean = torch.tensor([1.0-0.68688968, 1.0-0.44634704, 1.0-0.61367611])

std = torch.tensor([0.46521431,0.46922062,0.42265951])
# mean = torch.tensor([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304])

# std = torch.tensor([0.36357649, 0.49984502, 0.40477625])

# #mean: [0.96589806 0.9326964  0.95441414] , std: [0.30177967 0.4173849  0.33891267]
def open_image(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,after_open:Callable=None)->Image:

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin

        x = PIL.Image.open(fn).convert(convert_mode)

    if after_open: 

        x = after_open(x)

    x = pil2tensor(x,np.float32)

    if div: 

        x.div_(255)

    return cls(1.0-x) #invert image for zero padding



class MImage(ItemBase):

    def __init__(self, imgs):

        self.obj  = (imgs)

        self.data = [(imgs[i].data - mean[...,None,None])/std[...,None,None] for i in range(len(imgs))]

    

    def apply_tfms(self, tfms,*args, **kwargs):

        for i in range(len(self.obj)):

            self.obj[i] = self.obj[i].apply_tfms(tfms, *args, **kwargs)

            self.data[i] = (self.obj[i].data - mean[...,None,None])/std[...,None,None]

        return self

    

    def __repr__(self): 

        return f'{self.__class__.__name__} {img.shape for img in self.obj}'

    

    def to_one(self):

        img = torch.stack(self.data,1)

        img = img.view(3,-1,N,sz,sz).permute(0,1,3,2,4).contiguous().view(3,-1,sz*N)

        return Image(1.0 - (mean[...,None,None]+img*std[...,None,None]))



class MImageItemList(ImageList):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    

    def __len__(self)->int: return len(self.items) or 1 

    

    def get(self, i):

        fn = Path(self.items[i])

        fnames = [Path(str(fn)+'_'+str(i)+'.png')for i in range(N)]

        imgs = [open_image(fname, convert_mode=self.convert_mode, after_open=self.after_open)

               for fname in fnames]

        return MImage(imgs)



    def reconstruct(self, t):

        return MImage([mean[...,None,None]+_t*std[...,None,None] for _t in t])

    

    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(300,50), **kwargs):

        rows = min(len(xs),8)

        fig, axs = plt.subplots(rows,1,figsize=figsize)

        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):

            xs[i].to_one().show(ax=ax, y=ys[i], **kwargs)

        plt.tight_layout()

        



#collate function to combine multiple images into one tensor

def MImage_collate(batch:ItemsList)->Tensor:

    result = torch.utils.data.dataloader.default_collate(to_data(batch))

    if isinstance(result[0],list):

        result = [torch.stack(result[0],1),result[1]]

    return result
def get_data(fold=0):

    return (MImageItemList.from_df(df, path='.', folder=TRAIN, cols='image_id')

      .split_by_idx(df.index[df.split == fold].tolist())

      .label_from_df(cols=['isup_grade'])

      .transform(get_transforms(flip_vert=True,max_rotate=15),size=sz,padding_mode='zeros')

      .databunch(bs=bs,num_workers=5,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

data = get_data(0)

#data.show_batch()
# i[1] contains labels, indicating isup grades

# preds,y,losses = learn.get_preds(with_loss=True)

# interp = ClassificationInterpretation(learn, preds, y, losses)
for i in data.dl():

    print(i[0][0].shape,i[0][7][0].shape,i[1].shape)

    break
def display_image(img,is_mask=False):

    '''

    To display image/mask 

    args: img, image

          is_mask, boolean True if greyscale mask is passed

    '''

    from matplotlib import pyplot as plt

    %matplotlib inline

    if is_mask:

        plt.imshow(img,cmap='gray')

    else:

        plt.imshow(img)

    plt.show()            
# m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',model='resnext50_32x4d_ssl')

# #there are 8 blocks in resnext 50

# print(list(m.children())[:-2][0],'\n',list(m.children())[:-2][1],'\n',list(m.children())[:-2][2],'\n',list(m.children())[:-2][3])

# for i in [4,5,6,7]:

#     print(list(m.children())[:-2][i])

#     print("-"*100)#,list(m.children())[:-2][5],list(m.children())[:-2][6],list(m.children())[:-2][7]
from collections import Counter

prim_dict=Counter(df['prim_gleason'])

sec_dict= Counter(df['secon_gleason'])

#Method1

# prim_weights = [1 - (prim_dict[i] / sum(prim_dict.values())) for i in [0,1,2,3]]

# prim_weights = torch.FloatTensor(prim_weights).to('cuda')

# sec_weights = [1 - (sec_dict[i] / sum(sec_dict.values())) for i in [0,1,2,3]]

# sec_weights = torch.FloatTensor(sec_weights).to('cuda')



#Method2

prim_weights = torch.FloatTensor([2,1,1,10]).to('cuda')#torch.FloatTensor([max(prim_dict.values())/prim_dict[i]  for i in [0,1,2,3]]).to('cuda')

sec_weights = torch.FloatTensor([2,1,1,10]).to('cuda')#torch.FloatTensor([max(sec_dict.values())/sec_dict[i]  for i in [0,1,2,3]]).to('cuda')

prim_weights,sec_weights,prim_dict,sec_dict

# class ConfusionMatrix(Callback):

#     "Computes the confusion matrix."



#     def on_train_begin(self, **kwargs):

#         self.n_classes = 0



#     def on_epoch_begin(self, **kwargs):

#         self.cm = None



#     def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):

#         preds = last_output.argmax(-1).view(-1).cpu()

#         targs = last_target.cpu()

#         if self.n_classes == 0:

#             self.n_classes = last_output.shape[-1]

#         if self.cm is None: self.cm = torch.zeros((self.n_classes, self.n_classes), device=torch.device('cpu'))

#         cm_temp_numpy = self.cm.numpy()

#         np.add.at(cm_temp_numpy, (targs ,preds), 1)

#         self.cm = torch.from_numpy(cm_temp_numpy)



#     def on_epoch_end(self, **kwargs):

#         self.metric = self.cm

        

# @dataclass

# class KappaScore1(ConfusionMatrix):

#     "Computes the rate of agreement (Cohens Kappa)."

#     weights:Optional[str]=None      # None, `linear`, or `quadratic`

        

#     def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):

# #         print("Customised scoring function....")

# #         print(last_output[0].shape,last_output[1].shape,last_target.shape)

#         preds,targs = get_isup_preds_targs_2targets(last_output,last_target)#convert gleasons-->isup for evaluatio



#         if self.n_classes == 0:

#             self.n_classes = 6 #n_classes in isup_grade

#         if self.cm is None: 

#             #This executes only once

#             self.cm = torch.zeros((self.n_classes, self.n_classes), device=torch.device('cpu'))

#         cm_temp_numpy = self.cm.numpy()

#         np.add.at(cm_temp_numpy, (targs ,preds), 1)

#         self.cm = torch.from_numpy(cm_temp_numpy)

        

        

#     def on_epoch_end(self, last_metrics, **kwargs):

#         sum0 = self.cm.sum(dim=0)

#         sum1 = self.cm.sum(dim=1)

#         expected = torch.einsum('i,j->ij', (sum0, sum1)) / sum0.sum()

#         if self.weights is None:

#             w = torch.ones((self.n_classes, self.n_classes))

#             w[self.x, self.x] = 0

#         elif self.weights == "linear" or self.weights == "quadratic":

#             w = torch.zeros((self.n_classes, self.n_classes))

#             w += torch.arange(self.n_classes, dtype=torch.float)

#             w = torch.abs(w - torch.t(w)) if self.weights == "linear" else (w - torch.t(w)) ** 2

#         else: raise ValueError('Unknown weights. Expected None, "linear", or "quadratic".')

#         k = torch.sum(w * self.cm) / torch.sum(w * expected)

#         return add_metrics(last_metrics, 1-k)
def get_resnext(layers, pretrained, progress, **kwargs):

    from torchvision.models.resnet import ResNet, Bottleneck

    model = ResNet(Bottleneck, layers, **kwargs)

    model.load_state_dict(torch.load('../input/resnext-50-ssl/semi_supervised_resnext50_32x4-ddb3e555.pth'))

    return model





class GleasonISUP(nn.Module):

    def __init__(self):

        super().__init__()

        #m = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

        #m = get_resnext([3, 4, 6, 3], pretrained=True, progress=False, groups=32,width_per_group=4)

        m = se_resnext50_32x4d(4,loss='softmax', pretrained=True)

        self.enc = nn.Sequential(*list(m.children())[:-2])       

        nc = list(m.children())[-1].in_features

        self.head = nn.Sequential(AdaptiveConcatPool2d(),

                                  Flatten(),

#                                   nn.Linear(2*nc,512),

#                                   Mish(),

#                                   nn.BatchNorm1d(512), 

#                                   nn.Dropout(0.5),

                                  nn.Linear(2*nc,512),

                                  Mish(),

                                  nn.BatchNorm1d(512), 

                                  nn.Dropout(0.3))

        self.prim =  nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Dropout(0.3),nn.Linear(128,4))

        self.sec  =  nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Dropout(0.3),nn.Linear(128,4))

        self.isup  =  nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Dropout(0.3),nn.Linear(128,6))

      

        

        

    def forward(self, *x):

        

        

        shape = x[0].shape

        n = len(x)

        x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])

        #x: bs*N x 3 x 128 x 128

        

        

        x = self.enc(x)

        

        #x: bs*N x C x 4 x 4

        shape = x.shape

        #concatenate the output for tiles into a single map

        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])

        #x: bs x C x N*4 x 4

        x = self.head(x)

        #x: bs x n

        prim_gleason = self.prim(x)

        sec_gleason  = self.sec(x)

        isup = self.isup(x)

        preds = [prim_gleason,sec_gleason,isup]

        return preds
def get_resnext(layers, pretrained, progress, **kwargs):

    from torchvision.models.resnet import ResNet, Bottleneck

    model = ResNet(Bottleneck, layers, **kwargs)

    model.load_state_dict(torch.load('../input/resnext-50-ssl/semi_supervised_resnext50_32x4-ddb3e555.pth'))

    return model





class Gleason(nn.Module):

    def __init__(self):

        super().__init__()

        #m = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

        #m = get_resnext([3, 4, 6, 3], pretrained=True, progress=False, groups=32,width_per_group=4)

        m = se_resnext50_32x4d(4,loss='softmax', pretrained=True)

        self.enc = nn.Sequential(*list(m.children())[:-2])       

        nc = list(m.children())[-1].in_features

        self.head = nn.Sequential(AdaptiveConcatPool2d(),

                                  Flatten(),

                                  nn.Linear(2*nc,512),

                                  Mish(),

                                  nn.BatchNorm1d(512), 

                                  nn.Dropout(0.5),

                                  nn.Linear(512,256),

                                  Mish(),

                                  nn.BatchNorm1d(256), 

                                  nn.Dropout(0.3))

        self.prim =  nn.Sequential(nn.Linear(256,128),Mish(),nn.BatchNorm1d(128),nn.Dropout(0.3),nn.Linear(128,4))

        self.sec  =  nn.Sequential(nn.Linear(256,128),Mish(),nn.BatchNorm1d(128),nn.Dropout(0.3),nn.Linear(128,4))

      

      

        

        

    def forward(self, *x):

        

        

        shape = x[0].shape

        n = len(x)

        x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])

        #x: bs*N x 3 x 128 x 128

        

        

        x = self.enc(x)

        

        #x: bs*N x C x 4 x 4

        shape = x.shape

        #concatenate the output for tiles into a single map

        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])

        #x: bs x C x N*4 x 4

        x = self.head(x)

        #x: bs x n

        prim_gleason = self.prim(x)

        sec_gleason  = self.sec(x)

        preds = [prim_gleason,sec_gleason]

        return preds
def get_isup_preds_targs(preds,target):

    lookup_map = {(0,0):0,(1,1):1,(1,2):2,(2,1):3,(2,2):4,(3,1):4,(1,3):4,(2,3):5,(3,2):5,(3,3):5}

    lookup_map2 = {(0,1):1,(0,2):1,(0,3):2,(1,0):1,(2,0):3,(3,0):4}#or all 0

    prim_preds = preds[0].argmax(-1).view(-1,1)

    sec_preds  = preds[1].argmax(-1).view(-1,1)

    temp_preds = torch.cat([prim_preds,sec_preds],dim=1)

    

    count = 0

    errors = 0

    temp = []

    for i in np.array(temp_preds.cpu()):

        count+=1

        try:

            temp.append(lookup_map[tuple(i)])

        except KeyError:

            print(tuple(i)," missing")

            print('target={0},prediction={1}'.format(isup_target[i],isup_preds[i]))

            errors+=1

            temp.append(lookup_map2[tuple(i)])

    print("count={0},errors={1}".format(count,errors),"correct=",count-errors)

    isup_preds = torch.tensor(temp,dtype=torch.long,device='cpu')

    temp = []

    for i in np.array(target.cpu()):

        temp.append(lookup_map[tuple(i)])

    isup_targs = torch.tensor(temp,dtype=torch.long,device='cpu')    

    return isup_preds,isup_targs



class BiTaskGleasonLoss(nn.Module):

    def __init__(self, task_num):

        super(BiTaskGleasonLoss, self).__init__()

        self.task_num = task_num

        self.log_vars = nn.Parameter(torch.zeros((task_num)))



    def forward(self, preds, targets):

        temp=[]

        primloss = nn.CrossEntropyLoss(weight=prim_weights)

        prim_preds  = preds[0]

        prim_target = targets[:,0].long()

        loss0 = primloss(prim_preds,prim_target)

        temp.append(loss0)

        precision0 = torch.exp(-self.log_vars[0])

        loss0 = precision0*loss0 + self.log_vars[0]   

        

        secloss  = nn.CrossEntropyLoss(weight=sec_weights)

        sec_preds  = preds[1]

        sec_target = targets[:,1].long()

        loss1 = secloss(sec_preds,sec_target)

        temp.append(loss1)

        precision1 = torch.exp(-self.log_vars[1])

        loss1 = precision1*loss1 + self.log_vars[1]   

        

        

        print("precisions:", precision0,precision1)

        print(self.log_vars[0],self.log_vars[1].dtype)

        print("simple loss:",sum(temp))

        print("complicated loss:","loss0+loss1:",loss0+loss1,"loss0+loss1:",loss0+loss1)

        return loss1+loss0



def get_isup_preds_targs_2targets(preds,targs):

    #predictions

    prim_preds = preds[0].argmax(-1).view(-1,1)

    sec_preds  = preds[1].argmax(-1).view(-1,1)

    gleason_preds = np.array(torch.cat([prim_preds,sec_preds],dim=1).cpu())#converting to np.array for tuple()

    #targets

    target = np.array(targs.cpu())#converting to np.array to cast to tuple()

    gleason_target = target[:,0:2]

    

    lookup_map = {(0,0):0,(1,1):1,(1,2):2,(2,1):3,(2,2):4,(3,1):4,(1,3):4,(2,3):5,(3,2):5,(3,3):5}

    lookup_map2 = {(0,1):1,(0,2):1,(0,3):2,(1,0):1,(2,0):3,(3,0):4}

    #lookup_map's keys are indices of gleason scores and not gleason scores themselves!

    temp1 = []#for predictions

    temp2 = []#for targets

    count = 0

    errors = 0

    for i in range(len(gleason_preds)):

        count+=1

        try:

            temp1.append(lookup_map[tuple(gleason_preds[i])])

        except KeyError:

            errors+=1

            temp1.append(lookup_map2[tuple(gleason_preds[i])])

            print(">>>>>>>>>>missing<<<<<<<<<<<")

        finally:

            print('target={0},prediction={1}'.format(gleason_target[i],tuple(gleason_preds[i])))

            temp2.append(lookup_map[tuple(gleason_target[i])])

            print("-"*30)

            

    print("count={0},errors={1}".format(count,errors),"correct=",count-errors)

    final_preds = torch.tensor(temp1,dtype=torch.long,device='cpu')

    final_targs = torch.tensor(temp2,dtype=torch.long,device='cpu')    

    return final_preds,final_targs
# class TriTaskGleasonLoss(nn.Module):

#     def __init__(self, task_num):

#         super(TriTaskGleasonLoss, self).__init__()

#         self.task_num = task_num

#         self.log_vars = nn.Parameter(torch.zeros((task_num)))



#     def forward(self, preds, targets):

# #         print("type(preds):",type(preds),"| len(preds):",len(preds),"| preds[0].shape:",preds[0].shape,"| preds[1].shape:",preds[1].shape,"| preds[2].shape:",preds[2].shape)

# #         print("type(targets):",type(targets),"| targets[:,0].shape:",targets[:,0].shape,"| targets[:,1].shape:",targets[:,1].shape,"| targets[:,2].shape:",targets[:,2].shape)

        

#         temp=[]

#         primloss = nn.CrossEntropyLoss(weight=prim_weights)

#         prim_preds  = preds[0]

#         prim_target = targets[:,0].long()

#         loss0 = primloss(prim_preds,prim_target)

#         temp.append(loss0)

#         precision0 = torch.exp(-self.log_vars[0])

#         loss0 = precision0*loss0 + self.log_vars[0]   

        

#         secloss  = nn.CrossEntropyLoss(weight=sec_weights)

#         sec_preds  = preds[1]

#         sec_target = targets[:,1].long()

#         loss1 = secloss(sec_preds,sec_target)

#         temp.append(loss1)

#         precision1 = torch.exp(-self.log_vars[1])

#         loss1 = precision1*loss1 + self.log_vars[1]   

        

#         crossEntropy = nn.CrossEntropyLoss()

#         isup_preds  = preds[2]

#         isup_target = targets[:,2].long()

#         loss2 = crossEntropy(isup_preds,isup_target)

#         temp.append(loss2)

#         precision2 = torch.exp(-self.log_vars[2])

#         loss2 = precision2*loss2 + self.log_vars[2]   

#         print("precisions:", precision0,precision1,precision2)

#         print(self.log_vars[0],self.log_vars[1].dtype,self.log_vars[2])

#         print("simple loss:",sum(temp))

#         print("complicated loss:","loss0+loss1+loss2:",loss0+loss1+loss2,"loss0+loss1:",loss0+loss1)

#         print("-"*20)

#         return loss0*loss1*loss2



# def get_isup_preds_targs_3targets(preds,targs):

#     #predictions

#     prim_preds = preds[0].argmax(-1).view(-1,1)

#     sec_preds  = preds[1].argmax(-1).view(-1,1)

#     temp_preds = np.array(torch.cat([prim_preds,sec_preds],dim=1).cpu())#converting to np.array for tuple()

#     isup_preds = preds[2].argmax(-1).view(-1).cpu()

#     #targets

#     target = np.array(targs.cpu())#converting to np.array to cast to tuple()

#     gleason_target = target[:,0:2]

#     isup_target = target[:,2]

    

#     lookup_map = {(0,0):0,(1,1):1,(1,2):2,(2,1):3,(2,2):4,(3,1):4,(1,3):4,(2,3):5,(3,2):5,(3,3):5}

#     #lookup_map's keys are indices of gleason scores and not gleason scores themselves!

#     temp1 = []#for predictions

#     temp2 = []#for targets

#     count = 0

#     errors = 0

#     for i in range(len(temp_preds)):

# #         temp1.append(isup_preds[i])

# #         temp2.append(isup_target[i])

# #         print('target={0},prediction={1}'.format(isup_target[i],isup_preds[i]))

#         count+=1

#         try:

#             temp1.append(lookup_map[tuple(temp_preds[i])])

#             temp2.append(lookup_map[tuple(gleason_target[i])])

#         except KeyError:

#             print(tuple(temp_preds[i])," is missing!")

#             print('target={0},prediction={1}'.format(isup_target[i],isup_preds[i]))

#             errors+=1

#             temp1.append(isup_preds[i])

#             temp2.append(isup_target[i])

#     print("count={0},errors={1}".format(count,errors),"correct=",count-errors)

#     final_preds = torch.tensor(temp1,dtype=torch.long,device='cpu')

#     final_targs = torch.tensor(temp2,dtype=torch.long,device='cpu')    

#     return final_preds,final_targs
class Model(nn.Module):

    def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True):

        super().__init__()

        m = se_resnext50_32x4d(4,loss='softmax', pretrained=True)

        self.enc = nn.Sequential(*list(m.children())[:-2])       

        nc = list(m.children())[-1].in_features

        self.head = nn.Sequential(AdaptiveConcatPool2d(),

                                  Flatten(),

                                  nn.Linear(2*nc,512),

                                  Mish(),

                                  nn.BatchNorm1d(512), 

                                  nn.Dropout(0.5),

                                  nn.Linear(512,n))

        

        

    def forward(self, *x):

        shape = x[0].shape

        n = len(x)

        x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])

        #x: bs*N x 3 x 128 x 128

        x = self.enc(x)

        #x: bs*N x C x 4 x 4

        shape = x.shape

        #concatenate the output for tiles into a single map

        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])

        #x: bs x C x N*4 x 4

        x = self.head(x)

        #x: bs x n

        return x
# data  = get_data(3)

# model = Gleason()

# gleason_loss = BiTaskGleasonLoss(2)

# learn = Learner(data,model,loss_func=gleason_loss, opt_func=RAdam, metrics=[KappaScore(weights='quadratic')]).to_fp16()

# learn.loss_func=learn.loss_func.to('cuda')

# learn.model=learn.model.to('cuda')

# learn.clip_grad = 1.0

# learn.split([learn.model.enc[0],learn.model.enc[1],learn.model.enc[2],learn.model.enc[3],learn.model.enc[4],learn.model.head,nn.ModuleList([learn.model.prim,learn.model.sec])])

# learn.freeze()#first, train only the last layer(the heads)
data  = get_data(3)

model = Model()

gleason_loss = LabelSmoothingCrossEntropy()

learn = Learner(data,model,loss_func=gleason_loss, opt_func=Over9000, metrics=[KappaScore(weights='quadratic')]).to_fp16()

learn.loss_func=learn.loss_func.to('cuda')

learn.model=learn.model.to('cuda')

learn.clip_grad = 1.0

learn.split([learn.model.enc[0],learn.model.enc[1],learn.model.enc[2],learn.model.enc[3],learn.model.enc[4],learn.model.head])

learn.freeze()#first, train only the last layer(the heads)
torch.cuda.get_device_name(),torch.cuda.get_device_properties(0)
# learn.lr_find()

# # figures out what is the fastest I can train this neural network without making it zip off the rails and get blown apart

# learn.recorder.plot()

# plt.title("Loss Vs Learning Rate")
# learn.fit_one_cycle(5, max_lr=5e-2, div_factor=100, pct_start=0.0, callbacks =[SaveModelCallback(learn,name='stage1',monitor='kappa_score')])
learn.load('stage5')

learn.unfreeze()

# learn.lr_find()

# learn.recorder.plot()

# plt.title("Loss Vs Learning Rate")
learn.fit_one_cycle(8, max_lr=slice(5e-4,5e-2),div_factor=100,pct_start=0.0,callbacks = [SaveModelCallback(learn,name='stage5',monitor='kappa_score')])
# fname = 'RNXT50'

# pred,target = [],[]

# for fold in range(nfolds):#nfolds

#     print("-"*40,"Fold-",fold,"-"*40)

#     data  = get_data(fold)

#     model = GleasonModel()

#     gleason_loss = TriTaskGleasonLoss(3)

#     learn = Learner(data,model,loss_func=gleason_loss, opt_func=Over9000, metrics=[KappaScore(weights='quadratic')]).to_fp16()

#     learn.loss_func=learn.loss_func.to('cuda')

#     learn.model=learn.model.to('cuda')

#     learn.clip_grad = 1.0

#     learn.split([learn.model.enc[0:7],learn.model.enc[7],learn.model.head,nn.ModuleList([learn.model.prim,learn.model.sec,learn.model.isup])])

#     learn.freeze()#first, train only the last layer(the heads)

#     learn.fit_one_cycle(1, max_lr=1e-1, div_factor=25, pct_start=0.0, 

#       callbacks = [SaveModelCallback(learn,name='stage1_{0}'.format(fold),monitor='kappa_score')])

#     logger = CSVLogger(learn, f'log_{fname}_{fold}')

#     learn.load('stage1_{0}'.format(fold))

#     learn.freeze_to(1)

#     learn.fit_one_cycle(1, max_lr=slice(5e-6,5e-4), div_factor=25, pct_start=0.0, 

#       callbacks = [SaveModelCallback(learn,name='stage2_{0}'.format(fold),monitor='kappa_score')])

    

#     torch.save(learn.model.state_dict(), f'{fname}_{fold}.pth')

    

#     learn.model.eval()

#     with torch.no_grad():

#         for step, (x, y) in progress_bar(enumerate(data.dl(DatasetType.Valid)),total=len(data.dl(DatasetType.Valid))):

#             #print(len(x),x[0].shape)

#             p = learn.model(*x) 

#             preds,targs = get_isup_preds_targs(p,y)

#             pred.append(preds)

#             target.append(targs)
# p = torch.argmax(torch.cat(pred,dim=0),1)      

# t = torch.cat(target)

# print(cohen_kappa_score(t,p,weights='quadratic'))

# print(confusion_matrix(t,p))
# confusion_matrix(torch.cat(target),torch.cat(pred,dim=0))
# cohen_kappa_score(torch.cat(target),torch.cat(pred,dim=0),weights='quadratic')
# !rm -r 'cache'
# !pip install -q torchviz

# import torch

# from torchviz import make_dot

# model = Model()

# x = torch.randn(2, 3, 128, 128).requires_grad_(True)

# y = model(x)

# make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)])).render("attached")
# class Model(nn.Module):

#     def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True):

#         super().__init__()

#         #m = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

#         m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)

#         self.enc = nn.Sequential(*list(m.children())[:-2])       

#         nc = list(m.children())[-1].in_features

#         self.head = nn.Sequential(AdaptiveConcatPool2d(),

#                                   Flatten(),

#                                   nn.Linear(2*nc,128),

#                                   Mish(),

#                                   nn.BatchNorm1d(128), 

#                                   nn.Dropout(0.5),

#                                   nn.Linear(128,n))

        

        

#     def forward(self, *x):

#         shape = x[0].shape

#         n = len(x)

#         x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])

#         #x: bs*N x 3 x 128 x 128

#         x = self.enc(x)

#         #x: bs*N x C x 4 x 4

#         shape = x.shape

#         #concatenate the output for tiles into a single map

#         x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])

#         #x: bs x C x N*4 x 4

#         x = self.head(x)

#         #x: bs x n

#         return x
# #UNDERSAMPLING

# import seaborn as sns

# from imblearn.under_sampling import RandomUnderSampler

# rus = RandomUnderSampler(random_state=SEED)

# X_resampled, y_resampled = rus.fit_resample(df[['image_id', 'data_provider','split']].iloc[:8000], df['isup_grade'].iloc[:8000])

# df = pd.concat([pd.DataFrame(X_resampled,columns=['image_id', 'data_provider','split']),pd.DataFrame(y_resampled,columns=['isup_grade'])],axis=1)

# df = df.sample(frac=1,random_state=SEED).reset_index(drop=True)

# sns.countplot(df['isup_grade'])

# df.head()

# df = df.groupby(['split'], group_keys=False).apply(lambda x: x.sample(min(len(x), 1000),random_state=SEED))

# df[['isup_grade','split']].hist(bins=50)
# t1=torch.tensor([[-8.8875e+01,  3.9062e+01, -6.1406e+00, -6.6625e+01],

#         [-3.3781e+01,  1.4578e+01, -8.6133e-01, -2.5047e+01],

#         [-2.9297e+00,  1.2412e+00,  9.8584e-01, -2.6113e+00],

#         [-1.6992e-01,  3.9355e-01,  4.3701e-01, -1.1953e+00],

#         [-5.7910e-01,  6.8457e-01,  4.6216e-01, -1.4893e+00],

#         [-1.7639e-01,  4.2603e-01,  3.8843e-01, -1.2490e+00],

#         [-1.6297e+01,  7.1523e+00,  4.3018e-01, -1.2164e+01],

#         [-1.6333e-01,  4.3164e-01,  3.7207e-01, -1.2646e+00],

#         [-4.8309e-02,  4.4946e-01,  3.0298e-01, -1.1963e+00],

#         [-3.6844e+01,  1.6312e+01, -2.1621e+00, -2.7594e+01],

#         [        0,         1,         0,         0],

#         [-7.6721e-02,  4.3408e-01,  3.4644e-01, -1.1943e+00],

#         [-8.0875e+01,  3.5188e+01, -3.9707e+00, -5.9656e+01],

#         [-6.1625e+01,  2.7312e+01, -3.2949e+00, -4.6594e+01],

#         [-3.5906e+01,  1.5633e+01, -9.3945e-01, -2.6281e+01],

#         [-2.1641e+01,  9.4062e+00,  6.9385e-01, -1.5695e+01],

#         [-2.8781e+01,  1.2414e+01,  4.8706e-01, -2.0859e+01],

#         [-3.0266e+01,  1.3586e+01, -1.4209e+00, -2.2922e+01],

#         [        0,         1,         0,         0],

#         [-3.6531e+01,  1.6172e+01, -1.5479e+00, -2.7625e+01],

#         [-6.9580e-02,  4.5190e-01,  3.1958e-01, -1.2119e+00],

#         [-6.0181e-02,  4.5117e-01,  3.1421e-01, -1.2178e+00],

#         [-2.8711e-01,  3.8086e-01,  4.9854e-01, -1.2412e+00],

#         [-4.4373e-02,  4.3896e-01,  3.1714e-01, -1.1748e+00],

#         [-6.0028e-02,  4.4263e-01,  3.2349e-01, -1.1914e+00],

#         [-9.6741e-02,  4.4019e-01,  3.5449e-01, -1.2178e+00],

#         [-3.0762e+00,  1.5293e+00,  6.8506e-01, -2.8242e+00],

#         [-5.3345e-02,  4.5386e-01,  3.0737e-01, -1.2119e+00],

#         [-1.2711e+01,  5.7656e+00,  3.3203e-01, -9.6797e+00],

#         [-6.6406e-02,  4.4727e-01,  3.2471e-01, -1.1963e+00],

#         [-4.9042e-02,  4.3872e-01,  3.0420e-01, -1.1709e+00],

#         [-1.6641e+00,  1.1172e+00,  5.8203e-01, -2.0605e+00]])

# t2=torch.tensor([[-5.9812e+01,  1.6828e+01, -1.5188e+01,  4.7781e+01],

#         [-2.2938e+01,  7.0039e+00, -5.7188e+00,  1.8125e+01],

#         [-2.1074e+00,  8.9258e-01, -4.4336e-01,  1.7598e+00],

#         [-4.0796e-01,  5.6299e-01, -8.6121e-02, -2.2437e-01],

#         [-5.1025e-01,  8.9648e-01, -3.6938e-01, -2.2949e-02],

#         [-4.5142e-01,  6.2695e-01, -2.0935e-01, -1.9116e-01],

#         [-1.1297e+01,  3.5859e+00, -2.5879e+00,  8.7109e+00],

#         [-4.3408e-01,  6.1182e-01, -1.8848e-01, -2.2412e-01],

#         [-2.7954e-01,  5.9863e-01, -2.1533e-01, -3.5278e-01],

#         [-2.4562e+01,  7.7227e+00, -7.1094e+00,  2.0703e+01],

#         [        0,         1,         0,         0],

#         [-3.1030e-01,  5.6934e-01, -1.3806e-01, -3.4082e-01],

#         [-5.5281e+01,  1.5883e+01, -1.3398e+01,  4.2781e+01],

#         [-4.1875e+01,  1.1969e+01, -1.0406e+01,  3.2938e+01],

#         [-2.4562e+01,  7.4766e+00, -6.0117e+00,  1.9375e+01],

#         [-1.5336e+01,  5.4688e+00, -4.2070e+00,  1.1961e+01],

#         [-2.0141e+01,  6.2305e+00, -4.4219e+00,  1.5086e+01],

#         [-2.0578e+01,  6.3203e+00, -5.8203e+00,  1.7156e+01],

#         [        0,         1,         0,         0],

#         [-2.4766e+01,  7.2578e+00, -6.2344e+00,  1.9750e+01],

#         [-3.0591e-01,  6.0107e-01, -2.0142e-01, -3.3472e-01],

#         [-2.9712e-01,  5.8789e-01, -1.8433e-01, -3.5400e-01],

#         [-5.7617e-01,  5.8154e-01, -1.0052e-01, -8.4595e-02],

#         [-2.5415e-01,  5.9082e-01, -1.9971e-01, -3.5474e-01],

#         [-2.8394e-01,  6.0303e-01, -2.0618e-01, -3.4229e-01],

#         [-3.3081e-01,  5.8594e-01, -1.5564e-01, -3.2251e-01],

#         [-2.0703e+00,  9.4531e-01, -6.0254e-01,  1.8076e+00],

#         [-2.9077e-01,  5.9424e-01, -1.9482e-01, -3.5474e-01],

#         [-8.9766e+00,  2.7832e+00, -2.2129e+00,  7.0469e+00],

#         [-2.8809e-01,  6.1279e-01, -2.1301e-01, -3.4375e-01],

#         [-2.7637e-01,  6.0547e-01, -2.3462e-01, -3.2910e-01],

#         [-1.1592e+00,  1.0586e+00, -5.2344e-01,  7.5098e-01]])

# preds = [t1,t2]

# target = torch.tensor([[1., 2.],

#         [0., 0.],

#         [2., 3.],

#         [3., 3.],

#         [0., 0.],

#         [2., 1.],

#         [0., 0.],

#         [2., 2.],

#         [2., 3.],

#         [1., 1.],

#         [2., 3.],

#         [0., 0.],

#         [2., 1.],

#         [1., 2.],

#         [1., 1.],

#         [1., 2.],

#         [2., 2.],

#         [0., 0.],

#         [2., 2.],

#         [2., 3.],

#         [2., 2.],

#         [0., 0.],

#         [3., 3.],

#         [1., 2.],

#         [2., 1.],

#         [3., 3.],

#         [2., 3.],

#         [0., 0.],

#         [2., 2.],

#         [3., 2.],

#         [2., 2.],

#         [0., 0.]])