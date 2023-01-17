"""

!pip install -q kaggle



!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!kaggle competitions download -c humpback-whale-identification



!mkdir data

!mkdir ./data/train

!mkdir ./data/test

!mv *.zip ./data/

!mv *.csv ./data/

!unzip -q ./data/test.zip -d ./data/test/

!unzip -q ./data/train.zip -d ./data/train/



"""
#!kaggle competitions download -c humpback-whale-identification
from fastai import *

from fastai.vision import *

from fastai.callbacks import *

from fastai.layers import *

from fastai.metrics import accuracy_thresh



import warnings

warnings.filterwarnings("ignore")
path = Path('../input')

path_t=Path('../input/humpback-whale-identification/')

path1='.'

df = pd.read_csv(path_t/'train.csv'); 

#!pip install fastai=='1.0.39' --no-deps

import fastai

fastai.__version__

#path_t=Path('../input/humpback-whale-identification/')

#path1='.'

#df = pd.read_csv(path/'train.csv'); 



#!pip install fastai=='1.0.39'



import fastai

fastai.__version__
exclude_list=['0b1e39ff.jpg',

'0c11fa0c.jpg',

'1b089ea6.jpg',

'2a2ecd4b.jpg',

'2c824757.jpg',

'3e550c8a.jpg',

'56893b19.jpg',

'613539b4.jpg',

'6530809b.jpg',

'6b753246.jpg',

'6b9f5632.jpg',

'75c94986.jpg',

'7f048f21.jpg',

'7f7702dc.jpg',

'806cf583.jpg',

'95226283.jpg',

'a3e9070d.jpg',

'ade8176b.jpg',

'b1cfda8a.jpg',

'b24c8170.jpg',

'b7ea8be4.jpg',

'b9315c19.jpg',

'b985ae1e.jpg',

'baf56258.jpg',

'c4ad67d8.jpg',

'c5da34e7.jpg',

'c5e3df74.jpg',

'ced4a25c.jpg',

'd14f0126.jpg',

'e0b00a14.jpg',

'e6ce415f.jpg',

'e9bd2e9c.jpg',

'f4063698.jpg',

'f9ba7040.jpg']

new_whale_df = df[df.Id == "new_whale"] # only new_whale dataset

train_df = df[~(df.Id == "new_whale")] # no new_whale dataset, used for training

unique_labels = np.unique(train_df.Id.values)

trn_imgs=train_df.copy()

cnter = Counter(trn_imgs.Id.values)

trn_imgs['cnt']=trn_imgs['Id'].apply(lambda x: cnter[x])

#trn_imgs['target'] = 1

trn_imgs['target'] = 0 # 0 for same images

trn_imgs1 = trn_imgs.copy()

#trn_imgs1['target'] = 0

trn_imgs1['target'] = 1 # 1 for dissimilar images

#trn_imgs = trn_imgs.append(trn_imgs1)

target_col = 3

trn_imgs.head(1)

trn_imgs=trn_imgs[~trn_imgs.Image.isin(exclude_list)]
def is_even(num): return num % 2 == 0



class TripleDS(Dataset):

    def __init__(self, ds):

        self.ds = ds

        self.whale_ids = ds.y.items

        #bb_df=pd.read_csv('bounding_boxes.csv')

    def __len__(self):

        return len(self.ds)

    def __getitem__(self, idx):

        #if is_even(idx):

        img_a=self.ds[idx][0]

        img_p=self.sample_same(idx )

        img_n=self.sample_different(idx)

          #return self.sample_same(idx // 2)

        #print('-ve',img_n)

        #img_a=read_img(self.bb_df,)

        return self.construct_example(img_a,img_p,img_n,0)

          #return self.sample_different((idx-1) // 2)

    def sample_same(self, idx):

        whale_id = self.whale_ids[idx]        

        candidates = list(np.where(self.whale_ids == whale_id)[0])

        candidates.remove(idx) # dropping our current whale - we don't want to compare against an identical image!

        

        if len(candidates) == 0: # oops, there is only a single whale with this id in the dataset

            return self.sample_different(idx)

        

        np.random.shuffle(candidates)

        return  self.ds[candidates[0]][0]

    def sample_different(self, idx):

        whale_id = self.whale_ids[idx]

        candidates = list(np.where(self.whale_ids != whale_id)[0])

        np.random.shuffle(candidates)

        return self.ds[candidates[0]][0]

    

    def construct_example(self,im_A, im_B,im_C ,class_idx):

        b=[im_A,im_B,im_C],class_idx

        #print(b[0][2].shape)

        return [im_A,im_B,im_C],class_idx

classes = df.Id.unique()
#test_fnames[:5]

#data.test_ds



#data.train_ds.y.items

#data.train_ds[0]

bs=24

!ls -l
a=[1,2,3]

b=[3,4,5]

c=[]

c.append(a)

c.append(b)

np.stack(c).shape
def extract_embeddings(l,p):

    data_e = (ImageItemList.from_df(df=trn_imgs, path=p,folder='train')

         .no_split()

         .label_from_df(cols=1,classes=classes)

         #.add_test(test_fnames)

         .transform((trn_tfms,trn_tfms), size=224,resize_method=ResizeMethod.SQUISH)

         .databunch(bs=bs ))

    train_feats=[]

    # get train embeddings

    l.load('bestmodel_tnt98_1')

    l.model.eval()

    preds = torch.zeros((len(data_e.train_dl.dataset), 1024))

    

    with torch.no_grad():

        start=0

        

        for ims, t in data_e.train_dl:

        #train_feats.append(learn.model.process_features(learn.model.cnn(ims)).detach().cpu())

            #train_feats.append(l.model.head(learn.model.cnn(ims)).detach().cpu())

            #train_class_idxs.append(t)

            size=ims.shape[0]

            preds[start:start+size,:]=l.model.head(l.model.cnn(ims))

            start=start+size

    return preds,[os.path.basename(name) for name in data_e.train_ds.x.items],data_e.train_ds.xtra.Id.values
test_ids = list(sorted({fname.split('_')[0] for fname in os.listdir(path_t/'test')}))

test_fnames = [path_t/'test'/test_id for test_id in test_ids]

trn_tfms,_= get_transforms(do_flip=False, flip_vert=True, max_rotate=30., max_zoom=1.08,

                              max_lighting=0., max_warp=0. )



data = (ImageItemList.from_df(df=trn_imgs, path=path_t,folder='train')

         .random_split_by_pct(valid_pct=0.1, seed=42)

         .label_from_df(cols=1,classes=classes)

         .add_test(test_fnames)

         .transform((trn_tfms,trn_tfms), size=224,resize_method=ResizeMethod.SQUISH)

         .databunch(bs=bs ))

#data.test_ds.x.items
#data.x[0] #prints image

#data.train_dl

#len(data.x.items)

#data.xtra.Id



#data.valid_ds.x.items

#data_e.train_ds.x.items

#l=[os.path.basename(name) for name in data_e.train_ds.x.items]

#len(l)

data_e = (ImageItemList.from_df(df=trn_imgs, path=path_t,folder='train')

         .no_split()

         .label_from_df(cols=1,classes=classes)

         #.add_test(test_fnames)

         .transform((trn_tfms,trn_tfms), size=224,resize_method=ResizeMethod.SQUISH)

         .databunch(bs=bs ))

#data_e.train_ds.xtra.Id,data_e.train_ds.xtra.Image,

#trn_imgs[trn_imgs.Image=='0000e88ab.jpg']



#data_e.train_ds.x.items
#x,y=next(iter(data_bunch.train_dl))

#x[0][0]
import cv2



mean, std = torch.tensor(imagenet_stats)

class SiamImage(ItemBase):

    def __init__(self, img1, img2,img3): ## These should of Image type

        self.img1, self.img2 ,self.img3= img1, img2,img3

        #print(img1.data.shape)

        self.obj, self.data = (img1, img2,img3), [(img1.data-mean[...,None,None])/std[...,None,None]

                                                  , (img2.data-mean[...,None,None])/std[...,None,None]

                                                   , (img3.data-mean[...,None,None])/std[...,None,None]]

    def apply_tfms(self, tfms,*args, **kwargs):

        self.img1 = self.img1.apply_tfms(tfms, *args, **kwargs)

        self.img2 = self.img2.apply_tfms(tfms, *args, **kwargs)

        self.img3 = self.img3.apply_tfms(tfms, *args, **kwargs)

        self.data = [(self.img1.data-mean[...,None,None])/std[...,None,None]

                     , (self.img2.data-mean[...,None,None])/std[...,None,None]

                    , (self.img3.data-mean[...,None,None])/std[...,None,None]]

        return self

    def __repr__(self): return f'{self.__class__.__name__} {self.img1.shape, self.img2.shape}'

    def to_one(self):

        return Image(mean[...,None,None]+torch.cat(self.data,2)*std[...,None,None])

      

      

class SiamImageItemList(ImageItemList):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

#         self._label_cls=FloatList

    

    def __len__(self)->int: return len(self.items) or 1 

    

    def get(self, i):

        sz=224

        #match=1

        #if i>=len(self.items)//2:#"First set of iteration will generate similar pairs, next will generate different pairs"

            #match = 0

        fn = self.items[i]

        #print(i)

        img1 = super().get(i) # Returns Image class object

        #print(img1.shape,'img')

        #img1=np.asarray(img1)

        #img1 = PIL.Image.fromarray(read_img(fn[fn.rfind('/')+1:],bbox_df,img1) )

        #img1 = read_img(fn[fn.rfind('/')+1:],bbox_df,img1) 

        imgs = self.xtra.Image.values

        ids = self.xtra.Id.values

        wcls = ids[i]

        simgs = imgs[ids == wcls]

        dimgs = imgs[ids != wcls]

        

        if len(simgs)==1 :

            fn2=fn

        else:

        

            while True:

                np.random.shuffle(simgs)

                np.random.shuffle(dimgs)

                if simgs[0] != fn[fn.rfind('/')+1:]:

                    fn2 =simgs[0] #[simgs[0]  

                    break

        #np.random.shuffle(simgs)

            #print(fn2)

            fn2 = self.items[np.where(imgs==fn2)[0][0]]

        np.random.shuffle(dimgs)

        

        img2 = super().open(fn2) # Returns Image class object

        #img2=np.asarray(img2)

        fn3 = [dimgs[0] ]

        fn3 = self.items[np.where(imgs==fn3)[0][0]]

        img3 = super().open(fn3)

        #img2 = PIL.Image.fromarray(read_img(fn[fn.rfind('/')+1:],bbox_df,img2) )

        return SiamImage(img1, img2,img3)

    

    def reconstruct(self, t): 

      return SiamImage(mean[...,None,None]+t[0]*std[...,None,None], mean[...,None,None]+t[1]*std[...,None,None]

                      ,mean[...,None,None]+t[2]*std[...,None,None])

    

    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(9,10), **kwargs):

        rows = int(math.sqrt(len(xs)))

        fig, axs = plt.subplots(rows,rows,figsize=figsize)

        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):

            xs[i].to_one().show(ax=ax, y=ys[i], **kwargs)

        plt.tight_layout()
data1 = (SiamImageItemList.from_df(df=trn_imgs, path=path_t,folder='train')

         .random_split_by_pct(valid_pct=0.1, seed=42)

         .label_from_df(cols=1, classes=classes)

         #.add_test(test_fnames)

         .transform((trn_tfms,trn_tfms), size=224,resize_method=ResizeMethod.SQUISH)

         .databunch(bs=22,num_workers=0))
#x,y=next(iter(data_bunch.train_dl))

#(x[0][2].shape)

#x[2].shape

#is_listy(x)

#a=[x]

##a[1].shape

#len(x[0])

#xb, yb = cb_handler.on_batch_begin(x, y)



#learn.data

#y.shape

#x[2].shape

#data1.show_batch(2)

#fn=data1.items[0]

#fn[fn.rfind('/')+1:]
!pip install PyFunctional  --user

from functional import seq



from fastai.callbacks import *

from fastai.basic_train import *

from torch.autograd import Variable

from torchvision import models as m



#self.body = learner.create_body(self.arch, True, learner.cnn_config(self.arch)['cut'])

#self.head = learner.create_head(num_features_model(self.body) * 2, self.emb_sz, self.lin_ftrs, self.ps,self.bn_final)



class SiameseNetwork(nn.Module):

    def __init__(self, arch=m.densenet121):

        super().__init__() 

        #self.cnn = create_body(arch )

        body = create_body(arch,True,-1)

        h = create_head(num_features_model(body) * 2,512,[512],0.5,False)

        self.cnn=  nn.Sequential( body,h)

        self.head = nn.Linear(num_features_model(self.cnn), 1)

        

    def forward(self, im_A, im_B):

        # dl - distance layer

        #print(im_A.shape,im_B.shape)

        x1, x2 = seq(im_A, im_B).map(self.cnn).map(self.process_features)

        #dl = self.calculate_distance(x1, x2)

        #out = self.head(dl)

        return x1, x2

    def calculate_score(self,x1,x2,targs,thr=0.8):

        #sim= sim.cuda()

        #sim=nn.Linear(256, 1)

        #print(x1.shape,x2.shape,self.calculate_distance(x1, x2).shape)

        #out =  self.sim.cuda()(self.calculate_distance(x1, x2).cuda()).sigmoid_()

        out = self.head.cuda()(self.calculate_distance(x1, x2)).sigmoid_()

        #print(out)

        sc=(out>thr).int()

        #print(targs)

        #print(sc)

        score=(sc==targs.int()).float()

        #print(score.shape)

        return score.float().mean()

    def process_features(self, x): return x.reshape(*x.shape[:2], -1).max(-1)[0]

    def calculate_distance(self, x1, x2): 

        return F.pairwise_distance(x1,x2,keepdim=True)

        #return (x1 - x2).abs_()
!pip install PyFunctional  --user

from functional import seq



from fastai.callbacks import *

from fastai.basic_train import *

from torch.autograd import Variable

from torchvision import models as m



#self.body = learner.create_body(self.arch, True, learner.cnn_config(self.arch)['cut'])

#self.head = learner.create_head(num_features_model(self.body) * 2, self.emb_sz, self.lin_ftrs, self.ps,self.bn_final)



class TripleNetwork(nn.Module):

    def __init__(self, arch=m.densenet121):

        super().__init__() 

        #self.cnn = create_body(arch )

        body = create_body(arch,True,-1)

        h = create_head(num_features_model(body) * 2,512,[512],0.5,False)

        self.cnn=  nn.Sequential( body,h)

        self.head = nn.Linear(num_features_model(self.cnn), 1024)

        

    def forward(self, img_a, img_p,img_n):

        # dl - distance layer

        x1, x2,x3 = seq(img_a, img_p,img_n).map(self.cnn)#.map(self.process_features)

        #print(len(ops))

        #x1, x2,x3 = seq(ops[0][0], ops[0][1],ops[0][2]).map(self.cnn)#.map(self.process_features)

        #dl = self.calculate_distance(x1, x2)

        x1=self.head(x1)

        x2=self.head(x2)

        x3=self.head(x3)

        

        #out = self.head(dl)

        return x1, x2,x3

    def calculate_score(self,x1,x2,x3,targs,thr=0.8):

        #sim= sim.cuda()

        #sim=nn.Linear(256, 1)

        #print(x1.shape,x2.shape,self.calculate_distance(x1, x2).shape)

        #out =  self.sim.cuda()(self.calculate_distance(x1, x2).cuda()).sigmoid_()

        dp=self.calculate_distance(x1, x2).pow(2).cuda()

        dn=self.calculate_distance(x1, x3).pow(2).cuda()

        

        #out = self.head.cuda()(self.calculate_distance(x1, x2)).sigmoid_()

        #print(out)

        #sc=(out>thr).int()

        #print(targs)

        #print(sc)

        score=(dn>dp).float()

        #print(score.shape)

        return score.mean()

    def process_features(self, x): return x.reshape(*x.shape[:2], -1).max(-1)[0]

    def calculate_distance(self, x1, x2): 

        return F.pairwise_distance(x1,x2)

        #return (x1 - x2).abs_()

"""

class TripleLoss(nn.Module):

    #Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise

     

    def __init__(self, margin=5.):

        super(ContrastiveLoss, self).__init__()

        self.margin = margin

        self.wd=1e-4

    



    def forward(self, ops, label,size_average=True):

        ep = F.pairwise_distance(ops[0], ops[1])

        en = F.pairwise_distance(ops[0], ops[2])

        #loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +

        #                              (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        #label=label.float() # to overcome cuda/long mismatch issue

        #ndist =euclidean_distance*label

        #pdist =euclidean_distance*(1-label)

        losses = F.relu(ep - en + self.margin)

        return losses.mean() if size_average else losses.sum()

        #loss += self.wd*(euclidean_distance**2)

        #return loss.float().mean() # to overcome format issue

"""

        

class TripletLoss(nn.Module):

    """

    Triplet loss

    Takes embeddings of an anchor sample, a positive sample and a negative sample

    """



    def __init__(self, margin=5.):

        super(TripletLoss, self).__init__()

        self.margin = margin



    def forward(self, ops, label,size_average=True):

        #print(len(ops))

        #print(len(ops[0]))

        distance_positive = (ops[0] - ops[1]).pow(2).sum(1)  # .pow(.5)

        distance_negative = (ops[0] - ops[2]).pow(2).sum(1)  # .pow(.5)

        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean() if size_average else losses.sum()
#SiameseNetwork()

#learn.model.head

torch.log(tensor(12.0))
class ContrastiveLoss(nn.Module):

    """Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise

    """

    def __init__(self, margin=2.):

        super(ContrastiveLoss, self).__init__()

        self.margin = margin

        self.wd=1e-4

    



    def forward(self, ops, label,size_average=True):

        euclidean_distance = F.pairwise_distance(ops[0], ops[1])

        #loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +

        #                              (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        label=label.float() # to overcome cuda/long mismatch issue

        ndist =euclidean_distance*label

        pdist =euclidean_distance*(1-label)

        loss = 0.5* ((pdist**2) + (F.relu(self.margin-ndist)**2))

        #loss += self.wd*(euclidean_distance**2)

        return loss.mean() # to overcome format issue
import gc

gc.collect()

from fastai.callbacks import *

from fastai.basic_train import *

from torch.autograd import Variable



TripleLoss=TripletLoss().cuda()

#ContrastiveLoss().cuda()

#

learn = Learner(data1, TripleNetwork().cuda(), loss_func=TripleLoss, path=path1,

                metrics=[lambda preds, targs: TripleNetwork().calculate_score(preds[0],

                                                                     preds[1],preds[2] ,

                                                                     targs, thr=0.85)])

                #[lambda preds, targs: accuracy_thresh(preds.squeeze(), targs, sigmoid=False)])

#learn.split([learn.model.cnn[:6], learn.model.cnn[6:],learn.model.head])

#learn.split([learn.model.cnn[0][:6], learn.model.cnn[0][6:],learn.model.head])

learn.split([learn.model.cnn[0][0][:6], learn.model.cnn[0][0][6:],learn.model.head])

#apply_init(learn.model.cnn[1], nn.init.kaiming_normal_)

#apply_init(learn.model.head, nn.init.kaiming_normal_)



#learn.callback_fns.append(partial(GradientClipping,1))

learn.callback_fns.append(partial(SaveModelCallback,monitor='val_loss',mode='min'))

learn.callback_fns.append(partial(ReduceLROnPlateauCallback, min_delta=1e-5, patience=3))
#learn.model.cnn[0][0][:6]

#x=torch.rand(1,3,224,224).cuda()

#y=torch.rand(1,3,224,224).cuda()

#a=(x,y)

#a=SiameseNetwork().cuda().cnn(*a)

#learn.data

!mkdir models

#!cp /kaggle/input/hump-back-twoim/models/*.pth /kaggle/working/models/

#!cp /kaggle/input/triplet/*.pth /kaggle/working/models/

#!mkdir models

! cp /kaggle/input/hump-back-tn/models/*.pth /kaggle/working/models/
#learn.lr_find()



#learn.recorder.plot()



#data.show_batch(2)

#learn.recorder.plot()
#learn.recorder.losses[-1]

#learn.data.dl(data.train_ds[0])
#learn.lr_find()



#cb_handler = CallbackHandler()

#learn.recorder.plot()
#learn.save('freeze')

#!cp *.pth ./data/models

#push
#emb.reshape(-1,1024).shape

#len(names)

#names.shape

preds = torch.zeros((len(data_e.train_dl.dataset), 1024))

#preds.shape
#learn.show_results()

#xb,yb=next(iter(data_bunch.train_dl))

#xb1, yb1 = cb_handler.on_batch_begin(xb, yb)

from IPython.display import FileLink

#xb[2].shape

#data_bunch.train_dl

#push

emb,names,labels=extract_embeddings(learn,path_t)

trn_emb = pd.DataFrame({'files':names,'emb':emb.tolist(),'Id':labels})

trn_emb.emb = trn_emb.emb.map(lambda emb: ' '.join(list([str(i) for i in emb])))

trn_emb.to_csv('train_emb.csv', header=True, index=False)

#FileLink('train_emb.csv')
#trn_emb.to_csv('train_emb.csv', header=True, index=False)

#FileLink('train_emb.csv')



trn_emb.head(2)
import pandas as pd

trn_emb_df=pd.read_csv('train_emb.csv')

#emb.head(1)

#torch.from_numpy(emb.emb.values.reshape(15697,-1).astype('float16') )

#trn_emb.loc[trn_emb.Image == name,'emb'].tolist()[0]

#for i in emb.emb.values:

    #print(list(i))

    #break

trn_emb_df['emb'] = [[float(i) for i in s.split()] for s in trn_emb_df['emb']]

#trn_emb_df = trn_emb_df.set_index('files')

emb = np.array(trn_emb_df.emb.tolist())

emb=torch.from_numpy(emb) 

#torch.cat(emb.emb.values).size
trn_emb_df['seq'] = np.arange(len(trn_emb_df))

trn_emb_df = trn_emb_df.set_index('seq')

#len(trn_emb.loc[trn_emb_df.Id!='w_f48451c'].index.tolist())
#emb.size()

#trn_emb_df.set_index('seq',inplace=True)

#trn_emb_df.reset_index(drop=False)

trn_emb_df.head(2)
import gc



learn.load('bestmodel_tnt98_2')

learn.model.eval()

#%%time

sims = []

dist_dict={}

n_idx=[]

with torch.no_grad():

    

    for feat,i ,f in  zip(emb,trn_emb_df.Id.values,trn_emb_df.files.values):

        n_idx=[]

        dists = learn.model.calculate_distance(emb, feat.unsqueeze(0).repeat(15697, 1)).pow(2)

        predicted_similarity = dists.cuda()#learn.model.head(dists.cuda())

        seq_ids = trn_emb.loc[trn_emb_df.Id!=i].index.tolist()

        dist_sort,indx=torch.sort(predicted_similarity[seq_ids],descending=False)

        n_idx=[seq_ids[i] for i in indx[:90]]

        dist_dict[f]= [trn_emb_df.loc[idx,'files'] for idx in n_idx]

        gc.collect()

        #sims.append(predicted_similarity.squeeze().detach().cpu())

    #trn_emb = trn_emb.set_index('idx')

            
neg_img_pair=pd.DataFrame({'file':list(dist_dict.keys()) })

neg_img_pair['img']=list(dist_dict.values())



neg_img_pair.to_csv('neg_img_pair.csv',index=False,header=True)
#!ls -l ./data/train/
d={}

d['a']=[3,4,5,6]

d['b']=[4,5,6,7]

#import pandas as pd

#list(d.values() )

df=pd.DataFrame({'file':list(d.keys())})

#a['values']=list(d.values())

df

#import random

#random.sample(a.loc[a.file=='a','values'][0],2)[0]

#a.loc[a.file=='a','values'][0]