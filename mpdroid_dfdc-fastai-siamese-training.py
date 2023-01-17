import numpy as np
import pandas as pd
import os
import sys
import cv2
import glob
import fastai
import PIL
import torch
import glob
from functools import partial
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.basics import *
from fastai.vision import learner

from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms


tqdm.pandas()

INPUT_PATH = "../input/deepfake-detection-challenge"
VERBOSE = True
EPS = 1e-5
RUN_NOTEBOOK=False
FACES_PATH = 'faces'

os.makedirs(FACES_PATH, exist_ok=True)
from dfdc_face_extractor import *
from dfdc_fastai_reusables import *
pair_df = pd.read_csv(f'../input/fakereal-pairs-in-dfdc-test-videos/dfdc_test_video_pairs.csv')

extractor = DFDCVideoFaceExtractor(backend='CV2')
pair_df = pair_df[:20]
for index, row in tqdm(pair_df.iterrows(), total=len(pair_df)):
    video_filename = row["filename"]
    basename, _ = basename_and_ext(video_filename)
    video_path=f'{INPUT_PATH}/test_videos/{video_filename}'
    extractor.extract_faces(video_path, seq_length=10,stride=1, faces_path="faces", margin=1)

    video_filename = row["original"]
    basename, _ = basename_and_ext(video_filename)
    video_path=f'{INPUT_PATH}/test_videos/{video_filename}'
    extractor.extract_faces(video_path, seq_length=10,stride=1, faces_path="faces", margin=1)



mean, std = torch.tensor(imagenet_stats)

class SiamesePair(ItemBase):
    def __init__(self, img1, img2): ## These should of Image type
        self.img1, self.img2 = img1, img2
        self.obj, self.data = (img1, img2), [
            (img1.data-mean[...,None,None])/std[...,None,None],
            (img2.data-mean[...,None,None])/std[...,None,None]
        ]
    def apply_tfms(self, tfms,*args, **kwargs):
        self.img1 = self.img1.apply_tfms(tfms, *args, **kwargs)
        self.img2 = self.img2.apply_tfms(tfms, *args, **kwargs)
        self.data = [(self.img1.data-mean[...,None,None])/std[...,None,None],
                     (self.img2.data-mean[...,None,None])/std[...,None,None]
                    ]
        return self
    def __repr__(self): return f'{self.__class__.__name__} {self.img1.shape, self.img2.shape}'
    def to_one(self):
        return Image(mean[...,None,None]+torch.cat(self.data,-1)*std[...,None,None])

normalize = partial(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))

denormalize = partial(transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
))
class SiameseImageList(ImageList):
    @classmethod
    def from_df(cls, df,**kwargs):
        return cls(items=range(len(df)),inner_df=df, **kwargs)

    def to_img(self, path):
        m = Image.open(path)
        return m
        
    def get(self, i):
        row = self.inner_df.iloc[i]
        first = row['first']
        first = mp4_to_glob(first)
        # Randomly selects one face per video
        first_files = glob.glob(first)
        first = random.choice(first_files)
        
        second=row['second']
        second = mp4_to_glob(second)
        second_files = glob.glob(second)
        # Do not include first file if both sets are from the same fake video
        # second_files = [f for f in second_files if f != first ]
        second = random.choice(second_files)
        first = super().open(first)
        second = super().open(second)
        return SiamesePair(first, second)

    def reconstruct(self, t):
        return SiamesePair(mean[...,None,None]+t[0]*std[...,None,None], 
                            mean[...,None,None]+t[1]*std[...,None,None])
    
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(9,10), **kwargs):
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, y=ys[i], **kwargs)
        plt.tight_layout()

def get_siamese_data(bs=4, faces_path='faces', tfms=[[],[]] ):
    unlike_df = pd.read_csv(f'../input/fakereal-pairs-in-dfdc-test-videos/dfdc_test_video_pairs.csv')
    files = os.listdir(faces_path)
    videos = [jpg_to_mp4name(f) for f in files]
    
    unlike_df = unlike_df[unlike_df['filename'].isin(videos)].copy()
    unlike_df = unlike_df[unlike_df['original'].isin(videos)].copy()
    unlike_df.rename(columns={'filename':'first', 'original': 'second'}, inplace=True)
    unlike_df['label']=1
    unlike_df['is_valid']=False
    unlike_df['is_valid'].iloc[:5] = True

    like_df = unlike_df.copy()
    like_df['second']=like_df['first']
    like_df['label']=0
    
    siamese_df=pd.concat([unlike_df, like_df], axis=0)
    siamese_df.reset_index(inplace=True)
    siamese_df.drop(columns=['index'], inplace=True)
    siamese_df = siamese_df.sample(frac=1)
    databunch = SiameseImageList.from_df(siamese_df)\
                     .split_from_df(col='is_valid')\
                     .label_from_df(cols='label')\
                     .transform(tfms, size=(224,224))\
                    .databunch(bs=4)
    
    return databunch

tfms = get_dfdc_transforms()
data = get_siamese_data(tfms=tfms)
data.show_batch()
class SiameseLoss(nn.Module):
    """
    Custom loss designed to decrease distance between predictions for like pairs and\
    increase distance between predictions for unlike pairs.
    """
    def __init__(self, margin=5, eps=1e-3):
        super(SiameseLoss, self).__init__()
        self.margin=margin
        self.eps=eps
        
    def bce_loss(self, p,n, target, size_average=True):
        ps = torch.sigmoid(p)[:,-1]
        pt = torch.ones_like(p)
        ns = torch.sigmoid(n)[:,-1]
        nt = target.unsqueeze(-1)
        logps = torch.log(ps)
        logns = torch.log(1-ns)
        d1 = -pt*logps
        d2 = -(1-nt)*logns
        d = d1+d2
        return d.mean() if size_average else d.sum()
        
    def contrastive_loss(self, p,n, target, size_average=True):
        euclidean_distance = F.pairwise_distance(p, n, keepdim = True)
        tgt = target.unsqueeze(-1).float()
        term1 = (tgt) * torch.pow(euclidean_distance, 2)
        term2 = (1-tgt) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        t = torch.cat((tgt, euclidean_distance, term1, term2),dim=1)
        loss_contrastive = torch.mean(term1 + term2)
        return loss_contrastive
    
    def forward(self, siamese_out, target, size_average=True):
        p, n = siamese_out[0], siamese_out[1]
        return self.contrastive_loss(p,n, target, size_average=size_average)

class SiameseNet(nn.Module):
    def __init__(self, arch=models.resnet50, lin_ftrs=[256], emb_sz=128,ps=0., bn_final=True):
        super(SiameseNet, self).__init__()
        self.arch, self.emb_sz = arch, emb_sz
        self.lin_ftrs, self.ps, self.bn_final = lin_ftrs, ps, bn_final
        self.body = learner.create_body(self.arch, True, learner.cnn_config(self.arch)['cut'])
        self.head = learner.create_head(num_features_model(self.body)*2, self.emb_sz, self.lin_ftrs, self.ps,self.bn_final)
        self.cnn = nn.Sequential(self.body, self.head)
                                  
    def forward(self, fake, original):
        p = self.cnn(fake)
        n = self.cnn(original)
        return p, n

    def get_embedding(self, x):
        return self.cnn(x)
# Training Phase 1
model_dir = 'models/dfdc-siamese'
os.makedirs(model_dir, exist_ok=True)
data = get_siamese_data()
model = SiameseNet()

loss_func = SiameseLoss()
siamese_1_learn = Learner(data,
                model,
                loss_func=loss_func,
                model_dir=model_dir)

siamese_1_learn.fit(1)


# Training Phase 2
siamese_cnn = siamese_1_learn.model.cnn
head = bn_drop_lin(128,1,True,p=0.25)
siamese_2_net = nn.Sequential(siamese_cnn, *head)


model_dir = 'models/dfdc-siamese-2'
os.makedirs(model_dir, exist_ok=True)
data = get_deepfakeimagelist_data()


siamese_2_learn = Learner(data,
                model=siamese_2_net,
                loss_func=BCEWithLogitsFlat(),
                metrics=[DFDCAUROC(),RealLoss(),FakeLoss()],
                model_dir=model_dir)
siamese_2_learn.split( lambda m: m[1])
siamese_2_learn.unfreeze()
siamese_2_learn.freeze_to(1)
siamese_2_learn.fit(1)


