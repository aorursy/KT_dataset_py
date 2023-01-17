import numpy as np
import pandas as pd
import os
import sys
import cv2
import glob
import fastai
import PIL
import torch
from functools import partial
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.basics import *
from fastai.vision import learner

from tqdm import tqdm


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
    # extract_faces_with_cv2(video_path, basename, seq_length=10,stride=1, output_path="cv2_faces")
    extractor.extract_faces(video_path, seq_length=10,stride=1, faces_path="faces", margin=1)

    video_filename = row["original"]
    basename, _ = basename_and_ext(video_filename)
    video_path=f'{INPUT_PATH}/test_videos/{video_filename}'
    # extract_faces_with_cv2(video_path, basename, seq_length=10,stride=1, output_path="cv2_faces")
    extractor.extract_faces(video_path, seq_length=10,stride=1, faces_path="faces", margin=1)


class TripletImageList(DeepFakeImageList):
    resize_option = 2
       # 0 - No custom resize, resizing to be done with fastai transform 
       # 1 - keep original size, center and crop if too big or pad and reflect if too small
       # 2 - center and size to fit with same aspect ratio and reflect the border

    @classmethod
    def from_df(cls, df,**kwargs):
        return cls(items=range(len(df)),inner_df=df, **kwargs)

    def get_image(self, pth):
        im = PIL.Image.open(pth)
        if self.resize_option == 1:
            im = crop_pad(im)
        elif self.resize_option == 2:
            im = size_to_fit(im)
        return im
    

    def get(self, i):
        row = self.inner_df.iloc[i]
        fake=row['fake']
        fake = mp4_to_glob(fake)
        original=row['original']
        original = mp4_to_glob(original)
        # Randomly selects one face per video
        fake_files = glob.glob(fake)
        fake1 = random.choice(fake_files)
        
        # Commented out as this notebook is using small amount of demo data
        # fake2_files = [f for f in fake_files if f != fake1 ]
        # fake2 = random.choice(fake2_files)
        fake2 = random.choice(fake_files)

        original_files = glob.glob(original)
        original = random.choice(original_files)
        
        fake1 = self.get_image(fake1)
        fake2 = self.get_image(fake2)
        original = self.get_image(original)
        im = concat(fake1, fake2, original)
        im = to_fastai(im)
        return im
def get_triplet_data(bs=4, faces_path='faces', tfms=[[],[]] ):
    unlike_df = pd.read_csv(f'../input/fakereal-pairs-in-dfdc-test-videos/dfdc_test_video_pairs.csv')
    files = os.listdir(faces_path)
    videos = [jpg_to_mp4name(f) for f in files]
    
    unlike_df = unlike_df[unlike_df['filename'].isin(videos)].copy()
    unlike_df = unlike_df[unlike_df['original'].isin(videos)].copy()
    unlike_df.rename(columns={'filename':'fake'}, inplace=True)
    unlike_df['is_valid']=False
    unlike_df.reset_index(inplace=True)
    unlike_df['is_valid'].iloc[:4] = True
    unlike_df.drop(columns=['index'], inplace=True)
    unlike_df = unlike_df.sample(frac=1)
    databunch = TripletImageList.from_df(unlike_df)\
                     .split_from_df(col='is_valid')\
                     .label_empty()\
                     .transform(tfms)\
                    .databunch(bs=bs).normalize(imagenet_stats)
    
    return databunch
tfms = get_dfdc_transforms()
data = get_triplet_data(tfms=tfms)
data.show_batch()
class TripletLoss(nn.Module):
    "Loss designed to increase difference of pairwise distance between fake-fake(anchor-positive) and fake-real(anchor-negative)"
    def __init__(self, margin=1.):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def triplet_loss(self, a,p,n,  size_average=True):
        d = nn.PairwiseDistance(p=2)
        distance = d(a, p) - d(a, n) + self.margin 
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
        return loss

    def forward(self, triple_out, target, size_average=True):
        a, p, n = triple_out[0], triple_out[1], triple_out[2]
        return self.triplet_loss(a,p,n)
    
class DapDan(Callback):
    "Reports difference of pairwise distance between fake-fake(anchor-positive) and fake-real(anchor-negative)"
    def on_epoch_begin(self, **kwargs):
        self.distances = Tensor([])
        
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        a, p, n = last_output[0], last_output[1], last_output[2]
        d = nn.PairwiseDistance(p=2)
        distance = d(a, p) - d(a, n)
        self.distances = torch.cat((self.distances, distance.squeeze(-1).cpu()))
    
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self.distances.mean())
class TripleNet(nn.Module):
    # Passes 2 fakes and 1 real through the same encoder
    def __init__(self, arch=models.resnet50, lin_ftrs=[256], emb_sz=128,ps=0., bn_final=True):
        super(TripleNet, self).__init__()
        self.arch, self.emb_sz = arch, emb_sz
        self.lin_ftrs, self.ps, self.bn_final = lin_ftrs, ps, bn_final
        self.body = learner.create_body(self.arch, True, learner.cnn_config(self.arch)['cut'])
        self.head = learner.create_head(num_features_model(self.body)*2, self.emb_sz, self.lin_ftrs, self.ps,self.bn_final)
        self.cnn = nn.Sequential(self.body, self.head)
                                  
    def trivide(self, triplet):
        n = triplet.shape[-1] // 3
        return torch.split(triplet, n, dim=-1)
        
    def forward(self, triplet):
        fake0, fake1, original = self.trivide(triplet)
        a = self.cnn(fake0)
        p = self.cnn(fake1)
        n = self.cnn(original)
        return a, p, n

    def get_embedding(self, x):
        return self.cnn(x)
# Training Phase 1
model_dir = 'models/dfdc-triplet'
os.makedirs(model_dir, exist_ok=True)
data = get_triplet_data()
model = TripleNet()

loss_func = TripletLoss()
triplet_1_learn = Learner(data,
                model,
                loss_func=loss_func,
                metrics=[DapDan()],
                model_dir=model_dir)

triplet_1_learn.fit(1)

# Training Phase 2
triplet_cnn = triplet_1_learn.model.cnn
head = bn_drop_lin(128,1,True,p=0.25)
triplet_2_net = nn.Sequential(triplet_cnn, *head)


model_dir = 'models/dfdc-triplet-2'
os.makedirs(model_dir, exist_ok=True)
data = get_deepfakeimagelist_data()


triplet_2_learn = Learner(data,
                model=triplet_2_net,
                loss_func=BCEWithLogitsFlat(),
                metrics=[DFDCAUROC(),RealLoss(),FakeLoss()],
                model_dir=model_dir)
triplet_2_learn.split( lambda m: m[1])
triplet_2_learn.unfreeze()
triplet_2_learn.freeze_to(1)
triplet_2_learn.fit(1)
