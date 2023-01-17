# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        pass

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tqdm import tqdm
import re
import csv
import glob
import torch
#import tensorflow as tf
#import keras.backend as K
import concurrent.futures
import numpy as np
import os
from skimage.filters import threshold_otsu
import random
import time
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
try:
    from skimage.io import imsave as imwrite
    from skimage.io import imread
except ImportError:
    from cv2 import imread
    from cv2 import imwrite

import openslide as ops
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models
import numpy as np
import torch.nn as nn
from torchvision.transforms import transforms,Lambda
import torchvision.transforms.functional as TF
import random
import PIL
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
INPUT_DIR = '/kaggle/input/prostate-cancer-grade-assessment/'
train_csv = f'{INPUT_DIR}/train.csv'
test_csv = f'{INPUT_DIR}/test.csv'
sample_sub = f'{INPUT_DIR}/sample_submission.csv'
test_img_dir = f'{INPUT_DIR}/test_images'
train_img_dir = f'{INPUT_DIR}/train_images'
#model_weight_path = '/kaggle/input/approch-3-epoch-1/run_1_checkpoint_best_1.pth'
model_weight_path = '/kaggle/input/approach3epoch3/run_1_256_checkpoint_best_3.pth'
sample_df = pd.read_csv(sample_sub)
sample_df.head(10)
test_df = pd.read_csv(test_csv)
for imgId in test_df['image_id']:
    print(imgId)
train_df = pd.read_csv(train_csv,index_col ="image_id")

train_df.head(10)
class PANDAInferenceDataset():
    def __init__(self,imageName,
                 wsi_level_to_read_patch=0,
                 wsi_level_to_process_wsi=1,
                 patch_size=256,
                 transform_inference = None,
                 plot = True,
                 test_img_dir=None):
        
        self.imageName = imageName
        self.wsi_level_to_read_patch = wsi_level_to_read_patch
        self.wsi_level_to_process_wsi = wsi_level_to_process_wsi
        self.patch_size = patch_size
        self.test_img_dir = test_img_dir
        self.area_thd = 0.5
        self.transforms = transform_inference
        self.plot = plot
        
        self.yxfmtLevel0 = None
        self.get_locations()
    
    def get_locations(self):
        filepath = os.path.join(self.test_img_dir,f'{self.imageName}.tiff')
        self.wsiObj = ops.open_slide(filepath)
        tissueMask = self.generate_tissue_mask(is_otsu = True)
        
        self.scale_factor_est = self.wsiObj.level_dimensions[0][0] //self.wsiObj.level_dimensions[self.wsi_level_to_process_wsi][0]
        stride = self.patch_size // self.scale_factor_est
        yxpoints = self.get_xy_points(tissueMask, stride=stride)
        cord_factor = self.scale_factor_est
        self.yxfmtLevel0 = [(y * cord_factor, x * cord_factor) for y, x in yxpoints]
        #print('Total-points ',len(self.yxfmtLevel0))
        if self.plot: 
            plt.figure(1)
            plt.imshow(tissueMask)
            H, W = tissueMask.shape
            #cv2.imwrite(mask_path, np.uint8(tissueMask) * 255)
            yxfmt = [(y//cord_factor, x//cord_factor) for y, x in  self.yxfmtLevel0]
            if True:
                img_box = np.zeros([H, W, 3], np.uint8)
                img_box[:, :, 0] = np.uint8(tissueMask) * 255
                img_box[:, :, 1] = np.uint8(tissueMask) * 255
                img_box[:, :, 2] = np.uint8(tissueMask) * 255
                for y, x in yxfmt:
                    cv2.rectangle(img_box, (x, y), (x + stride, y + stride), (0, 0, 255), 2)

                #mask_path = os.path.join(self.masksaveFolder, f'{filename}_mask_level_{wsiLevel}_box.png')
                #cv2.imwrite(mask_path, img_box)
            plt.figure(figsize=(15,9))
            plt.imshow(img_box)
            
        
        #stride = self.patch_size // (2**self.wsi_level_to_process_wsi)
        #yxpoints = self.get_xy_points(tissueMask, stride=stride)
        #cord_factor = 2 ** (self.wsi_level_to_process_wsi)
        #self.yxfmtLevel0 = [(y * cord_factor, x * cord_factor) for y, x in yxpoints]
        
    def generate_tissue_mask(self,is_otsu = True):
        grayData = np.array(self.wsiObj.read_region(size=self.wsiObj.level_dimensions[self.wsi_level_to_process_wsi],
                                               location=(0, 0), level=self.wsi_level_to_process_wsi).convert('L'))
        # grayData = cv2.cvtColor(tissue_at_level, cv2.COLOR_BGR2GRAY)
        if is_otsu:
            try:
                thd = threshold_otsu(grayData)
            except:
                thd = 255
            tissueMask = grayData < thd
        return tissueMask
    
    def get_xy_points(self,maskImg,stride):
        H,W = maskImg.shape
        area_thd = stride * stride * self.area_thd
        yxpairList = []
        if True:
            x_grid = np.arange(0, W , stride)
            y_grid = np.arange(0, H , stride)
            for y in y_grid:
                for x in x_grid:
                    if np.sum(maskImg[y:y + stride, x:x + stride]) > area_thd:
                        yxpairList.append((y,x))
        return yxpairList
    
    def __len__(self):
        return len(self.yxfmtLevel0)
    
    def __getitem__(self, index):
        Y, X = self.yxfmtLevel0[index]
        imgPatch = self.wsiObj.read_region(location=(X, Y),
                                           level=self.wsi_level_to_read_patch,
                                            size=(self.patch_size, self.patch_size)).convert('RGB')

        # already a PIL Image
        imgPatch = np.asarray(imgPatch)

        if self.transforms is not None:
            imgPatch = self.transforms(imgPatch)
        
        else:
            raise print('Give valid transform')
        
        return imgPatch
inferenceTransfom = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])])
#Get model
num_class = 4
model = models.resnet50(pretrained=False)
if False:
    wgtdict = torch.load(modelWeightsPath)
    model.load_state_dict(wgtdict, False)
    model.fc = nn.Linear(model.fc.in_features, num_class)
    print(model)
else:
    model.fc = nn.Linear(model.fc.in_features, num_class)
    wgtdict = torch.load(model_weight_path)
    model.load_state_dict(wgtdict['state_dict'], True)
def Inference(model,dataloder,dataLength,batch_size,device,debug=False,folder_path = './'):
    nLength = dataLength
    result_dict = {'0':0,'1':0,'2':0,'3':0}
    if debug:
        probs = torch.cuda.FloatTensor(nLength,self.num_classes) if torch.cuda.is_available() else torch.FloatTensor(nLength,self.num_classes)
    else:
        probs = torch.cuda.FloatTensor(nLength,2) if torch.cuda.is_available() else torch.FloatTensor(nLength,2)
    with torch.no_grad():
        for batchIdx, imgdata in enumerate(dataloder):
            inputs = imgdata.float().to(device)
            outputs = model(inputs)
            predict_proba = outputs.softmax(dim=1)
            predProbMax, predictID = torch.max(predict_proba.data, 1)
            #print('Valid-Inference\tEpoch: [{}/{}]\tPatches: [{}/{}]'.format(epoch, self.num_epoch, self.batch_size*(batchIdx + 1), nLength))
            #output = F.softmax(model(input), dim=1)
            #print(inputs.size(0),predict_proba.size(0))
            if debug:
                probs[batchIdx * batch_size:batchIdx * batch_size + inputs.size(0),:] = predict_proba.detach()[0:inputs.size(0), :].clone()
            else:
                probs[batchIdx * batch_size:batchIdx * batch_size + inputs.size(0),0] = predProbMax.detach()[0:inputs.size(0)].clone()
                probs[batchIdx * batch_size:batchIdx * batch_size + inputs.size(0),1] = predictID.detach()[0:inputs.size(0)].clone()
        return probs.cpu().numpy()
label_to_gleason_score = {'0':0,'1':3,'2':4,'3':5} #0==negative
gleason_to_isup_score = {'3+3':1,'3+4':2,'4+3':3,'4+4':4,'3+5':4,'5+3':4,'4+5':5,'5+4':5,'5+5':5} #otherwise =0
def get_isup_score(prob_freq):
    g0 = prob_freq[0]
    g3 = prob_freq[1]
    g4 = prob_freq[2]
    g5 = prob_freq[3]
    if g3==0 and g4==0 and g5==0:
        glason_score = '0+0'
        try:
            isup_score =  gleason_to_isup_score[glason_score]
        except:
            isup_score = 0
    elif g3!=0 and g4==0 and g5==0:
        glason_score = '3+3'
        try:
            isup_score =  gleason_to_isup_score[glason_score]
        except:
            isup_score = 0
    elif g3==0 and g4!=0 and g5==0:
        glason_score = '4+4'
        try:
            isup_score =  gleason_to_isup_score[glason_score]
        except:
            isup_score = 0
    elif g3==0 and g4==0 and g5!=0:
        glason_score = '5+5'
        try:
            isup_score =  gleason_to_isup_score[glason_score]
        except:
            isup_score = 0
    else:
        #case of 2 non -zero or all non zeros
        tumors = prob_freq[1:]
        max_idxs = np.argsort(tumors)
        max_idxs = max_idxs+1
        gleason_1 = label_to_gleason_score[str(max_idxs[-1])] 
        gleason_2 = label_to_gleason_score[str(max_idxs[-2])] 
        glason_score = f'{gleason_1}+{gleason_2}'
        try:
            isup_score = gleason_to_isup_score[glason_score]
        except:
            isup_score = 0
    
    return isup_score,glason_score
test_df = pd.read_csv(test_csv,index_col ="image_id")
batch_size = 32
num_worker = 4
pathc_size = 256
plot = False
#print(test_df.shape)
#plt.figure(figsize=(15,9))
use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
model.to(device)
model.eval()
results = []
if os.path.exists(test_img_dir):
    for imgId in test_df.index:#['image_id'][:10]:
        dsInfer = PANDAInferenceDataset(imageName=imgId,
                                        wsi_level_to_read_patch=0,
                                        wsi_level_to_process_wsi=1,
                                        patch_size=pathc_size,
                                        plot = plot,
                                        transform_inference = inferenceTransfom,
                                        test_img_dir=test_img_dir)
        dlInfer = DataLoader(dsInfer,
                             batch_size=batch_size,
                             num_workers= num_worker,
                             pin_memory=False, 
                             shuffle=False,
                             drop_last=False)
        dataLength = dsInfer.__len__()
        prob_info = Inference(model=model,
                              dataloder=dlInfer,
                              dataLength=dataLength,
                              batch_size=batch_size,
                              device=device,
                              debug=False,
                              folder_path = './')
        yx = np.where(prob_info[:,0] > 0.90)
        filter_probs = prob_info[yx]
        #nd = np.lexsort((filter_probs[:,1],filter_probs[:,0])) 
        #print(filter_probs[nd,0],filter_probs[nd,1])
        #need to decide fiter on hightest prob or highers number
        prob_freq = np.bincount(filter_probs[:,1].astype(np.int64),minlength=4)
    #     max_idxs = np.argsort(prob_freq)
    #     gleason_1 = label_to_gleason_score[str(max_idxs[-1])] if prob_freq[max_idxs[-1]] > 0 else 0
    #     gleason_2 = label_to_gleason_score[str(max_idxs[-2])] if prob_freq[max_idxs[-2]] > 0 else gleason_1
    #     isup_key = f'{gleason_1}+{gleason_2}'
    #     try:
    #         score = gleason_to_isup_score[isup_key]
    #     except:
    #         score = 0
        isup,gleason = get_isup_score(prob_freq)
        #print(imgId,train_df.loc[imgId]['isup_grade'],train_df.loc[imgId]['gleason_score'],isup,gleason,prob_freq)
        result = {
                'image_id': imgId,
                'isup_grade': isup}
        results.append(result)

    

test_df = pd.DataFrame(results, columns=['image_id', 'isup_grade'])
test_df.head(10)
test_df.to_csv('submission.csv', index=False)
if False:
    img = next(iter(dlInfer))
    print(img.shape)
    img1 = np.uint8((img[10,...].permute(1,2,0).cpu().numpy()+0.5)*255)
    for i in range(5):
        img1 = np.uint8((img[i,...].permute(1,2,0).cpu().numpy()+0.5)*255)
        plt.imshow(img1)
        plt.show()



