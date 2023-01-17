

import os

import sys

import numpy as np

import pandas as pd

import cv2



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

from torch.utils.data.sampler import *

print('torch version:', torch.__version__)



import warnings

warnings.filterwarnings('ignore')
os.listdir('../input/hengmodels')
DATA_DIR = '../input/bengaliai-cv19'

SUBMISSION_CSV_FILE = 'submission.csv'

MYFILE_DIR = '../input/myfile'
sys.path.append(MYFILE_DIR)

from etc import *

from densenet_model import Net as DenseNet


def do_predict(net, input):



    def logit_to_probability(logit):

        probability=[]

        for l in logit:

            p = F.softmax(l,1)

            probability.append(p)

        return probability



    #-----

    num_ensemble = len(net)

    for i in range(num_ensemble):

        net[i].eval()





    probability=[0,0,0]

    #----

    for i in range(num_ensemble):

        logit = net[i](input)

        prob  = logit_to_probability(logit)

        probability = [p+q for p,q in zip(probability,prob)]



    #----

    probability = [p/num_ensemble for p in probability]

    predict = [torch.argmax(p,-1) for p in probability]

    predict = [p.data.cpu().numpy() for p in predict]

    predict = np.array(predict).T

    predict = predict.reshape(-1)



    return predict
DESENET_CHECKPOINT_FILE = [

        '../input/hengmodels/swa_fold1_no_bn_model.pth',

    ]

TASK_NAME = [ 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic' ]


## load net -----------------------------------

net = []

for checkpoint_file in DESENET_CHECKPOINT_FILE:

    n = DenseNet().cuda()

    n.load_state_dict(torch.load(checkpoint_file, map_location=lambda storage, loc: storage),strict=True)

    net.append(n)


def run_make_submission_csv():



    row_id=[]

    target=[]

    batch_size= 32



    print('\nstart here !!!!')

    for i in range(4):

        start_timer = timer()

        df  = pd.read_parquet(DATA_DIR+'/test_image_data_%d.parquet'%i, engine='pyarrow')

        #df  = pd.read_parquet(DATA_DIR+'/train_image_data_%d.parquet'%i, engine='pyarrow') #use this to test timing

        print('pd.read_parquet() = %s'%(time_to_str((timer() - start_timer),'sec')))



        start_timer = timer()

        num_test = len(df)

        for b in range(0,num_test,batch_size):

            if b%1000==0:

                print('test_image_data_%d.parquet @%06d, %s'%(i,b,time_to_str((timer() - start_timer),'sec')))

            #----

            B = min(num_test,b+batch_size)-b

            image = df.iloc[b:b+B, range(1,32332+1)].values

            image_id = df.iloc[b:b+B, 0].values



            image = image.reshape(B,1,137, 236)

            image = np.tile(image, (1,3,1,1))

            image = image.astype(np.float32)/255



            #----

            input = torch.from_numpy(image).float().cuda()

            predict = do_predict(net, input)

            #----



            image_id = np.tile(image_id.reshape(B,1), (1,3,)) + ['_']  + TASK_NAME

            image_id = image_id.reshape(-1)

            row_id.append(image_id)

            target.append(predict)

        print('')



    row_id = np.concatenate(row_id)

    target = np.concatenate(target)

    #---------



    df = pd.DataFrame(zip(row_id, target), columns=['row_id', 'target'])

    df.to_csv(SUBMISSION_CSV_FILE, index=False)

run_make_submission_csv()