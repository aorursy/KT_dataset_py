import child_model as CM

import utils as U

import dataa as D

import endgame as E 



import torch

import torch.optim as optim

import torch.nn.functional as F



import tarfile



import os

# extract CIFAR10 dataset

tar = tarfile.open("../input/cifar10-python/cifar-10-python.tar.gz", "r:gz")

tar.extractall()

tar.close()
BATCH_SIZE = 100

TEST_BATCH_SIZE = 200

cifar10 = D.CIFAR10(batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, path="./")
folder = '../input/enas-winner-checkpoints/'

files = os.listdir( folder ) 

print('Files:')

for f in files:

    print(f)



checkpoint = folder + 'enas_feedback_run1_2019_9_27_5_40_51.pt'

#checkpoint = torch.load( checkpoint )
#child = E.pick_best_from_checkpoint(checkpoint=checkpoint, weight_init='nudge')