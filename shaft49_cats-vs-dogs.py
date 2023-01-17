import shutil
import os
from PIL import Image
def make_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
import zipfile


DATASET_PATH = '../output/dogs-vs-cats'

make_directory(DATASET_PATH)

# extract train data
with zipfile.ZipFile('../input/dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall(DATASET_PATH)

# extract test data
with zipfile.ZipFile('../input/dogs-vs-cats/test1.zip', 'r') as zip_ref:
    zip_ref.extractall(DATASET_PATH)

TRAIN_DATA_PATH = os.path.sep.join([DATASET_PATH, 'train'])
TEST_DATA_PATH = os.path.sep.join([DATASET_PATH, 'test1'])
train_files = os.listdir(TRAIN_DATA_PATH)
test_files = os.listdir(TEST_DATA_PATH)

print(len(train_files), len(test_files))
Image.open(os.path.sep.join([TRAIN_DATA_PATH, train_files[0]])) 
Image.open(os.path.sep.join([TEST_DATA_PATH, test_files[0]])) 
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
class CatsVsDogsDataset(Dataset):
    def __init__(self):
        pass
    
    def __getitem(self):
        pass
    
    def __len__(self):
        pass