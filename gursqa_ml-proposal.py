

import numpy as np 
import pandas as pd

import os
print(os.listdir('../input/chest-xray-pneumonia/chest_xray/'))

train_dir = '../input/chest-xray-pneumonia/chest_xray/train'
test_dir = '../input/chest-xray-pneumonia/chest_xray/test'
valDir = '../input/chest-xray-pneumonia/chest_xray/val'


def make_data(DIR):
    print(os.listdir(DIR))
    
make_data(train_dir)
    