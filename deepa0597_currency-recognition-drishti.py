# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import pandas as panda

TRAIN_DIR= '../input/testing-currency/testing/Testing/'
TEST_DIR='../input/currency-test/'
IMG_SIZE=50
LR = 1e-3

MODEL_NAME = 'currency-{}-{}.model'.format(LR, '2conv-basic')


def label_img(img):
    word_label = img.split('.')[-6]
    if word_label == 'fifty': return [1,0,0,0]
    elif word_label == 'hundred': return [0,1,0,0]
    elif word_label == 'ten': return [0,0,1,0]
    elif word_label == 'twenty': return [0,0,0,1]
    
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
