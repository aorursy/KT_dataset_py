import os

import glob



import numpy as np

import pandas as pd



from PIL import Image

import cv2

import matplotlib.pyplot as plt



metadata = pd.read_csv('../input/metadata.csv')





def image_open(dname, car_id, idx):    

    i = cv2.cvtColor(cv2.imread('../input/{}/{}_{}.jpg'.format(dname, car_id, idx)), cv2.COLOR_BGR2RGB)

    if dname == 'train':

        m = Image.open('../input/{}_masks/{}_{}_mask.gif'.format(dname, car_id, idx))

        m = np.array(m.getdata()).reshape(i.shape[:2])

        for c in range(3):

            i[:,:,c] = np.bitwise_and(i[:,:,c], 255*m)

    return i



    

def display(dname, car_id):

    plt.figure(figsize=(20, 10))

    plt.imshow(np.hstack([image_open(dname, car_id, str(x).zfill(2)) for x in range(1, 8)]))

    

    plt.figure(figsize=(20, 10))

    plt.imshow(np.hstack([image_open(dname, car_id, str(x).zfill(2)) for x in range(8, 17)]))



train_ids = set([os.path.basename(x).split('_')[0] for x in glob.glob('../input/train/*jpg')])

print('train ids: ', len(train_ids))



example = list(train_ids)[0]

display('train', example)



test_ids = set([os.path.basename(x).split('_')[0] for x in glob.glob('../input/test/*jpg')])

print('test ids: ', len(test_ids))



example = list(test_ids)[0]

display('test', example)

  
example = list(test_ids)[1]

display('test', example)