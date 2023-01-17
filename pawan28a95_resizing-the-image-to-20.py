import pandas as pd

import numpy as np

import os

import cv2

import matplotlib.pyplot as plt

from pathlib import Path

import time
x_size = 0.2

y_size = 0.2
os.getcwd()

os.listdir('../input/landmark-recognition-2020/')
train = pd.read_csv('../input/landmark-recognition-2020/train.csv')
start = time.time()



for d1 in os.listdir('../input/landmark-recognition-2020/train/'):

    print("d1:", d1)

    for d2 in os.listdir('../input/landmark-recognition-2020/train/'+d1+'/')[0]:

        print("d2:",d2)

        for d3 in os.listdir('../input/landmark-recognition-2020/train/'+d1+'/'+d2+'/'):

            print("d3:",d3)

            for i in os.listdir('../input/landmark-recognition-2020/train/'+d1+'/'+d2+'/'+d3+'/'):

                img = cv2.imread('../input/landmark-recognition-2020/train/'+d1+'/'+d2+'/'+d3+'/'+i)

                small = cv2.resize(img, (0,0), fx=x_size, fy=y_size)

                target = str(train[train['id'] == i.replace('.jpg','')].landmark_id.values[0])

                

                Path(str(target)).mkdir(parents=True, exist_ok=True)

                cv2.imwrite(os.path.join(target,i), small)

        

end = time.time()

print(end - start)