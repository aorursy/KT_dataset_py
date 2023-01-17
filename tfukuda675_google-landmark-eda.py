import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random

from scipy import stats

#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/landmark-retrieval-2020/train.csv")
train.head()
landmark_id_list = train.landmark_id.unique()
len(train.landmark_id.unique())
print(landmark_id_list)
path='../input/landmark-retrieval-2020/train/'

def plot_landmarks(landmark_id, df, ncolumns=4, max_figs = 96):
    landmark_df = df[df['landmark_id']==landmark_id]
    nrows = int(math.ceil(len(landmark_df)/ncolumns))
    plt.rcParams["axes.grid"] = False
    f, ax = plt.subplots(ncols=ncolumns, nrows=nrows, figsize=(int(max_figs/ncolumns), int(max_figs/ncolumns)), squeeze=False)
    f.set_size_inches(18, 6*nrows)
    
    pos = 0
    count = 0
    for i, row in landmark_df.iterrows():
        image_id =  row['id']
        img = cv2.imread(path+'/'+image_id[0]+'/'+image_id[1]+'/'+image_id[2]+'/'+image_id+'.jpg')
        img = img[:,:,::-1]
        
        col = count%ncolumns
        ax[pos,col].imshow(img)
        if col == int(ncolumns - 1):
            pos += 1
        count += 1
plot_landmarks(landmark_id=183115, df=train)
plot_landmarks(landmark_id=1, df=train)
random_id = landmark_id_list[random.randrange(len(train.landmark_id.unique()))]
print(random_id)
plot_landmarks(landmark_id=random_id, df=train)
