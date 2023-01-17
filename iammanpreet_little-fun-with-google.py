import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import glob
for i in glob.glob('../input/landmark-retrieval-2020/*'):
    print(i)
train_csv=pd.read_csv('../input/landmark-retrieval-2020/train.csv')
train_csv.tail()
train_csv.info()
sns.distplot(train_csv.landmark_id.unique())
sns.distplot(train_csv['landmark_id'])
train_csv[train_csv['landmark_id']==1]['id'].values
fig=plt.figure(figsize=(20,20))
length_of_matched_id=len(train_csv[train_csv['landmark_id']==203092]['id'])
ii=1
for i in train_csv[train_csv['landmark_id']==203092]['id'].values:
    img=cv2.imread('../input/landmark-retrieval-2020/train/'+i[0]+'/'+i[1]+'/'+i[2]+'/'+i+'.jpg')
    fig.add_subplot(4,2,ii)
    plt.imshow(img)
    fig.add_subplot
    ii=ii+1
    
    
