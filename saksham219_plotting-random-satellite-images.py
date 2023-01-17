import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import cv2
print(os.listdir("../input"))
labels = pd.read_csv('../input/train_v2.csv')
print('training imagees: ', labels.shape[0])
labels.head()
fig,ax = plt.subplots(3,3, figsize = (20,20))
img_path_base = '../input/train-jpg/train_{}.jpg'
for i,j in enumerate(np.random.randint(0,labels.shape[0],9)):
    image_path = img_path_base.format(str(j))
    img = cv2.imread(image_path)
    ax[int(i/3),i%3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[int(i/3),i%3].set_title(labels.iloc[j,1])
