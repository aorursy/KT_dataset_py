import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt 
import scipy.io as io
from os import listdir
from os.path import join, isfile

# Any results you write to the current directory are saved as output.
path_img = '/kaggle/input/train2/train2/img2'
path_dmap = '/kaggle/input/train2/train2/dmap2'
path_save = '/kaggle/output/'

li_img = [f for f in listdir(path_img) if isfile(join(path_img,f))]
print(len(li_img))

img = cv2.imread(path_img+'/'+li_img[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(path_save+'test.jpg', img)
plt.imshow(img)
plt.show()
cd /kaggle/output
ls
