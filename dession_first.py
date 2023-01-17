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
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import multiprocessing
import cv2, glob

from skimage.segmentation import mark_boundaries
%matplotlib inline
train_label = glob.glob('../input/train_label/**')
train = pd.DataFrame(glob.glob('../input/train_color/**'), columns=['Path'])
train['LabelPath'] = train['Path'].map(lambda x: str(x).replace('.jpg','_instanceIds.png').replace('train_color','train_label'))
from IPython.display import display

im = Image.open(train.Path[0])
tlabel = (np.asarray(Image.open(train.LabelPath[0])) / 1000).astype('uint8')

plt.figure(figsize=(20,20))
plt.subplot(151)
plt.imshow(im)

plt.subplot(152)
tlabel[tlabel != 0] = 255
plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.5))

plt.subplot(153)
bound_img = mark_boundaries(image = im, label_img = tlabel, color = (1,0,0), background_label = 255, mode = 'thick')
plt.imshow(bound_img)

plt.subplot(154)
xd, yd = np.where(tlabel>0)
plt.imshow(bound_img[xd.min():xd.max(), yd.min():yd.max(),:])



# display(plt.show())
submission = pd.DataFrame()
submission['ImageId'] = 1
submission['LabelId'] = 33
submission['Confidence'] = 1
submission['PixelCount'] = 300
submission['EncodedPixels'] = 1
submission.to_csv('1.csv', index=False)
submission.head(5)
from IPython.display import FileLink
#%cd $LESSON_HOME_DIR
FileLink('1.csv')