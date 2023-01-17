# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import sys
import subprocess
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import skimage.transform
import skimage.io
def read_images(folder):
    str1 = str(subprocess.check_output(["ls" ,"../input/flowers/flowers/"+folder]))
    str_list = str1.split("\\n")
    str_list[0] = str_list[0][2:]
    img1 = skimage.io.imread("../input/flowers/flowers/"+folder+"/"+str_list[0])
    plt.imshow(img1)
    img_np = np.reshape(skimage.transform.resize(img1,[120,120]),(1,120,120,3))
    y_np = [folder]
    for i,img in enumerate(str_list[1:-1]):
        print(i)
        try:
            img1 = skimage.io.imread("../input/flowers/flowers/"+folder+"/"+img)
            img1 = np.reshape(skimage.transform.resize(img1,[120,120]),(1,120,120,3))
            img_np = np.append(img_np,img1,axis=0)
            y_np.append(folder)
        except:
            continue
    return img_np,np.array(y_np)
img_np_daisy,y_daisy = read_images("daisy")
img_np = img_np_daisy
y_np = y_daisy

img_np_dan,y_dan = read_images("dandelion")
img_np = np.append(img_np,img_np_dan,axis=0)
y_np = np.append(y_np,y_dan)

img_np_rose,y_rose = read_images("rose")
img_np = np.append(img_np,img_np_rose,axis=0)
y_np = np.append(y_np,y_rose)

img_np_sun,y_sun = read_images("sunflower")
img_np = np.append(img_np,img_np_sun,axis=0)
y_np = np.append(y_np,y_sun)

img_np_tu,y_tu = read_images("tulip")
img_np = np.append(img_np,img_np_tu,axis=0)
y_np = np.append(y_np,y_tu)
np.save('img_np.npy',img_np)
np.save('y_np.npy',y_np)
from IPython.display import FileLink, FileLinks
FileLinks('.') 
from keras.models import Sequential

