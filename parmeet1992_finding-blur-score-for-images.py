# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
!pwd
# Any results you write to the current directory are saved as output.
!ls ../input
import pandas as pd
from sklearn import preprocessing
import os
import cv2
from PIL import Image
import pickle
dir_num = '../input' 
import numpy as np
def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    image = np.array(image) 
    return cv2.Laplacian(image, cv2.CV_64F).var()
list_of_images = os.listdir(dir_num)
print(list_of_images)
mapping = {}
def function1(file):
    dir_num = '../input'
    file_name = file.split(".")[0]
    fm = 0
    if file.split(".")[1]!='jpg':
        return fm
    try:
        gray = Image.open(os.path.abspath('/kaggle/working/'+dir_num+"/"+file)).convert('LA')
        fm = variance_of_laplacian(gray)
        #print(fm)
        #mapping[file_name] = fm 
    except:
        print("Error out")
    return (file,fm)
import multiprocessing
import sys
pool = multiprocessing.Pool()
#blur_list = pool.map(function1,list_of_images)
blur_list = []
for i, output in enumerate(pool.imap_unordered(function1,list_of_images), 1):
    sys.stderr.write('\rdone {0:%}'.format(i/len(list_of_images)))
    blur_list.append(output)
print(blur_list)
# for idx,file in enumerate(list_of_images):
#     print(idx)
#     file_name = file.split(".")[0]
#     if file.split(".")[1]!='jpg':
#         continue
#     try:
#         gray = Image.open(os.path.abspath('../'+dir_num+"/"+file)).convert('LA')
#         fm = variance_of_laplacian(gray)
#         print(fm)
#         mapping[file_name] = fm 
#     except:
#         print("Error out")
