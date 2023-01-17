# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/lits17/lits/LiTS"))



# Any results you write to the current directory are saved as output.
import cv2

from scipy import ndimage



path = r"../input/lits17/lits/LiTS/mask"

for filename in os.listdir(path):

    if filename.endswith('.png'):

        print(filename)

        oriimg = cv2.imread(path+'/'+filename)

        newX,newY = 400,400

        newimg = cv2.resize(oriimg,(int(newX),int(newY)))

        cv2.imwrite(str(filename),newimg)        

        print('Image saved')
path = r"../input/brats17/2017/2017/training/mask"

for filename in os.listdir(path):

    if filename.endswith('.png'):

        print(filename)

        oriimg = cv2.imread(path+'/'+filename)

        newX,newY = 400,400

        newimg = cv2.resize(oriimg,(int(newX),int(newY)))

        new = newimg[::-1, :]

        cv2.imwrite(str(filename+"-flip"),new)        

        print('Image saved') 
"""import cv2

from skimage import util

path = r"../input/brats17/2017/2017/training/mask"

for filename in os.listdir(path):

    if filename.endswith('.png'):

        print(filename)

        oriimg = cv2.imread(path+'/'+filename)

        newX,newY = 400,400

        newimg = cv2.resize(oriimg,(int(newX),int(newY)))

        new = util.invert(newimg)

        cv2.imwrite(str(filename+"-Sinversion"),new)        

        print('Image saved')    """               
"""import cv2

from skimage import exposure

path = r"../input/brats17/2017/2017/training/mask"

for filename in os.listdir(path):

    if filename.endswith('.png'):

        print(filename)

        oriimg = cv2.imread(path+'/'+filename)

        newX,newY = 400,400

        newimg = cv2.resize(oriimg,(int(newX),int(newY)))

        new = exposure.adjust_gamma(newimg, gamma=0.4, gain=0.9)

        cv2.imwrite(str(filename+"-gamma"),new)        

        print('Image saved')"""
"""import cv2

import numpy as np

from skimage import exposure

path = r"../input/brats17/2017/2017/training/mask"

for filename in os.listdir(path):

    if filename.endswith('.png'):

        print(filename)

        oriimg = cv2.imread(path+'/'+filename)

        newX,newY = 400,400

        newimg = cv2.resize(oriimg,(int(newX),int(newY)))

        v_min, v_max = np.percentile(newimg, (0.2, 99.8))

        new = exposure.rescale_intensity(newimg, in_range=(v_min, v_max))

        cv2.imwrite(str(filename+"-cont"),new)        

        print('Image saved')    """

"""import numpy as np

path = r"../input/brats17/2017/2017/training/mask"

for filename in os.listdir(path):

    if filename.endswith('.png'):

        print(filename)

        oriimg = cv2.imread(path+'/'+filename)

        newX,newY = 400,400

        newimg = cv2.resize(oriimg,(int(newX),int(newY)))

        cv2.imwrite(str(filename+"-resized"),newimg)        

        print('Image saved')    """
