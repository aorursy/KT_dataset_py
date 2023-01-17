import sys

import os

import subprocess

from six import string_types



# Make sure you have all of these packages installed, e.g. via pip

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import scipy

from skimage import io

from scipy import ndimage

from IPython.display import display

%matplotlib inline
PLANET_KAGGLE_ROOT = os.path.abspath("../input/")

PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')

PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train.csv')

assert os.path.exists(PLANET_KAGGLE_ROOT)

assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)

assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)
def load_image(filename):

    for dirname in os.listdir(PLANET_KAGGLE_ROOT):

        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))

        if os.path.exists(path):

            print('Found image {}'.format(path))

            return io.imread(path)

    # if you reach this line, you didn't find the image you're looking for

    print('Load failed: could not find image {}'.format(path))
def calibrate_image(rgb_image):

    # Transform test image to 32-bit floats to avoid 

    # surprises when doing arithmetic with it 

    calibrated_img = rgb_image.copy().astype('float32')



    # Loop over RGB

    for i in range(3):

        # Subtract mean 

        calibrated_img[:,:,i] = calibrated_img[:,:,i]-np.mean(calibrated_img[:,:,i])

        # Normalize variance

        calibrated_img[:,:,i] = calibrated_img[:,:,i]/np.std(calibrated_img[:,:,i])

        # Scale to reference 

        calibrated_img[:,:,i] = calibrated_img[:,:,i]*ref_stds[i] + ref_means[i]

        # Clip any values going out of the valid range

        calibrated_img[:,:,i] = np.clip(calibrated_img[:,:,i],0,255)



    # Convert to 8-bit unsigned int

    return calibrated_img.astype('uint8')
# Pull a list of 20000 image names

jpg_list = os.listdir(PLANET_KAGGLE_JPEG_DIR)[:20000]

# Select a random sample of 100 among those

np.random.shuffle(jpg_list)

jpg_list = jpg_list[:100]
ref_colors = [[],[],[]]

for _file in jpg_list:

    # keep only the first 3 bands, RGB

    _img = mpimg.imread(os.path.join(PLANET_KAGGLE_JPEG_DIR, _file))[:,:,:3]

    # Flatten 2-D to 1-D

    _data = _img.reshape((-1,3))

    # Dump pixel values to aggregation buckets

    for i in range(3): 

        ref_colors[i] = ref_colors[i] + _data[:,i].tolist()

    

ref_colors = np.array(ref_colors)
ref_means = [np.mean(ref_colors[i]) for i in range(3)]

ref_stds = [np.std(ref_colors[i]) for i in range(3)]
def show_img(path):

    img = load_image(path)[:,:,:3]

    if '.tif' in path:

        img = calibrate_image(img)

    fig = plt.figure()

    a = fig.add_subplot(1, 1, 1)

    a = a.set_title(path)

    plt.imshow(img)
from random import shuffle



# allegedly hazy images labeled as clear :)

hazy = ['train_645.jpg', 'train_3323.jpg', 'train_786.jpg', 'train_3204.jpg', 'train_3030.jpg', 'train_3444.jpg', 'train_3790.jpg', 'train_2383.jpg', 'train_1119.jpg', 'train_1572.jpg', 'train_2992.jpg', 'train_2434.jpg', 'train_6.jpg', 'train_186.jpg', 'train_1166.jpg', 'train_379.jpg', 'train_415.jpg', 'train_1816.jpg', 'train_536.jpg', 'train_2474.jpg', 'train_1417.jpg', 'train_944.jpg', 'train_1683.jpg', 'train_3736.jpg', 'train_750.jpg', 'train_1423.jpg', 'train_1196.jpg', 'train_2415.jpg', 'train_326.jpg', 'train_1851.jpg', 'train_3887.jpg', 'train_3326.jpg', 'train_2822.jpg', 'train_313.jpg', 'train_3016.jpg', 'train_1660.jpg', 'train_1923.jpg', 'train_2527.jpg', 'train_1272.jpg', 'train_1694.jpg', 'train_1327.jpg', 'train_609.jpg', 'train_2612.jpg', 'train_1185.jpg', 'train_3343.jpg', 'train_841.jpg', 'train_960.jpg', 'train_2879.jpg', 'train_2436.jpg', 'train_2619.jpg', 'train_574.jpg', 'train_2820.jpg', 'train_1294.jpg', 'train_1532.jpg', 'train_3886.jpg', 'train_1897.jpg', 'train_2507.jpg', 'train_1444.jpg', 'train_3980.jpg', 'train_3728.jpg', 'train_3404.jpg', 'train_2031.jpg', 'train_195.jpg', 'train_421.jpg', 'train_2851.jpg', 'train_3448.jpg', 'train_3893.jpg', 'train_2722.jpg', 'train_2050.jpg', 'train_2162.jpg', 'train_2856.jpg', 'train_813.jpg', 'train_1865.jpg', 'train_3640.jpg', 'train_2212.jpg', 'train_3983.jpg', 'train_933.jpg', 'train_2106.jpg', 'train_1659.jpg', 'train_982.jpg', 'train_3236.jpg', 'train_568.jpg', 'train_3072.jpg', 'train_524.jpg', 'train_2986.jpg', 'train_159.jpg', 'train_3046.jpg', 'train_2278.jpg', 'train_1590.jpg', 'train_1885.jpg', 'train_767.jpg', 'train_3494.jpg', 'train_830.jpg', 'train_119.jpg', 'train_3518.jpg', 'train_2894.jpg', 'train_806.jpg', 'train_2065.jpg', 'train_3628.jpg', 'train_1145.jpg', 'train_2311.jpg', 'train_2423.jpg', 'train_1741.jpg', 'train_2874.jpg', 'train_3361.jpg', 'train_1898.jpg', 'train_1537.jpg', 'train_3900.jpg', 'train_2490.jpg', 'train_2544.jpg', 'train_2132.jpg', 'train_2637.jpg', 'train_498.jpg', 'train_3299.jpg', 'train_2547.jpg', 'train_831.jpg', 'train_1582.jpg', 'train_1787.jpg', 'train_2167.jpg', 'train_1234.jpg', 'train_3122.jpg', 'train_1682.jpg', 'train_883.jpg', 'train_1822.jpg', 'train_253.jpg', 'train_2521.jpg', 'train_1655.jpg', 'train_358.jpg', 'train_3936.jpg', 'train_2410.jpg', 'train_3563.jpg', 'train_3183.jpg', 'train_3342.jpg', 'train_1314.jpg', 'train_2137.jpg', 'train_1912.jpg', 'train_3719.jpg', 'train_557.jpg', 'train_1826.jpg', 'train_1004.jpg', 'train_919.jpg', 'train_1810.jpg', 'train_2059.jpg', 'train_2698.jpg', 'train_3994.jpg', 'train_3984.jpg', 'train_1997.jpg', 'train_9.jpg', 'train_1955.jpg', 'train_1556.jpg', 'train_1936.jpg', 'train_2112.jpg', 'train_3596.jpg', 'train_2263.jpg', 'train_2845.jpg', 'train_1623.jpg', 'train_2225.jpg', 'train_2115.jpg', 'train_3433.jpg', 'train_3810.jpg', 'train_3514.jpg', 'train_1815.jpg', 'train_2085.jpg', 'train_1966.jpg', 'train_20.jpg', 'train_1270.jpg', 'train_3998.jpg', 'train_1402.jpg', 'train_330.jpg', 'train_3288.jpg', 'train_411.jpg', 'train_2728.jpg', 'train_1720.jpg', 'train_1482.jpg', 'train_1061.jpg', 'train_1819.jpg', 'train_155.jpg', 'train_127.jpg', 'train_3068.jpg', 'train_2183.jpg', 'train_2191.jpg', 'train_3095.jpg', 'train_3757.jpg', 'train_2123.jpg', 'train_23.jpg', 'train_3791.jpg', 'train_66.jpg', 'train_2095.jpg', 'train_3876.jpg', 'train_1347.jpg', 'train_2772.jpg', 'train_364.jpg', 'train_389.jpg', 'train_2972.jpg', 'train_1974.jpg', 'train_2919.jpg', 'train_1975.jpg', 'train_751.jpg', 'train_1886.jpg', 'train_2343.jpg', 'train_2299.jpg', 'train_950.jpg', 'train_2250.jpg', 'train_2146.jpg', 'train_2170.jpg', 'train_1725.jpg', 'train_2956.jpg', 'train_3324.jpg', 'train_1401.jpg', 'train_3733.jpg', 'train_316.jpg', 'train_3314.jpg', 'train_338.jpg', 'train_275.jpg', 'train_1472.jpg', 'train_597.jpg', 'train_3053.jpg', 'train_1308.jpg', 'train_633.jpg', 'train_281.jpg', 'train_1445.jpg', 'train_126.jpg', 'train_1420.jpg', 'train_755.jpg', 'train_1393.jpg', 'train_2111.jpg', 'train_3944.jpg', 'train_36.jpg', 'train_556.jpg', 'train_1950.jpg', 'train_2737.jpg', 'train_2913.jpg', 'train_1690.jpg', 'train_1518.jpg', 'train_1137.jpg', 'train_1109.jpg', 'train_2798.jpg', 'train_495.jpg', 'train_1241.jpg', 'train_327.jpg', 'train_1669.jpg', 'train_2852.jpg', 'train_3741.jpg', 'train_1945.jpg', 'train_2156.jpg', 'train_3011.jpg', 'train_3006.jpg', 'train_82.jpg', 'train_3811.jpg', 'train_335.jpg', 'train_3111.jpg', 'train_2937.jpg', 'train_3519.jpg', 'train_3694.jpg', 'train_131.jpg', 'train_723.jpg', 'train_2769.jpg', 'train_2305.jpg', 'train_677.jpg', 'train_2452.jpg', 'train_2513.jpg', 'train_2810.jpg', 'train_613.jpg', 'train_2693.jpg', 'train_2604.jpg', 'train_1301.jpg', 'train_927.jpg', 'train_1204.jpg', 'train_1918.jpg', 'train_1120.jpg', 'train_1708.jpg', 'train_992.jpg', 'train_1561.jpg', 'train_931.jpg', 'train_1332']



shuffle(hazy)



n = 0

for i in hazy:

    show_img(i)

    

    if n == 25:

        break

        

    n += 1
import copy



def show_shuffle(lst, n):

    lst_copy = copy.deepcopy(lst)

    shuffle(lst_copy)



    for i in lst_copy[:n]:

        show_img(i)
show_shuffle(hazy, 25)
show_shuffle(hazy, 25)
show_shuffle(hazy, 25)