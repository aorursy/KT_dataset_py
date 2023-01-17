import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
from tqdm import tqdm
%matplotlib inline

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('dark_background')
def cieluv(img, target):
    # adapted from https://www.compuphase.com/cmetric.htm
    img = img.astype('int')
    aR, aG, aB = img[:,:,0], img[:,:,1], img[:,:,2]
    bR, bG, bB = target
    rmean = ((aR + bR) / 2.).astype('int')
    r2 = np.square(aR - bR)
    g2 = np.square(aG - bG)
    b2 = np.square(aB - bB)
    
    # final sqrt removed for speed; please square your thresholds accordingly
    result = (((512+rmean)*r2)>>8) + 4*g2 + (((767-rmean)*b2)>>8)
    
    return result
def process_image(f, plot=True):
    img = plt.imread(f)
    img = np.round(img * 255).astype('ubyte')[:,:,:3]
    if plot:
        plt.figure(1)
        plt.subplot(141)
        plt.imshow(img)
        plt.title('Raw Image')
    img_filter = (
        (cieluv(img, (71, 86, 38)) > 1600)
        & (cieluv(img, (65,  79,  19)) > 1600)
        & (cieluv(img, (95,  106,  56)) > 1600)
        & (cieluv(img, (56,  63,  43)) > 500)
    )
    img[img_filter] = 0
    
    if plot:
        plt.subplot(142)
        plt.imshow(img)
        plt.title('CIELUV Color Thresholding')
    
    img = cv2.medianBlur(img, 9)
    
    if plot:
        plt.subplot(143)
        plt.imshow(img)
        plt.title('Median filter')
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype('uint8')
    
    if plot:
        plt.subplot(144)
        plt.imshow(img)
        plt.title('Black and White')
    return img
i = process_image('../input/test/fe9e87b78.png')
i = process_image('../input/test/1821eb11a.png')
i = process_image('../input/train/Black-grass/b4b8b1507.png')
i = process_image('../input/train/Maize/5363a9f84.png')
i = process_image('../input/train/Fat Hen/35083f3c2.png')
i = process_image('../input/train/Common Chickweed/495d1a520.png')