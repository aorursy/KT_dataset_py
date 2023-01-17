import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
def RLenc(img,order='F',format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not
    
    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = [] ## list of run lengths
    r = 0     ## the current run length
    pos = 1   ## count starts from 1 per WK
    for c in bytes:
        if ( c == 0 ):
            if r != 0:
                runs.append((pos, r))
                pos+=r
                r=0
            pos+=1
        else:
            r+=1

    #if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''
    
        for rr in runs:
            z+='{} {} '.format(rr[0],rr[1])
        return z[:-1]
    else:
        return runs
from itertools import chain;
def run_length(label):
    x = label.transpose().flatten();
    y = np.where(x>0.5)[0];
    if len(y)<10:# consider as empty
        return [];
    z = np.where(np.diff(y)>1)[0]
    start = np.insert(y[z+1],0,y[0])
    end = np.append(y[z],y[-1])
    leng = end - start;
    res = [[s+1,l+1] for s,l in zip(list(start),list(leng))]
    res = list(chain.from_iterable(res));
    return res;
masks= np.array([plt.imread(img) for img in glob.glob("../input/train/*.tif") if 'mask' in img])

plt.imshow(masks.mean(axis=0))
masks_mean = masks.mean(axis=0)
masks_mean = masks_mean/masks_mean.max()
masks_bin = masks>0
masks_bin.shape
def dice(x, y):
    x = x>0
    y = y>0
    if y.sum()==0 & x.sum() == 0:
        return 1
    return 2*(x*y).sum()/(x.sum() + y.sum())
np.zeros(10)
def obj(mask):
    N = masks_bin.shape[0]
    res = np.zeros(N)
    for i in range(N):
        res[i] = dice(mask, masks_bin[i])
    return res.mean()
x  =masks_mean
y = masks_mean
a = [1,2,3,1, 0, 0, 0]
a = np.array(a) > 0
a.sum()
optimal_mask = (masks_mean > 0.45)
obj(optimal_mask)
plt.imshow(optimal_mask)
obj(optimal_mask)
RLenc(optimal_mask)
RLenc(optimal_mask, 'C')
run_length(optimal_mask)
plt.imshow(optimal_mask)
result = pd.csvread('../input/sample_submission.csv')
plt.hist(masks_mean.ravel())
dice(masks_mean>0.6, masks_mean>0.59)
images = np.array([plt.imread(img) for img in glob.glob("../input/train/*.tif") if 'mask' not in img])
plt.imshow(masks.mean(axis=0)>40)
masks.shape
import pandas as pd
pd.Series(masks.mean(axis=0).ravel()).describe()

plt.imshow(masks[8,:,:])
res = glob.glob("../input/train/*")
len(res)
res[:30]
import os