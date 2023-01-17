# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import cv2

matplotlib.style.use('fivethirtyeight')


path = "../input"
image_path = os.path.join(path, 'train')
file_masks = pd.read_csv(os.path.join(path , 'train_ship_segmentations.csv'))
img_ids = file_masks.groupby('ImageId').size().reset_index(name='counts')

file_data = dict(zip(img_ids['ImageId'], [[] for x in range(0, len(img_ids))] ));
for i,code in  file_masks.values:
    file_data[i].append(code)

data =  [(i.split('.')[0],
        os.path.join(image_path,'{}'.format(i)),
        code,
        ) for i,code in  file_data.items()  ] 

print(len(data))
print(data[0])

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    if not isinstance( mask_rle, str ):
        return np.zeros( shape )
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def lincomb(im1,im2,mask,alpha=0.5):
    im = im1.copy()      
    row, col = np.where(mask != 0)
    for i in range( len(row) ):
        r,c = row[i],col[i]
        im[r,c,0] = im1[r,c,0]*(1-alpha) + im2[r,c,0]*(alpha)
        im[r,c,1] = im1[r,c,1]*(1-alpha) + im2[r,c,1]*(alpha)
        im[r,c,2] = im1[r,c,2]*(1-alpha) + im2[r,c,2]*(alpha)
    return im

def setcolor(im, mask, color):    
    tmp=im.copy()
    tmp=np.reshape( tmp, (-1, im.shape[2])  )   
    mask = np.reshape( mask, (-1,1))      
    tmp[ np.where(mask>0)[0] ,:] = color
    im=np.reshape( tmp, (im.shape)  )
    return im

def makecolormask( masks ):
    cmap = plt.get_cmap('jet_r')
    colormask = np.zeros( (masks.shape[1], masks.shape[2], 3) )    
    for i,mask in enumerate(masks):
        color = cmap(float(i)/masks.shape[0] )
        colormask = setcolor(colormask, mask, color[:3] )        
    return colormask

def codes2masks( codes ):
    masks = []
    for code in codes:
        masks.append( rle_decode(code, image.shape[:2]) )
    masks = np.stack(masks, axis=0)  
    return masks

idname, ipath, codes = data[ 50 ]
image = cv2.imread(ipath)[:,:,(2,1,0)]
masks = codes2masks(codes)

plt.figure( figsize=(8,8) )
plt.imshow( lincomb( image/255, makecolormask(masks), masks.max(axis=0) , alpha=0.8 ) )
plt.axis('off')

plt.show()

matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['figure.figsize'] = (12,19)

numRows = 9; numCols = 5

plt.figure()
for k in range(numRows*numCols):
    idname, ipath, codes = data[ np.random.randint( len(data) ) ]
    image = cv2.imread(ipath)[:,:,(2,1,0)]
    masks = codes2masks(codes)    
    plt.subplot(numRows,numCols,k+1); 
    plt.imshow( lincomb( image/255, makecolormask(masks), masks.max(axis=0) , alpha=0.8 ) )
    plt.title( idname ); 
    plt.axis('off')
    
    


