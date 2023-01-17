# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd 
from PIL import Image
from PIL import ImageStat
import PIL
import matplotlib.pyplot as plt
import glob
import cv2
import math
import time
train = pd.read_csv('../input/pollutionvision/train_data.csv')
train.head()
train.shape
train = train[['Temp(C)','Pressure(kPa)','Rel. Humidity','Errors','Alarm Triggered','Dilution Factor','Dead Time','Image_file','Wind_Speed','Distance_to_Road','Camera_Angle','Elevation','Total']]
train = train[train['Errors'] == 0][train['Alarm Triggered'] == 0][train['Dead Time'] <= 0.01][train['Dilution Factor'] == 1].reset_index(drop=True)
train['Image_day'] = train['Image_file'].apply(lambda x:x[:13])
train['Image_day'].unique()
train.head()
train.shape
im0 = PIL.Image.open('/kaggle/input/pollutionvision/frames/frames/video08052020_2771.jpg') # read the image using PIL
im0.mode,im0.size
plt.imshow(im0) # Plot the image read by PIL
im00 = cv2.imread('/kaggle/input/pollutionvision/frames/frames/video08052020_2771.jpg',1) # read the image using OpenCV
pil_img = Image.fromarray(cv2.cvtColor(im00,cv2.COLOR_BGR2RGB)) # Convert GRB format to RGB format
plt.imshow(pil_img) # Plot the image read by OpenCV
b1,g1,r1 = cv2.split(im00) # Remember that open consider images channel to be in order of B,G,R and not in RGB.
l1 = np.mean(cv2.cvtColor(im00, cv2.COLOR_BGR2GRAY))

imnp = np.array(im0)
r2 = imnp[:, :, 0]
g2 = imnp[:, :, 1]
b2 = imnp[:, :, 2]
im2 = im0.convert('L')
l2 = ImageStat.Stat(im2).mean[0]
print("OpenCV",r1.mean()/255,g1.mean()/255,b1.mean()/255,l1)
print("PIL",r2.mean()/255,g2.mean()/255,b2.mean()/255,l2)
# First we have to define several functions
# Resource: https://github.com/He-Zhang/image_dehaze.git
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(np.max([math.floor(imsz/1000),1]))
    darkvec = dark.reshape(imsz,1);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res
# Take one image as example to show what is haze removed and what is transmission
src = cv2.imread('/kaggle/input/pollutionvision/frames/frames/video08072020_1.jpg')
I = src.astype('float64')/255;
dcI = DarkChannel(I,15);
A = AtmLight(I,dcI);
te = TransmissionEstimate(I,A,15);
t = TransmissionRefine(src,te);
J = Recover(I,t,A,0.6);
dcJ = DarkChannel(J,15);
# The original image
plt.imshow(Image.fromarray(cv2.cvtColor((src).astype('uint8'),cv2.COLOR_BGR2RGB))) 
# The dehazed image
plt.imshow(Image.fromarray(cv2.cvtColor((J*255).astype('uint8'),cv2.COLOR_BGR2RGB))) 
# The dehazed ROI ([:350,:] in this case)
plt.imshow(Image.fromarray(cv2.cvtColor((J*255)[:350,:].astype('uint8'),cv2.COLOR_BGR2RGB)))
# The transmission image
plt.imshow(Image.fromarray(cv2.cvtColor((t*255)[:350,:].astype('uint8'),cv2.COLOR_BGR2RGB))) 
print('the amount of haze removed:',((dcI[:350,:]- dcJ[:350,:])**2).mean(),'; trasmission:',t[:350,:].mean()) 
# Those two numbers are what we would add to the dataframe
# Note that I only focus on the ROI
start = time.time()
import skimage.measure    
count = 60000 # 0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000  
for i in train['Image_file'][count:]:
    src = cv2.imread('/kaggle/input/pollutionvision/frames/frames/'+i)
    
    ### Contrast
    Y = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # compute min and max of Y
    Y_min = int(np.min(Y))
    Y_max = int(np.max(Y))
    # compute contrast
    train.loc[count,'Contrast_minmax'] = (Y_max-Y_min)/(Y_max+Y_min)
    train.loc[count,'Contrast_RMS'] = Y.std()
    
    
    ### RGB and Luminance
    b,g,r = cv2.split(src)
    avg_r = np.mean(r)/255
    avg_g = np.mean(g)/255
    avg_b = np.mean(b)/255
    avg_lum = np.mean(Y)
    train.loc[count,'Red'] = avg_r
    train.loc[count,'Green'] = avg_g
    train.loc[count,'Blue'] = avg_b
    train.loc[count,'Luminance'] = avg_lum

    
    ### Entropy
    train.loc[count,'Entropy'] = skimage.measure.shannon_entropy(src)
    
    
    ### Haze removed and Trnasmission
    if i[:13] == 'video08052020' or i[:13] == 'video06192020':
        src = src[150:,:]
        I = src.astype('float64')/255
        dcI = DarkChannel(I,15)
        A = AtmLight(I,dcI)
        te = TransmissionEstimate(I,A,15)
        t = TransmissionRefine(src,te)
        J = Recover(I,t,A,0.8)
        dcJ = DarkChannel(J,15)
             
    elif i[:13] == 'video08072020'or i[:13] == 'video06182020':
        src = src[:350,:]
        I = src.astype('float64')/255
        dcI = DarkChannel(I,15)
        A = AtmLight(I,dcI)
        te = TransmissionEstimate(I,A,15)
        t = TransmissionRefine(src,te)
        J = Recover(I,t,A,0.6)
        dcJ = DarkChannel(J,15)
    
    elif i[:13] == 'video06152020':
        src = src[:230,400:]
        I = src.astype('float64')/255
        dcI = DarkChannel(I,15)
        A = AtmLight(I,dcI)
        te = TransmissionEstimate(I,A,15)
        t = TransmissionRefine(src,te)
        J = Recover(I,t,A,0.6)
        dcJ = DarkChannel(J,15) 
        
    elif i[:13] == 'video08062020':
        src = src[:280,400:]
        I = src.astype('float64')/255
        dcI = DarkChannel(I,15)
        A = AtmLight(I,dcI)
        te = TransmissionEstimate(I,A,15)
        t = TransmissionRefine(src,te)
        J = Recover(I,t,A,0.6)
        dcJ = DarkChannel(J,15)        
        
    elif i[:13] == 'video06172020' or i[:13] == 'video06082020':
        src = src
        I = src.astype('float64')/255
        dcI = DarkChannel(I,15)
        A = AtmLight(I,dcI)
        te = TransmissionEstimate(I,A,15)
        t = TransmissionRefine(src,te)
        J = Recover(I,t,A,0.7)
        dcJ = DarkChannel(J,15)        
        
    elif i[:13] == 'video06112020':
        src = src[:200,800:]
        I = src.astype('float64')/255
        dcI = DarkChannel(I,15)
        A = AtmLight(I,dcI)
        te = TransmissionEstimate(I,A,15)
        t = TransmissionRefine(src,te)
        J = Recover(I,t,A,0.7)
        dcJ = DarkChannel(J,15)        
        
    elif i[:13] == 'video08102020':
        src = src[:200,:]
        I = src.astype('float64')/255
        dcI = DarkChannel(I,15)
        A = AtmLight(I,dcI)
        te = TransmissionEstimate(I,A,15)
        t = TransmissionRefine(src,te)
        J = Recover(I,t,A,0.9)
        dcJ = DarkChannel(J,15)

    train.loc[count,'Haze_removed'] = ((dcI- dcJ)**2).mean()
    train.loc[count,'Transmission'] = t.mean()
    
    
    count += 1
train
end = time.time()
print ('runtime:',end - start,'sec.')
train.to_csv("Train_imageinfo.csv",index=False,sep=',')