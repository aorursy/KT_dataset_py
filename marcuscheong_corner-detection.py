import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
path = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
path2 = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___healthy/00907d8b-6ae6-4306-bfd7-d54471981a86___RS_HL 5709.JPG"
img = cv2.imread(path)
img2 = cv2.imread(path2)
def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    kH,kW = conv_filter.shape
    (imH,imW) = img.shape
    result = np.zeros(img.shape)
    pad = int((kH-1)/2)
    
    for y in range(imH-kH):
        for x in range(imW-kW):
            window = img[y:y+kH,x:x+kW]
            result[y+pad,x+pad] = (conv_filter * window).sum()

    return result


#https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def gauss_filter(shape = (3,3), sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def harris(img):

    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    g = gauss_filter((5,5),1)

    dx = np.array([[-1,0,1],
                   [-1,0,1],
                   [-1,0,1]])
    dy = dx.transpose()

    Ix = conv2(bw,dx)
    Iy = conv2(bw,dy)

    Ix2 = conv2(np.power(Ix,2),g)
    Iy2 = conv2(np.power(Iy,2),g)
    Ixy = conv2((Ix*Iy),g)
    
    det = (Ix2 * Iy2) - (np.power(Ixy,2))

    trace = Ix2 + Iy2
    k = 0.04
    r = det - k*(np.power(trace,2))

    #Non-Max Suppression
    maxima = r.max()
    corner_points = np.array([])
    detected_img = img.copy()
    thresh = 0.01

    window_size = 3
    gap = window_size - 1 // 2
    row,col = r.shape

    for h in range(gap, row-(gap + 1)):
        for w in range(gap, col-(gap + 1)):
                #Define the 2D space to focus on (window)
            window = r[h-gap:h+(gap+1),w-gap:w+(gap+1)]
                #Condition to meet : Value must be the largest within the 2D window and is larger than the product of maxima and threshold
                #This is to prevent multiple detections around the same corner
            if r[h,w] > maxima * thresh and r[h,w] == np.max(window):
                    #Creating a red box around the detected corner
                detected_img[h-1:h+1,w-1:w+1] = [255,0,0]
                if(corner_points.size == 0):
                    corner_points = np.array([h,w])
                else:
                    corner_points = np.vstack((corner_points,[h,w]))
    return corner_points, detected_img
corner, detected = harris(img)
corner2, detected2 = harris(img2)
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(detected)

plt.subplot(122)
plt.imshow(detected2)

plt.show()
