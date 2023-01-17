! pip install --upgrade imutils

import numpy as np

import struct

import os

import array

import cv2

import matplotlib.pylab as plt

from IPython.display import clear_output

from scipy.signal import savgol_filter

from time import sleep

from skimage.feature import peak_local_max

from skimage.morphology import watershed

from scipy import ndimage

import numpy as np

import imutils
# path ="../input/data2/data/seq-P04-M04-A0001-G01-C00-S0033/seq-P04-M04-A0001-G01-C00-S0033.gt"

# path1="../input/data2/data/seq-P04-M04-A0001-G01-C00-S0033/seq-P04-M04-A0001-G01-C00-S0033.z16"

path="../input/data2/data/seq-P08-M02-A0001-G02-C00-S0029/seq-P08-M02-A0001-G02-C00-S0029.gt"

path1="../input/data2/data/seq-P08-M02-A0001-G02-C00-S0029/seq-P08-M02-A0001-G02-C00-S0029.z16"
size = 512*424

file = None
def getFrame(frame,size=512*424):

    arr=[]

    global file

    if file is None:

          file= open(path1, "rb")

    file.seek(size*2*frame)

    buffer = file.read()

    outx = np.frombuffer(buffer,dtype=np.int16,count=size).reshape(424,512)

    #temp= np.copy(outx)

    out = np.uint8(outx * (255 / np.max(outx)))

    out = 255-out

    out[out>240] =0

    out[out<80] = 0

    return out
def findHead(image,kernel_size=91,thresh_rem=248):

    kernel = np.ones((kernel_size,kernel_size),np.uint8)

    thresh = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh[thresh==255]=1

    dilated = cv2.dilate(image,kernel,iterations=1)

    res= dilated*thresh

    res= res-image

    res[thresh==0]=255

    res= 255-res

    res[res< thresh_rem] =0

    kernel2 = np.ones((11,11),np.uint8)

    res= cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel2)

    return res
for i in range (420,510):

    #print(i)

    out=getFrame(i)

    final=findHead(out)

    cnts = cv2.findContours(final.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    contours=[]

    for cnt in cnts:

        area = cv2.contourArea(cnt)

        if area > 300:

            #print(area)

            contours.append(cnt)

        

    #c = max(cnts, key=cv2.contourArea)

    count = 0

    #cv2.rectangle(out,(50,75),(480,350),(255,255,255),3)

    

    for c in contours:

        count = count + 1

        ((x, y), r) = cv2.minEnclosingCircle(c)

        cv2.circle(out, (int(x), int(y)), int(r), (255, 255, 255), 2)

        cv2.putText(out, "#{}".format(str(count)), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    plt.figure(figsize=(10, 8))

    plt.imshow(out)

    plt.show()

    #sleep(0.2)

    clear_output(wait=True)