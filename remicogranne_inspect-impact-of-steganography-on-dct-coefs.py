# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import random

import matplotlib.pyplot as plt

import glob

import os

from PIL import Image 



os.mkdir('./jpegio/')

# Downloaded from https://github.com/dwgoon/jpegio

# Author: Daewon Lee, github: @dwgoon

# Wheel packaged for python 3.7

!curl https://github.com/dwgoon/jpegio/blob/master/dist/jpegio-0.2.1-cp37-cp37m-linux_x86_64.whl > ./jpegio/jpegio-0.2.1-cp37-cp37m-linux_x86_64.whl

# Once downloaded install the package

!pip install ./jpegio/jpegio-0.2.1-cp37-cp37m-linux_x86_64.whl

import jpegio as jio

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



coverDir = "/kaggle/input/alaska2-image-steganalysis/Cover/"

JMiPODDir = "/kaggle/input/alaska2-image-steganalysis/JMiPOD/"

JUNIWARDDir = "/kaggle/input/alaska2-image-steganalysis/JUNIWARD/"

UERDDir = "/kaggle/input/alaska2-image-steganalysis/UERD/"



train_filenames = np.array(sorted(os.listdir(coverDir)))

train_filenames[0:10]



#imgList = sorted(os.listdir(coverDir))

imgList = os.listdir(coverDir)

random.shuffle(imgList)

for imgIdx in range( len(imgList) ): 

    c_struct=jio.read(coverDir + imgList[imgIdx] )

    coverDCT = np.zeros([512,512,3]) ; coverDCT[:,:,0] = c_struct.coef_arrays[0] ; coverDCT[:,:,1] = c_struct.coef_arrays[1] ; coverDCT[:,:,2] = c_struct.coef_arrays[2]

    coverQTbl = c_struct.quant_tables[0];

    plt.imshow( abs(coverDCT) )

    plt.show()



    coverPixels = np.array( Image.open( os.path.join(coverDir , imgList[imgIdx] ) ) ).astype('float')

    plt.imshow( coverPixels.astype('uint8') )

    plt.show()





    stego = jio.read( os.path.join( JMiPODDir , imgList[imgIdx] ) )

    stegoDCT = np.zeros([512,512,3]) ; stegoDCT[:,:,0] = stego.coef_arrays[0] ; stegoDCT[:,:,1] = stego.coef_arrays[1] ; stegoDCT[:,:,2] = stego.coef_arrays[2]

    stegoQTbl = stego.quant_tables[0]; #Of course this is the same as coverQTbl

    imgDiff = coverDCT - stegoDCT;

    NbChanges = np.sum( abs( imgDiff ) )

    plt.imshow( abs(imgDiff) )

    plt.show()

    

    stegoPixels = np.array( Image.open( os.path.join( JMiPODDir , imgList[imgIdx] ) ) ).astype('float')

    pixelsDiff = coverPixels - stegoPixels;

    plt.imshow( abs( pixelsDiff ) )

    plt.show()

    



    stego = jio.read( os.path.join( JUNIWARDDir , imgList[imgIdx] ) )

    stegoDCT = np.zeros([512,512,3]) ; stegoDCT[:,:,0] = stego.coef_arrays[0] ; stegoDCT[:,:,1] = stego.coef_arrays[1] ; stegoDCT[:,:,2] = stego.coef_arrays[2]

    imgDiff = coverDCT - stegoDCT;

    NbChanges = np.sum( abs( imgDiff ) )

    plt.imshow( abs(imgDiff) )

    plt.show()



    stegoPixels = np.array( Image.open( os.path.join( JUNIWARDDir , imgList[imgIdx] ) ) ).astype('float')

    pixelsDiff = coverPixels - stegoPixels;

    plt.imshow( abs( pixelsDiff ) )

    plt.show()





    stego = jio.read( os.path.join( UERDDir , imgList[imgIdx] ) )

    stegoDCT = np.zeros([512,512,3]) ; stegoDCT[:,:,0] = stego.coef_arrays[0] ; stegoDCT[:,:,1] = stego.coef_arrays[1] ; stegoDCT[:,:,2] = stego.coef_arrays[2]

    imgDiff = coverDCT - stegoDCT;

    NbChanges = np.sum( abs( imgDiff ) )

    plt.imshow( abs(imgDiff) )

    plt.show()



    stegoPixels = np.array( Image.open( os.path.join( UERDDir , imgList[imgIdx] ) ) ).astype('float')

    pixelsDiff = coverPixels - stegoPixels;

    plt.imshow( abs( pixelsDiff ) )

    plt.show()

    

    

# Any results you write to the current directory are saved as output.
