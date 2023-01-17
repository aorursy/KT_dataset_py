! git clone https://github.com/dwgoon/jpegio

# Once downloaded install the package

!pip install jpegio/.

import jpegio as jio

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import numpy as np

import jpegio as jpio

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

#This code extract YCbCr channels from a jpeg object

def JPEGdecompressYCbCr(jpegStruct):

    

    nb_colors=len(jpegStruct.coef_arrays)

        

    [Col,Row] = np.meshgrid( range(8) , range(8) )

    T = 0.5 * np.cos(np.pi * (2*Col + 1) * Row / (2 * 8))

    T[0,:] = T[0,:] / np.sqrt(2)

    

    sz = np.array(jpegStruct.coef_arrays[0].shape)

    

    imDecompressYCbCr = np.zeros([sz[0], sz[1], nb_colors]);

    szDct = (sz/8).astype('int')

    

    

    

    for ColorChannel in range(nb_colors):

        tmpPixels = np.zeros(sz)

    

        DCTcoefs = jpegStruct.coef_arrays[ColorChannel];

        if ColorChannel==0:

            QM = jpegStruct.quant_tables[ColorChannel];

        else:

            QM = jpegStruct.quant_tables[1];

        

        for idxRow in range(szDct[0]):

            for idxCol in range(szDct[1]):

                D = DCTcoefs[idxRow*8:(idxRow+1)*8 , idxCol*8:(idxCol+1)*8]

                tmpPixels[idxRow*8:(idxRow+1)*8 , idxCol*8:(idxCol+1)*8] = np.dot( np.transpose(T) , np.dot( QM * D , T ) )

        imDecompressYCbCr[:,:,ColorChannel] = tmpPixels;

    return imDecompressYCbCr





import os

print(os.listdir('../input/alaska2-image-steganalysis'))
for i, img in enumerate(os.listdir('../input/alaska2-image-steganalysis/Cover')[:10]):

    imgRGB = mpimg.imread('../input/alaska2-image-steganalysis/Cover/' + img)

    jpegStruct = jpio.read('../input/alaska2-image-steganalysis/Cover/' + img)

    imDecompressYCbCr = JPEGdecompressYCbCr(jpegStruct)



    plt.subplot(2, 4, 1) ; plt.imshow(imgRGB)

    plt.subplot(2, 4, 2) ; plt.imshow(imgRGB[:,:,0] , cmap='gray')

    plt.subplot(2, 4, 3) ; plt.imshow(imgRGB[:,:,1] , cmap='gray')

    plt.subplot(2, 4, 4) ; plt.imshow(imgRGB[:,:,2] , cmap='gray')



    plt.subplot(2, 4, 5) ; plt.imshow(imDecompressYCbCr.astype('int'))

    plt.subplot(2, 4, 6) ; plt.imshow(imDecompressYCbCr[:,:,0] , cmap='gray')

    plt.subplot(2, 4, 7) ; plt.imshow(imDecompressYCbCr[:,:,1] , cmap='gray')

    plt.subplot(2, 4, 8) ; plt.imshow(imDecompressYCbCr[:,:,2] , cmap='gray')

    plt.show()



#Now to get an idea of what the quality factor is ...

for i, img in enumerate(os.listdir('../input/alaska2-image-steganalysis/Cover')[:10]):

    #Let's have a look at the jpeg struct :

    jpegStruct = jpio.read('../input/alaska2-image-steganalysis/Cover/' + img)

    if (jpegStruct.quant_tables[0][0,0]==2):

        print('Quality Factor is 95')

    elif (jpegStruct.quant_tables[0][0,0]==3):

        print('Quality Factor is 90')

    elif (jpegStruct.quant_tables[0][0,0]==8):

        print('Quality Factor is 75')

    print(jpegStruct.quant_tables[0])

#You can note that this matrix is used in the decompression (function JPEGdecompressYCbCr) as follows:

    #tmpPixels[idxRow*8:(idxRow+1)*8 , idxCol*8:(idxCol+1)*8] = np.dot( np.transpose(T) , np.dot( QM * D , T ) )

#In brief, the JPEG (mainly) acts in three steps

# 1. Transform color from RGB to YCbCr

# 2. Split pixels (YCbCr) into blocks of 8x8 pixels denoted Pix and apply over each block the DCT transform 

#DCT = T * Pix * T'

# Where T is an orthonormal basis change matrix (computed at the begining of function JPEGdecompressYCbCr)

# 3. The final part consist in dividing each DCT coefs by a specific factor and then round 

#DCTquantized = round( DCT / QM) = round( ( T * Pix * T' ) / QM )

# In order to get Pix from DCTquantized you have to undo all step (except quantization) which yields:

# T' * DCTquantized * QM * T  Pix

#which is what I do here :

    #tmpPixels[idxRow*8:(idxRow+1)*8 , idxCol*8:(idxCol+1)*8] = np.dot( np.transpose(T) , np.dot( QM * D , T ) )

# The largest the terms in quatization matrix QM, the largest the division and the more rough is the quantization,

# such matrix is usually determined from a standard matrix as  follows



table0 = np.array(

    [ [ 16,  11,  10,  16,  24,  40,  51,  61 ],

    [ 12,  12,  14,  19,  26,  58,  60,  55 ],

    [ 14,  13,  16,  24,  40,  57,  69,  56 ],

    [ 14,  17,  22,  29,  51,  87,  80,  62 ],

    [ 18,  22,  37,  56,  68, 109, 103,  77 ],

    [ 24,  35,  55,  64,  81, 104, 113,  92 ],

    [ 49,  64,  78,  87, 103, 121, 120, 101 ],

    [ 72,  92,  95,  98, 112, 100, 103,  99 ] ] )

table1 = np.array(

    [ [ 17,  18,  24,  47,  99,  99,  99,  99 ],

    [ 18,  21,  26,  66,  99,  99,  99,  99 ],

    [ 24,  26,  56,  99,  99,  99,  99,  99 ],

    [ 47,  66,  99,  99,  99,  99,  99,  99 ],

    [ 99,  99,  99,  99,  99,  99,  99,  99 ],

    [ 99,  99,  99,  99,  99,  99,  99,  99 ],

    [ 99,  99,  99,  99,  99,  99,  99,  99 ],

    [ 99,  99,  99,  99,  99,  99,  99,  99 ] ] )



qualityFactor = [95, 90, 75]

for QF in qualityFactor:

    quality = 200 - QF*2



    QMY = np.floor( (table0 * quality + 50) /100 )

    QMY[QMY<1] = 1 



    QMC = np.floor( (table1 * quality + 50) /100 )

    QMC[QMC<1] = 1 



    print('Quantization Matrix (Y-Luminance) for Quality Factor ' , QF)

    print(QMY)

    print('Quantization Matrix (CbCr-Chrominance) for Quality Factor ' , QF)

    print(QMC)


