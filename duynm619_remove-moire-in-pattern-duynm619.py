# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2

import matplotlib.pyplot as plt
img1 = cv2.imread('/kaggle/input/hw4_radiograph_1.jpg')

img2 = cv2.imread('/kaggle/input/hw4_radiograph_2.jpg')
plt.imshow(img1)

plt.show()

print("imgae 1")

plt.imshow(img2)

plt.show()

print("imgae 2")
# Trying with image 1

plt.imshow(img1)

plt.show()

print("imgae 1 (original)")

for i in range(1,21,2):

    img1Fix = cv2.medianBlur(img1,i)

    plt.imshow(img1Fix)

    plt.show()

    print("Applied filter %i" %i)
# Trying with image 2

plt.imshow(img2)

plt.show()

print("imgae 2 (original)")

for i in range(1,21,2):

    img2Fix = cv2.medianBlur(img2,i)

    plt.imshow(img2Fix)

    plt.show()

    print("Applied filter %i" %i)
img1Fix = cv2.medianBlur(img1,7)

fig = plt.figure(num=None, figsize=(80, 80), dpi=80, facecolor='w', edgecolor='k')

fig1 = fig.add_subplot(10, 10, 1)

fig2 = fig.add_subplot(10, 10, 2)

fig1.title.set_text('Orignal image')

fig1.imshow(img1)

fig2.title.set_text('Applied filter 7x7')

fig2.imshow(img1Fix)

plt.show()
img2Fix = cv2.medianBlur(img2,13)

fig = plt.figure(num=None, figsize=(80, 80), dpi=80, facecolor='w', edgecolor='k')

fig1 = fig.add_subplot(10, 20, 1)

fig2 = fig.add_subplot(10, 20, 2)

fig1.title.set_text('Orignal image')

fig1.imshow(img2)

fig2.title.set_text('Applied filter 13x13')

fig2.imshow(img2Fix)

plt.show()
img = cv2.imread('/kaggle/input/hw4_radiograph_1.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)



magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))



plt.subplot(121),plt.imshow(img, cmap = 'gray')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()
rows, cols = img.shape

crow,ccol = rows//2 , cols//2

# create a mask first, center square is 1, remaining all zeros

mask = np.zeros((rows,cols,2),np.uint8)

mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT

fshift = dft_shift*mask

f_ishift = np.fft.ifftshift(fshift)

img_back = cv2.idft(f_ishift)

img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(img_back, cmap = 'gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()
f = np.fft.fft2(img)

fshift = np.fft.fftshift(f)

magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])



rows, cols = img.shape

crow,ccol = rows//2 , cols//2

fshift[crow-30:crow+31, ccol-30:ccol+31] = 0

f_ishift = np.fft.ifftshift(fshift)

img_back = np.fft.ifft2(f_ishift)

img_back = np.real(img_back)

plt.subplot(131),plt.imshow(img, cmap = 'gray')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132),plt.imshow(img_back, cmap = 'gray')

plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])

plt.subplot(133),plt.imshow(img_back)

plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

plt.show()
img = cv2.imread('/kaggle/input/hw4_radiograph_2.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)



magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))



plt.subplot(121),plt.imshow(img, cmap = 'gray')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()
rows, cols = img.shape

crow,ccol = rows//2 , cols//2

# create a mask first, center square is 1, remaining all zeros

mask = np.zeros((rows,cols,2),np.uint8)

mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT

fshift = dft_shift*mask

f_ishift = np.fft.ifftshift(fshift)

img_back = cv2.idft(f_ishift)

img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(img_back, cmap = 'gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()
# https://github.com/glasgio/homomorphic-filter

# High-frequency filters implemented

# + butterworth

# + gaussian



class HomomorphicFilter:



    def __init__(self, a = 0.5, b = 1.5):

        self.a = float(a)

        self.b = float(b)



    # Filters

    def __butterworth_filter(self, I_shape, filter_params):

        P = I_shape[0]/2

        Q = I_shape[1]/2

        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')

        Duv = (((U-P)**2+(V-Q)**2)).astype(float)

        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])

        return (1 - H)



    def __gaussian_filter(self, I_shape, filter_params):

        P = I_shape[0]/2

        Q = I_shape[1]/2

        H = np.zeros(I_shape)

        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')

        Duv = (((U-P)**2+(V-Q)**2)).astype(float)

        H = np.exp((-Duv/(2*(filter_params[0])**2)))

        return (1 - H)



    # Methods

    def __apply_filter(self, I, H):

        H = np.fft.fftshift(H)

        I_filtered = (self.a + self.b*H)*I

        return I_filtered



    def filter(self, I, filter_params, filter='butterworth', H = None):

        #  Validating image

        if len(I.shape) is not 2:

            raise Exception('Improper image')



        # Take the image to log domain and then to frequency domain 

        I_log = np.log1p(np.array(I, dtype="float"))

        I_fft = np.fft.fft2(I_log)



        # Filters

        if filter=='butterworth':

            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)

        elif filter=='gaussian':

            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)

        elif filter=='external':

            print('external')

            if len(H.shape) is not 2:

                raise Exception('Invalid external filter')

        else:

            raise Exception('Selected filter not implemented')

        

        # Apply filter on frequency domain then take the image back to spatial domain

        I_fft_filt = self.__apply_filter(I = I_fft, H = H)

        I_filt = np.fft.ifft2(I_fft_filt)

        I = np.exp(np.real(I_filt))-1

        return np.uint8(I)

# End of class HomomorphicFilter
img = cv2.imread('/kaggle/input/hw4_radiograph_1.jpg',0)

img = cv2.medianBlur(img,7)                                             # -> remove moire (1)

homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)

img_filtered = homo_filter.filter(I=img, filter_params=[30,2])

plt.imshow(img_filtered)

plt.show()
img = cv2.imread('/kaggle/input/hw4_radiograph_2.jpg',0)

img = cv2.medianBlur(img,13)                                            # -> remove moire (1)

homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)

img_filtered = homo_filter.filter(I=img, filter_params=[30,2])

plt.imshow(img_filtered)

plt.show()
from scipy import fftpack

im = cv2.imread('/kaggle/input/hw4_radiograph_1.jpg',0)

im = cv2.medianBlur(im,7)                             

F1 = fftpack.fft2((im).astype(float))

F2 = fftpack.fftshift(F1)

w,h = im.shape

for i in range(60, w, 135):

    for j in range(100, h, 200):

        if not (i == 330 and j == 500):

            F2[i-10:i+10, j-10:j+10] = 0

for i in range(0, w, 135):

    for j in range(200, h, 200):

        if not (i == 330 and j == 500):

            F2[max(0,i-15):min(w,i+15), max(0,j-15):min(h,j+15)] = 0

plt.figure(figsize=(6.66,10))

plt.imshow( (20*np.log10( 0.1 + F2)).astype(int), cmap=plt.cm.gray)

plt.show()

im1 = fftpack.ifft2(fftpack.ifftshift(F2)).real

plt.figure(figsize=(10,10))

plt.imshow(im1, cmap='gray')

plt.axis('off')

plt.show()
from scipy import fftpack

im = cv2.imread('/kaggle/input/hw4_radiograph_2.jpg',0)

im = cv2.medianBlur(im,7)                             

F1 = fftpack.fft2((im).astype(float))

F2 = fftpack.fftshift(F1)

w,h = im.shape

for i in range(60, w, 135):

    for j in range(100, h, 200):

        if not (i == 330 and j == 500):

            F2[i-10:i+10, j-10:j+10] = 0

for i in range(0, w, 135):

    for j in range(200, h, 200):

        if not (i == 330 and j == 500):

            F2[max(0,i-15):min(w,i+15), max(0,j-15):min(h,j+15)] = 0

plt.figure(figsize=(6.66,10))

plt.imshow( (20*np.log10( 0.1 + F2)).astype(int), cmap=plt.cm.gray)

plt.show()

im1 = fftpack.ifft2(fftpack.ifftshift(F2)).real

plt.figure(figsize=(10,10))

plt.imshow(im1, cmap='gray')

plt.axis('off')

plt.show()