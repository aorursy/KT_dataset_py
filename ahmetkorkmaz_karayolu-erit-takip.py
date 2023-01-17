# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

calibration_file = {}

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import cv2

fourcc = cv2.VideoWriter_fourcc(*'XVID')

from scipy import ndimage

from scipy.ndimage.filters import convolve

from scipy import misc
class cannyEdgeDetector:

    def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=70, strong_pixel=200, lowthreshold=0.10, highthreshold=0.10):

        self.img = imgs

        self.img_smoothed = None

        self.gradientMat = None

        self.thetaMat = None

        self.nonMaxImg = None

        self.thresholdImg = None

        self.weak_pixel = weak_pixel

        self.strong_pixel = strong_pixel

        self.sigma = sigma

        self.kernel_size = kernel_size

        self.lowThreshold = lowthreshold

        self.highThreshold = highthreshold

        return 

    

    def gaussian_kernel(self, size, sigma=1):

        size = int(size) * 3

        x, y = np.mgrid[-size:size+1, -size:size+1]

        normal = 1 / (2.0 * np.pi * sigma**2)

        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

        return g

    

    def sobel_filters(self, img):

        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)

        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)



        Ix = ndimage.filters.convolve(img, Kx)

        Iy = ndimage.filters.convolve(img, Ky)



        G = np.hypot(Ix, Iy)

        G = G / G.max() * 255

        theta = np.arctan2(Iy, Ix)

        return (G, theta)

    



    def non_max_suppression(self, img, D):

        M, N = img.shape

        Z = np.zeros((M,N), dtype=np.int32)

        angle = D * 180. / np.pi

        angle[angle < 0] += 180





        for i in range(1,M-1):

            for j in range(1,N-1):

                try:

                    q = 255

                    r = 255



                   #angle 0

                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):

                        q = img[i, j+1]

                        r = img[i, j-1]

                    #angle 45

                    elif (22.5 <= angle[i,j] < 67.5):

                        q = img[i+1, j-1]

                        r = img[i-1, j+1]

                    #angle 90

                    elif (67.5 <= angle[i,j] < 112.5):

                        q = img[i+1, j]

                        r = img[i-1, j]

                    #angle 135

                    elif (112.5 <= angle[i,j] < 157.5):

                        q = img[i-1, j-1]

                        r = img[i+1, j+1]



                    if (img[i,j] >= q) and (img[i,j] >= r):

                        Z[i,j] = img[i,j]

                    else:

                        Z[i,j] = 0





                except IndexError as e:

                    pass



        return Z



    def threshold(self, img):



        highThreshold = img.max() * self.highThreshold;

        lowThreshold = highThreshold * self.lowThreshold;



        M, N = img.shape

        res = np.zeros((M,N), dtype=np.int32)



        weak = np.int32(self.weak_pixel)

        strong = np.int32(self.strong_pixel)



        strong_i, strong_j = np.where(img >= highThreshold)

        zeros_i, zeros_j = np.where(img < lowThreshold)



        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))



        res[strong_i, strong_j] = strong

        res[weak_i, weak_j] = weak



        return (res)



    def hysteresis(self, img):



        M, N = img.shape

        weak = self.weak_pixel

        strong = self.strong_pixel



        for i in range(1, M-1):

            for j in range(1, N-1):

                if (img[i,j] == weak):

                    try:

                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)

                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)

                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):

                            img[i, j] = strong

                        else:

                            img[i, j] = 0

                    except IndexError as e:

                        pass



        return img

    

    def detect(self):

        self.img_smoothed = convolve(self.img, self.gaussian_kernel(self.kernel_size, self.sigma))

        self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)

        self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)

        self.thresholdImg = self.threshold(self.nonMaxImg)

        img_final = self.hysteresis(self.thresholdImg)

        return img_final
def binary_image(gray_img,treshold):

    M, N = gray_img.shape

    Z = np.zeros((M,N), dtype=np.int32)

    for i in range(1,M-1):

        for j in range(1,N-1):

            if gray_img[i,j]>treshold:

                Z[i,j]=255

    return Z

            
def bgr2gray(bgr):



    b, g, r = bgr[:,:,0], bgr[:,:,1], bgr[:,:,2]

    gray = 0.333 * b + 0.334 * g + 0.333 * r



    return gray
def visualize(imgs, format=None, gray=False):

    plt.figure(figsize=(20, 40))

    plt.imshow(imgs, format)

    plt.show()
img_mp = mpimg.imread('/kaggle/input/yol-resimleri/resim-4.png')

img_mp = cv2.resize(img_mp, (250,444), interpolation = cv2.INTER_AREA)

img_gray = bgr2gray(img_mp)

plt.imshow(img_mp)

#visualize(img_mp)

#visualize(img_gray)
detector=cannyEdgeDetector(img_gray)

canny_image=detector.detect()

plt.imshow(canny_image)

#visualize(canny_image)

def reigon_of_interest(image):

    height=len(image)

    width=len(image[0])

    polygons=np.array([

        [(int(width*0.30),int(height*0.90)),(int(width/2),int(height*0.75)),(int(width*0.70),int(height*0.90))]

    ])

    mask=np.zeros_like(image)

    cv2.fillPoly(mask,polygons,255)

    masked_img=cv2.bitwise_and(image,mask)

    return masked_img

roi_image1=reigon_of_interest(canny_image)

plt.imshow(roi_image1)

#visualize(roi_image1)
def get_hist(img):

    hist = np.sum(img[img.shape[0]//2:,:], axis=0)

    return hist
img_binary=binary_image(roi_image1,127)

img_hist=get_hist(img_binary)

plt.plot(img_hist, label="beyaz",color='r')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
def orta_toplam(img_roi):

    sum=0

    M, N = img_roi.shape

    for i in range(1,M-1):

        for j in range(1,N-1):

            sum=sum+img_roi[i,j]

    return (sum)

ihlal=orta_toplam(roi_image1)

print(ihlal)
#lines_edges = cv2.addWeighted(img, 1.0, line_image, 1, 5)

font = cv2.FONT_HERSHEY_SIMPLEX

if ihlal>35000:

    cv2.putText(img_mp,'Serit ihlali var',(15,40), font, 1.0,(255,0,0),2,cv2.LINE_AA)

else:

    cv2.putText(img_mp,'Serit ihlali yok',(15,40), font, 1.0,(0,255,0),2,cv2.LINE_AA)

plt.imshow(img_mp)
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (250,444))

cap = cv2.VideoCapture('/kaggle/input/video-karabuk/VIDEO-2020-01-09-15-45-48.mp4')

sayac=0

h=0

w=0

h2=0

w2=0

while(cap.isOpened()):

    ret, frame = cap.read()

    if ret == True:

        sayac=sayac+1

        if sayac >1850:

            img_vid = frame

            img_vid = cv2.resize(img_vid, (444,250), interpolation = cv2.INTER_AREA)



            (w, h) = img_vid.shape[:2]

            '''

            center = (125, 222)

            angle90 = 90

            scale = 1.0

            M = cv2.getRotationMatrix2D(center, angle90, scale)

            img_vid = cv2.warpAffine(img_vid, M, (h, w))

            '''

            img_vid=cv2.rotate(img_vid, cv2.ROTATE_90_COUNTERCLOCKWISE)

            (w2, h2) = img_vid.shape[:2]

            img_gray_vid = bgr2gray(img_vid)

            detector=cannyEdgeDetector(img_gray_vid)

            canny_image_vid=detector.detect()

            roi_image_vid = reigon_of_interest(canny_image_vid)

            ihlal_vid = orta_toplam(roi_image_vid)

            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(img_vid,str(ihlal_vid),(15,20), font, 1.0,(255,0,0),2,cv2.LINE_AA)

            if ihlal_vid>35000:

                cv2.putText(img_vid,'Serit ihlali var',(15,60), font, 1.0,(0,0,255),2,cv2.LINE_AA)

            else:

                cv2.putText(img_vid,'Serit ihlali yok',(15,60), font, 1.0,(0,255,0),2,cv2.LINE_AA)



            out.write(img_vid)

    else:

        print(h,h2,w,w2)

        break

cap.release()

out.release()

cv2.destroyAllWindows()