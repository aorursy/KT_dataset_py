# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import cv2

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.data import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("C:\\Users\\admin\\AppData\\Roaming\\jupyter\\input"))
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = os.listdir('../input/train')
print(len(train))

test = os.listdir('../input/test')
print(len(test))
submission = pd.read_csv('../input/sample_submission.csv')
submission.head()
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction
masks = pd.read_csv('../input/train_ship_segmentations.csv')
masks.head()
from sklearn.cluster import KMeans
class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    IMAGES = None
    INPUT = None
    FREQUENCY = None
    HSV = None
    BGR = None
    GRAY = None
    
    def __init__(self, image, clusters=8):
        self.IMAGES = {}
        self.CLUSTERS = clusters
        self.IMAGE = image
        self.INPUT = cv2.imread(self.IMAGE)
        self.IMAGES['00_original RGB'] = cv2.cvtColor(self.INPUT, cv2.COLOR_BGR2RGB)
        self.INPUT = cv2.cvtColor(self.INPUT, cv2.COLOR_BGR2HLS)
        self.IMAGES['10_converted CMAP'] = self.INPUT

    def dominantColors(self):
        self.SMALL = cv2.pyrDown(self.INPUT)
        Z = self.SMALL.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,self.LABELS,self.COLORS=cv2.kmeans(Z,self.CLUSTERS,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        #print(self.COLORS)
        #print('HSV color:', self.COLORS)        

    def calcFrequency(self):
        #labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)
        #create frequency count tables    
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        
        #appending frequencies to cluster centers
        colors = self.COLORS
        
        #descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()]
        self.MAX = hist[0]
        return zip(colors,hist)

    def equalizeHistogram(self):
        H, S, V = cv2.split(cv2.cvtColor(cv2.cvtColor(self.INPUT, cv2.COLOR_HLS2BGR), cv2.COLOR_BGR2HSV))
        eq_V = cv2.equalizeHist(V)
        self.INPUT = cv2.merge([H, S, eq_V])
        self.INPUT = cv2.cvtColor(cv2.cvtColor(self.INPUT, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2HLS)
        self.IMAGES['20_equalized HLS'] = self.INPUT

    def denoiseImage(self):
        self.INPUT = cv2.fastNlMeansDenoisingColored(self.INPUT,None,8,10,7,21)
        self.IMAGES['30_denoised HLS'] = self.INPUT        
    
    def filterMeans(self, COLOR):
        lower_h = np.clip(COLOR[0]-7,  0, 180)
        upper_h = np.clip(COLOR[0]+7,  0, 180)
        lower_color = np.array([lower_h,1,1])
        upper_color = np.array([upper_h,255,255])
        mask_cv = cv2.inRange(self.INPUT, lower_color, upper_color)
        mask_cv = 255-mask_cv
        self.INPUT = cv2.bitwise_and(self.INPUT,self.INPUT,mask=mask_cv)
        self.IMAGES['40_means_removed HLS'] = self.INPUT 

    def filterClouds(self):
        lower_h = 0
        lower_l = 240
        lower_s = 0
        upper_h = 255
        upper_l = 255
        upper_s = 255
        lower_color = np.array([lower_h,lower_l,lower_s])
        upper_color = np.array([upper_h,upper_l,upper_s])
        mask_kmeans = cv2.inRange(self.INPUT, lower_color, upper_color)
        mask_kmeans = 255-mask_kmeans
        self.INPUT = cv2.bitwise_and(self.INPUT,self.INPUT,mask=mask_kmeans)
        self.IMAGES['41_clouds_removed HLS'] = self.INPUT
        
    # processImage
    def plotImage(self):
        self.plotHistogram()
        #self.denoiseImage()
        #self.equalizeHistogram()
        for COLOR, FREQUENCY in self.calcFrequency():
            #print(ImageId, ': ', COLOR, FREQUENCY)
            if FREQUENCY > 0.022:
                self.filterMeans(COLOR)
                print('Removed color:', COLOR, 'with frequency:', FREQUENCY)   
            # finally filter clouds, prevents black from becoming biggest k-mean
            #self.filterClouds()

        self.BLUR = cv2.medianBlur(self.INPUT,5)        
        #self.BLUR = self.INPUT
                
        ### find contours
        kernel = np.ones((5,5),np.uint8)
        cimage = cv2.cvtColor(self.BLUR, cv2.COLOR_HLS2BGR)
        cimage = cv2.cvtColor(cimage, cv2.COLOR_BGR2GRAY)
        closing = cv2.morphologyEx(cimage, cv2.MORPH_GRADIENT, kernel,iterations = 2 )
        self.IMAGES['50_morph_gradient'] = closing.astype(np.uint8)
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.erode(closing,kernel,iterations = 2)
        
        im2,contours,hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        closing = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10 and area < 15000:
                
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(closing,[box],0,(0,0,255),2)
        self.IMAGES['99_result'] = closing
        self.dominantColors()
        self.calcFrequency()
        if self.MAX > 0.98:
            self.plotHistogram()
            for key,OUTPUTIMAGE in self.IMAGES.items():
                print(key)
                if key.endswith('HLS'):
                    plt.figure()
                    plt.axis("off")
                    plt.imshow(cv2.cvtColor(OUTPUTIMAGE, cv2.COLOR_HLS2RGB))
                    plt.show()   
                elif key.endswith('BGR'):
                    plt.figure()
                    plt.axis("off")
                    plt.imshow(cv2.cvtColor(OUTPUTIMAGE, cv2.COLOR_BGR2RGB))
                    plt.show()   
                elif key.endswith('CMAP'):
                    plt.figure()
                    plt.axis("off")
                    plt.imshow(OUTPUTIMAGE,cmap='hot')
                    plt.show()   
                else:
                    plt.figure()
                    plt.axis("off")
                    plt.imshow(OUTPUTIMAGE)
                    plt.show()   

    # optional
    def plotHistogram(self):
        #labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)
        #create frequency count tables    
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        
        #appending frequencies to cluster centers
        colors = self.COLORS
        
        #descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()] 

        #creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        #creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500
            
            #getting rgb values
            h = int(colors[i][0])
            s = int(colors[i][1])
            v = int(colors[i][2])
            
            #using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (h,s,v), -1)
            start = end

        #display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(cv2.cvtColor(chart, cv2.COLOR_HLS2RGB))
        plt.show()

def remove_white(img_hsv, sensitivity):
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])    

    # Threshold the HSV image to get only white colors
    mask_cv = cv2.inRange(img_hsv, lower_white, upper_white)
    mask_cv = 255-mask_cv
    # Bitwise-OR mask and original image
    return cv2.bitwise_or(img_hsv,img_hsv,mask=mask_cv)

images = os.listdir('../input/train')[0:20]
all = len(images)
count = 0
for ImageId in images:    # masks.head(200)['ImageId']: # .head(200) 
    count = count + 1
    img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
    if len(img_masks)>0:
        print('Done %s / %s' % (count,all))
        # Take the individual ship masks and create a single mask array for all ships
        all_masks = np.zeros((768, 768))
        for mask in img_masks:
            try:
                all_masks += rle_decode(mask)
            except:
                break

        print('=================================================================')
        print('Initializing...')
        dc = DominantColors('../input/train/' + ImageId, 8) 
        print('Dominant colors...')
        colors = dc.dominantColors()
        print('Plotting...')
        dc.plotImage()
        print('=================================================================')
        



