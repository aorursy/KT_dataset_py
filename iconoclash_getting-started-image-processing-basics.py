# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#Importing the other necessary libraries
import os
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#Importing OpenCV - the computer vision library
import cv2
# Glob the training data and load a single image path
training_paths = pathlib.Path('../input/stage1_train').glob('*/images/*.png')
training_sorted = sorted([x for x in training_paths])
im_path = training_sorted[45]
#To read the image 
bgrimg = cv2.imread(str(im_path))
plt.imshow(bgrimg)
plt.xticks([]) #To get rid of the x-ticks and y-ticks on the image axis
plt.yticks([])
print('Original Image Shape',bgrimg.shape)
#To see the structure of the image let's display one row of the image matrix
print('The entire shape of the image is',bgrimg.shape)
print('The first row of the image matrix contains',len(bgrimg[1]),'pixels')
print('The first row of the image matrix contains',bgrimg.shape[1],'pixels') #for alternative
print(bgrimg[1])
#To transfrom the colorspace from BGR to grayscale so as to make things simpler
grayimg = cv2.cvtColor(bgrimg,cv2.COLOR_BGR2GRAY)
#To plot the image
plt.imshow(grayimg,cmap='gray') #cmap has been used as matplotlib uses some default colormap to plot grayscale images
plt.xticks([]) #To get rid of the x-ticks and y-ticks on the image axis
plt.yticks([])
print('New Image Shape',grayimg.shape)
#To understand this further, let's display one entire row of the image matrix
print('The first row of the image matrix contains',len(grayimg[1]),'pixels')
print(grayimg[1])
#Okay let's look at the distribution of the intensity values of all the pixels
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.distplot(grayimg.flatten(),kde=False)#This is to flatten the matrix and put the intensity values of all the pixels in one single row vector
plt.title('Distribution of intensity values')

#To zoom in on the distribution and see if there is more than one prominent peak 
plt.subplot(1,2,2)
sns.distplot(grayimg.flatten(),kde=False) 
plt.ylim(0,20000) 
plt.title('Distribution of intensity values (Zoomed In)')
from skimage.filters import threshold_otsu
thresh_val = threshold_otsu(grayimg)
print('The optimal separation value is',thresh_val)
mask= np.where(grayimg>thresh_val,1,0)
mask
# mask= np.where(grayimg>thresh_val,1,3)
# mask
#To plot the original image and mask side by side
plt.figure(figsize=(12,12))

plt.subplot(1,2,1)
plt.imshow(grayimg,cmap='viridis')
plt.title('Original Image')

plt.subplot(1,2,2)
maskimg = mask.copy()
plt.imshow(maskimg, cmap='viridis')
plt.title('Mask')
newmask= np.where(grayimg>10,1,0)

#To plot the original image and mask side by side
plt.figure(figsize=(12,12))

plt.subplot(1,2,1)
plt.imshow(grayimg,cmap='viridis')
plt.title('Original Image')

plt.subplot(1,2,2)
newmaskimg = newmask.copy()
plt.imshow(newmaskimg, cmap='viridis')
plt.title('Mask with manual tresholding')
#Let's see if K-Means does a good job on this data 
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2) #2 as we're still trying to seperate the lighter coloured nuclei from the darker coloured background 
kmeans.fit(grayimg.reshape(grayimg.shape[0]*grayimg.shape[1],1))

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(kmeans.labels_.reshape(520,696),cmap='magma')
plt.title('K-Means')

plt.subplot(1,2,2)
plt.imshow(maskimg, cmap='viridis')
plt.title('Mask with Otsu Seperation')
grayimg.shape
grayimg.reshape(grayimg.shape[0]*grayimg.shape[1],1).shape
#To check if there's any difference
sum((kmeans.labels_.reshape(520,696)==mask).flatten())/(mask.shape[0]*mask.shape[1])
from scipy import ndimage
#To see this at a matrix level
matrix = np.array([[0,0,1,1,1,1],
                  [0,0,0,0,1,1],
                  [1,1,0,1,1,1],
                  [1,1,0,1,1,1]])
matrix
#Applying the ndimage.label function
ndimage.label(matrix)
labels,nlabels=ndimage.label(mask)
print('There are',nlabels,'distinct nuclei in the mask.')
print(labels.shape)
print(nlabels)
#Since we need to create a seperate mask for every nucelus, let's store the masks in an iterable like a list 
label_array=[]
#We need to iterate from 1 as ndimage.label encodes every object starting from number 1
for i in range(1,nlabels+1):
    label_mask = np.where(labels==i,1,0)
    label_array.append(label_mask)
#To see one such mask
label_array[68]
plt.imshow(label_array[68])
#Function for rle encoding
def rle(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])
#Running RLE on the last label_mask in label_array gives us 
rle(label_mask)
#To take a look at the different parts
im_path.parts
#Now defining a function that is applicable to all images
def basic(im_path):
    #Reading the image
    im_id=im_path.parts[-3] #To extract the image ID
    bgr = cv2.imread(str(im_path)) #Reading it in OpenCV
    gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY) #Converting everything to grayscale from BGR

    #To remove the background
    thresh_val = threshold_otsu(gray) #Using Otsu's method to seperate the foreground objects from the background
    mask = np.where(gray > thresh_val, 1, 0) #Coding objects with intensity values higher than background as 1
    
    #Extracting connected objects
    test_rle=pd.DataFrame()
    labels, nlabels = ndimage.label(mask) #labels gives us the label of the different objects in every image starting from 1 and nlabels gives us the total number of objects in every image
    for i in range(1,nlabels+1): #Iterating through every object/label
        label_mask = np.where(labels==i,1,0) #Individual masks for every nucleus
        RLE = rle(label_mask) #RLE for every mask
        solution = pd.Series({'ImageId': im_id, 'EncodedPixels': RLE})
        test_rle = test_rle.append(solution, ignore_index=True)
    
    #Return the dataframe
    return(test_rle)
        
#Defining a function that takes a list of image paths (pathlib.Path objects), analyzes each and returns a submission ready DataFrame
def list_of_images(im_path_list):
    all_df = pd.DataFrame()
    for im_path in im_path_list: #We'll use this for the test images
        im_df = basic(im_path) #Creating one dataframe for every image 
        all_df = all_df.append(im_df, ignore_index=True) #Appending all these dataframes
    
    #Returing the submission ready dataframe
    return (all_df)
#Final submission
test_images = pathlib.Path('../input/stage1_test/').glob('*/images/*.png')
basic_solution = list_of_images(list(test_images))
basic_solution.to_csv('basic_solution.csv', index=None)
#cv2.Sobel arguments - the image, output depth, order of derivative of x, order of derivative of y, kernel/filter matrix size
sobelx = cv2.Sobel(grayimg,int(cv2.CV_64F),1,0,ksize=3) #ksize=3 means we'll be using the 3x3 Sobel filter
sobely = cv2.Sobel(grayimg,int(cv2.CV_64F),0,1,ksize=3)

#To plot the vertical and horizontal edge detectors side by side
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(sobelx,cmap='gray')
plt.title('Sobel X (vertical edges)')
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(sobely,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Sobel Y (horizontal edges)')
#Plotting the original image
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(grayimg,cmap='gray')
plt.title('Original image')

#Now to combine the 2 sobel filters
sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
plt.subplot(1,2,2)
plt.imshow(sobel,cmap='gray')
plt.title('Sobel Filter')
#To highlight the problem areas
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(grayimg[350:450,485:530],cmap='gray')
plt.title('Original image (zoomed in)')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,2)
plt.imshow(sobel[350:450,485:530],cmap='gray')
plt.title('Sobel Filter (zoomed in)')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(maskimg[350:450,485:530], cmap='gray')
plt.title('Otsu/K-Means (zoomed in)')
plt.xticks([])
plt.yticks([])
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(grayimg,cmap='gray')
plt.title('Original image')
plt.xticks([])
plt.yticks([])

#Let's see how the Canny Edge Detector does on the image
plt.subplot(1,2,2)
canny = cv2.Canny(grayimg,0,21)
plt.imshow(canny,cmap='gray')
plt.title('Canny Edge Detection')
plt.xticks([])
plt.yticks([])
#Using contouring to create the masks
canny_cont=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1] #Using an approximation function to obtain the contour points and retreiving only the external contours

#To show the contour points
plt.figure(figsize=(14,8))
plt.imshow(canny,cmap='gray')
plt.title('Canny Edge Detection with contours')
plt.xticks([])
plt.yticks([])

for i in (range(len(canny_cont))):
    plt.scatter(canny_cont[i].flatten().reshape(len(canny_cont[i]),2)[:,0],
         canny_cont[i].flatten().reshape(len(canny_cont[i]),2)[:,1])
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(grayimg, cmap='gray')
plt.title('Original Image')

#Now to create masks with contours
background=np.zeros(grayimg.shape)
canny_mask=cv2.drawContours(background,canny_cont,-1,255,-1)

plt.subplot(1,2,2)
plt.imshow(canny_mask,cmap='gray')
plt.title('Creating masks with contours')
plt.xticks([])
plt.yticks([])
canny_mask_copy=canny_mask.copy()
canny_mask_clabels=ndimage.label(canny_mask_copy)[0]
for label_ind, label_mat in enumerate(ndimage.find_objects(canny_mask_clabels)):
    cell = canny_mask_clabels[label_mat]
    #Toheck if the label size is too small
    if np.product(cell.shape) < 100:
        canny_mask_clabels[np.where(canny_mask_clabels==label_ind+1)]=1
canny_mask_clabels=np.where(canny_mask_clabels>1,0,canny_mask_clabels)

#To show the original mask
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(canny_mask,cmap='gray')
plt.title('Masks created with edge plus contour detection')
plt.xticks([])
plt.yticks([])

#To plot the problem areas
plt.subplot(1,2,2)
plt.imshow(canny_mask_clabels,cmap='gray')
plt.title('Incomplete Masks')
plt.xticks([])
plt.yticks([])
#For convolving 2D arrays
from scipy import signal
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.distplot(np.where(canny_mask==255,1,0).flatten())
plt.title('Canny Mask')

plt.subplot(1,2,2)
#To smooth the canny_mask by convolving with a matrix that has all values = 1/9
canny_mask_smooth=signal.convolve2d(np.where(canny_mask==255,1,0),np.full((3,3),1/9),'same')
sns.distplot(canny_mask_smooth.flatten())
canny_mask_smooth_thresh=threshold_otsu(canny_mask_smooth)
plt.axvline(x=canny_mask_smooth_thresh)
plt.title('Smoothened Canny Mask with Otsu threshold value')
plt.figure(figsize=(12,6))
plt.imshow(canny_mask_smooth,cmap='gray')
plt.title('Smoothened canny mask')
plt.xticks([])
plt.yticks([])
#Setting all values above otsu's threshold as 0 in the matrix and in this image matrix setting all values above 0 as 1 
plt.figure(figsize=(12,6))
canny_conv1=np.where(np.where(canny_mask_smooth>canny_mask_smooth_thresh,0,canny_mask_smooth)>0,1,0)
plt.imshow(canny_conv1,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('After 1 convolution')
plt.figure(figsize=(12,6))
canny_mask_smooth2=signal.convolve2d(canny_conv1,np.full((3,3),1/9),'same')
canny_mask_smooth_thresh2=threshold_otsu(canny_mask_smooth2)
canny_conv2=np.where(canny_mask_smooth2>canny_mask_smooth_thresh2,1,0)
plt.imshow(canny_conv2,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('After 2 convolutions')
#Combing the 2 convolutions 
canny_cont=cv2.findContours(cv2.convertScaleAbs(canny_conv2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
background=np.zeros(grayimg.shape)
canny_mask=cv2.drawContours(background,canny_cont,-1,255,-1)

plt.figure(figsize=(12,6))
plt.imshow(canny_mask,cmap='gray')
plt.title('Contour detection after 2 convolutions')
plt.xticks([])
plt.yticks([])
#Let's try the same parameters for canny edge on other types of images - starting with another black background and white foreground image
for i in range(len(training_sorted)):
    if training_sorted[i].parts[-1]=='feffce59a1a3eb0a6a05992bb7423c39c7d52865846da36d89e2a72c379e5398.png':
        bwimg=cv2.imread(str(training_sorted[i]))
        bwimg=cv2.cvtColor(bwimg,cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20,8))
        plt.subplot(1,3,1)
        plt.imshow(bwimg)
        plt.title('Black background and white foreground')
        
        plt.subplot(1,3,2)
        bwimg=cv2.cvtColor(bwimg,cv2.COLOR_RGB2GRAY)
        bwimg_canny=cv2.Canny(bwimg,0,21)
        plt.imshow(bwimg_canny,cmap='gray')
        plt.title('Canny edge detection')
        
        plt.subplot(1,3,3)
        bwimg_cont=cv2.findContours(bwimg_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
        #Now to create masks with contours
        bwimg_bg=np.zeros(bwimg.shape)
        bwimg_mask=cv2.drawContours(bwimg_bg,bwimg_cont,-1,255,-1)
        
        #Convolving once
        bwimg_mask_smooth=signal.convolve2d(np.where(bwimg_mask==255,1,0),np.full((3,3),1/9),'same')
        bwimg_mask_smooth_thresh=threshold_otsu(bwimg_mask_smooth)
        bwimg_conv1=np.where(np.where(bwimg_mask_smooth>bwimg_mask_smooth_thresh,0,bwimg_mask_smooth)>0,1,0)
        
        #Convolving again
        bwimg_mask_smooth2=signal.convolve2d(bwimg_conv1,np.full((3,3),1/9),'same')
        bwimg_mask_smooth_thresh2=threshold_otsu(bwimg_mask_smooth2)
        bwimg_conv2=np.where(bwimg_mask_smooth2>bwimg_mask_smooth_thresh2,1,0)
        
        #Now to create masks with contours after 2 convolutions
        bwimg_cont=cv2.findContours(cv2.convertScaleAbs(bwimg_conv2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
        bwimg_bg=np.zeros(bwimg.shape)
        bwimg_mask=cv2.drawContours(bwimg_bg,bwimg_cont,-1,255,-1)

        plt.imshow(bwimg_mask,cmap='gray')
        plt.title('Contour detection after 2 convolutions')
#Purple background and purple foreground
for i in range(len(training_sorted)):
    if training_sorted[i].parts[-1]=='0e21d7b3eea8cdbbed60d51d72f4f8c1974c5d76a8a3893a7d5835c85284132e.png':
        ppimg=cv2.imread(str(training_sorted[i]))
        ppimg=cv2.cvtColor(ppimg,cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20,8))
        plt.subplot(1,3,1)
        plt.imshow(ppimg)
        plt.title('Purple background and purple foreground')
        
        plt.subplot(1,3,2)
        ppimg=cv2.cvtColor(ppimg,cv2.COLOR_RGB2GRAY)
        ppimg_canny=cv2.Canny(ppimg,20,100)
        plt.imshow(ppimg_canny,cmap='gray')
        plt.title('Canny edge detection')
        
        plt.subplot(1,3,3)
        ppimg_cont=cv2.findContours(ppimg_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
        #Now to create masks with contours
        ppimg_bg=np.zeros(ppimg.shape)
        ppimg_mask=cv2.drawContours(ppimg_bg,ppimg_cont,-1,255,-1)
        
        #Convolving once
        ppimg_mask_smooth=signal.convolve2d(np.where(ppimg_mask==255,1,0),np.full((3,3),1/9),'same')
        ppimg_mask_smooth_thresh=threshold_otsu(ppimg_mask_smooth)
        ppimg_conv1=np.where(np.where(ppimg_mask_smooth>ppimg_mask_smooth_thresh,0,ppimg_mask_smooth)>0,1,0)
        
        #Convolving again
        ppimg_mask_smooth2=signal.convolve2d(ppimg_conv1,np.full((3,3),1/9),'same')
        ppimg_mask_smooth_thresh2=threshold_otsu(ppimg_mask_smooth2)
        ppimg_conv2=np.where(ppimg_mask_smooth2>ppimg_mask_smooth_thresh2,1,0)
        
        #Now to create masks with contours after 2 convolutions
        ppimg_cont=cv2.findContours(cv2.convertScaleAbs(ppimg_conv2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
        ppimg_bg=np.zeros(ppimg.shape)
        ppimg_mask=cv2.drawContours(ppimg_bg,ppimg_cont,-1,255,-1)

        plt.imshow(ppimg_mask,cmap='gray')
        plt.title('Contour detection after 2 convolutions')
#White background and purple foreground
for i in range(len(training_sorted)):
    if training_sorted[i].parts[-1]=='0121d6759c5adb290c8e828fc882f37dfaf3663ec885c663859948c154a443ed.png':
        wpimg=cv2.imread(str(training_sorted[i]))
        wpimg=cv2.cvtColor(wpimg,cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20,8))
        plt.subplot(1,3,1)
        plt.imshow(wpimg)
        plt.title('White background and purple foreground')
        
        plt.subplot(1,3,2)
        wpimg=cv2.cvtColor(wpimg,cv2.COLOR_RGB2GRAY)
        wpimg_canny=cv2.Canny(wpimg,20,100)
        plt.imshow(wpimg_canny,cmap='gray')
        plt.title('Canny edge detection')
        
        plt.subplot(1,3,3)
        wpimg_cont=cv2.findContours(wpimg_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
        #Now to create masks with contours
        wpimg_bg=np.zeros(wpimg.shape)
        wpimg_mask=cv2.drawContours(wpimg_bg,wpimg_cont,-1,255,-1)
        
        #Convolving once
        wpimg_mask_smooth=signal.convolve2d(np.where(wpimg_mask==255,1,0),np.full((3,3),1/9),'same')
        wpimg_mask_smooth_thresh=threshold_otsu(wpimg_mask_smooth)
        wpimg_conv1=np.where(np.where(wpimg_mask_smooth>wpimg_mask_smooth_thresh,0,wpimg_mask_smooth)>0,1,0)
        
        #Convolving again
        wpimg_mask_smooth2=signal.convolve2d(wpimg_conv1,np.full((3,3),1/9),'same')
        wpimg_mask_smooth_thresh2=threshold_otsu(wpimg_mask_smooth2)
        wpimg_conv2=np.where(wpimg_mask_smooth2>wpimg_mask_smooth_thresh2,1,0)
        
        #Now to create masks with contours after 2 convolutions
        wpimg_cont=cv2.findContours(cv2.convertScaleAbs(wpimg_conv2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
        wpimg_bg=np.zeros(wpimg.shape)
        wpimg_mask=cv2.drawContours(wpimg_bg,wpimg_cont,-1,255,-1)

        plt.imshow(wpimg_mask,cmap='gray')
        plt.title('Contour detection after 2 convolutions')
#White background and black foreground
for i in range(len(training_sorted)):
    if training_sorted[i].parts[-1]=='08275a5b1c2dfcd739e8c4888a5ee2d29f83eccfa75185404ced1dc0866ea992.png':
        wbimg=cv2.imread(str(training_sorted[i]))
        wbimg=cv2.cvtColor(wbimg,cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20,8))
        plt.subplot(1,3,1)
        plt.imshow(wbimg)
        plt.title('White background and black foreground')
        
        plt.subplot(1,3,2)
        wbimg=cv2.cvtColor(wbimg,cv2.COLOR_RGB2GRAY)
        wbimg_canny=cv2.Canny(wbimg,20,100)
        plt.imshow(wbimg_canny,cmap='gray')
        plt.title('Canny edge detection')
        
        plt.subplot(1,3,3)
        wbimg_cont=cv2.findContours(wbimg_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
        #Now to create masks with contours
        wbimg_bg=np.zeros(wbimg.shape)
        wbimg_mask=cv2.drawContours(wbimg_bg,wbimg_cont,-1,255,-1)
        
        #Convolving once
        wbimg_mask_smooth=signal.convolve2d(np.where(wbimg_mask==255,1,0),np.full((5,5),1/25),'same')
        wbimg_mask_smooth_thresh=threshold_otsu(wbimg_mask_smooth)
        wbimg_conv1=np.where(np.where(wbimg_mask_smooth>wbimg_mask_smooth_thresh,0,wbimg_mask_smooth)>0,1,0)
        
        #Convolving again
        wbimg_mask_smooth2=signal.convolve2d(wbimg_conv1,np.full((5,5),1/25),'same')
        wbimg_mask_smooth_thresh2=threshold_otsu(wbimg_mask_smooth2)
        wbimg_conv2=np.where(wbimg_mask_smooth2>wbimg_mask_smooth_thresh2,1,0)
        
        #Now to create masks with contours after 2 convolutions
        wbimg_cont=cv2.findContours(cv2.convertScaleAbs(wbimg_conv2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
        wbimg_bg=np.zeros(wbimg.shape)
        wbimg_mask=cv2.drawContours(wbimg_bg,wbimg_cont,-1,255,-1)

        plt.imshow(wbimg_conv2,cmap='gray')
        plt.title('Contour detection after 2 convolutions')
#There are some images in the test set with a yellow background and purple foreground
test_images = pathlib.Path('../input/stage1_test/').glob('*/images/*.png')
testing_sorted=sorted([x for x in test_images])
for i in range(len(testing_sorted)):
    if testing_sorted[i].parts[-1]=='9f17aea854db13015d19b34cb2022cfdeda44133323fcd6bb3545f7b9404d8ab.png':
        ypimg=cv2.imread(str(testing_sorted[i]))
        ypimg=cv2.cvtColor(ypimg,cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20,8))
        plt.subplot(1,3,1)
        plt.imshow(ypimg)
        plt.title('Yellow background and purple foreground')
        
        plt.subplot(1,3,2)
        ypimg=cv2.cvtColor(ypimg,cv2.COLOR_RGB2GRAY)
        ypimg_canny=cv2.Canny(ypimg,100,200)
        plt.imshow(ypimg_canny,cmap='gray')
        plt.title('Canny edge detection')
        
        plt.subplot(1,3,3)
        ypimg_cont=cv2.findContours(ypimg_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
        #Now to create masks with contours
        ypimg_bg=np.zeros(ypimg.shape)
        ypimg_mask=cv2.drawContours(ypimg_bg,ypimg_cont,-1,255,-1)
        
        #Convolving once
        ypimg_mask_smooth=signal.convolve2d(np.where(ypimg_mask==255,1,0),np.full((3,3),1/9),'same')
        ypimg_mask_smooth_thresh=threshold_otsu(ypimg_mask_smooth)
        ypimg_conv1=np.where(np.where(ypimg_mask_smooth>ypimg_mask_smooth_thresh,0,ypimg_mask_smooth)>0,1,0)
        
        #Convolving again
        ypimg_mask_smooth2=signal.convolve2d(ypimg_conv1,np.full((3,3),1/9),'same')
        ypimg_mask_smooth_thresh2=threshold_otsu(ypimg_mask_smooth2)
        ypimg_conv2=np.where(ypimg_mask_smooth2>ypimg_mask_smooth_thresh2,1,0)
        
        #Now to create masks with contours after 2 convolutions
        ypimg_cont=cv2.findContours(cv2.convertScaleAbs(ypimg_conv2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
        ypimg_bg=np.zeros(ypimg.shape)
        ypimg_mask=cv2.drawContours(ypimg_bg,ypimg_cont,-1,255,-1)

        plt.imshow(ypimg_conv2,cmap='gray')
        plt.title('Contour detection after 2 convolutions')
train_path = '../input/stage1_train/'
test_path = '../input/stage1_test/'
train_ids = os.listdir(train_path)
def LabelMerge(imgpath):
    #to get all the png files
    png_files = [f for f in os.listdir(imgpath) if f.endswith('.png')]
    #to load the image as a grayscale
    img = cv2.imread(imgpath+'/'+png_files[0],0)
    for i in png_files[1:]:
        temp_img = cv2.imread(imgpath+'/'+i,0)
        img = img+temp_img
    return(img)
path = train_path+training_sorted[45].parts[-3]+'/masks/'
combined_mask=LabelMerge(path)
plt.imshow(combined_mask,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Combined Mask')
objects=ndimage.label(canny_mask)[0]
plt.figure(figsize=(16,8))
plt.subplot(1,3,1)
plt.imshow(grayimg[ndimage.find_objects(objects)[20]],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Nuclei in the original image')

plt.subplot(1,3,2)
plt.imshow(canny_mask[ndimage.find_objects(objects)[20]],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Created mask')

plt.subplot(1,3,3)
plt.imshow(combined_mask[ndimage.find_objects(objects)[20]],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Label from the combined mask')
#To get one dataframe for all the pixels within all the bounding boxes in an image
pixels_gs=pd.DataFrame()
columns=[]
for i in range(9):
    columns.append('pixel-'+str(i))
columns=columns+['label']
bounding=ndimage.find_objects(objects)
for bbox in bounding:
    for i in range(1,canny_mask[bbox].shape[0]-1):
        for j in range(1,canny_mask[bbox].shape[1]-1):
            pixel0=grayimg[bbox][i][j] #center pixel
            pixel1=grayimg[bbox][i-1][j-1] #top left pixel
            pixel2=grayimg[bbox][i-1][j] #pixel above the center pixel
            pixel3=grayimg[bbox][i-1][j+1] #top right pixel
            pixel4=grayimg[bbox][i][j-1] #pixel to the left of center pixel
            pixel5=grayimg[bbox][i][j+1] #pixel to the right of center pixel
            pixel6=grayimg[bbox][i+1][j-1] #bottom left pixel
            pixel7=grayimg[bbox][i+1][j] #pixel to the bottom of center pixel 
            pixel8=grayimg[bbox][i+1][j+1] #bottom right pixel
            label=combined_mask[i][j] #label of the center pixel
            neighbors = pd.Series({a:b for (a,b) in zip(columns,[pixel0,pixel1,pixel2,pixel3,pixel4,pixel5,pixel6,pixel7,pixel8,label])})
            pixels_gs = pixels_gs.append(neighbors, ignore_index=True)
#To see the head of the dataframe
pixels_gs.head()
pixels_gs['label'].value_counts()
#To divide the data into training and testing sets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(pixels_gs.drop('label',axis=1),pixels_gs['label'],test_size=0.3,random_state=101)
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
rfc_pred=rfc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
predicted=np.zeros((canny_mask.shape))
bbox=[]
bbox_dim_prod=[0]
rfc_pred = rfc.predict(pixels_gs.drop('label',axis=1))
for i in range(len(bounding)):
    bbox_dim=np.array(list(background[bounding[i]].shape))-2 #Since we are taking 1 to (n-1) rows and 1 to (n-1) columns
    bbox_dim_prod.append(np.product(bbox_dim)) #for indexing
    bbox_pred=rfc_pred[sum(bbox_dim_prod[0:i+1]):sum(bbox_dim_prod[0:i+1])+np.product(bbox_dim)].reshape(bbox_dim[0],bbox_dim[1]) #for reshaping the predicted labels into the reduced dimensions of the bounding box 
    bbox.append(bbox_pred)
    predicted[bounding[i]][1:predicted[bounding[i]].shape[0]-1,1:predicted[bounding[i]].shape[1]-1]=bbox[i]
plt.figure(figsize=(13,7))
plt.subplot(1,2,1)
plt.imshow(combined_mask,cmap='gray')
plt.title('Combined Mask')
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(predicted,cmap='gray')
plt.title('Predicted Mask')
plt.xticks([])
plt.yticks([])