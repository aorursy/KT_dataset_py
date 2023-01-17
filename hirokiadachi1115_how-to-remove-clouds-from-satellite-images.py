import pandas as pd

import os

import matplotlib.pyplot as plt

import glob

import numpy as np
USA=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/USA.npy')

USA.shape
plt.figure(figsize=(16,16))

for i, image in enumerate(USA[0:15]):

    plt.subplot(5,3,i+1)

    plt.imshow(image, 'gray')
# calculate the total NTL value and add time axis

USA_total = []

for i, image in enumerate(USA):

    USA_total.append(image.sum())



dates = pd.date_range(start='1/1/2020', periods=len(USA), freq='D')

USA_total = pd.Series(USA_total, index=dates)



plt.figure(figsize=(16,4))

plt.plot(USA_total)

plt.title("total value of NTL", fontsize=16)
# simple seasonal decomposition

import statsmodels.api as sm

res = sm.tsa.seasonal_decompose(USA_total, period=30)



plt.figure(figsize=(16, 8)) 



plt.subplot(411)

plt.plot(USA_total)

plt.ylabel('Original')



plt.subplot(412)

plt.plot(res.trend)

plt.ylabel('Trend')



plt.subplot(413) 

plt.plot(res.seasonal)

plt.ylabel('Seasonality')



plt.subplot(414)

plt.plot(res.resid)

plt.ylabel('Residuals')
plt.figure(figsize=(16,16))

for i, image in enumerate(USA[-45:-30]):

    plt.subplot(5,3,i+1)

    plt.imshow(image, 'gray')
import cv2

# external_contours is a black image

external_contours = np.zeros(USA[-39,:,:].shape)

temp_image = USA[-39,:,:]



# original

plt.figure(figsize=(16,16))

plt.subplot(1,3,1)

plt.imshow(temp_image, 'gray')

plt.title('original')



#  If the pixel value is smaller than the maimum value, set it to 0.

ret, thresh = cv2.threshold(temp_image,temp_image.max()-1,temp_image.max(),cv2.THRESH_BINARY)

plt.subplot(1,3,2)

plt.imshow(thresh, 'gray')

plt.title('thresh_binary')



# find countours 

contours, hierarchy = cv2.findContours(thresh.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)



# calculate each countour area and draw if it is larger than 500 pixels

for i, cnt in enumerate(contours):

    area = cv2.contourArea(cnt)

    if area >= 500:

        cv2.drawContours(external_contours, contours, i, 255, -1)



plt.subplot(1,3,3)

plt.imshow(external_contours,cmap='gray')

plt.title('contour area which is larger than 500 pixels') 
USA_nan = np.empty((0,794,1740), int)

for i, image in enumerate(USA):

    image_nan = image.copy()



    ret, thresh = cv2.threshold(image,image.max()-1,image.max(),cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)



    for j, cnt in enumerate(contours):

        area = cv2.contourArea(cnt)

        if area >= 500:

            image_nan = cv2.fillConvexPoly(image_nan, contours[j], (np.nan))

    image_nan = image_nan[np.newaxis,:,:]

    USA_nan = np.append(USA_nan, image_nan, axis=0)
plt.figure(figsize=(16,16))

for i, image in enumerate(USA_nan[-45:-30]):

    plt.subplot(5,3,i+1)

    plt.imshow(image, 'gray')
# calculate the average NTL value and add time axis

USA_nan_avg = []

for i, image in enumerate(USA_nan):

    USA_nan_avg.append(np.nanmean(image))



dates = pd.date_range(start='1/1/2020', periods=len(USA), freq='D')

USA_nan_avg = pd.Series(USA_nan_avg, index=dates)



plt.figure(figsize=(16,4))

plt.plot(USA_nan_avg)

plt.title("average value of NTL", fontsize=16)
z,y,x = USA_nan.shape

USA_resized = np.empty((0,int(y/10),int(x/10)), int)



for i, image in enumerate(USA_nan):

    image_resized = cv2.resize(image,None,fx=0.1,fy=0.1,interpolation=cv2.INTER_AREA) 

    image_resized = image_resized[np.newaxis,:,:]

    USA_resized = np.append(USA_resized, image_resized, axis=0)
plt.figure(figsize=(16,16))

for i, image in enumerate(USA_resized[-45:-30]):

    plt.subplot(5,3,i+1)

    plt.imshow(image, 'gray')
z, y, x = USA_resized.shape



dates = pd.date_range(start='1/1/2020', periods=z, freq='D')

USA_changed = pd.DataFrame()



for i in range(y):

    for j in range(x):

        time_series = pd.Series(USA_resized[:,i,j], index=dates)

        # Check each pixel in chronological order, and if even one has a value of 0, set all to 0.

        if (np.nanmin(time_series) == 0):

            pixel_value = pd.Series(0,index=dates)

        else:

            # interpolate NAN value and replace it with the trend value

            time_series.interpolate(method='index',limit_direction='both',inplace=True)

            res = sm.tsa.seasonal_decompose(time_series, period=30)

            pixel_value = res.trend

        USA_changed = USA_changed.append(pixel_value, ignore_index=True)
plt.figure(figsize=(16,16))

for i, image in enumerate(range(15)):

    plt.subplot(5,3,i+1)

    plt.imshow(USA_changed.iloc[:,i-45].values.reshape(y,x),'gray')
plt.figure(figsize=(16, 4)) 

plt.plot(USA_changed.dropna(axis=1).sum())
# seems to be good, save the result

pd.to_pickle(USA_changed, "../working/USA_trend.pkl")