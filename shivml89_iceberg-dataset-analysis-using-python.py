%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import signal
train = pd.read_json('../input/train.json')
index_ship=np.where(train['is_iceberg']==0)

index_ice=np.where(train['is_iceberg']==1)
def plots(band,index,title):

    plt.figure(figsize=(12,10))

    for i in range(12):

        plt.subplot(3,4,i+1)

        plt.xticks(())

        plt.yticks(())

        plt.xlabel((title))

        plt.imshow(np.reshape(train[band][index[0][i]], (75,75)),cmap='gist_heat')

    plt.show()  
plots('band_1',index_ship,'band1 ship')
plots('band_2',index_ship,'band2 ship')
plots('band_1',index_ice,'band1 ice')
plots('band_2',index_ice,'band2 ice')
def plot_conv(band,index,xFilt,yFilt,title):

    plt.figure(figsize=(12,10))

    for i in range(12):

        plt.subplot(3,4,i+1)

        plt.xticks(())

        plt.yticks(())

        plt.xlabel((title))

        img=np.reshape(train[band][index[0][i]], (75,75))

        img_Gx=signal.convolve2d(img,xFilt,mode='valid')

        img_Gy=signal.convolve2d(img,yFilt,mode='valid')

        img_filt=np.hypot(img_Gx,img_Gy)



        plt.imshow((img_filt) ,cmap='gist_heat')

    plt.show()

    
##sobel operator



Gx_sobel=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

Gy_sobel=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

lpf=1.0/25*np.ones((5,5))

plot_conv('band_1', index_ice,Gx_sobel,Gy_sobel,'band1 ice using sobel')
plot_conv('band_2', index_ice,Gx_sobel,Gy_sobel,'band2 ice using sobel')
plot_conv('band_1', index_ship,Gx_sobel,Gy_sobel,'band1 ship using sobel')
plot_conv('band_2', index_ship,Gx_sobel,Gy_sobel,'band2 ship using sobel')
def plot_Avgconv(band,index,lpf,xFilt,yFilt,title):

    plt.figure(figsize=(12,10))

    for i in range(12):

        plt.subplot(3,4,i+1)

        plt.xticks(())

        plt.yticks(())

        plt.xlabel((title))

        img=np.reshape(train[band][index[0][i]], (75,75))

        img_lpf=signal.convolve2d(img,lpf,mode='valid')

        img_Gx=signal.convolve2d(img_lpf,xFilt,mode='valid')

        img_Gy=signal.convolve2d(img_lpf,yFilt,mode='valid')

        img_filt=np.hypot(img_Gx,img_Gy)



        plt.imshow((img_filt) ,cmap='gist_heat')

    plt.show()
##avg+sobel operator

plot_Avgconv('band_1', index_ice,lpf,Gx_sobel,Gy_sobel,'band1 ice using avg+sobel')
plot_Avgconv('band_2', index_ice,lpf,Gx_sobel,Gy_sobel,'band2 ice using avg+sobel')
plot_Avgconv('band_1', index_ship,lpf,Gx_sobel,Gy_sobel,'band1 ship using avg+sobel')
plot_Avgconv('band_2', index_ship,lpf,Gx_sobel,Gy_sobel,'band2 ship using avg+sobel')
##Prewitt operator + averaging



Gx_pre=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

Gy_pre=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

plot_Avgconv('band_1', index_ship,lpf,Gx_pre,Gy_pre,'band1 ship using avg+prewitt')
plot_Avgconv('band_1', index_ice,lpf,Gx_pre,Gy_pre,'band1 ice using avg+prewitt')
Gx_rob=np.array([[1,0],[0,-1]])

Gy_rob=np.array([[0,1],[-1,0]])

plot_Avgconv('band_1', index_ship,lpf,Gx_rob,Gy_rob,'band1 ship using avg+roberts')
plot_Avgconv('band_1', index_ice,lpf,Gx_rob,Gy_rob,'band1 ice using avg+roberts')
plt.figure(figsize=(14,12))

k=0

for i in range(4):

    k+=1

    plt.subplot(4,3,k)

    plt.xticks(())

    plt.yticks(())

    plt.xlabel(('band1 ice using avg+sobel'))

    img=np.reshape(train['band_1'][index_ice[0][i]], (75,75))

    img_lpf=signal.convolve2d(img,lpf,mode='valid')

    img_Gx=signal.convolve2d(img_lpf,Gx_sobel,mode='valid')

    img_Gy=signal.convolve2d(img_lpf,Gy_sobel,mode='valid')

    img_filt=np.hypot(img_Gx,img_Gy)

  

    plt.imshow((img_filt) ,cmap='gist_heat')

                

    k+=1

    plt.subplot(4,3,k)

    plt.xticks(())

    plt.yticks(())

    plt.xlabel(('band1 ice using avg+prewitt'))

    img=np.reshape(train['band_1'][index_ice[0][i]], (75,75))

    img_lpf=signal.convolve2d(img,lpf,mode='valid')

    img_Gx=signal.convolve2d(img_lpf,Gx_pre,mode='valid')

    img_Gy=signal.convolve2d(img_lpf,Gy_pre,mode='valid')

    img_filt=np.hypot(img_Gx,img_Gy)

  

    plt.imshow((img_filt) ,cmap='gist_heat')

                

    k+=1

    plt.subplot(4,3,k)

    plt.xticks(())

    plt.yticks(())

    plt.xlabel(('band1 ice using avg+roberts'))

    img=np.reshape(train['band_1'][index_ice[0][i]], (75,75))

    img_lpf=signal.convolve2d(img,lpf,mode='valid')

    img_Gx=signal.convolve2d(img_lpf,Gx_rob,mode='valid')

    img_Gy=signal.convolve2d(img_lpf,Gy_rob,mode='valid')

    img_filt=np.hypot(img_Gx,img_Gy)

  

    plt.imshow((img_filt) ,cmap='gist_heat')

plt.show()
plt.figure(figsize=(14,12))

k=0

for i in range(4):

    k+=1

    plt.subplot(4,3,k)

    plt.xticks(())

    plt.yticks(())

    plt.xlabel(('band1 ship using avg+sobel'))

    img=np.reshape(train['band_1'][index_ship[0][i]], (75,75))

    img_lpf=signal.convolve2d(img,lpf,mode='valid')

    img_Gx=signal.convolve2d(img_lpf,Gx_sobel,mode='valid')

    img_Gy=signal.convolve2d(img_lpf,Gy_sobel,mode='valid')

    img_filt=np.hypot(img_Gx,img_Gy)

  

    plt.imshow((img_filt) ,cmap='gist_heat')

                

    k+=1

    plt.subplot(4,3,k)

    plt.xticks(())

    plt.yticks(())

    plt.xlabel(('band1 ship using avg+prewitt'))

    img=np.reshape(train['band_1'][index_ship[0][i]], (75,75))

    img_lpf=signal.convolve2d(img,lpf,mode='valid')

    img_Gx=signal.convolve2d(img_lpf,Gx_pre,mode='valid')

    img_Gy=signal.convolve2d(img_lpf,Gy_pre,mode='valid')

    img_filt=np.hypot(img_Gx,img_Gy)

  

    plt.imshow((img_filt) ,cmap='gist_heat')

                

    k+=1

    plt.subplot(4,3,k)

    plt.xticks(())

    plt.yticks(())

    plt.xlabel(('band1 ship using avg+roberts'))

    img=np.reshape(train['band_1'][index_ship[0][i]], (75,75))

    img_lpf=signal.convolve2d(img,lpf,mode='valid')

    img_Gx=signal.convolve2d(img_lpf,Gx_rob,mode='valid')

    img_Gy=signal.convolve2d(img_lpf,Gy_rob,mode='valid')

    img_filt=np.hypot(img_Gx,img_Gy)

  

    plt.imshow((img_filt) ,cmap='gist_heat')

plt.show()