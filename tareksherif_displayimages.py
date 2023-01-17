import os
import cv2
import numpy as np
def DisplayImage(image,title,cols=1):
  
    import matplotlib.pyplot as plt
    
    image_no=len(image)+1
    postion=0

    plt.figure(figsize=(8, 8))
    for i in range(1,image_no):
        postion+=1
        plt.subplot(1,cols,postion),plt.imshow(image[i-1],cmap = 'gray'), plt.title(title[i-1]), plt.axis('off')
       
        if ( i%cols==0):
            plt.show(),plt.figure(figsize=(8, 8))
            postion=0
images=[]
titles=[]
for dirname, _, filenames in os.walk('../input/digits-in-noise/Test'):
    for filename in filenames[0:9]:
       
        images.append( cv2.imread(dirname+"/"+ filename,0))
        titles.append("1")
        
        
DisplayImage(images,titles,3)
