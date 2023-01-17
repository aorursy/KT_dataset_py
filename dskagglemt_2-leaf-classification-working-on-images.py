import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

# Image manipulation.
import PIL.Image
from IPython.display import display
# from zipfile import ZipFile
# image_dir = '../input/leaf-classification/images.zip'
# image_folder = ZipFile(image_dir, 'r')
# image_folder.namelist()[0:5]
# image_folder.namelist()[1:2]
# image_folder
# importing required modules 
from zipfile import ZipFile 
  
# specifying the zip file name 
images_zip = '../input/leaf-classification/images.zip'
  
# opening the zip file in READ mode 
with ZipFile(images_zip, 'r') as zip: 
    # printing all the contents of the zip file 
#     zip.printdir() 
  
    # extracting all the files 
    print('Extracting all the files now...') 
    zip.extractall() 
    print('Done!') 
image_dir = './images' # From output folder.
img = image_dir + '/' + str(100) + '.jpg'
img
img = mpimg.imread(img)
imgplot = plt.imshow(img)
plt.show()
type(imgplot)
# imgplot.shape
type(img), img.shape
# img = img.resize((160, 240), mpimg.ANTIALIAS)

# type(img), img.shape
# show some random images
plt.figure(figsize=(12,12))

cnt = range(100)

for k in range(28):
    randomID = np.random.randint(len(cnt))
    
    imageFilename = image_dir + '/' + str(randomID) + '.jpg' 
    
    plt.subplot(4,7,k+1); 
    
    plt.imshow(mpimg.imread(imageFilename), cmap='gray')

    plt.axis('off')
import cv2 

new_width = 28
new_height = 28

# resized_image = cv2.resize(original_image, (new_width, new_height), 
#                            interpolation=cv2.INTER_NEAREST)
# show some random images
plt.figure(figsize=(12,12))

cnt = range(100)

for k in range(28):
    randomID = np.random.randint(len(cnt))
    
    imageFilename = image_dir + '/' + str(randomID) + '.jpg' 
    
    plt.subplot(4,7,k+1); 
    
    src = cv2.imread(imageFilename, cv2.IMREAD_UNCHANGED)
    
    resized_image = cv2.resize(src, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    plt.imshow(resized_image, cmap='gray')
    
#     plt.imshow(mpimg.imread(resized_image), cmap='gray')

    plt.axis('off')
    