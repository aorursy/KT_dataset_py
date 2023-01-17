# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from glob import glob
import os
PATH = os.path.abspath(os.path.join('..', 'input'))

# ../input/sample/images/
SOURCE_IMAGES = os.path.join(PATH, "sample", "images")

# ../input/sample/images/*.png
images = glob(os.path.join(SOURCE_IMAGES, "*.png"))

# Load labels
labels = pd.read_csv('../input/sample_labels.csv')
# Keep only two columns I need
labels.drop(['Follow-up #', 'Patient ID',
       'Patient Age', 'Patient Gender', 'View Position', 'OriginalImageWidth',
       'OriginalImageHeight', 'OriginalImagePixelSpacing_x',
       'OriginalImagePixelSpacing_y'],1,inplace = True)

#Remaining Labels are image index and Finding Labels
labels.head()
# #resize
# import numpy as np
# imgResize = cv2.resize(img,(224,224),cv2.INTER_AREA)
# plt.imshow(imgResize)
# print(imgResize.shape)
import matplotlib.pyplot as plt
def resizeHistEqual(images, labels):
    resizedImages = []
    modLabels = []
    for s in images:
        modLabels.append([labels['Image Index'][labels['Image Index'] == os.path.basename(s)].values[0],
                     labels['Finding Labels'][labels['Image Index'] == os.path.basename(s)].values[0]])
    for x in images:
        img = cv2.imread(x)
        clahe = cv2.createCLAHE(clipLimit = 20.0, tileGridSize = (8,8))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = clahe.apply(img)
        resizedImages.append(cv2.resize(img, (224,224)))
    return resizedImages, modLabels

resizeHistEqImages, labels = resizeHistEqual(images,labels)
plt.imshow(resizeHistEqImages[0], cmap = 'gray')  
np.savez("resizeCLHEimages", resizeHistEqImages,)
np.savez("indexLabels", labels)
# def histEqual(images,labels):
#     subsample = random.sample(images, 5) #np.random.choice(np.arange(len(images)))
# #     print(subsample)
#     modLabels = []
#     for s in subsample:
#         modLabels.append([labels['Image Index'][labels['Image Index'] == os.path.basename(s)].values[0],
#                      labels['Finding Labels'][labels['Image Index'] == os.path.basename(s)].values[0]])

# #     print(subsample)
#     subsample = [cv2.imread(x) for x in subsample]
#     x = [cv2.resize(x,(224,224), cv2.INTER_AREA) for x in subsample ]
    
#     equalImages = []
#     clahe = cv2.createCLAHE(clipLimit = 20.0, tileGridSize = (8,8))
#     plt.figure(figsize=(10,5))
#     for i in range(len(subsample)):
#         img = cv2.cvtColor(x[i], cv2.COLOR_BGR2GRAY)
#         equalImages.append(clahe.apply(img))
#         plt.subplot(1,len(subsample),i+1)
#         plt.imshow(equalImages[i], cmap = 'gray')
                           
#     plt.show()
    
#     return equalImages, modLabels
# import random 
# randomSamp = random.sample(images,5)

# def histEqual(images,labels):
#     subsample = random.sample(images, 5) #np.random.choice(np.arange(len(images)))
# #     print(subsample)
#     modLabels = []
#     for s in subsample:
#         modLabels.append([labels['Image Index'][labels['Image Index'] == os.path.basename(s)].values[0],
#                      labels['Finding Labels'][labels['Image Index'] == os.path.basename(s)].values[0]])

# #     print(subsample)
#     subsample = [cv2.imread(x) for x in subsample]
#     x = [cv2.resize(x,(224,224), cv2.INTER_AREA) for x in subsample ]
    
#     equalImages = []
#     clahe = cv2.createCLAHE(clipLimit = 20.0, tileGridSize = (8,8))
#     plt.figure(figsize=(10,5))
#     for i in range(len(subsample)):
#         img = cv2.cvtColor(x[i], cv2.COLOR_BGR2GRAY)
#         equalImages.append(clahe.apply(img))
#         plt.subplot(1,len(subsample),i+1)
#         plt.imshow(equalImages[i], cmap = 'gray')
                           
#     plt.show()
    
#     return equalImages, modLabels
# newHistImages, modLabels = histEqual(images,labels)
# print(modLabels)
# np.savez("modImages", equalImages,)
# np.savez("yModLabels", modLabels)
# def randomAffine(images, labels):
#     subsample = random.sample(images, 5 ) #np.random.choice(np.arange(len(images)))
    
#     modLabels = []
#     for s in subsample:
#         modLabels.append([labels['Image Index'][labels['Image Index'] == os.path.basename(s)].values[0],
#                      labels['Finding Labels'][labels['Image Index'] == os.path.basename(s)].values[0]])
    
#     subsample = [cv2.imread(x) for x in subsample]
#     x = [cv2.resize(x,(224,224), cv2.INTER_AREA) for x in subsample ]
    
#     affImages = []
#     plt.figure(figsize=(10,10))
#     for i in range(len(subsample)):
#         M = np.float32([[1,0,np.random.randint(-50,50)],[0,1,np.random.randint(-50,50)]])
#         dst = cv2.warpAffine(x[i],M, (224,224)) 
#         affImages.append(dst)
        
#         plt.subplot(1,len(subsample),i+1)
#         plt.imshow(dst)
#     plt.show()
    
#     return affImages, modLabels
# newAffImages, newAffLabels = randomAffine(images,labels)
# #Perspective Transform
# pts1 = np.float32([[56,65],[190,30],[15,221],[222,224]])
# pts2 = np.float32([[0,0],[224,0],[0,224],[224,224]])

# M = cv2.getPerspectiveTransform(pts1,pts2)

# dst = cv2.warpPerspective(imgResize,M,(224,224))

# plt.subplot(121),plt.imshow(imgResize),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()
