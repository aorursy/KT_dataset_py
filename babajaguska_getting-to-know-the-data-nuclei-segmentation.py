import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.io

from tqdm import tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataTrainPath="../input/stage1_train/"
dataTestPath="../input/stage1_test/"

root,dirs,files=os.walk(dataTrainPath).__next__()
print(dirs[0:5],"\n")
print("And there are {} of them".format(len(dirs)))
print(os.listdir(root+dirs[0]))
rIm,dIm,fIm=os.walk(root+dirs[0]+"/images/").__next__()
print("This is an example image file name: ",fIm)
rM,dM,fM=os.walk(root+dirs[0]+"/masks/").__next__()
print("Example mask file name: \n ",fM)
print("Well, look at that, multiple masks for one image file. \nAll .png format.")
print("This is an image")
im=mpimg.imread(rIm+fIm[0])
plt.imshow(im)
plt.show()
print("These are the corresponding masks")
for mask in fM:
    iMask=mpimg.imread(rM+mask)
    plt.imshow(iMask)
    plt.show()
maskList=[rM+mask for mask in fM]
masks=skimage.io.imread_collection(maskList).concatenate()
s=masks.shape
print("Shape of this squashed masks ndarray: ",s) #so number of masks, dimension1, dimension2

# Start with a matrix of size the same as the image (or masks)
labels=np.zeros((s[1],s[2]),np.uint16)

for i in range(s[0]):
    labels[masks[i]>0]=i+1

# And there you have it
print("MASKS all together:")
plt.imshow(labels)
plt.show()

# Just to compare again with the original image:
print("IMAGE:")
plt.imshow(im)
plt.show()
numMasks=np.zeros(len(dirs),np.uint16)
for i,dirName in enumerate(dirs):
    tempR,tempD,tempF=os.walk(root+dirName+"/masks/").__next__()
    numMasks[i]=len(tempF)
plt.hist(numMasks,40)
plt.xlabel("Number of masks per image")
plt.show()

dfNumMasks=pd.DataFrame(numMasks,columns=["NumNuclei"])
dfNumMasks.describe()
# Let me adapt the above code for merging masks into a function
def mergeNPlotMasks(imageRoot,imageDirName):
    imageFileName=imageRoot+imageDirName+"/images/"+imageDirName+".png"
    image=skimage.io.imread(imageFileName)
    
    rM,dM,fM=os.walk(imageRoot+imageDirName+"/masks/").__next__()
    maskList=[rM+mask for mask in fM]
    masks=skimage.io.imread_collection(maskList).concatenate()
    s=masks.shape
    
   # Start with a matrix of size the same as the image (or masks)
    labels=np.zeros((s[1],s[2]),np.uint16)
    for i in range(s[0]):
        labels[masks[i]>0]=i+1

    # And there you have it
    print("MASKS all together:")
    plt.imshow(labels)
    plt.show()

    # Just to compare again with the original image:
    print("IMAGE:")
    plt.imshow(image)
    plt.show()
for i,dirName in enumerate(dirs):
    tempR,tempD,tempF=os.walk(root+dirName+"/masks/").__next__()
    if len(tempF)>300:
        print("#",i)
        mergeNPlotMasks(root,dirName)
imageAreas=[]
imageLevels=[]
for dirName in dirs:
    tempImage=skimage.io.imread(root+dirName+"/images/"+dirName+".png")
    imageAreas.append(tempImage.shape[0]*tempImage.shape[1])
    imageLevels.append(tempImage.shape[2])
plt.hist(imageAreas,20)
plt.xlabel("Image area [pixels]")
plt.show()

plt.hist(imageLevels,20)
plt.xlabel("Levels")
plt.show()

dfImageSizes=pd.DataFrame(imageAreas,columns=["image_area_pxls"])
dfImageSizes.describe()


alphaMeans=[]
for dirName in tqdm(dirs):
    tempImage=skimage.io.imread(root+dirName+"/images/"+dirName+".png")
    tempAlpha=[tempImage[i,j,3] for i in range(0,tempImage.shape[0])for j in range(0,tempImage.shape[1])]
    alphaMeans.append(np.mean(tempAlpha))
dfAlphaMeans=pd.DataFrame(alphaMeans,columns=["Alpha levels"])
dfAlphaMeans.describe()