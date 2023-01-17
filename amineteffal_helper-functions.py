import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os #operating system functionality
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from scipy import misc
#Let's first read an image from the train folder
ImageId='0ea221716cf13710214dcd331a61cea48308c3940df1d28cfc7fd817c83714e1'
folder='../input/stage1_train/'
f = misc.imread(folder+ImageId+'/images/'+ImageId+'.png',flatten=False)
heigh=f.shape[0]
width=f.shape[1]
print('Image : ', ImageId, ' ',heigh,'*',width)
plt.imshow(f)
plt.show()
# Now let's plot the masks for this image
#But first let's read the file containing the labels
labels = pd.read_csv('../input/stage1_train_labels.csv',sep=',')
labels.head(5)
#The labelImage function
def labelImage(ImageId,imageSize):
    #define the array where to store the labels
    y=np.zeros((imageSize,),dtype=bool)
    #select just the encoded pixels for the given image
    temp=list(labels[labels['ImageId']==ImageId]['EncodedPixels'])
    for w in temp:
        x=w.split()
        x=[int(x[i]) for i in range(0,len(x)) ]
        for i in range(0,len(x),2):
            y[x[i]:min((x[i]+x[i+1]),len(y)-1)] = [True]*(min((x[i]+x[i+1]),len(y)-1)-x[i])
        
    return y
y=labelImage(ImageId,heigh*width)
plt.imshow(y.reshape((heigh,width),order='F'),cmap='gray')
plt.show()
plt.imshow(f)
plt.show()
#Now let's try to use a simple algorithm to detect the Nucleis in this image
#But let use this function to plot an image encoded in one-dimension arra
def showDigitImage(x, h=0,Type='gray', order_='F'):
    if h<=0:
        dimension_1 = int(np.sqrt(len(x)))
        dimension_2=dimension_1
    else:
        dimension_1=h
        dimension_2=int(len(x)/h)
        
    print('(',dimension_1,dimension_2,')',sep=' ')
    plt.imshow(x.reshape((dimension_1,dimension_2),order=order_),cmap='gray')
    plt.show()
showDigitImage(y,heigh)
#The binarize function
def binarize(x, trashold=128, h=0,Type='gray'):
    x2=x.copy()
    for i in range(0,len(x)):
        if x[i]>trashold :
            x2[i]=1
        else:
            x2[i]=0
    showDigitImage(x2,h,Type)
    
    return x2
#But we need to grayScale the image first
def grayScaleImage(ImageId,folder='../input/stage1_train/'):
    f = misc.imread(folder+ImageId+'/images/'+ImageId+'.png',flatten=False)
    l=f.shape[0]
    array_image=f.flatten()
    array_image_mean=[]
    for i in range(0,len(array_image),f.shape[2]):
        array_image_mean.append((0.0+array_image[i]+array_image[i+1]+array_image[i+2]+0)/3)
    n=len(array_image_mean)
    image_features = []
    for i in range(0,len(array_image_mean)):
        image_features.append([array_image_mean[i]])
    
    print('Gray scale image : ')
    plt.imshow(f)
    plt.show()
    return np.array(image_features)

plt.hist(grayScaleImage(ImageId))
binarize(grayScaleImage(ImageId),27,heigh)