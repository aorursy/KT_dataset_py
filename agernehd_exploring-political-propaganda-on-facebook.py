import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core import display as ICD
from torchvision import transforms
import imageio
import PIL
import cv2
import os
from glob import glob
import shutil
import random
import zipfile
path = "../input/russian political influence campaigns/Russian Political Influence Campaigns/Supplementary Data/From Russian Ad Explorer/images.zip"
with zipfile.ZipFile(path,"r") as z:
    z.extractall(".")
dataPath = '../input/russian political influence campaigns/Russian Political Influence Campaigns/Supplementary Data/From Data World/FacebookAds.csv'
dataset = pd.read_csv(dataPath)
adCount = dataset['Clicks'].count()
AdSpendCount = dataset['AdSpend'].sum()
AdSpendCountDol = AdSpendCount/63
clickCount = dataset['Clicks'].sum()
clickCountPerAd = clickCount/adCount
ClickRatioDol = AdSpendCountDol/clickCount
AdSpendCountRatio = AdSpendCountDol/adCount
print('Total # of Ads Purchased by IRA between 2015 to 2017:',adCount)
print('Total $ Spent by IRA between 2015 to 2017: $',AdSpendCountDol)
print('Avg Cost per Ad: $',AdSpendCountRatio)
print('Avg # of Clicks Per Ad: ',clickCountPerAd)
print('Avg Cost per Click: $',ClickRatioDol)
def load_image(path, size):
    img = PIL.Image.open(path)
    normalise = transforms.ToTensor()
    img_tensor = normalise(img).unsqueeze(0)
    img_np = img_tensor.numpy()
    return img, img_tensor, img_np
inputImage = 'images/2016-06/P10002392.-000.png'
input_img, input_tensor, input_np = load_image(inputImage, size=[1024, 1024])
input_img
inputImage = 'images/2015-06/P10002571.-001.png'
input_img, input_tensor, input_np = load_image(inputImage, size=[1024, 1024])
input_img
inputImage = 'images/2016-12/P10006344.-000.png'
input_img, input_tensor, input_np = load_image(inputImage, size=[1024, 1024])
input_img
inputImage = 'images/2017-05/P10000708.-000.png'
input_img, input_tensor, input_np = load_image(inputImage, size=[1024, 1024])
input_img
inputImage = 'images/2017-05/P10005183.-000.png'
input_img, input_tensor, input_np = load_image(inputImage, size=[1024, 1024])
input_img2=np.asarray(input_img)
imageio.imwrite('outfile.jpg', input_img2)
input_img
multipleImages = glob('images/2016-06/**')
def plotImages(path,begin,end):
    i_ = 0
    plt.rcParams['figure.figsize'] = (25.0, 25.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    for l in multipleImages[begin:end]:
        im = cv2.imread(l)
        im = cv2.resize(im, (1024, 1024)) 
        plt.subplot(3, 3, i_+1)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
        i_ += 1
    plt.show()
plotImages(multipleImages,0,9)  
plotImages(multipleImages,9,18)  
plotImages(multipleImages,18,27)  
multipleImages = glob('images/2016-12/**')
plotImages(multipleImages,0,9)  
plotImages(multipleImages,9,18)  
plotImages(multipleImages,18,27) 
multipleImages = glob('images/2017-03/**')
plotImages(multipleImages,0,9)  
plotImages(multipleImages,9,18)  
plotImages(multipleImages,18,27) 
multipleImages = glob('images/2017-04/**')
plotImages(multipleImages,0,9)  
plotImages(multipleImages,9,18)  
plotImages(multipleImages,18,27) 
print('Groups Targeted -- # of Ads Targeted at Specific Groups\n')
groupCounts = dataset['FriendsOfConnections'].value_counts()
groupCounts2 = dataset['PeopleWhoMatch'].value_counts()
print(groupCounts.head(5),'\n\n',groupCounts2.head(5),'\n')
path = 'images'
shutil.rmtree(path)