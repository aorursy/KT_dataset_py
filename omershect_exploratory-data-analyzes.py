import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

os.listdir("../input/severstal-steel-defect-detection")

from matplotlib import pyplot as plt

from itertools import cycle, islice

from tqdm import tqdm, tqdm_notebook

import seaborn as sns; sns.set_style("white")

import random

import cv2

from PIL import Image
train_data = pd.read_csv("../input/severstal-steel-defect-detection/train.csv")

train_data.head()
train_data['EncodedPixels'].fillna(-1, inplace=True)

train_data['ClassId'] = train_data['ImageId_ClassId'].apply(lambda x: int(x[-1:]))

train_data['ImageName'] = train_data['ImageId_ClassId'].apply(lambda x: x[:-6] +  '.jpg' )

train_data['Defect'] =np.where(train_data['EncodedPixels']==-1, 0, 1) 

train_data['ClassId'] =np.where(train_data['EncodedPixels']==-1,  0,train_data['ClassId']) 

train_data.head()
colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(train_data)))

DF_Defect = train_data[train_data['EncodedPixels']!=-1]

ax = DF_Defect['ClassId'].value_counts().plot(kind='bar',

                                    figsize=(14,8),color=colors)

ax.set_xlabel("Class")

ax.set_ylabel("Frequency")

plt.xticks(rotation=360)

plt.show()
train_data.groupby(['ImageName'])['Defect'].sum().hist()
path_img_train = "../input/severstal-steel-defect-detection/train_images/"

path_img_test = "../input/severstal-steel-defect-detection/test_images/"
#This function show a gride of images, and their histogram 



def show_images_with_Histograms(images, cols = 2, titles = None):

   

    n_images = len(images)

    nrows = int(n_images/cols)

    fig, ax = plt.subplots(nrows, 2*  cols )

    

    assert((titles is None)or (len(images) == len(titles)))

    

    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    

    row = 0

    col = 0 

    for n, (image, title) in enumerate(zip(images, titles)):

        

        if image.ndim == 2:

            plt.gray()

        ax[row,col].imshow(image)

        ax[row,col].set_title(title,fontsize=42)

        

        col +=1 

        if col == 2 * cols : 

            col =0

            row +=1

        

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        

        ax[row,col].hist(img_gray.ravel(),30,[0,256])

       

        col +=1 

        if col == 2 * cols : 

            col =0

            row +=1

            

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images   )

    fig.subplots_adjust(hspace = 0.8)

    #plt.show()
SEED = 87

imgs = []

titles = []

plt.rcParams.update({'font.size': 36})

for i in range(10):

    #plt.figure(figsize=(5,5))

    random.seed(SEED + i)

    id = random.choice(os.listdir(path_img_train))

    id_code = id.split(".")[0]

    imgs.append(np.asarray(Image.open(os.path.join(path_img_train, id))))

    

    titles.append("Image:"+str(id))

show_images_with_Histograms(imgs, cols = 1,titles = titles)

    #imgplot = plt.imshow(imgs[i])

    #plt.show()
SEED = 34

imgs = []

titles = []

 

for i in range(10):

    #plt.figure(figsize=(5,5))

    random.seed(SEED + i)

    id = random.choice(os.listdir(path_img_test))

    id_code = id.split(".")[0]

    imgs.append(np.asarray(Image.open(os.path.join(path_img_test, id))))

    

    titles.append("Image:"+str(id))

show_images_with_Histograms(imgs, cols = 1,titles = titles)

    #imgplot = plt.imshow(imgs[i])

    #plt.show()
SEED = 12

imgs = []

titles = []

DF_Defect_only = train_data[train_data['Defect']!= 0]



for i in range(10):

    

    

    DF = DF_Defect_only.sample(n = 1, replace = False , random_state = SEED + i) 

    id = DF['ImageName'].values[0]

    label = DF['ClassId'].values[0]

    imgs.append(np.asarray(Image.open(os.path.join(path_img_train, id))))

    

    titles.append("Image:"+str(id)+" Class is:"+str(label))

show_images_with_Histograms(imgs, cols = 1,titles = titles)

    #imgplot = plt.imshow(imgs[i])

    #plt.show()
#Calculate the Active area of the image 



def Calc_Active_Area(img):

    Thrshold = 10 

    ImageSize = img.shape[0] * img.shape[1]

    Black_size = np.count_nonzero(img < Thrshold )

   

    

    active_area = ((ImageSize - Black_size)/ImageSize)*100

    

    

    return int(active_area)
SEED = 12

imgs = []

titles = []

DF_Defect_only = train_data[train_data['Defect']!= 0]

plt.rcParams.update({'font.size': 45})

for i in range(10):

    

    

    DF = DF_Defect_only.sample(n = 1, replace = False , random_state = SEED + i) 

    id = DF['ImageName'].values[0]

    label = DF['ClassId'].values[0]

    Active_area = Calc_Active_Area(np.asarray(Image.open(os.path.join(path_img_train, id)).convert('L')))

    imgs.append(np.asarray(Image.open(os.path.join(path_img_train, id))))

    

    titles.append("Image:"+str(id)+"  Active Area  is:"+str(Active_area)+'%')

show_images_with_Histograms(imgs, cols = 1,titles = titles)

    #imgplot
#Calculate the Saturated area of the image 



def Calc_Saturate_Area(img):

    Thrshold = 240 

    ImageSize = img.shape[0] * img.shape[1]

    saturate_area = np.count_nonzero(img > Thrshold )

   

    

    saturate_area = ((saturate_area)/ImageSize)*100

    

    

    return int(saturate_area)
SEED = 44

imgs = []

titles = []

DF_Defect_only = train_data[train_data['Defect']!= 0]

plt.rcParams.update({'font.size': 42})

for i in range(10):

    

    

    DF = train_data.sample(n = 1, replace = False , random_state = SEED + i) 

    id = DF['ImageName'].values[0]

   

    Saturate_Area = Calc_Saturate_Area(np.asarray(Image.open(os.path.join(path_img_train, id)).convert('L')))

    imgs.append(np.asarray(Image.open(os.path.join(path_img_train, id))))

    

    titles.append("Image:"+str(id)+" Saturated Area  is:"+str(Saturate_Area)+'%')

show_images_with_Histograms(imgs, cols = 1,titles = titles)
def Calculate_Active_area_from_file_name(FileName):

    img = np.asarray(Image.open(os.path.join(path_img_train, FileName)).convert('L'))

    result = Calc_Active_Area(img)

    return result 





def Calculate_Saturated_area_from_file_name(FileName):

    img = np.asarray(Image.open(os.path.join(path_img_train, FileName)).convert('L'))

    result = Calc_Saturate_Area(img)

    return result 
DF = train_data.drop_duplicates(subset='ImageName')

DF['ActiveArea'] = DF['ImageName'].apply(lambda x:Calculate_Active_area_from_file_name(x))





DF['SaturatedArea'] = DF['ImageName'].apply(lambda x:Calculate_Saturated_area_from_file_name(x))

DF.head()
DF['ActiveArea'].hist()
DF['SaturatedArea'].hist()
DF.head(100)
def rle2mask(mask_rle, shape=(1600,256)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (width,height) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T
mask = (rle2mask(DF['EncodedPixels'][20]))

print(mask.shape)



id = DF['ImageName'].values[5]

img = np.asarray(Image.open(os.path.join(path_img_train, id)))
plt.figure(figsize = (20,20)) 

plt.rcParams.update({'font.size': 15})

plt.imshow(mask)



plt.figure(figsize = (20,20)) 

plt.imshow(img)