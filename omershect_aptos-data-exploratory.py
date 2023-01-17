# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from PIL import Image

import seaborn as sns; sns.set_style("white")

import random

import cv2

import time 

from tqdm import tqdm, tqdm_notebook





 

df_train = pd.read_csv("../input/train.csv")

df_train.head()
df_train.shape
ax = df_train['diagnosis'].value_counts().plot(kind='bar',

                                    figsize=(14,8),

                                    title="Count Diagnosis Labels  ")

ax.set_xlabel("Diagnossis Label")

ax.set_ylabel("Frequency")

ax.set_xticklabels( labels = list(('No DR', 'Moderate', 'Mild', 'Proliferative DR', 'Severe')),rotation=45)

plt.show()

%time

PATH = "../input/train_images"

image_size_list=[]

images_files = os.listdir(PATH)

for image in images_files :

    image_size_list.append(Image.open(os.path.join(PATH, image)).size)

 
images_size = np.array(image_size_list)

images_area =  images_size[:,0] * images_size[:,1]

DF = pd.DataFrame(images_size,columns=['Width','Height'])

DF.head()
ax = DF['Width'].value_counts().plot(kind='bar',

                                    figsize=(14,8),

                                    title="Image's Width    ")

ax.set_xlabel("Width")

ax.set_ylabel("Frequency")



plt.show()
ax = DF['Height'].value_counts().plot(kind='bar',

                                    figsize=(14,8),

                                    title="Image's Height    ")

ax.set_xlabel("Height")

ax.set_ylabel("Frequency")



plt.show()


#This function show a gride of images, and heir histogram 



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

        ax[row,col].set_title(title,fontsize=50)

        

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

            

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images  )

    #plt.show()
SEED = 1234

PATH = "../input/train_images"

imgs = []

titles = []



for i in range(12):

    #plt.figure(figsize=(5,5))

    random.seed(SEED + i)

    id = random.choice(os.listdir(PATH))

    id_code = id.split(".")[0]

    imgs.append(np.asarray(Image.open(os.path.join(PATH, id))))

    titles.append(" ".join([str("Label ="),str((df_train.loc[df_train.id_code == id_code, 'diagnosis']).item()),str("Image id is:"),str(id_code)]))

   



show_images_with_Histograms(imgs, cols = 2, titles = titles)
#Calculate the Active area of the image 

Thrshold = 50 

def Calc_Active_Area(img):



    img = np.where(img < 50, 0, img) 

        

    # Mask of non-black pixels (assuming image has a single channel).

    mask = img > 0



    # Coordinates of non-black pixels.

    coords = np.argwhere(mask)



    # Bounding box of non-black pixels.

    y0, x0 = coords.min(axis=0)

    y1 , x1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    

    active_area = (abs(x1-x0) * abs(y1-y0)) / (img.shape[0] * img.shape[1])*100

    

    return active_area
%time

PATH = "../input/train_images"

images_active_list=[]





tk0 = tqdm_notebook(list(images_files))





for  i , image in enumerate(tk0) :

    

    

    img = cv2.imread(os.path.join(PATH, image))

    

    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    active_area = Calc_Active_Area(np.asarray(img_gray))

    images_active_list.append(active_area)
DFActiveList  = pd.DataFrame(images_active_list,columns=['ActiveArea'])

DFActiveList.shape
bins = [0, 1, 5, 10, 20,30,40, 50,60,70,80,85,90,95,99, 100]

DFActiveList['binned'] = pd.cut(DFActiveList['ActiveArea'], bins)

ax = DFActiveList['binned'].value_counts().plot(kind='bar',

                                    figsize=(14,8),

                                    title="Image's Active Area %    ")

ax.set_xlabel("% Active Area")

ax.set_ylabel("Frequency")

labels = ax.get_xticklabels()

ax.set_xticklabels(labels = labels ,rotation=45)



plt.show()