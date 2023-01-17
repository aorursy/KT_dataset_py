

import numpy as np 

import pandas as pd 



import os

total_files=[]



fp="/kaggle/input/global-wheat-detection/train/"

for dirname, _, filenames in os.walk(fp):

    total_files=filenames

        







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")
from skimage import data

import numpy as np

import skimage.filters as filters

import matplotlib.pyplot as plt

from skimage import io

fig,ax=plt.subplots(1,5,figsize=(24,24))

for en,i in enumerate([1.5,2,2.5,3,3.5]):    

    image = io.imread('/kaggle/input/global-wheat-detection/train/'+str(train_df.loc[0,"image_id"])+".jpg")

    if en==0:

        ax[en].imshow(image)

    

        continue

    

    gimage=filters.unsharp_mask(image,amount=i)

    ax[en].imshow(gimage)





import skimage.segmentation as seg

import skimage.filters as filters

import skimage.draw as draw

import skimage.color as color

import skimage.transform as tfm



def image_show(image, nrows=1, ncols=1,cmap="gray"):

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))

    ax.imshow(image,cmap="gray")

    ax.axis('off')

    return fig, ax

fig,ax=plt.subplots(1,1)

ax.hist(image.ravel(),bins=64,range=[0,256])

ax.set_xlim(0,256)

#image_show(image>20)


fig,ax=plt.subplots(1,5,figsize=(12,12))

for en,i in enumerate([0,50,60,80,100]):

            

    image = io.imread('/kaggle/input/global-wheat-detection/train/'+str(train_df.loc[0,"image_id"])+".jpg")

    if en==0:

        

        ax[en].imshow(tfm.resize(image,(512,512)))

        continue

   

    mask=image<i

    image[mask]=0

    

    gimage=color.rgb2gray(image)

    ax[en].imshow(filters.unsharp_mask(gimage,radius=0.05),cmap="gray")

#image_show(fimage)



#segmented

#image.shape
t_filter=[filters.threshold_li,

filters.threshold_triangle,

          filters.threshold_yen,

          filters.threshold_isodata,

          filters.threshold_sauvola

         ]

fig,ax=plt.subplots(1,5,figsize=(32,32))

fig.tight_layout(pad=3.0)

for en,i in enumerate(t_filter):

    tit=str(i)       

    image = io.imread('/kaggle/input/global-wheat-detection/train/'+str(train_df.loc[0,"image_id"])+".jpg")

   

    i=i(image)

    mask=image<i

    image[mask]=0

    

    gimage=(color.rgb2gray(filters.unsharp_mask(image,amount=3)))

    ax[en].imshow(gimage,cmap="gray")

    ax[en].set_title(tit,fontsize=20)

#image_show(fimage)
t_filter=[

filters.threshold_mean,

          filters.threshold_minimum,

          filters.threshold_niblack,

          filters.threshold_yen

         ]



fig,ax=plt.subplots(1,4,figsize=(22,22))

fig.tight_layout(pad=3.0)

for en,i in enumerate(t_filter):

    tit=str(i)       

    image = io.imread('/kaggle/input/global-wheat-detection/train/'+str(train_df.loc[0,"image_id"])+".jpg")

   

    

    i=i(image)

    mask=image<i

    image[mask]=0

    

    gimage=(color.rgb2gray(filters.unsharp_mask(image,amount=3)))

    ax[en].imshow(gimage,cmap="gray")

    ax[en].set_title(tit)

#image_show(fimage)
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches



from skimage import data

from skimage.filters import threshold_yen

from skimage.segmentation import clear_border

from skimage.measure import label, regionprops

from skimage.morphology import closing, square,rectangle,convex_hull_object

from skimage.color import label2rgb



def make_rect(name):

    

    n_image = io.imread('/kaggle/input/global-wheat-detection/test/'+str(name))



    image = color.rgb2gray(n_image)





    thresh = filters.threshold_yen(image)

    

    bw=closing(image < thresh, square(4))

    

    #bw = convex_hull_object(bw)

    



    

    label_image = label(bw,connectivity=1,background=255)



    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)



    





    fig, ax = plt.subplots(figsize=(10, 6))

    regionprops(label_image)

    ax.imshow(n_image)



    listrect=[]

    for region in regionprops(label_image):

        # take regions with large enough areas

        if region.area >= 500:

            # draw rectangle around segmented coins

            minr, minc, maxr, maxc = region.bbox

            listrect.append([minc,minr,maxc-minc,maxr-minr])

            

            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,

                                      fill=False, edgecolor='red', linewidth=2)

            ax.add_patch(rect)



    

    return listrect

    

d=dict()

fp="/kaggle/input/global-wheat-detection/test/"

for dirname, _, filenames in os.walk(fp):

    total_files=filenames

data_list=list(set(train_df["image_id"].values))

for k in range(len(data_list)):

    data_list[k]=data_list[k]+".jpg"

notthere=[]

for i in filenames:

    if i not in data_list:

        notthere.append(i)

print(len(notthere))

    



#for i in filenames
temp_df=pd.DataFrame({"image_id":[],"width":[],"height":[],"bbox":[],"source":[]})



for i in notthere:

    rects=make_rect(i)

    rects=rects

    #print(rects)

    rects=" ".join(str(item) for innerlist in rects for item in innerlist)

    #print(rects)

    d={"image_id":[],"width":[],"height":[],"bbox":[],"source":[]}

    #print(rects)

    

    

    d["bbox"].append(rects)

    d["image_id"].append(i[:-4])

    d["width"].append(1024)

    d["height"].append(1024)

    d["source"].append("MyOwnWhoCaresGonnaDropItAnyWay")

       # print(pd.DataFrame(d))  

    temp_df=temp_df.append(pd.DataFrame(d),ignore_index=True)

    
#temp_df["image_id"]=temp_df["image_id"].apply(lambda x: x[:-4])

temp_df["bbox"]
fp='/kaggle/input/global-wheat-detection/sample_submission.csv'

df=pd.read_csv(fp)
df["PredictionString"]=temp_df["bbox"]



        
df.to_csv("submission.csv",index=False)
\