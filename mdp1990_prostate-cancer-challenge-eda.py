import os

import PIL

import time

import math

import warnings

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import seaborn as sns

import openslide



import warnings

warnings.filterwarnings('ignore')

import os

import cv2

import numpy as np

import pandas as pd 

import json

import skimage.io

import matplotlib.pyplot as plt



train = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/train.csv")

test = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/test.csv")

print("Train Data Type",train.dtypes)

print("Test Data Type",test.dtypes)



print("unique ids : ", len(train.image_id.unique()))

print("unique data provider : ", len(train.data_provider.unique()))

print("unique isup_grade(target) : ", len(train.isup_grade.unique()))

print("unique gleason_score : ", len(train.gleason_score.unique()))
"""

Labels are somewhat unbalanced.

"""

train['isup_grade'].hist(figsize = (10, 5))



"""

Gleason Score Patterns

"""

train[['primary Gleason', 'secondary Gleason']] = train.gleason_score.str.split('+',expand=True)

train[['primary Gleason', 'secondary Gleason']]
train['primary Gleason'].hist(figsize = (10, 5))

train['secondary Gleason'].hist(figsize = (10, 5))





print(train['gleason_score'].unique())

"""

Lets see what does this label "0+0" means , whene we have "negative" for no cancer data . They both map to zero. In such case

those two labels can be mapped to one label.

"""

print(train[train['gleason_score']=='0+0']['isup_grade'].unique())

print(train[train['gleason_score']=='negative']['isup_grade'].unique())



"""

Data points haveing no cancer cells , the proportion is around 2892/10616 = 27%

"""

print(len(train[train['gleason_score']=='0+0']['isup_grade']))

print(len(train[train['gleason_score']=='negative']['isup_grade']))

"""

Mapping of ISUP_grade to gleason score.

3+4 and 4+3 map to different ISUP scores while other pairs like 3-5 and 5-3 , 4-5 and 5-4 map to same ISUP 

"""

print(train[(train['gleason_score'] == "3+4") | (train['gleason_score'] == "4+3")]['isup_grade'].unique())

print(train[(train['gleason_score']=='3+5') | (train['gleason_score']=='5+3')]['isup_grade'].unique())

print(train[(train['gleason_score']=='5+4') | (train['gleason_score']=='4+5')]['isup_grade'].unique())
print(train[train['gleason_score'] =="3+4"]['isup_grade'].unique())

print(train[train['gleason_score'] =="4+3"]['isup_grade'].unique())
"""

Identified as an anomaly record

"""

print(train[(train['gleason_score'] == "4+3") & (train['isup_grade'] == 2)])

train.drop([7273],inplace=True)
"""

Mapping Negative to 0+0 gleason score

"""



train['gleason_score'] = train['gleason_score'].apply(lambda x: "0+0" if x == "negative" else x)

train.gleason_score.values



"""

Test Data

"""

print("shape : ", test.shape)

print("unique ids : ", len(test.image_id.unique()))

print("unique data provider : ", len(test.data_provider.unique()))

data = train.groupby('isup_grade').count()['image_id'].reset_index().sort_values(by='image_id',ascending=False)



fig = px.bar(data, x='isup_grade', y='image_id',

             hover_data=['image_id', 'isup_grade'], color='image_id',height=400)

fig.show()



fig = go.Figure(go.Funnelarea(

    text =data.isup_grade,

    values = data.image_id,

    title = {"position": "top center", "text": "ISUP_grade Distribution"}

    ))

fig.show()
data_gleason_score = train.groupby('gleason_score').count()['image_id'].reset_index().sort_values(by='image_id',ascending=False)



fig = go.Figure(go.Funnelarea(

    text =data_gleason_score.gleason_score,

    values = data.image_id,

    title = {"position": "top center", "text": "Gleaseon Score Distribution"}

    ))

fig.show()



fig = px.bar(data_gleason_score, x='gleason_score', y='image_id',

             hover_data=['image_id', 'gleason_score'], color='image_id',height=500)

fig.show()
train.groupby(['data_provider','gleason_score'])['image_id'].size().reset_index()



'''

Visualizing the GLEASON_SCORE distribution wrt Data_providers

'''



fig = plt.figure(figsize=(10,6))

ax = sns.countplot(x="gleason_score", hue="data_provider", data=train)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/10616),

                ha="center")
example = openslide.OpenSlide(os.path.join("/kaggle/input/prostate-cancer-grade-assessment/train_images", '005e66f06bce9c2e49142536caf2f6ee.tiff'))



patch = example.read_region((17800,19500), 0, (256, 256))



# Display the image

display(patch)



# Close the opened slide after use

example.close()



ex_img_path = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'+train['image_id'][np.random.choice(len(train))]+'.tiff'

example_image = openslide.OpenSlide(ex_img_path)

example_image.properties
img = example_image.read_region(location=(0,0),level=2,size=(example_image.level_dimensions[2][0],example_image.level_dimensions[2][1]))

print(img.size)

img
train = train.set_index('image_id')

train.head()
def get_values(image,max_size=(600,400)):

    slide = openslide.OpenSlide(os.path.join("/kaggle/input/prostate-cancer-grade-assessment/train_images", f'{image}.tiff'))

    

    # Here we compute the "pixel spacing": the physical size of a pixel in the image.

    # OpenSlide gives the resolution in centimeters so we convert this to microns.

    f,ax =  plt.subplots(2 ,figsize=(6,16))

    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)

    patch = slide.read_region((1780,1950), 0, (256, 256)) #ZOOMED FUGURE

    ax[0].imshow(patch) 

    ax[0].set_title('Zoomed Image')

    

    

    ax[1].imshow(slide.get_thumbnail(size=max_size)) #UNZOOMED FIGURE

    ax[1].set_title('Full Image')

    

    

    print(f"File id: {slide}")

    print(f"Dimensions: {slide.dimensions}")

    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")

    print(f"Number of levels in the image: {slide.level_count}")

    print(f"Downsample factor per level: {slide.level_downsamples}")

    print(f"Dimensions of levels: {slide.level_dimensions}\n\n")

    

    print(f"ISUP grade: {train.loc[image, 'isup_grade']}")

    print(f"Gleason score: {train.loc[image, 'gleason_score']}")

    

get_values('0a4b7a7499ed55c71033cefb0765e93d')
def display_images(images):

    '''

    This function takes in input a list of images. It then iterates through the image making openslide objects , on which different functions

    for getting out information can be called later

    '''

    f, ax = plt.subplots(5,3, figsize=(18,22))

    for i, image in enumerate(images):

        slide = openslide.OpenSlide(os.path.join("/kaggle/input/prostate-cancer-grade-assessment/train_images", f'{image}.tiff')) # Making Openslide Object

        #Here we compute the "pixel spacing": the physical size of a pixel in the image,

        #OpenSlide gives the resolution in centimeters so we convert this to microns

        spacing = 1/(float(slide.properties['tiff.XResolution']) / 10000)

        patch = slide.read_region((1780,1950), 0, (256, 256)) #Reading the image as before betweeen x=1780 to y=1950 and of pixel size =256*256

        ax[i//3, i%3].imshow(patch) #Displaying Image

        slide.close()       

        ax[i//3, i%3].axis('off')

        

        image_id = image

        data_provider = train.loc[image, 'data_provider']

        isup_grade = train.loc[image, 'isup_grade']

        gleason_score = train.loc[image, 'gleason_score']

        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")



    plt.show() 

    

images = [

'07a7ef0ba3bb0d6564a73f4f3e1c2293',

    '037504061b9fba71ef6e24c48c6df44d',

    '035b1edd3d1aeeffc77ce5d248a01a53',

    '059cbf902c5e42972587c8d17d49efed',

    '06a0cbd8fd6320ef1aa6f19342af2e68',

    '06eda4a6faca84e84a781fee2d5f47e1',

    '0a4b7a7499ed55c71033cefb0765e93d',

    '0838c82917cd9af681df249264d2769c',

    '046b35ae95374bfb48cdca8d7c83233f',

    '074c3e01525681a275a42282cd21cbde',

    '05abe25c883d508ecc15b6e857e59f32',

    '05f4e9415af9fdabc19109c980daf5ad',

    '060121a06476ef401d8a21d6567dee6d',

    '068b0e3be4c35ea983f77accf8351cc8',

    '08f055372c7b8a7e1df97c6586542ac8']



display_images(images)
slide = '0005f7aaab2800f6170c399693a96917'

mask_slide = openslide.OpenSlide(os.path.join("/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/", f'{slide}_mask.tiff')) # Making Openslide Obje

display(mask_slide.get_thumbnail(size=(600,400)))
import matplotlib

def display_masks(slides):    

    f, ax = plt.subplots(2,3, figsize=(18,22))

    for i, slide in enumerate(slides):

        

        mask = openslide.OpenSlide(os.path.join("/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/", f'{slide}_mask.tiff'))

        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

        cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])



        ax[i//3, i%3].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) 

        mask.close()       

        ax[i//3, i%3].axis('off')

        

        image_id = slide

        data_provider = train.loc[slide, 'data_provider']

        isup_grade = train.loc[slide, 'isup_grade']

        gleason_score = train.loc[slide, 'gleason_score']

        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")

        f.tight_layout()

        

    plt.show()
display_masks(images[:6]) #Visualizing Only six Examples
def mask_img(image,max_size=(600,400)):

    slide = openslide.OpenSlide(os.path.join("/kaggle/input/prostate-cancer-grade-assessment/train_images", f'{image}.tiff'))

    mask =  openslide.OpenSlide(os.path.join("/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/", f'{image}_mask.tiff'))

    # Here we compute the "pixel spacing": the physical size of a pixel in the image.

    # OpenSlide gives the resolution in centimeters so we convert this to microns.

    f,ax =  plt.subplots(1,2 ,figsize=(18,22))

    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)

    img = slide.get_thumbnail(size=(600,400)) #IMAGE 

    

    mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

    

    ax[0].imshow(img) 

    #ax[0].set_title('Image')

    

    

    ax[1].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) #IMAGE MASKS

    #ax[1].set_title('Image_MASK')

    

    

    image_id = image

    data_provider = train.loc[image, 'data_provider']

    isup_grade = train.loc[image, 'isup_grade']

    gleason_score = train.loc[image, 'gleason_score']

    ax[0].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score} IMAGE")

    ax[1].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score} IMAGE_MASK")

    

    

mask_img('08f055372c7b8a7e1df97c6586542ac8')

dims, spacings, level_counts = [], [], []

down_levels, level_dims = [], []



for i in train.reset_index().image_id:

    slide = openslide.OpenSlide("/kaggle/input/prostate-cancer-grade-assessment/train_images/"+i+".tiff")

    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)

    dims.append(slide.dimensions)

    spacings.append(spacing)

    level_counts.append(slide.level_count)

    down_levels.append(slide.level_downsamples)

    level_dims.append(slide.level_dimensions)

    slide.close()

    del slide



train['width']  = [i[0] for i in dims]

train['height'] = [i[1] for i in dims]