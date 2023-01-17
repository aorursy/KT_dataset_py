from IPython.display import IFrame, YouTubeVideo

YouTubeVideo('UuG__lpn8qQ',width=600, height=400)
#BASIC

import numpy as np 

import pandas as pd 

import os



# DATA visualization

import matplotlib.pyplot as plt

import seaborn as sns

import PIL

from IPython.display import Image, display

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff



import openslide

BASE_FOLDER = "/kaggle/input/prostate-cancer-grade-assessment/"

!ls {BASE_FOLDER}
mask_dir = f'{BASE_FOLDER}/train_label_masks'
train = pd.read_csv(BASE_FOLDER+"train.csv")

test = pd.read_csv(BASE_FOLDER+"test.csv")

sub = pd.read_csv(BASE_FOLDER+"sample_submission.csv")
train.head()
print("unique ids : ", len(train.image_id.unique()))

print("unique data provider : ", len(train.data_provider.unique()))

print("unique isup_grade(target) : ", len(train.isup_grade.unique()))

print("unique gleason_score : ", len(train.gleason_score.unique()))
train['gleason_score'].unique()
print(train[train['gleason_score']=='0+0']['isup_grade'].unique())

print(train[train['gleason_score']=='negative']['isup_grade'].unique())
print(len(train[train['gleason_score']=='0+0']['isup_grade']))

print(len(train[train['gleason_score']=='negative']['isup_grade']))
print(train[(train['gleason_score']=='3+4') | (train['gleason_score']=='4+3')]['isup_grade'].unique())

print(train[(train['gleason_score']=='3+5') | (train['gleason_score']=='5+3')]['isup_grade'].unique())

print(train[(train['gleason_score']=='5+4') | (train['gleason_score']=='4+5')]['isup_grade'].unique())
print(train[train['gleason_score']=='3+4']['isup_grade'].unique())

print(train[train['gleason_score']=='4+3']['isup_grade'].unique())
train[(train['isup_grade'] == 2) & (train['gleason_score'] == '4+3')]
train.drop([7273],inplace=True)
train['gleason_score'] = train['gleason_score'].apply(lambda x: "0+0" if x=="negative" else x)
print("shape : ", test.shape)

print("unique ids : ", len(test.image_id.unique()))

print("unique data provider : ", len(test.data_provider.unique()))
temp = train.groupby('isup_grade').count()['image_id'].reset_index().sort_values(by='image_id',ascending=False)

temp.style.background_gradient(cmap='Purples')
fig = go.Figure(go.Funnelarea(

    text =temp.isup_grade,

    values = temp.image_id,

    title = {"position": "top center", "text": "Funnel-Chart of ISUP_grade Distribution"}

    ))

fig.show()
fig = px.bar(temp, x='isup_grade', y='image_id',

             hover_data=['image_id', 'isup_grade'], color='image_id',

             labels={'pop':'population of Canada'}, height=400)

fig.show()
fig = plt.figure(figsize=(10,6))

ax = sns.countplot(x="isup_grade", hue="data_provider", data=train)

for p in ax.patches:

    '''

    Courtesy of Rohit Singh for teaching me this

    https://www.kaggle.com/rohitsingh9990/panda-eda-better-visualization-simple-baseline

    '''

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2,

                height +3,

                '{:1.2f}%'.format(100*height/10616),

                ha="center")
temp = train.groupby('gleason_score').count()['image_id'].reset_index().sort_values(by='image_id',ascending=False)

temp.style.background_gradient(cmap='Reds')
fig = go.Figure(go.Funnelarea(

    text =temp.gleason_score,

    values = temp.image_id,

    title = {"position": "top center", "text": "Funnel-Chart of ISUP_grade Distribution"}

    ))

fig.show()
fig = px.bar(temp, x='gleason_score', y='image_id',

             hover_data=['image_id', 'gleason_score'], color='image_id',

             labels={'pop':'population of Canada'}, height=400)

fig.show()
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
'''

Example for using Openslide to display an image

'''





# Open the image (does not yet read the image into memory)

example = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", '005e66f06bce9c2e49142536caf2f6ee.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = example.read_region((17800,19500), 0, (256, 256))



# Display the image

display(patch)



# Close the opened slide after use

example.close()
train = train.set_index('image_id')

train.head()
def get_values(image,max_size=(600,400)):

    slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image}.tiff'))

    

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
get_values('07a7ef0ba3bb0d6564a73f4f3e1c2293')
def display_images(images):

    '''

    This function takes in input a list of images. It then iterates through the image making openslide objects , on which different functions

    for getting out information can be called later

    '''

    f, ax = plt.subplots(5,3, figsize=(18,22))

    for i, image in enumerate(images):

        slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image}.tiff')) # Making Openslide Object

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

    '08f055372c7b8a7e1df97c6586542ac8'

]



display_images(images)
example_mask =  openslide.OpenSlide(os.path.join(mask_dir, f'{"00412139e6b04d1e1cee8421f38f6e90"}_mask.tiff'))

display(example_mask.get_thumbnail(size=(600,400)))
import matplotlib

def display_masks(slides):    

    f, ax = plt.subplots(2,3, figsize=(18,22))

    for i, slide in enumerate(slides):

        

        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{slide}_mask.tiff'))

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

    slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image}.tiff'))

    mask =  openslide.OpenSlide(os.path.join(mask_dir, f'{image}_mask.tiff'))

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
mask_img('07a7ef0ba3bb0d6564a73f4f3e1c2293')
images1= [

    '08ab45297bfe652cc0397f4b37719ba1',

    '090a77c517a7a2caa23e443a77a78bc7'

]



for image in images1:

    mask_img(image)
def differentiate_cancerous(image_mask):

    mask =  openslide.OpenSlide(os.path.join(mask_dir, f'{image_mask}_mask.tiff'))

    mask_level = mask.read_region((0,0),mask.level_count - 1,mask.level_dimensions[-1]) #Selecting the level

    mask_data = np.asarray(mask_level)[:,:,0] #SELECTING R from RGB

    

    mask_background = np.where(mask_data == 0, 1, 0).astype(np.uint8) # SELECTING BG

    mask_benign = np.where(mask_data == 1, 1, 0).astype(np.uint8) #SELECTING BENIGN LABELS

    

    if train.loc[image_mask,'data_provider'] == 'karolinska':

        mask_cancerous = np.where(mask_data == 2, 1, 0).astype(np.uint8) #SELECTING CANCEROUS LABELS

    elif train.loc[image_mask,'data_provider'] == 'radboud':

        mask_cancerous = np.where(mask_data == 5, 1, 0).astype(np.uint8) #SELECTING NON-CANCEROUS LABELS

        

    return mask_background,mask_benign,mask_cancerous
image2 =[ '07a7ef0ba3bb0d6564a73f4f3e1c2293',

    'ffdc59cd580a1468eac0e6a32dd1ff2d']



for image in image2:

    background,benign,cancerous = differentiate_cancerous(image)



    #if train.loc[image,'data_provider'] == 'karolinska'

    fig, ax = plt.subplots(1, 3, figsize=(18, 12))



    ax[0].imshow(background.astype(float), cmap=plt.cm.gray)

    ax[0].axis('off')

    ax[0].set_title('background,'+'  '+'data_provider:'+train.loc[image]["data_provider"]);

    ax[1].imshow(benign.astype(float), cmap=plt.cm.gray)

    ax[1].axis('off')

    ax[1].set_title('benign');

    ax[2].imshow(cancerous.astype(float), cmap=plt.cm.gray)

    ax[2].axis('off')

    ax[2].set_title('cancerous')
def overlay_mask_on_slide(images, center='radboud', alpha=0.8, max_size=(800, 800)):

    """Show a mask overlayed on a slide."""

    f, ax = plt.subplots(2,3, figsize=(18,22))

    

    

    for i, image_id in enumerate(images):

        slide = openslide.OpenSlide(os.path.join(BASE_FOLDER+"train_images", f'{image_id}.tiff'))

        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{image_id}_mask.tiff'))

        slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])

        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

        mask_data = mask_data.split()[0]

        

        # Create alpha mask

        alpha_int = int(round(255*alpha))

        if center == 'radboud':

            alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)

        elif center == 'karolinska':

            alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)



        alpha_content = PIL.Image.fromarray(alpha_content)

        preview_palette = np.zeros(shape=768, dtype=int)



        if center == 'radboud':

            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}

            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)

        elif center == 'karolinska':

            # Mapping: {0: background, 1: benign, 2: cancer}

            preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)



        mask_data.putpalette(data=preview_palette.tolist())

        mask_rgb = mask_data.convert(mode='RGB')

        overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)

        overlayed_image.thumbnail(size=max_size, resample=0)



        

        ax[i//3, i%3].imshow(overlayed_image) 

        slide.close()

        mask.close()       

        ax[i//3, i%3].axis('off')

        

        data_provider = train.loc[image_id, 'data_provider']

        isup_grade = train.loc[image_id, 'isup_grade']

        gleason_score = train.loc[image_id, 'gleason_score']

        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
overlay_mask_on_slide(images[:6])
dims, spacings, level_counts = [], [], []

down_levels, level_dims = [], []



for i in train.reset_index().image_id:

    slide = openslide.OpenSlide(BASE_FOLDER+"train_images/"+i+".tiff")

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
plt.figure(figsize=(12,6))

p1=sns.kdeplot(train['width'], shade=True, color="b").set_title('KDE of Width and Height of images')

p2=sns.kdeplot(train['height'], shade=True, color="r")

plt.legend(labels=['width','height'])