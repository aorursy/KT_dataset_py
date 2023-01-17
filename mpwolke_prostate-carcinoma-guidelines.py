#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRcAo69vF1onCRMd_XyDxUJifE_Yr18s8Zd9WiKDgFdzyzNG-cn&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import PIL

from IPython.display import Image, display

import plotly.express as px

import seaborn

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6634264/bin/CMAR-11-6521-g0001.jpg',width=400,height=400)
psa = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")

psa.head()
fig = px.bar(psa, 

             x='gleason_score', y='isup_grade', color_discrete_sequence=['purple'],

             title='Prostate Carcinoma Guidelines', text='gleason_score')

fig.show()
fig = px.scatter(psa.dropna(), x='gleason_score',y='image_id', trendline="data_provider", color_discrete_sequence=['purple'])

fig.show()
fig = px.density_contour(psa, x="gleason_score", y="isup_grade", color_discrete_sequence=['purple'])

fig.show()
px.histogram(psa, x='gleason_score', color='isup_grade')
seaborn.set(rc={'axes.facecolor':'magenta', 'figure.facecolor':'magenta'})

# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Prostate Carcinoma Guidelines")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=psa.index, y=psa['gleason_score'])



# Add label for vertical axis

plt.ylabel("Prostate Carcinoma Guidelines")
fig = px.line(psa, x="isup_grade", y="data_provider",  color_discrete_sequence=['purple'],  

              title="Prostate Carcinoma Guidelines")

fig.show()
seaborn.set(rc={'axes.facecolor':'#27F1E7', 'figure.facecolor':'#27F1E7'})

sns.countplot(psa["isup_grade"])

plt.xticks(rotation=90)

plt.show()
import openslide

import skimage.io

import PIL

from IPython.display import Image, display

import plotly.graph_objs as go
# Location of the training images

data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images'

mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks'
# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, '274187a867ebf7834ab44372da776439.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((17800,19500), 0, (256, 256))



# Display the image

display(patch)



# Close the opened slide after use

image.close()
biopsy = openslide.OpenSlide(os.path.join(data_dir, '07c52f877d1c531c1da6ea32b3c6bff5.tiff'))



x = 5150

y = 21000

level = 0

width = 512

height = 512



region = biopsy.read_region((x,y), level, (width, height))

display(region)
x = 5140

y = 21000

level = 1

width = 512

height = 512



region = biopsy.read_region((x,y), level, (width, height))

display(region)
def print_mask_details(slide, center='radboud', show_thumbnail=True, max_size=(400,400)):

    """Print some basic information about a slide"""



    if center not in ['radboud', 'karolinska']:

        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")



    # Generate a small image thumbnail

    if show_thumbnail:

        # Read in the mask data from the highest level

        # We cannot use thumbnail() here because we need to load the raw label data.

        mask_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])

        # Mask data is present in the R channel

        mask_data = mask_data.split()[0]

        # To show the masks we map the raw label values to RGB values

        preview_palette = np.zeros(shape=768, dtype=int)

        if center == 'radboud':

            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}

            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)

        elif center == 'karolinska':

            # Mapping: {0: background, 1: benign, 2: cancer}

            preview_palette[0:9] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 0, 0]) * 255).astype(int)

        mask_data.putpalette(data=preview_palette.tolist())

        mask_data = mask_data.convert(mode='RGB')

        mask_data.thumbnail(size=max_size, resample=0)

        display(mask_data)

        

            # Compute microns per pixel (openslide gives resolution in centimeters)

    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)

    

    print(f"File id: {slide}")

    print(f"Dimensions: {slide.dimensions}")

    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")

    print(f"Number of levels in the image: {slide.level_count}")

    print(f"Downsample factor per level: {slide.level_downsamples}")

    print(f"Dimensions of levels: {slide.level_dimensions}")
mask = openslide.OpenSlide(os.path.join(mask_dir, '78bd2af7d449cc2185aaf861f61531aa_mask.tiff'))

print_mask_details(mask, center='radboud')

mask.close()
mask = openslide.OpenSlide(os.path.join(mask_dir, '878be7ee5bdda3b29e2417f6dc93af64_mask.tiff'))

print_mask_details(mask, center='karolinska')

mask.close()
from wordcloud import WordCloud

def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in psa["data_provider"]])

wordcloud = WordCloud(max_font_size=None,colormap='Set3', background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='Prostate Carcinoma Guidelines')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQrQRag1f6EWlByk91cVmHzKQQdkxEA-tWH29Za3Ai9C_rCAG8y&usqp=CAU',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTtlFrQsdG2PaVnf3qUQXYec6nfW8PBKlWtTaTgxdQ4VFM47UYB&usqp=CAU',width=400,height=400)