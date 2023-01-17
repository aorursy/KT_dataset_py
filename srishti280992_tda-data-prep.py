# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import cv2

from skimage.filters import prewitt_h,prewitt_v

from skimage.filters import sato

from skimage.filters import sobel

from skimage import feature

from skimage import measure

from skimage.feature import hog

from skimage.filters.rank import entropy

from skimage.morphology import disk

from skimage.measure.entropy import shannon_entropy

from skimage.feature import corner_harris, corner_subpix, corner_peaks

from skimage import data, segmentation, color, filters, io

from skimage.future import graph



from skimage import color



# for dirname, _, filenames in os.walk('/kaggle/input/lego-parts/parts_148/parts_148'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
parts=pd.read_csv("/kaggle/input/lego-parts/parts.csv")

parts['file_name_poss']= parts['part_num']+".png"

parts.head()
parts.shape
#selecting parts with both entries in csv and part drawings

parts_files_path= "/kaggle/input/lego-parts/parts_148/parts_148"

possibilities=parts['file_name_poss'].tolist()

diagrams=os.listdir(parts_files_path)

complete_parts=list(set(possibilities) & set(diagrams))

len(complete_parts)
parts_select=parts[parts['file_name_poss'].isin(complete_parts)]

parts_select.shape

parts_select
parts_select[parts_select.duplicated('file_name_poss')]

# 5 part_names are duplicated
duplicated_parts= ["4591.png","4588.png","4590.png","4740.png","4742.png"]

parts_select[parts_select['file_name_poss'].isin(duplicated_parts)]
parts_select.drop_duplicates(subset ="file_name_poss", 

                     keep = False, inplace = True)

parts_select.shape
#Distribution of data across part categories

percent_distr = round(parts_select["part_cat_id"].value_counts() / len(parts_select["part_cat_id"]) * 100,2)



print("Data Percent: ")

print(percent_distr)
len(np.unique(parts_select["part_cat_id"]))
complete_parts = [x for x in complete_parts if x not in duplicated_parts]
len(complete_parts)
for part in complete_parts[:20]:

    im = cv2.imread("/kaggle/input/lego-parts/parts_148/parts_148/" + part, cv2.IMREAD_GRAYSCALE)

    

    plt.imshow(im, cmap='gray')

    plt.show()
def load_data(dir_data,part_df):

    ''' Load each of the image files into memory 



    While this is feasible with a smaller dataset, for larger datasets,

    not all the images would be able to be loaded into memory

    '''

    parts_df  = part_df

    labels    = parts_df.part_cat_id.values

    ids       = parts_df.part_num.values

    data      = []

    for identifier in ids:

        fname     = dir_data + identifier + '.png'

        image     = mpl.image.imread(fname)

        data.append(image)

    data = np.array(data) # Convert to Numpy array

    return data, labels
dir_images="../input/lego-parts/parts_148/parts_148/"

data, labels = load_data(dir_images,parts_select)
# using Luminance to obtain grayscale images

# luminance is the weighted average of RGB values

data_gray = [ color.rgb2gray(i) for i in data]
hor_edges = []

ver_edges = []

for image in data_gray:

    #calculating horizontal edges using prewitt kernel

    edges_prewitt_horizontal = prewitt_h(image)

    #calculating vertical edges using prewitt kernel

    edges_prewitt_vertical = prewitt_v(image)

    hor_edges.append(edges_prewitt_horizontal)

    ver_edges.append(edges_prewitt_vertical)

# plot image after applying vertical edge filter to grayscale image

plt.imshow(ver_edges[855])

plt.axis('off')

plt.title('Image after applying vertical edge detector')
# plot image after applying vertical edge filter to grayscale image

plt.imshow(hor_edges[855])

plt.axis('off')

plt.title('Image after applying horizontal edge detector')
canny_edges_for_image = []

for image in data_gray:

    # running this bit to get the images after canny edge detector is applied- to visualize

    edges_sigma1 = feature.canny(image, sigma=3)

    canny_edges_for_image.append(edges_sigma1)

    

len(canny_edges_for_image)
# plot image after applying canny edge detector to grayscale image

plt.imshow(canny_edges_for_image[855])

plt.axis('off')

plt.title('Image after applying Canny Edge Detector')
#corner detection algos

harris_corner_for_image = []

corner_peaks_res=[]

corner_subpix_res=[]

for image in data_gray:

    coords = corner_harris(image)

    coords_peaks = corner_peaks(corner_harris(image), min_distance=5, threshold_rel=0.02)

    coords_subpix = corner_subpix(image, coords_peaks, window_size=13)

    harris_corner_for_image.append(coords)

    corner_peaks_res.append(coords_peaks)

    corner_subpix_res.append(coords_peaks)

    

len(harris_corner_for_image)
# plot image after applying corner detection algos

plt.imshow(harris_corner_for_image[855])

plt.axis('off')

plt.title('Image after applying Harris Corner Detector')
# plot image after applying corner detection algos

plt.imshow(corner_peaks_res[855])

plt.axis('off')

plt.title('Image after applying Find peaks in corner measure response image')
# plot image after applying corner detection algos

plt.imshow(corner_subpix_res[855])

plt.axis('off')

plt.title('Image after applying subpixel position of corners')
#compute entropies- apply entropies filter

entropies = []

for image in data_gray:

    #getting images with entropy filter applied to the images

    e1 =entropy(image, disk(10))

    entropies.append(e1)

    

len(entropies)
# plot image after applying entropy filter to grayscale image

plt.imshow(entropies[855])

plt.axis('off')

plt.title('Image after applying entropy filter')
# apply HOG (Histogram of Oriented Gradient) transformation

ppc = 16

hog_images = []

hog_features = []

for image in data_gray:

    # extracting hog transformed images and the feature arrays

    fd,hog_image = hog(image, orientations=10, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)

    hog_images.append(hog_image)

    hog_features.append(fd)
# plot image after applying HOG filter to grayscale image

plt.imshow(hog_images[855])

plt.axis('off')

plt.title('Image after applying HOG filter')
sato_res = []

for image in data_gray:

    #getting images with entropy filter applied to the images

    st1 =sato(image)

    sato_res.append(st1)

    

len(sato_res)
# plot image after applying sato tube filter

plt.imshow(sato_res[855])

plt.axis('off')

plt.title('Image after applying Sato Tube filter')
# import skimage

# for img in data_gray:

#     labels = segmentation.slic(img)

#     edge_map = filters.sobel(color.rgb2gray(img))

#     rag = graph.rag_boundary(labels, edge_map)

 

    

# NEED TO WORK ON THIS