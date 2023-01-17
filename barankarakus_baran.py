import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely  # for working with WKT format
import tifffile # for working with TIFF or TIF image format
import zipfile  # for working with massive .zip file
# loading grid sizes data
# not entirely sure what this data is - something to do with co-ordinate grids of images
grid_sizes = pd.read_csv('/kaggle/input/dstl-satellite-imagery-feature-detection/grid_sizes.csv.zip', index_col=0)
grid_sizes.index.name = 'image_id'
print('450 image IDs. These are the grid sizes:')
grid_sizes.head()
train_wkt = pd.read_csv('/kaggle/input/dstl-satellite-imagery-feature-detection/train_wkt_v4.csv.zip', index_col=0)
print('Here is the training data "labels".')
print('25 training images, 10 classes per image, so 250 rows.')
train_wkt
# dataset should be 'three_band' or 'sixteen_band'; if 'sixteen_band' then need code 'P', 'M', or 'A'.
# returns BytesIO object which can be passed as a file to any image reading function.
def get_img_file(dataset, imageid='6120_2_3', code='P'):
    if dataset == 'sixteen_band':
        if code in ['P', 'M', 'A']:
            filename = f'{dataset}/{imageid}_{code}.tif'
        else:
            print('Need code to be P, M, or A.')
            return
    elif dataset == 'three_band':
        filename = f'{dataset}/{imageid}.tif'
    else:
        print('Need dataset to be three_band or sixteen_band')
    archive = zipfile.ZipFile(f'../input/dstl-satellite-imagery-feature-detection/{dataset}.zip', 'r')
    img_as_bytes = archive.read(filename)
    img_file = io.BytesIO(img_as_bytes) # can now pass this to any img reader as a file
    return img_file

def stretch2(band, lower_percent=1, higher_percent=99):
    a = 0 #np.min(band)
    b = 255  #np.max(band)
    c = np.percentile(band, lower_percent)
    d = np.percentile(band, higher_percent)        
    out = a + (band - c) * (b - a) / (d - c)    
    out[out<a] = a
    out[out>b] = b
    return out

# def plot_rgb(imageid='6120_2_3'):
#     rgb_img_file = get_img_file('three_band', imageid)
#     img_as_arr = tifffile.imread(rgb_img_file)  # dtype = uint16, shape (3, 3348, 3403)
#     img_as_arr = np.rollaxis(img_as_arr, 0, 3)  # changing shape to (3348, 3403, 3) (necessary to plot with pyplot)
#     # 
#     return img_as_arr
imageid='6010_0_0'
rgb_img_file = get_img_file('three_band', imageid)
img_as_arr = tifffile.imread(rgb_img_file)
img_as_arr = np.rollaxis(img_as_arr, 0, 3)
# assuming pixel intensities range from 0 to 2^11 - 1 
img1 = (img_as_arr/(2**11) * 256).astype(np.uint8)



# for i in range(3):
#     img_as_arr[:, :, i] = img_as_arr[:, :, i].astype(float)/img_as_arr[:, :, i].max()
img2 = np.array([img_as_arr[:, :, i]/img_as_arr[:, :, i].max() for i in range(3)])
img2 = np.rollaxis(img2, 0, 3)
plt.imshow(img2)

# print(img_as_arr[:, :, 0].max()) 
# print(img_as_arr[:, :, 1].max()) 
# print(img_as_arr[:, :, 2].max()) 
imageid='6010_0_0'
rgb_img_file = get_img_file('three_band', imageid)
img_as_arr = tifffile.imread(rgb_img_file)
tifffile.imshow(img_as_arr)
img_file = get_img_file('three_band', '6010_0_0')
img_file

import tifffile
import cv2

tif_arr = np.rollaxis(tifffile.imread(img_file), 0, 3)

tif_arr2 = tif_arr.astype(float)
plt.imshow(stretch2(tif_arr2).astype(np.uint8))
plt.imshow(((tif_arr/(2**11))*256).astype(np.uint8))
2**11
plt.imshow(rgb, vmin=0, vmax=2**16)

np.rollaxis(tif_arr, 0, 3).shape

from PIL import Image
file='../input/dstl-satellite-imagery-feature-detection/sixteen_band/6120_2_2_A.tif'
#imBandA = Image.open(rb'../input/three_band/6120_2_2.tif')
#imBandA.show()

#from scipy import misc
#raster = misc.imread('../input/sixteen_band/6120_2_2_A.tif')
#type(raster)

import os
os.listdir('../input/dstl-satellite-imagery-feature-detection/three_band')

with open('../input/dstl-satellite-imagery-feature-detection/three_band/6120_2_2.tif',encoding='utf-8', errors='ignore') as f:
    print(f.readlines())

imarray = plt.imread('image.tif')
'sfad '.strip(' ')
def plot_rgb(imageid):
    img_as_stream = get_img_file('three_band', 'image_id')
    
# for example, using tifffile:
import tifffile
img_as_arr = tifffile.imread(img_as_stream) #
img_as_arr.min()
2**16
plt.imshow(img_as_arr[7])

plt.imshow(rgb[:, :, 0])
img_as_arr[0]
img_as_arr[0]
def stretch2(band, lower_percent=1, higher_percent=99):
    a = 0 #np.min(band)
    b = 255  #np.max(band)
    c = np.percentile(band, lower_percent)
    d = np.percentile(band, higher_percent)        
    out = a + (band - c) * (b - a) / (d - c)    
    out[out<a] = a
    out[out>b] = b
    return out
def adjust_contrast(x):    
    for i in range(3):
        x[:,:,i] = stretch2(x[:,:,i], 0, 100)
    return x.astype(np.uint8) 


plt.imshow(rgb2.astype(np.uint8))
help(plt.imshow)
rgb[:, :, 0]
import io

stream = io.BytesIO(imgdata)
"""
Author : amanbh

- Set up some basic functions to load/manipulate image data
- Visualize/Summarize cType counts, training data, and true classes
- Plot Polygons with holes correctly by using descartes package

Based on Kernel by
    Author : Oleg Medvedev
    Link   : https://www.kaggle.com/torrinos/dstl-satellite-imagery-feature-detection/exploration-and-plotting/run/553107
"""

import pandas as pd
import numpy as np

from shapely.wkt import loads as wkt_loads  # for working with WKT format
from matplotlib.patches import Polygon, Patch

# decartes package makes plotting with holes much easier
from descartes.patch import PolygonPatch

import matplotlib.pyplot as plt
import tifffile as tiff  # for working with .tif and .tiff files

import pylab
# turn interactive mode on so that plots immediately
# See: http://stackoverflow.com/questions/2130913/no-plot-window-in-matplotlib
# pylab.ion()
import zipfile

inDir = '../input/dstl-satellite-imagery-feature-detection'  

# Give short names, sensible colors and z-orders to object types
CLASSES = {
        1 : 'Bldg',
        2 : 'Struct',
        3 : 'Road',
        4 : 'Track',
        5 : 'Trees',
        6 : 'Crops',
        7 : 'Fast H20',
        8 : 'Slow H20',
        9 : 'Truck',
        10 : 'Car',
        }
COLORS = {
        1 : '0.7',
        2 : '0.4',
        3 : '#b35806',
        4 : '#dfc27d',
        5 : '#1b7837',
        6 : '#a6dba0',
        7 : '#74add1',
        8 : '#4575b4',
        9 : '#f46d43',
        10: '#d73027',
        }
ZORDER = {
        1 : 5,
        2 : 5,
        3 : 4,
        4 : 1,
        5 : 3,
        6 : 2,
        7 : 7,
        8 : 8,
        9 : 9,
        10: 10,
        }
# z-orders determine overlap order for the polygons

# read the training data from train_wkt_v4.csv
df = pd.read_csv(inDir + '/train_wkt_v4.csv.zip')
print('training data:')
print(df.head())
print('\n')

# grid size will also be needed later..
gs = pd.read_csv(inDir + '/grid_sizes.csv.zip', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
print('grid sizes data:')
print(gs.head())
print('\n')
# imageIds in an array
allImageIds = gs.ImageId.unique()
trainImageIds = df.ImageId.unique()
def get_images(imageId):
    '''
    Load images correspoding to imageId

    Parameters
    ----------
    imageId : str
        imageId as used in grid_size.csv
    img_key : {None, '3', 'A', 'M', 'P'}, optional
        Specify this to load single image
        None loads all images and returns in a dict
        '3' loads image from three_band/
        'A' loads '_A' image from sixteen_band/
        'M' loads '_M' image from sixteen_band/
        'P' loads '_P' image from sixteen_band/

    Returns
    -------
    images : dict
        A dict of image data from TIFF files as numpy array
        Keys: ['3', 'A', 'M', 'P']
    '''
    images = dict()
    # getting 3-band image
    archive = zipfile.ZipFile(f'{inDir}/three_band.zip', 'r')
    img_as_bytes = archive.read(f'three_band/{imageId}.tif')
    img_file = io.BytesIO(img_as_bytes)  # byte stream
    images['3'] = tiff.imread(img_file)
    # getting 16-band images
    archive = zipfile.ZipFile(f'{inDir}/sixteen_band.zip', 'r')
    for k in ['P', 'M', 'A']:
        img_as_bytes = archive.read(f'sixteen_band/{imageId}_{k}.tif')
        img_file = io.BytesIO(img_as_bytes) 
        images[k] = tiff.imread(img_file)
    return images

def get_size(imageId):
    """
    Get the grid size of the image

    Parameters
    ----------
    imageId : str
        imageId as used in grid_size.csv
    """
    xmax, ymin = gs[gs.ImageId == imageId].iloc[0,1:].astype(float)
    W, H = get_images(imageId)['3'].shape[1:]
    return (xmax, ymin, W, H)

def is_training_image(imageId):
    '''
    Returns
    -------
    is_training_image : bool
        True if imageId belongs to training data
    '''
    return any(trainImageIds == imageId)

def plot_image(fig, ax, imageId, img_key, selected_channels=None):
    '''
    Plot get_images(imageId)[img_key] on axis/fig supplied
    Optional: select which channels of the image are used (used for sixteen_band/ images)
    Parameters
    ----------
    img_key : str, {'3', 'P', 'N', 'A'}
        See get_images for description.
    '''
    images = get_images(imageId)
    img = images[img_key]
    title_suffix = ''
    if selected_channels is not None:
        img = img[selected_channels]
        title_suffix = ' (' + ','.join([ repr(i) for i in selected_channels ]) + ')'
    if len(img.shape) == 2:
        new_img = np.zeros((3, img.shape[0], img.shape[1]))
        new_img[0] = img
        new_img[1] = img
        new_img[2] = img
        img = new_img
    
    tiff.imshow(img, figure=fig, subplot=ax)
    ax.set_title(imageId + ' - ' + img_key + title_suffix)
    ax.set_xlabel(img.shape[-2])
    ax.set_ylabel(img.shape[-1])
    ax.set_xticks([])
    ax.set_yticks([])

def plot_polygons(fig, ax, polygonsList):
    '''
    Plot descrates.PolygonPatch from list of polygons objs for each CLASS
    '''
    legend_patches = []
    for cType in polygonsList:
        print('{} : {} \tcount = {}'.format(cType, CLASSES[cType], len(polygonsList[cType])))
        legend_patches.append(Patch(color=COLORS[cType],
                                    label='{} ({})'.format(CLASSES[cType], len(polygonsList[cType]))))
        for polygon in polygonsList[cType]:
            mpl_poly = PolygonPatch(polygon,
                                    color=COLORS[cType],
                                    lw=0,
                                    alpha=0.7,
                                    zorder=ZORDER[cType])
            ax.add_patch(mpl_poly)
    # ax.relim()
    ax.autoscale_view()
    ax.set_title('Objects')
    ax.set_xticks([])
    ax.set_yticks([])
    return legend_patches

def visualize_image(imageId, plot_all=True):
    '''         
    Plot all images and object-polygons
    
    Parameters
    ----------
    imageId : str
        imageId as used in grid_size.csv
    plot_all : bool, True by default
        If True, plots all images (from three_band/ and sixteen_band/) as subplots.
        Otherwise, only plots Polygons.
    '''         
    df_image = df[df.ImageId == imageId]
    xmax, ymin, W, H = get_size(imageId)
    
    if plot_all:
        fig, axArr = plt.subplots(figsize=(10, 10), nrows=3, ncols=3)
        ax = axArr[0][0]
    else:
        fig, axArr = plt.subplots(figsize=(10, 10))
        ax = axArr
    if is_training_image(imageId):
        print('ImageId : {}'.format(imageId))
        polygonsList = {}
        for cType in CLASSES.keys():
            polygonsList[cType] = wkt_loads(df_image[df_image.ClassType == cType].MultipolygonWKT.values[0])
        legend_patches = plot_polygons(fig, ax, polygonsList)
        ax.set_xlim(0, xmax)
        ax.set_ylim(ymin, 0)
        ax.set_xlabel(xmax)
        ax.set_ylabel(ymin)
    if plot_all:
        plot_image(fig, axArr[0][1], imageId, '3')
        plot_image(fig, axArr[0][2], imageId, 'P')
        plot_image(fig, axArr[1][0], imageId, 'A', [0, 3, 6])
        plot_image(fig, axArr[1][1], imageId, 'A', [1, 4, 7])
        plot_image(fig, axArr[1][2], imageId, 'A', [2, 5, 0])
        plot_image(fig, axArr[2][0], imageId, 'M', [0, 3, 6])
        plot_image(fig, axArr[2][1], imageId, 'M', [1, 4, 7])
        plot_image(fig, axArr[2][2], imageId, 'M', [2, 5, 0])

    if is_training_image(imageId):
        ax.legend(handles=legend_patches,
                   # loc='upper center',
                   bbox_to_anchor=(0.9, 1),
                   bbox_transform=plt.gcf().transFigure,
                   ncol=5,
                   fontsize='x-small',
                   title='Objects-' + imageId,
                   # mode="expand",
                   framealpha=0.3)
    return (fig, axArr, ax)

# # Loop over few training images and save to files
# for imageId in trainImageIds:
#     fig, axArr, ax = visualize_image(imageId, plot_all=False)
#     plt.savefig('Objects--' + imageId + '.png')
#     plt.clf()


# Optionally, view images immediately:
# pylab.show()
# Uncomment to show plot when interactive mode is off 
# (this function blocks till fig is closed)
trainImageIds
visualize_image('6120_2_2')