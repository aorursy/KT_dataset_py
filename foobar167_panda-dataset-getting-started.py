import os
# Open images with OpenSlide
import openslide

# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
from IPython.display import display, FileLink, HTML
# Location of the training images
data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images'
mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks'

# Location of training labels
train = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv').set_index('image_id')
test = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/test.csv').set_index('image_id')

train_images = os.listdir(data_dir)
train_masks = os.listdir(mask_dir)

print('Number of images: ', len(train))
print('Number of masks: ', len(train))
print('Shape of the training data: ', train.shape)
print('Shape of the test data: ', test.shape)

display(train.head(8))
display(test.head())
import IPython.display
import PIL.Image

palette_radboud = [0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]
palette_karolinska = [0, 0, 0, 0.5, 0.5, 0.5, 1, 0, 0]

a = np.zeros( (100, 100, 3), dtype=np.uint8)

a[:,:] = (np.array(palette_radboud[0:3]) * 255).astype(int)
b = np.copy(a)
a[:,:] = (np.array(palette_radboud[3:6]) * 255).astype(int)
b = np.concatenate((b, a), axis=1)
a[:,:] = (np.array(palette_radboud[6:9]) * 255).astype(int)
b = np.concatenate((b, a), axis=1)

a[:,:] = (np.array(palette_radboud[9:12]) * 255).astype(int)
c = np.copy(a)
a[:,:] = (np.array(palette_radboud[12:15]) * 255).astype(int)
c = np.concatenate((c, a), axis=1)
a[:,:] = (np.array(palette_radboud[15:18]) * 255).astype(int)
c = np.concatenate((c, a), axis=1)

d = np.concatenate((b, c), axis=0)

IPython.display.display(PIL.Image.fromarray(d))

print('--- Radboud dataset ---')
print('Black\tbackground\t\t0')
print('Gray\tstroma\t\t\t1')
print('Green\tbenign epithelium\t2')
print('Yellow\tGleason-3 cancer\t3')
print('Orange\tGleason-4 cancer\t4')
print('Red\tGleason-5 cancer\t5')

a[:,:] = (np.array(palette_karolinska[0:3]) * 255).astype(int)
b = np.copy(a)
a[:,:] = (np.array(palette_karolinska[3:6]) * 255).astype(int)
b = np.concatenate((b, a), axis=1)
a[:,:] = (np.array(palette_karolinska[6:9]) * 255).astype(int)
b = np.concatenate((b, a), axis=1)

IPython.display.display(PIL.Image.fromarray(b))

print('--- Karolinska dataset ---')
print('Black\tbackground\t0')
print('Gray\tbenign\t\t1')
print('Red\tcancer\t\t2')
del(a); del(b); del(c); del(d)
import shutil

workdir = '/kaggle/working/'  # directory with write access
size = (256, 256)  # patch size

def get_patch(image_id, coor, size=size, data=data_dir):
    """ Show patch from the whole-slide image
            image_id - image ID
            coor - (x, y) coordinates of the upper left corner of the patch
            size - (w, h) width and height of the patch to get
            data - dataset, WSI image or its mask
    """
    # Open the image (does not yet read the image into memory)
    image = openslide.OpenSlide(os.path.join(data, f'{image_id}.tiff'))
    # Read a specific region of the image starting at upper left coordinate
    # 'coor' on level 0 and extracting a 'size' pixel patch.
    # At this point image data is read from the file and loaded into memory.
    patch = image.read_region(coor, 0, size)
    patch = patch.convert('RGB')  # convert from RGBA to RGB
    image.close()  # close the opened slide after use
    del(image)
    return patch

def overlay_images(patch, mask, dataset):
    """ Overlay 2 images together
            patch - 1st image, patch
            mask - 2nd image, mask
            dataset - radboud or karolinska
    """
    # Create alpha mask
    alpha_int = round(255 * 0.4)  # opacity
    if dataset == 'radboud':
        alpha_mask = 255 + (np.less(mask, 2).astype('uint8') - 1) * alpha_int
    else:  # dataset == 'karolinska'
        alpha_mask = 255 + (np.less(mask, 1).astype('uint8') - 1) * alpha_int
    alpha_mask = PIL.Image.fromarray(alpha_mask)
    #
    return PIL.Image.composite(
        image1=patch,
        image2=mask.convert(mode='RGB'),  # convert from RGBA to RGB
        mask=alpha_mask)

def show_patch(image_id, coor, size=size):
    """ Show patch from the whole-slide image """
    patch = get_patch(image_id, coor, size)
    mask = get_mask(image_id, coor, size)
    dataset = train.loc[image_id]['data_provider']  # get dataset name
    overlayed = overlay_images(patch, mask, dataset)
    # Concatenate 3 images together
    output = PIL.Image.new('RGBA', (patch.width*3, patch.height))
    output.paste(patch, (0, 0))
    output.paste(overlayed, (patch.width, 0))
    output.paste(mask, (patch.width*2, 0))
    display(output)

def patch_name(image_id, coor, size):
    """ Create patch file name like:
            0005f7aa_kar_0+0_0_03328_03328_1.png 
            00f6ea01_rad_3+3_1_04096_03840_3.png
        Format:
            {uid}_{rad/kar}_{i+j}_{ISUP}_{x_y}_{n}.png
        where:
            uid - first 8 characters from WSI {image_id} name
            'rad' - radboud or 'kar' - karolinska from {data_provider}
            i+j - Gleason score. If 'negative' then '0+0'.
            ISUP - ISUP grade. The label.
            x and y upper left corner patch coordinates on WSI image
            n - patch class, max number of its mask
    """
    row = train.loc[image_id]  # get row by index
    uid = image_id[:8]  # first 8 characters are enough for ID
    rad_kar = row['data_provider'][:3]
    gleason = row['gleason_score']
    if gleason == 'negative':
        gleason = '0+0'
    isup = row['isup_grade']
    x_y = f'{coor[0]:05d}_{coor[1]:05d}'
    mask = get_patch(image_id+'_mask', coor, size, data=mask_dir)  # get mask
    n = np.ndarray.max(np.array(mask))  # max number of the mask
    filename = f'{uid}_{rad_kar}_{gleason}_{isup}_{x_y}_{n}.png'
    return filename

def save_patch(image_id, coor, size=size):
    """ Save patch from the whole-slide image to the file system """
    patch = get_patch(image_id, coor, size)
    filename = patch_name(image_id, coor, size)  # get file name
    patch.save(workdir + filename)  # save patch image
    #
    mask = get_mask(image_id, coor, size)
    mask_file = filename[:-4] + '_mask.png'
    mask.save(workdir + mask_file)  # save mask image
    #
    dataset = train.loc[image_id]['data_provider']  # get dataset name
    overlayed = overlay_images(patch, mask, dataset)
    overlayed_file = filename[:-4] + '_overlay.png'
    overlayed.save(workdir + overlayed_file)  # save overlayed image
    return filename, overlayed_file, mask_file

def load_patch(image_id, coor, size=size):
    """ Load images from Jupyter Notebook """
    images = save_patch(image_id, coor, size)
    # Change directory to workdir otherwise it'll not work
    os.chdir(workdir)
    html = (f'<a href={images[0]} target="_blank">{images[0]}</a><br />'
            f'<a href={images[1]} target="_blank">{images[1]}</a><br />'
            f'<a href={images[2]} target="_blank">{images[2]}</a>')
    return HTML(html)

def get_mask(image_id, coor, size=size):
    """ Get mask from the whole-slide mask """
    mask = get_patch(f'{image_id}_mask', coor, size, data=mask_dir)
    mask = mask.split()[0]  # mask is present in the R channel
    dataset = train.loc[image_id]['data_provider']
    if dataset == 'radboud':
        palette = (np.array(palette_radboud) * 255).astype(int)
    else:  # dataset == 'karolinska':
        palette = (np.array(palette_karolinska) * 255).astype(int)
    mask.putpalette(data=palette.tolist())  # see --> PIL.Image.putpalette
    return mask

def show_mask(image_id, coor, size=size):
    """ Show mask from the whole-slide mask """
    mask = get_mask(image_id, coor, size)
    display(mask)

def get_wsi_mask(image_id, level=0, alpha=None):
    """ Get WSI mask
            image_id - image ID
            level - zoom level, can be [0, 1, 2]
    """
    mask = openslide.OpenSlide(os.path.join(mask_dir, f'{image_id}_mask.tiff'))
    dataset = train.loc[image_id]["data_provider"]  # get dataset name
    # Load data from the level
    mask_data = mask.read_region((0,0), level, mask.level_dimensions[level])
    mask_data = mask_data.split()[0]  # mask data is present in the R channel
    # Create alpha mask
    if alpha is None:
        alpha_mask = None
    else:
        alpha_int = round(255 * alpha)
        if dataset == 'radboud':
            alpha_mask = 255 + (np.less(mask_data, 2).astype('uint8') - 1) * alpha_int
        else:  # dataset == 'karolinska'
            alpha_mask = 255 + (np.less(mask_data, 1).astype('uint8') - 1) * alpha_int
        alpha_mask = PIL.Image.fromarray(alpha_mask)
    # Set palette
    dataset = train.loc[image_id]['data_provider']
    if dataset == 'radboud':
        palette = (np.array(palette_radboud) * 255).astype(int)
    else:  # dataset == 'karolinska':
        palette = (np.array(palette_karolinska) * 255).astype(int)
    mask_data.putpalette(data=palette.tolist())
    mask_data = mask_data.convert(mode='RGB')  # convert from RGBA to RGB
    mask.close()
    del(mask)
    return mask_data, alpha_mask

def overlay_wsi(image_id, level=0, alpha=0.5):
    """ Overlay WSI image with its mask.
            image_id - image ID
            level - zoom level, can be [0, 1, 2]
    """
    image = openslide.OpenSlide(os.path.join(data_dir, f'{image_id}.tiff'))
    image_data = image.read_region((0,0), level, image.level_dimensions[level])
    mask_data, alpha_mask = get_wsi_mask(image_id, level, alpha)
    overlayed_image = PIL.Image.composite(image1=image_data, image2=mask_data, mask=alpha_mask)
    image.close()
    del(image)
    return overlayed_image

def show_wsi(image_id, show=True, show_mask=True,
             size=None, link=True, info=True,
             alpha=0.5, level=2):
    """ Show WSI image and information about it.
            show - show image or not
            show_mask - overlay image with the mask
            size - thumbnail size, None - original size
            link - show link to download WSI image
            info - show info about image
            alpha - overlay opacity of the mask
            level - zoom level, can be [0, 1, 2]
    """
    image = openslide.OpenSlide(os.path.join(data_dir, f'{image_id}.tiff'))

    # Get some info about image
    row = train.loc[image_id]
    dataset = row["data_provider"]  # get dataset name
    if info:  # show info
        # Here we compute the "pixel spacing": the physical size of a pixel in the image.
        # OpenSlide gives the resolution in centimeters so we convert this to microns.
        resolution = 10000 / float(image.properties['tiff.XResolution'])
        print(f'dataset:\t{dataset}')
        print(f'size:\t\t{image.dimensions}')
        print(f'resolution:\t{resolution:.4f}')
        print(f'ISUP grade:\t{row["isup_grade"]}')
        print(f'Gleason score:\t{row["gleason_score"]}')
    
    if show:  # show image
        if show_mask:  # show image with mask
            overlayed_image = overlay_wsi(image_id, level, alpha)
            if size is not None:
                overlayed_image.thumbnail(size=size, resample=0)
            display(overlayed_image)
            if link:
                filename = f'{image_id}_overlay.jpg'  # overlayed image
                overlayed_image.save(workdir + filename)  # save the image
        else:  # show image without mask
            if size is not None:
                image_data = image.get_thumbnail(size=size)
                display(image_data)
                if link:
                    filename = f'{image_id}_thumbnail.jpg'  # thumbnail image
                    image_data.save(workdir + filename)
            else:  # show image without mask and with original size
                image_data = image.read_region((0,0), level, image.level_dimensions[level])
                display(image_data)
                if link:
                    filename = f'{image_id}.tiff'  # original image
                    shutil.copy(os.path.join(data_dir, f'{image_id}.tiff'), workdir)
    else:  # don't show image
        if link:
            filename = f'{image_id}.tiff'  # original image
            shutil.copy(os.path.join(data_dir, f'{image_id}.tiff'), workdir)

    image.close()
    del(image)

    if link:
        # Change directory to workdir otherwise it'll not work
        os.chdir(workdir)
        return FileLink(filename)

def show_wsi_mask(image_id, size=(600, 800)):
    """ Show mask for the WSI image """
    mask_data, _ = get_wsi_mask(image_id, level=2)  # get mask
    mask_data.thumbnail(size=size, resample=0)  # create thumbnail
    display(mask_data)

def load_wsi(image_id, alpha=0.3):
    """ Get link to overlayed WSI image """
    overlayed_image = overlay_wsi(image_id, level=0, alpha=alpha)
    filename = f'{image_id}_review.jpg'  # overlayed image
    overlayed_image.save(workdir + filename)  # save the image
    # Change directory to workdir otherwise it'll not work
    os.chdir(workdir)
    return FileLink(filename)

def link_wsi(image_id):
    """ Get downloadable link on WSI file """
    filename = f'{image_id}.tiff'
    shutil.copy(os.path.join(data_dir, filename), workdir)
    os.chdir(workdir)
    html = f'<a href={filename} target="_blank">{filename}</a>'
    return HTML(html)
size = (4000, 17000)
show_patch('00928370e2dfeb8a507667ef1d4efcbb', size)
load_patch('00928370e2dfeb8a507667ef1d4efcbb', size)
show_wsi('08ab45297bfe652cc0397f4b37719ba1', size=(600, 600))
show_wsi_mask('08ab45297bfe652cc0397f4b37719ba1', size=(300, 400))
# Open and review WSI with masks in the browser
# Sometimes there is an error
load_wsi('08ab45297bfe652cc0397f4b37719ba1')
import math
import shutil
import multiprocessing
import tqdm.notebook as tqdm

from joblib import Parallel, delayed

cut_size = 256  # size of resultant images
cut_level = 2  # slide level resolution, use 0 to cut original WSI
down_samples = [1, 4, 16]  # down samples list available in any tiff WSI
cpu_cores = multiprocessing.cpu_count()  # number of cores on the CPU

def test_cut_wsi(image_id, size=cut_size, level=cut_level):
    """ Test toy function to cut the given WSI into multiple images """
    image = openslide.OpenSlide(os.path.join(data_dir, f'{image_id}.tiff'))

    # Get the size of the image on the zoom level
    width, height = image.level_dimensions[level]

    # Get the number of smaller images
    h_cuts = math.ceil(width / size)
    v_cuts = math.ceil(height / size)

    patches = []
    for v in range(v_cuts):
        for h in range(h_cuts):
            x_location = h * size * down_samples[level]
            y_location = v * size * down_samples[level]
            patch = image.read_region((x_location, y_location), level, (size, size))
            patches.append(patch)
    image.close()
    del(image)
    return patches, h_cuts, v_cuts

def cut_image(image_id, dirname):
    """ Cut image into multiple patches """
    size = 256  # size of the patch
    image = openslide.OpenSlide(os.path.join(data_dir, f'{image_id}.tiff'))
    w, h = image.dimensions
    for v in range(0, h, size):
        for h in range(0, w, size):
            patch = image.read_region((h, v), 0, (size, size))  # get patch
            patch = patch.convert(mode='RGB')  # convert from RGBA to RGB

            # Consider gray color as "empty" background
            r, g, b = patch.split()  # split on red, green, blue colors
            r, g, b = np.array(r)/255, np.array(g)/255, np.array(b)/255
            mean = (r + g + b) / 3  # get mean value
            # Get standard deviation
            deviation = np.sqrt(((r-mean)**2 + (g-mean)**2 + (b-mean)**2) / 3)
            # EXPERIMENT with the deviation threshold
            emptiness = (np.count_nonzero(deviation < 0.01)) / deviation.size

            # EXPERIMENT with empty images
            """
            if emptiness > 0.65 and emptiness < 0.75:
                print(emptiness)  # show coefficient
                display(patch)  # show patch
            """

            if emptiness < 0.65:  # ignore 65% empty images
                # Save output image
                name = patch_name(image_id, (h, v), (size, size))  # get patch name
                patch.save(os.path.join(dirname, name))  # save patch image
    image.close()  # close image to free the RAM
    del(image)

def cut_wsi(image_id):
    """ Cut WSI image(s) into multiple patches
        and get a link to a ZIP archive with patches.
    """
    if isinstance(image_id, str):  # one WSI image
        zip_name = f'{image_id}'
        image_id = [image_id]  # convert string to list
    # Non-empty list of WSI images
    elif isinstance(image_id, list) and len(image_id):
        zip_name = 'dataset'
    else:  # wrong parameter
        return None

    dirname = os.path.join(workdir, zip_name)
    shutil.rmtree(dirname, ignore_errors=True)  # remove previous dir
    os.makedirs(dirname)  # create empty dir

    Parallel(n_jobs=cpu_cores)(delayed(cut_image)(im_id, dirname)
                for im_id in tqdm.tqdm(image_id))

    # Create a ZIP archive
    os.chdir(workdir)
    shutil.make_archive(zip_name, 'zip', dirname)
    # Get downloadable link to the ZIP archive
    html = f'<a href={zip_name}.zip target="_blank">{zip_name}.zip</a>'
    return HTML(html)
# Lets cut several images. Cutting all images will take many hours on Kaggle.
ids = ['ffc70bf605de30aaa936533397a29d9c',
       'fca1f600d0d453a15d9251eb505523fc',
       'f96885e98ce7050f21a86bb312f90c89',]

import time

start = time.time()
link = cut_wsi(ids)
end = time.time()

print('time:', end - start)
display(link)
cut_wsi('08ab45297bfe652cc0397f4b37719ba1')
# These patches are excluded from the dataset
show_patch('08ab45297bfe652cc0397f4b37719ba1', (0, 1024))
# NOTE! For many patches it'll be extremely slow visualization
patches, h_cuts, v_cuts = test_cut_wsi('08ab45297bfe652cc0397f4b37719ba1')
_, axs = plt.subplots(nrows=v_cuts, ncols=h_cuts, figsize=(12, 12))
axs = axs.flatten()
for patch, ax in zip(patches, axs):
    ax.axis('on')
    ax.grid(False)
    ax.imshow(patch)
plt.show()
# No NaN values in train.csv file - Ok
print('NaN or empty values:', train.isna().any().any())
# Proof that only 8 characters of {image_id} is enough for ID
arr = np.array(train.index)  # get all indices
l = []
for a in arr:
    l.append(a[:8])  # truncate to 8 chars
_, counts = np.unique(l, return_counts=True)
print('All unique:', max(counts) == 1)  # first 7 or 8 chars are enough for ID
# Information about TIFF image
image = openslide.OpenSlide(os.path.join(data_dir, '005e66f06bce9c2e49142536caf2f6ee.tiff'))

p = image.properties
for x in p:
    print (x, '\t', p[x])

image.close()
import seaborn as sns
palette = ['#440154FF','#20A387FF']

sns.set(style='whitegrid', font_scale=1.5)
fig, ax = plt.subplots(1, 2, figsize=(15,5))
train['data_provider'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], colors=palette,
                                               pctdistance=1.1, labeldistance=1.2)
ax[0].set_ylabel('')
sns.countplot('data_provider', data=train, ax=ax[1], palette=palette)
fig.suptitle('Distribution of Images between the Data Providers', fontsize = 14)
plt.show()
import tqdm.notebook as tqdm

def get_size(image_id):
    """ Get image size """
    image = openslide.OpenSlide(os.path.join(data_dir, f'{image_id}.tiff'))
    size = image.dimensions  # get image size

    try:
        err = None
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{image_id}_mask.tiff'))
        
        size = image.dimensions  # get image size
        size_mask = mask.dimensions  # get mask size
        if size != size_mask:
            print(image_id, size, size_mask)
    
    except openslide.OpenSlideUnsupportedFormatError:
        # Unsupported or missing image file
        err = image_id

    resolution = 10000 / float(image.properties['tiff.XResolution'])
    image.close()  # close the opened slide after use
    del(image)  # maybe this should free some RAM
    return size, resolution, err

width = []
height = []
resolution = []
corrupted_mask = []
arr = np.array(train.index)  # get all indices

for a in tqdm.tqdm(arr):
    size, res, err = get_size(a)
    width.append(size[0])
    height.append(size[1])
    resolution.append(res)
    if err is not None:
        corrupted_mask.append(err)

train['width'] = width  # new column 'width'
train['height'] = height  # new column 'height'
train['resolution'] = resolution  # new column 'resolution'

print('Biggest image width:\t', max(width))
print('Smallest image width:\t', min(width))
print('Biggest image height:\t', max(height))
print('Smallest image height:\t', min(height))

# Clean data to save RAM
import gc
del(width); del(height); del(resolution); del(arr)
_ = gc.collect()
# Clean data to save RAM
import gc
_ = gc.collect()
# As I can see, we can not open these masks
print('Number of corrupted masks:', len(corrupted_mask))
pd.set_option('display.max_rows', None, 'display.max_columns', None)
train.loc[corrupted_mask]
rows, cols = train.shape
print('Rows before:\t', rows)
train.drop(corrupted_mask, errors='ignore', inplace=True)  # exclude indices from the table
rows, cols = train.shape
print('Rows after:\t', rows)
sns.set(style='whitegrid', font_scale=1.5)

fig = plt.figure(figsize=(18,8))
ax = sns.scatterplot(x='width', y='height', data=train, hue='data_provider',
                     palette=palette, alpha=0.5)
ax.tick_params(labelsize=14)
plt.title('Dimensions of Images')
plt.show()
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(15, 5)

sns.stripplot(train['width'],train['data_provider'],ax=ax[0], palette=palette,jitter=True)
sns.stripplot(train['height'],train['data_provider'],ax=ax[1], palette=palette,jitter=True)

ax[0].tick_params(labelsize=14)
ax[1].tick_params(labelsize=14)
ax[0].tick_params(labelrotation=90)
ax[1].tick_params(labelrotation=90)
plt.show()
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(18, 8)

sns.stripplot(train['resolution'],train['data_provider'],ax=ax[0], palette=palette,jitter=True)
sns.countplot('resolution', data=train, ax=ax[1], palette=palette)

ax[0].tick_params(labelsize=14)
ax[1].tick_params(labelsize=14)
ax[0].tick_params(labelrotation=90)
ax[1].tick_params(labelrotation=90)
plt.show()
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(15, 6)

sns.countplot(x='isup_grade', data=train, ax=ax[0], palette='viridis',
              order=train['isup_grade'].value_counts().index)
sns.countplot(x='gleason_score', data=train, ax=ax[1], palette='viridis',
              order=train['gleason_score'].value_counts().index)

ax[0].set_title('ISUP Grade (target variable)', y=1.0, fontsize=14)
ax[1].set_title('Gleason Score', y=1.0, fontsize=14)

for axis in ['top', 'bottom', 'left', 'right']:
    ax[0].spines[axis].set_linewidth(0.7)
    ax[1].spines[axis].set_linewidth(0.7)
    
ax[0].tick_params(labelsize=14)
ax[1].tick_params(labelsize=14)

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
plt.show()
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(15, 6)

sns.countplot(x='isup_grade', data=train, hue='resolution',
              ax=ax[0], palette='viridis',
              order=train['isup_grade'].value_counts().index)
sns.countplot(x='gleason_score', data=train, hue='resolution',
              ax=ax[1], palette='viridis',
              order=train['gleason_score'].value_counts().index)

ax[0].set_title('ISUP Grade (target variable)', y=1.0, fontsize=14)
ax[1].set_title('Gleason Score', y=1.0, fontsize=14)

for axis in ['top', 'bottom', 'left', 'right']:
    ax[0].spines[axis].set_linewidth(0.7)
    ax[1].spines[axis].set_linewidth(0.7)
    
ax[0].tick_params(labelsize=14)
ax[1].tick_params(labelsize=14)

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
plt.show()
train_grouped = train.groupby('isup_grade')['gleason_score'].unique().to_frame().reset_index()
display(train_grouped)
# Display wrong Gleason score 4+3
train[(train.isup_grade == 2) & (train.gleason_score == '4+3')].reset_index()
show_wsi('b0a92a74cb53899311acc30b7405e101', size=(300, 300))
# Can not load WSI - MemoryError: Integer overflow in ysize
# It seems too big
#load_wsi('b0a92a74cb53899311acc30b7405e101')
# Get downloadable link instead. You can open it with ASAP viewer.
link_wsi('b0a92a74cb53899311acc30b7405e101')
# Exclude index from the table
train.drop(['b0a92a74cb53899311acc30b7405e101'], errors='ignore', inplace=True)
train_grouped = train.groupby('isup_grade')['gleason_score'].unique().to_frame().reset_index()
display(train_grouped)
fig = plt.figure(figsize=(10, 5))
sns.set(style='whitegrid', font_scale=1.5)

sns.countplot(x='isup_grade', hue='data_provider', data=train, palette=palette)
plt.title('ISUP grade by Data Provider')
plt.tick_params(labelsize=14)

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(10, 5))
sns.set(style='whitegrid', font_scale=1.5)

sns.countplot(x='gleason_score', hue='data_provider', data = train, palette=palette)
plt.title('Gleason score by Data Provider')
plt.tick_params(labelsize=14)

plt.tight_layout()
plt.show()
# Change 'negative' to '0+0'
train.gleason_score.replace('negative', '0+0', inplace=True)

# Check it
train_grouped = train.groupby('isup_grade')['gleason_score'].unique().to_frame().reset_index()
display(train_grouped)
compression_opts = dict(method='zip', archive_name='panda_train.csv')
train.to_csv('panda_train.zip', compression=compression_opts)