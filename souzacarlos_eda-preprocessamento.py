import datetime, os

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.image as mpimg

import cv2

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

# Import skimage modules

from skimage import data, img_as_float

from skimage import exposure

from skimage.io import imread, imsave

from skimage import exposure, color

from skimage.transform import resize

from skimage.util.dtype import dtype_range

from skimage.util import img_as_ubyte

from skimage.morphology import disk

from skimage.filters import rank
df_train = pd.read_csv("../input/i2a2-brasil-pneumonia-classification/train.csv")

print('------------------------------')

print(df_train.info())

print('------------------------------')

print(df_train.head())

print('------------------------------')

print(df_train.shape)
images_dir = os.path.join('..', 'input', 'i2a2-brasil-pneumonia-classification', 'images')



df_train['path'] = df_train['fileName'].map(lambda x: os.path.join(images_dir))



df_train.head()
df_train['path'] = df_train['path'].str.cat(df_train[['fileName']], sep='/')



df_train['exists'] = df_train['path'].map(os.path.exists)



print('------------------------------')

print(df_train['exists'].sum(), 'images found of', df_train.shape[0], 'total')

print('------------------------------')

print(df_train.head())

print('------------------------------')

print(df_train.shape)

for row in df_train.iterrows():

    img_name = row[1][0]

    img = mpimg.imread("../input/i2a2-brasil-pneumonia-classification/images/"+img_name)

    df_train['height'] = img.shape[0]

    df_train['width'] = img.shape[1]

    

print('----------------------------')

#basic statistics

print('Statistics')

print(df_train.describe())

print('----------------------------')
df_train[['pneumonia']].hist(figsize = (10, 5));
plt.figure(figsize=(20, 10))



ax = sns.kdeplot(df_train['pneumonia'], label='Global Distribution')

ax.set_title('Pneumonia Distribution', color='b')

ax.set_xlabel('pneumonia', color='b')

ax.set_xlabel('frequency', color='b')

ax.tick_params(labelcolor='b')
for fileName, pneumonia in df_train[['fileName','pneumonia']].sample(5).values:

    img_name = str(fileName)

    img = mpimg.imread("../input/i2a2-brasil-pneumonia-classification/images/"+img_name)

    plt.imshow(img)

    plt.title('Image: {} Pneumonia: {}'.format(fileName, pneumonia))

    plt.show()
fig, m_axs = plt.subplots(pneumonia, 3, figsize = (18, 12))

for c_ax, (_, c_row) in zip(m_axs.flatten(), 

                            df_train.sort_values(['pneumonia']).iterrows()):

    c_ax.imshow(mpimg.imread(c_row['path']), cmap = 'hot')

    c_ax.axis('off')

    c_ax.set_title('Image: {} Pneumonia: {}'.format(fileName, pneumonia))
def plot_img_and_hist(image, axes, bins=256):

    """Plot an image along with its histogram and cumulative histogram.

    """

    ax_img, ax_hist = axes

    ax_cdf = ax_hist.twinx()



    # Display image

    ax_img.imshow(image, cmap=plt.cm.gray)

    ax_img.set_axis_off()



    # Display histogram

    ax_hist.hist(image.ravel(), bins=bins)

    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    ax_hist.set_xlabel('Pixel intensity')



    xmin, xmax = dtype_range[image.dtype.type]

    ax_hist.set_xlim(xmin, xmax)



    # Display cumulative distribution

    img_cdf, bins = exposure.cumulative_distribution(image, bins)

    ax_cdf.plot(bins, img_cdf, 'r')



    return ax_img, ax_hist, ax_cdf
# Load an example image

for c_ax, (_, c_row) in zip(m_axs.flatten(), 

                            df_train.sort_values(['pneumonia']).iterrows()):

    img = img_as_ubyte(imread(c_row['path'])/255)



# Global equalize

img_rescale = exposure.equalize_hist(img)



# Equalization

selem = disk(30)

img_eq = rank.equalize(img, selem=selem)





# Display results

fig = plt.figure(figsize=(18, 12))

axes = np.zeros((2, 3), dtype=np.object)

axes[0, 0] = plt.subplot(2, 3, 1)

axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])

axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])

axes[1, 0] = plt.subplot(2, 3, 4)

axes[1, 1] = plt.subplot(2, 3, 5)

axes[1, 2] = plt.subplot(2, 3, 6)



ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])

ax_img.set_title('Low contrast image')

ax_hist.set_ylabel('Number of pixels')



ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])

ax_img.set_title('Global equalise')



ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])

ax_img.set_title('Local equalize')

ax_cdf.set_ylabel('Fraction of total intensity')





# prevent overlap of y-axis labels

fig.tight_layout()

plt.show()



images=pd.DataFrame([])
# Gamma

gamma_corrected = exposure.adjust_gamma(img, 2)



# Logarithmic

logarithmic_corrected = exposure.adjust_log(img, 1)



# Display results

fig = plt.figure(figsize=(18, 12))

axes = np.zeros((2, 3), dtype=np.object)

axes[0, 0] = plt.subplot(2, 3, 1)

axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])

axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])

axes[1, 0] = plt.subplot(2, 3, 4)

axes[1, 1] = plt.subplot(2, 3, 5)

axes[1, 2] = plt.subplot(2, 3, 6)



ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])

ax_img.set_title('Low contrast image')



y_min, y_max = ax_hist.get_ylim()

ax_hist.set_ylabel('Number of pixels')

ax_hist.set_yticks(np.linspace(0, y_max, 5))



ax_img, ax_hist, ax_cdf = plot_img_and_hist(gamma_corrected, axes[:, 1])

ax_img.set_title('Gamma correction')



ax_img, ax_hist, ax_cdf = plot_img_and_hist(logarithmic_corrected, axes[:, 2])

ax_img.set_title('Logarithmic correction')



ax_cdf.set_ylabel('Fraction of total intensity')

ax_cdf.set_yticks(np.linspace(0, 1, 5))



# prevent overlap of y-axis labels

fig.tight_layout()

plt.show()



# Contrast stretching

p2, p98 = np.percentile(img, (2, 98))

img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))



# Equalization

img_eq = exposure.equalize_hist(img)



# Adaptive Equalization

img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)



# Display results

fig = plt.figure(figsize=(18, 12))

axes = np.zeros((2, 4), dtype=np.object)

axes[0, 0] = fig.add_subplot(2, 4, 1)

for i in range(1, 4):

    axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])

for i in range(0, 4):

    axes[1, i] = fig.add_subplot(2, 4, 5+i)



ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])

ax_img.set_title('Low contrast image')



y_min, y_max = ax_hist.get_ylim()

ax_hist.set_ylabel('Number of pixels')

ax_hist.set_yticks(np.linspace(0, y_max, 5))



ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])

ax_img.set_title('Contrast stretching')



ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])

ax_img.set_title('Histogram equalization')



ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])

ax_img.set_title('Adaptive equalization')



ax_cdf.set_ylabel('Fraction of total intensity')

ax_cdf.set_yticks(np.linspace(0, 1, 5))



# prevent overlap of y-axis labels

fig.tight_layout()

plt.show()
# global thresholding

ret1,th1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)



# Otsu's thresholding

ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)



# Otsu's thresholding after Gaussian filtering

blur = cv2.GaussianBlur(img,(3,3),0)

ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)



# Display results

fig = plt.figure(figsize=(18, 12))

axes = np.zeros((2, 3), dtype=np.object)

axes[0, 0] = plt.subplot(2, 3, 1)

axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])

axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])

axes[1, 0] = plt.subplot(2, 3, 4)

axes[1, 1] = plt.subplot(2, 3, 5)

axes[1, 2] = plt.subplot(2, 3, 6)



ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])

ax_img.set_title('Low contrast image')



# plot all the images and their histograms

images = [img, 0, th1,

          img, 0, th2,

          blur, 0, th3]

titles = ['Original Noisy Image','Histogram','Global Thresholding (v=150)',

          'Original Noisy Image','Histogram',"Otsu's Thresholding",

          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]



for i in range(3):

    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')

    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)

    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')

    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()
df_train = df_train.head(15)
# saving processed images

os.makedirs(f'/kaggle/output/kaggle/working/global_eq_images')
# Global Equalize

def global_equalize(filename):

    image = cv2.imread(f"/kaggle/input/i2a2-brasil-pneumonia-classification/images/{filename}", 0)

    norm_img = np.zeros((524,948))

    img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)

    img_rescale = exposure.equalize_hist(img)

    cv2.imwrite(f"/kaggle/output/kaggle/working/global_eq_images/{filename}", img_rescale)
%%time

df_train.apply(lambda row : global_equalize(row['fileName']),axis = 1)
images_dir_g = os.path.join('..', 'output', 'kaggle', 'working', 'global_eq_images')



df_train['path_g'] = df_train['fileName'].map(lambda x: os.path.join(images_dir_g))



df_train.head()
df_train['path_g'] = df_train['path_g'].str.cat(df_train[['fileName']], sep='/')



df_train['exists_g'] = df_train['path_g'].map(os.path.exists)



print(df_train['exists_g'].sum(), 'images found of', df_train.shape[0], 'total')



df_train.head()
df = df_train



for filename in df['fileName']:

  plt.title('Before')

  before = mpimg.imread(f"../input/i2a2-brasil-pneumonia-classification/images/{filename}")

  imgplot = plt.imshow(before)

  plt.show()



  plt.title('After')

  after = mpimg.imread(f"../output/kaggle/working/global_eq_images/{filename}")

  imgplot = plt.imshow(after)

  plt.show()    
# saving processed images

os.makedirs(f'/kaggle/output/kaggle/working/linear_eq_images')
# Linear Equalization

def linear_equalization(filename):

    image = cv2.imread(f"/kaggle/input/i2a2-brasil-pneumonia-classification/images/{filename}", 0)

    norm_img = np.zeros((524,948))

    img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)

    # Equalization

    selem = disk(30)

    img_eq = rank.equalize(img, selem=selem)

    cv2.imwrite(f"/kaggle/output/kaggle/working/linear_eq_images/{filename}", img_eq)
%%time

df_train.apply(lambda row : linear_equalization(row['fileName']),axis = 1)
images_dir_l = os.path.join('..', 'output', 'kaggle', 'working', 'linear_eq_images')



df_train['path_l'] = df_train['fileName'].map(lambda x: os.path.join(images_dir_l))



df_train.head()
df_train['path_l'] = df_train['path_l'].str.cat(df_train[['fileName']], sep='/')



df_train['exists_l'] = df_train['path_l'].map(os.path.exists)



print(df_train['exists_l'].sum(), 'images found of', df_train.shape[0], 'total')



df_train.head()
# df = df_train.head(15)



for filename in df['fileName']:

  plt.title('Before')

  before = mpimg.imread(f"../input/i2a2-brasil-pneumonia-classification/images/{filename}")

  imgplot = plt.imshow(before)

  plt.show()



  plt.title('After')

  after = mpimg.imread(f"../output/kaggle/working/linear_eq_images/{filename}")

  imgplot = plt.imshow(after)

  plt.show()    
# saving processed images

os.makedirs(f'/kaggle/output/kaggle/working/gamma_images')
# Gamma

def gamma_correction(filename):

    image = cv2.imread(f"/kaggle/input/i2a2-brasil-pneumonia-classification/images/{filename}", 0)

    norm_img = np.zeros((524,948))

    img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)

    gamma_corrected = exposure.adjust_gamma(img, 2)

    cv2.imwrite(f"/kaggle/output/kaggle/working/gamma_images/{filename}", gamma_corrected)
%%time

df_train.apply(lambda row : gamma_correction(row['fileName']),axis = 1)
images_dir_gm = os.path.join('..', 'output', 'kaggle', 'working', 'gamma_images')



df_train['path_gm'] = df_train['fileName'].map(lambda x: os.path.join(images_dir_gm))



df_train.head()
df_train['path_gm'] = df_train['path_gm'].str.cat(df_train[['fileName']], sep='/')



df_train['exists_gm'] = df_train['path_gm'].map(os.path.exists)



print(df_train['exists_gm'].sum(), 'images found of', df_train.shape[0], 'total')



df_train.head()
# df = df_train.head(15)



for filename in df['fileName']:

  plt.title('Before')

  before = mpimg.imread(f"../input/i2a2-brasil-pneumonia-classification/images/{filename}")

  imgplot = plt.imshow(before)

  plt.show()



  plt.title('After')

  after = mpimg.imread(f"../output/kaggle/working/gamma_images/{filename}")

  imgplot = plt.imshow(after)

  plt.show()    
# saving processed images

os.makedirs(f'/kaggle/output/kaggle/working/log_corr_images')
# Logarithmic

def log_correction(filename):

    image = cv2.imread(f"/kaggle/input/i2a2-brasil-pneumonia-classification/images/{filename}", 0)

    norm_img = np.zeros((524,948))

    img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)

    logarithmic_corrected = exposure.adjust_log(img, 1)

    cv2.imwrite(f"/kaggle/output/kaggle/working/log_corr_images/{filename}", logarithmic_corrected)
%%time

df_train.apply(lambda row : log_correction(row['fileName']),axis = 1)
images_dir_lc = os.path.join('..', 'output', 'kaggle', 'working', 'log_corr_images')



df_train['path_lc'] = df_train['fileName'].map(lambda x: os.path.join(images_dir_lc))



df_train.head()
df_train['path_lc'] = df_train['path_lc'].str.cat(df_train[['fileName']], sep='/')



df_train['exists_lc'] = df_train['path_lc'].map(os.path.exists)



print(df_train['exists_lc'].sum(), 'images found of', df_train.shape[0], 'total')



df_train.head()
# df = df_train.head(15)



for filename in df['fileName']:

  plt.title('Before')

  before = mpimg.imread(f"../input/i2a2-brasil-pneumonia-classification/images/{filename}")

  imgplot = plt.imshow(before)

  plt.show()



  plt.title('After')

  after = mpimg.imread(f"../output/kaggle/working/log_corr_images/{filename}")

  imgplot = plt.imshow(after)

  plt.show()   
# saving processed images

os.makedirs(f'/kaggle/output/kaggle/working/contrast_images')
# Contrast stretching

def contrast_stretching(filename):

    image = cv2.imread(f"/kaggle/input/i2a2-brasil-pneumonia-classification/images/{filename}", 0)

    norm_img = np.zeros((524,948))

    img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)

    p2, p98 = np.percentile(img, (2, 98))

    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    cv2.imwrite(f"/kaggle/output/kaggle/working/contrast_images/{filename}", img_rescale)
%%time

df_train.apply(lambda row : contrast_stretching(row['fileName']),axis = 1)
images_dir_cs = os.path.join('..', 'output', 'kaggle', 'working', 'contrast_images')



df_train['path_cs'] = df_train['fileName'].map(lambda x: os.path.join(images_dir_cs))



df_train.head()
df_train['path_cs'] = df_train['path_cs'].str.cat(df_train[['fileName']], sep='/')



df_train['exists_cs'] = df_train['path_cs'].map(os.path.exists)



print(df_train['exists_cs'].sum(), 'images found of', df_train.shape[0], 'total')



df_train.head()
# df = df_train.head(15)



for filename in df['fileName']:

  plt.title('Before')

  before = mpimg.imread(f"../input/i2a2-brasil-pneumonia-classification/images/{filename}")

  imgplot = plt.imshow(before)

  plt.show()



  plt.title('After')

  after = mpimg.imread(f"../output/kaggle/working/contrast_images/{filename}")

  imgplot = plt.imshow(after)

  plt.show()    
# saving processed images

os.makedirs(f'/kaggle/output/kaggle/working/equalization_images')
# Equalization

def hist_equalization(filename):

    image = cv2.imread(f"/kaggle/input/i2a2-brasil-pneumonia-classification/images/{filename}", 0)

    norm_img = np.zeros((524,948))

    img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)

    img_eq = exposure.equalize_hist(img)

    cv2.imwrite(f"/kaggle/output/kaggle/working/equalization_images/{filename}", img_eq)
%%time

df_train.apply(lambda row : hist_equalization(row['fileName']),axis = 1)
images_dir_ha = os.path.join('..', 'output', 'kaggle', 'working', 'equalization_images')



df_train['path_ha'] = df_train['fileName'].map(lambda x: os.path.join(images_dir_ha))



df_train.head()
df_train['path_ha'] = df_train['path_ha'].str.cat(df_train[['fileName']], sep='/')



df_train['exists_ha'] = df_train['path_ha'].map(os.path.exists)



print(df_train['exists_ha'].sum(), 'images found of', df_train.shape[0], 'total')



df_train.head()
# df = df_train.head(15)



for filename in df['fileName']:

  plt.title('Before')

  before = mpimg.imread(f"../input/i2a2-brasil-pneumonia-classification/images/{filename}")

  imgplot = plt.imshow(before)

  plt.show()



  plt.title('After')

  after = mpimg.imread(f"../output/kaggle/working/equalization_images/{filename}")

  imgplot = plt.imshow(after)

  plt.show()  
# saving processed images

os.makedirs(f'/kaggle/output/kaggle/working/adp_eq_images')
# Adaptive Equalization

def adapt_equalization(filename):

    image = cv2.imread(f"/kaggle/input/i2a2-brasil-pneumonia-classification/images/{filename}", 0)

    norm_img = np.zeros((524,948))

    img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)

    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    cv2.imwrite(f"/kaggle/output/kaggle/working/adp_eq_images/{filename}", img_adapteq)
%%time

df_train.apply(lambda row : adapt_equalization(row['fileName']),axis = 1)
images_dir_a = os.path.join('..', 'output', 'kaggle', 'working', 'adp_eq_images')



df_train['path_a'] = df_train['fileName'].map(lambda x: os.path.join(images_dir_a))



df_train.head()
df_train['path_a'] = df_train['path_a'].str.cat(df_train[['fileName']], sep='/')



df_train['exists_a'] = df_train['path_a'].map(os.path.exists)



print(df_train['exists_a'].sum(), 'images found of', df_train.shape[0], 'total')



df_train.head()
# df = df_train.head(15)



for filename in df['fileName']:

  plt.title('Before')

  before = mpimg.imread(f"../input/i2a2-brasil-pneumonia-classification/images/{filename}")

  imgplot = plt.imshow(before)

  plt.show()



  plt.title('After')

  after = mpimg.imread(f"../output/kaggle/working/adp_eq_images/{filename}")

  imgplot = plt.imshow(after)

  plt.show()    
# saving processed images

os.makedirs(f'/kaggle/output/kaggle/working/gthresh_images')
# global thresholding

def global_thresholding(filename):

    image = cv2.imread(f"/kaggle/input/i2a2-brasil-pneumonia-classification/images/{filename}", 0)

    norm_img = np.zeros((524,948))

    img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)

    ret1,th1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)

    cv2.imwrite(f"/kaggle/output/kaggle/working/gthresh_images/{filename}", th1)
%%time

df_train.apply(lambda row : global_thresholding(row['fileName']),axis = 1)
images_dir_gt = os.path.join('..', 'output', 'kaggle', 'working', 'gthresh_images')



df_train['path_gt'] = df_train['fileName'].map(lambda x: os.path.join(images_dir_gt))



df_train.head()
df_train['path_gt'] = df_train['path_gt'].str.cat(df_train[['fileName']], sep='/')



df_train['exists_gt'] = df_train['path_gt'].map(os.path.exists)



print(df_train['exists_gt'].sum(), 'images found of', df_train.shape[0], 'total')



df_train.head()
# df = df_train.head(15)



for filename in df['fileName']:

  plt.title('Before')

  before = mpimg.imread(f"../input/i2a2-brasil-pneumonia-classification/images/{filename}")

  imgplot = plt.imshow(before)

  plt.show()



  plt.title('After')

  after = mpimg.imread(f"../output/kaggle/working/gthresh_images/{filename}")

  imgplot = plt.imshow(after)

  plt.show()  
# saving processed images

os.makedirs(f'/kaggle/output/kaggle/working/othresh_images')
# Otsu's thresholding

def otsus_thresholding(filename):

    image = cv2.imread(f"/kaggle/input/i2a2-brasil-pneumonia-classification/images/{filename}", 0)

    norm_img = np.zeros((524,948))

    img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)

    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imwrite(f"/kaggle/output/kaggle/working/othresh_images/{filename}", th2)
%%time

df_train.apply(lambda row : otsus_thresholding(row['fileName']),axis = 1)
images_dir_ot = os.path.join('..', 'output', 'kaggle', 'working', 'othresh_images')



df_train['path_ot'] = df_train['fileName'].map(lambda x: os.path.join(images_dir_ot))



df_train.head()
df_train['path_ot'] = df_train['path_ot'].str.cat(df_train[['fileName']], sep='/')



df_train['exists_ot'] = df_train['path_ot'].map(os.path.exists)



print(df_train['exists_ot'].sum(), 'images found of', df_train.shape[0], 'total')



df_train.head()
# df = df_train.head(15)



for filename in df['fileName']:

  plt.title('Before')

  before = mpimg.imread(f"../input/i2a2-brasil-pneumonia-classification/images/{filename}")

  imgplot = plt.imshow(before)

  plt.show()



  plt.title('After')

  after = mpimg.imread(f"../output/kaggle/working/othresh_images/{filename}")

  imgplot = plt.imshow(after)

  plt.show()  
# saving processed images

os.makedirs(f'/kaggle/output/kaggle/working/ogthresh_images')
# Otsu's thresholding after Gaussian filtering

def otsus_gaussian_thresholding(filename):

    image = cv2.imread(f"/kaggle/input/i2a2-brasil-pneumonia-classification/images/{filename}", 0)

    norm_img = np.zeros((524,948))

    img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)

    blur = cv2.GaussianBlur(img,(3,3),0)

    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imwrite(f"/kaggle/output/kaggle/working/ogthresh_images/{filename}", th3)
%%time

df_train.apply(lambda row : otsus_gaussian_thresholding(row['fileName']),axis = 1)
images_dir_ogt = os.path.join('..', 'output', 'kaggle', 'working', 'ogthresh_images')



df_train['path_ogt'] = df_train['fileName'].map(lambda x: os.path.join(images_dir_ogt))



df_train.head()
df_train['path_ogt'] = df_train['path_ogt'].str.cat(df_train[['fileName']], sep='/')



df_train['exists_ogt'] = df_train['path_ogt'].map(os.path.exists)



print(df_train['exists_ogt'].sum(), 'images found of', df_train.shape[0], 'total')



df_train.head()
# df = df_train.head(15)



for filename in df['fileName']:

  plt.title('Before')

  before = mpimg.imread(f"../input/i2a2-brasil-pneumonia-classification/images/{filename}")

  imgplot = plt.imshow(before)

  plt.show()



  plt.title('After')

  after = mpimg.imread(f"../output/kaggle/working/ogthresh_images/{filename}")

  imgplot = plt.imshow(after)

  plt.show()  