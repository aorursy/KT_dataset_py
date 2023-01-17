from pylab import rcParams

rcParams['figure.figsize'] = 10, 20
import imageio

import matplotlib.pyplot as plt

im = imageio.imread("../input/tcia-chest-ct-sample/chest-220.dcm")
print("Type of image {}".format(type(im)))

print("shape of image {}".format(type(im.shape)))
plt.imshow(im)

plt.axis("off")
print(im.meta.keys())

print("----------------------------------")

print(im.meta)
print("Type of the image {}".format(type(im)))

print("Shape of the image {}".format(im.shape))

plt.imshow(im,cmap='gray')

plt.axis("off")
plt.imshow(im,cmap = "gray",vmin=-200,vmax=200)

plt.show()
### create a 3d image 

import imageio

import numpy as np



# Read in each 2D image

im1 = imageio.imread('../input/tcia-chest-ct-sample/chest-220.dcm')

im2 = imageio.imread('../input/tcia-chest-ct-sample/chest-221.dcm')

im3 = imageio.imread('../input/tcia-chest-ct-sample/chest-222.dcm')



# Stack images into a volume

vol = np.stack([im1,im2,im3])

print('Volume dimensions:', vol.shape)
## we can directly load a image folder

import imageio

vol = imageio.volread("../input/tcia-chest-ct-sample/")

print("Avilable metadata: {} ".format(vol.meta.keys()))

print("Shape of the image Array: {} ".format(vol.shape))
print(vol.shape)
#image shape: number of element on each axis

n0,n1,n2 = vol.shape 

#sampling rate : physical space covered by each sample

d0,d1,d2 = vol.meta['sampling'] 

# field of view : physical space covered along each axis

print("Physical space covered by each axis")

print(n0*d0,n1*d1,n2*d2)
## pltting image with subplot

import imageio

col = imageio.volread("../input/tcia-chest-ct-sample/")

fig,axis = plt.subplots(nrows=1,ncols=3)

axis[0].imshow(vol[0],cmap="gray")

axis[1].imshow(vol[1],cmap="gray")

axis[2].imshow(vol[2],cmap="gray")
### plotting in row

import matplotlib.pyplot as plt

fig,axis = plt.subplots(nrows=2,ncols=1)

im1 = imageio.imread('../input/tcia-chest-ct-sample/chest-220.dcm')

im2 = imageio.imread('../input/tcia-chest-ct-sample/chest-221.dcm')

axis[0].imshow(im1,cmap="gray")

axis[1].imshow(im2,cmap="gray")

plt.show()
## how tosee the image from different dimension

## first lock one dimension and then 

## take the rest of the data and plot it

fig,axis = plt.subplots(nrows=1,ncols=4)

for dimension in range(4):

    im = vol[dimension,:,:]

    axis[dimension].imshow(im,cmap="gray")

    axis[dimension].axis("off")

plt.show()
## calculating aspect ratio

d0,d1,d2 = vol.meta['sampling']

asp1 = d0/d1

asp2 = d0/d2



## plot the image with the aspect ratio

fig,axis = plt.subplots(nrows=1,ncols=2)

axis[0].imshow(im1,cmap="gray",aspect = asp1)

axis[1].imshow(im2,cmap="gray",aspect = asp1)
import imageio

im = imageio.imread("../input/hand.png")
print(im.dtype)

print(im.size)
## chaning dtpye and size

im_int64 = im.astype(np.uint64)

print(im_int64.size)
plt.imshow(im)
## make a histogram to plot the pixel and the intensity of the value throughout image

import scipy.ndimage as ndi

hist = ndi.histogram(im,min=0,max=255,bins=256)

plt.plot(hist)
plt.hist(hist)
## thats show that this is a skewed image

## because most of the background is blak and some of the portion (the hand portion)

## is is image take hisger pixel thats why it is a skewed image

## we can redistribute this with the 

## Equalization

## distribution often skewed to the background image in medical image

## but we can redistribute this  intensity to optimize the full intensity length

## you can redistribute this thing with CDF (cumulative distributuion function)

## VDF can show that then range that your image is build in the total pixel range

## so lets redistribute this image
import scipy.ndimage as ndi

hist = ndi.histogram(im,min=0,max=255,bins=256)



## calculate the cdf

## find the cumulative sum and then devie witht eh histogram sum

## to make a cumulative histogram

cdf = hist.cumsum()/hist.sum()

cdf.shape
im_equilized = cdf[im]*255
## lets plot the both image

fig,axes = plt.subplots(nrows=2,ncols=1)

axes[0].imshow(im)

axes[1].imshow(im_equilized)
## this is good but you can see that it does the redistribute this image

## but it also brighten up the background of the image
# Load the hand radiograph

im = imageio.imread("../input/hand.png")

print('Data type:', im.dtype)

print('Min. value:', im.min())

print('Max value:', im.max())



# Plot the grayscale image

plt.imshow(im,cmap="gray",vmin=0,vmax=255)

plt.colorbar()
import scipy.ndimage as ndi

hist = ndi.histogram(im,min=0,max=255,bins=256)



## calculate the cdf

## find the cumulative sum and then devie witht eh histogram sum

## to make a cumulative histogram

cdf = hist.cumsum()/hist.sum()

fig, axes = plt.subplots(2, 1, sharex=True)

axes[0].plot(hist, label='Histogram')

axes[1].plot(cdf, label='CDF')
## masking image

## if you see the distribution you can see that

## only a part of the pixel have value high

## that means this image is themain feature of theimage 

## so lets extract the image
plt.plot(hist)
## so we can see that after 35 (apporx ) the curve went down

## lets filter it down

mask1 = im > 35
plt.imshow(mask1)
## so you can see we can extract theonlt the hand part

## but the main challange is can we show the bones too ??????

## because bones are the hoghest intensity tissue in an xray
## lets make a narder one 

mask2 = im>70
plt.imshow(mask2)
## can you plot both the bone and the tissue
# masking that are in the mask1 but ont in mask2

mask3 = mask1 & ~mask2
plt.imshow(mask3)
## now you can detect the skin and the bone too
## same work you can do it with numpy too

import numpy as np

im_bone_np = np.where(im>70,im,0)
plt.imshow(im_bone_np,cmap="gray")

plt.show()
im_bone_np = ndi.binary_dilation(im_bone_np,iterations=5)

plt.imshow(im_bone_np,cmap="gray")

plt.show()
## now you can do the opposite thing that ts pixel erosion

im_bone_np_down = ndi.binary_erosion(im_bone_np,iterations=5)

plt.imshow(im_bone_np_down,cmap="gray")

plt.show()
im = imageio.imread("../input/hand.png")

# Create skin and bone masks

mask_bone = im>=45

mask_skin = (im >=45) & (im<=145)



# Plot the skin (0) and bone (1) masks

fig, axes = plt.subplots(1,2)

axes[0].imshow(mask_bone,cmap="gray")

axes[1].imshow(mask_skin,cmap="gray")
# Import SciPy's "ndimage" module

import scipy.ndimage as ndi

#im_bone_np = np.where(im>70,im,0)

# Screen out non-bone pixels from "im"

mask_bone = im>=145

im_bone = np.where(im>-145, im, 0)



# Get the histogram of bone intensities

hist = ndi.histogram(im_bone,min=1,max=255,bins=255)



# Plot masked image and histogram

fig, axes = plt.subplots(2,1)

axes[0].imshow(im_bone)

axes[1].plot(hist)
# Create and tune bone mask

# mask_bone = im >= 145

# mask_dilate = ndi.binary_dilation(mask_bone, iterations=1)

# mask_closed = ndi.binary_closing(mask_bone, iterations=1)



# Plot masked images

# fig, axes = plt.subplots(1,3)

# axes[0].imshow(mask_bone)

# axes[1].imshow(mask_dilate)

# axes[2].imshow(mask_closed)
import imageio

import scipy.ndimage as ndi

im = imageio.imread("../input/hand.png")

## adding weight

##  we emphasize the middle one

weight = [[.11,.11,.11],

          [.11,.12,.11],

          [.11,.11,.11]]

im_fit = ndi.convolve(im,weight)
## plot the original and the convolution filter

fig,axes = plt.subplots(2,1)

axes[0].imshow(im,cmap="gray")

axes[1].imshow(im_fit,cmap="gray")
### apply median filter

# apply median filter to the  image

## this is the uniform filter

# Set filter weights

weights = [[0.11, 0.11,.11],

           [0.11, .11, 0.11], 

            [0.11, .11, 0.11]]

im_filt = ndi.convolve(im,weights)

plt.imshow(im_filt)
## smooting image with gaussian filter

## it will smooth the image with reducig the surge of pixel value

# Smooth "im" with Gaussian filters

im_s1 = ndi.gaussian_filter(im, sigma=1)

im_s3 = ndi.gaussian_filter(im, sigma=3)
fig, axes = plt.subplots(1,3)

axes[0].imshow( im)

axes[1].imshow(im_s1)

axes[2].imshow(im_s3)
## weight increase at the top and bottom

im = imageio.imread('../input/hand.png')

weights = [[+1,+1,+1],

          [0 ,0 ,0 ],

          [-1,-1,-1]]

edges = ndi.convolve(im,weights)

plt.imshow(edges)
## applying default sovel filter to findn the edge

im = imageio.imread('../input/hand.png')

edges = ndi.sobel(im,axis=1)

plt.imshow(edges,cmap="seismic")
# Set weights to detect vertical edges

weights = [[1,0,-1],

          [0 ,0 ,0 ],

          [-1,-1,-1]]



# Convolve "im" with filter weights

edges = ndi.convolve(im,weights)

plt.imshow(edges,cmap="seismic",vmin=-150,vmax=150)

plt.colorbar()
## image of the heart pumping blod we need to extract the left ventricle image

## the one in the circled

im = imageio.imread("../input/sunnybrook-cardiac-mr/SCD2001_006/SCD2001_MR_117.dcm")

plt.imshow(im)
## apply gausian filter to smooth the image

import scipy.ndimage as ndi

filt = ndi.gaussian_filter(im,sigma=2)

## not apply the filter

mask = filt >100

## lets see where we going

plt.imshow(mask,cmap="rainbow")
##extract the label and the number of  the label

labels,nlabels = ndi.label(mask)
print(nlabels)
## you can now select individual objects

plt.imshow(np.where(labels==3,im,0))
## ok we got the circullar image

## make  abounding box

## extract objects from the larger image

boxes = ndi.find_objects(labels)
###plotting the box

plt.imshow(im[boxes[2]])

# Label the image "mask"

labels, nlabels = ndi.label(mask)



# Select left ventricle pixels

lv_val = labels[128, 128]

lv_mask = np.where(labels==lv_val,1,np.nan)



# Overlay selected label

plt.imshow(lv_mask, cmap='rainbow')

plt.show()
# Create left ventricle mask

labels, nlabels = ndi.label(mask)

lv_val = labels[128,128]



lv_mask = np.where(labels==lv_val, 1, 0)
plt.imshow(lv_mask)
## getting a mean intensity of the entire vol 3d image

import imageio

import scipy.ndimage as ndi

vol = imageio.volread('../input/sunnybrook-cardiac-mr/SCD2001_006/')
## find the mean of all the pixel

print(ndi.mean(vol))
## if you provde a mask this mean will be avilabe for ony the non zero pixel

print(ndi.mean(vol,labels))
## you can find for a specfic one

print(ndi.mean(vol,labels,index=1))

print(ndi.mean(vol,labels,index=[1,2]))
## you can add this directly to a histogram plot

hist = ndi.histogram(vol,min=0,max=255,bins=256,labels=labels,index=[1,2])
len(hist)
plt.hist(hist)


## finding varience of all the picture 

## vs some labels picture

# Variance for all pixels

var_all = ndi.variance(vol, labels=None, index=None)

print('All pixels:', var_all)



# Variance for labeled pixels

var_labels = ndi.variance(vol, labels, index=None)

print('Labeled pixels:', var_labels)



# Variance for each object

var_objects = ndi.variance(vol, labels, index=[1,2])

print('Left ventricle:', var_objects[0])
# Create histograms for selected pixels

hist1 = ndi.histogram(vol, min=0, max=255, bins=256)

hist2 = ndi.histogram(vol, 0, 255, 256, labels=labels)

hist3 = ndi.histogram(vol, 0, 255, 256, labels=labels, index=1)
fig,axis = plt.subplots(nrows=1,ncols=3)

axis[0].hist(hist1)

axis[1].hist(hist2)

axis[2].hist(hist3)
## calculate the volume by multipleplying each voxel

## you can find it in the samppling in dicom images

d0,d1,d2 = vol.meta['sampling']

dvoxels = d0*d1*d2
## we want to count the number of vosxels in left ventricle

## this one is tricky

## first you assign the numebr 1 in the label and index 1 and then sum it that

## make the portion of the left ventricle

## then you mulltiply with the dvoxels thats how you find the volume of the portion

## volume of the portion = portion*total_volume

## nvoxels = ndi.sum(1,label,index=1)

## volume = nvoxels*dvoxels
#1)

labels,nlabels = ndi.label(mask)

labels_of_the_left_vn=np.where(labels==3,im,0)

plt.imshow(labels_of_the_left_vn)
# 2)

## calculate the volum in mm^3 in a  specfic time point

d0,d1,d2 = vol.meta['sampling']

dvoxels = d0*d1*d2

## instantiate an empty list

ts = np.zeros(20)



## loop through the volume time series

for t in range(20):

    nvoxels = ndi.sum(1,labels_of_the_left_vn[t],index=3)

    ts[t] = nvoxels*dvoxels

#3)

tmax = np.argmax(ts)

tmin = np.argmin(ts)

#4)# Calculate ejection fraction

ej_vol = ts.max() - ts.min()

ej_frac = ej_vol / ts.max()

print('Est. ejection volume (mm^3):', ej_vol)

print('Est. ejection fraction:', ej_frac)
import imageio

import scipy.ndimage as ndi

im = imageio.imread('../input/hand.png')

plt.imshow(im)
## find the center of mass to rotate

com = ndi.center_of_mass(im)
print(com)
## we set the target point

target = 128

## thenfor row and column we calculate the difference

d0 = target - com[0]

d2 = target - com[1]
## shift the image 

xfm = ndi.shift(im,shift=[d0,d1])
plt.imshow(xfm)
## rotate the imgae

plt.imshow(ndi.rotate(im,angle=25,axes=(0,1)))
## rotate the imgae

plt.imshow(ndi.rotate(im,angle=25,axes=(0,1),reshape=False))
## resclaling image

mat = [[.8,0,-20],[0,.8,-10],[0,0,1]]

xfm = ndi.affine_transform(im,mat)

plt.imshow(xfm)
## resampling (slicing the imge ,changes the shape of the array )

## downsampling

## upsampling
down_im = ndi.zoom(im,zoom=.5)
plt.imshow(down_im)
## the image is downsamopled and the quality is reduced 

## you can see the image scattered a little bit

print("shpe of image before downsampling {}".format(im.shape))

print("shpe of image after downsampling {}".format(down_im.shape))
## upsampling is not as same as downsampling

## you are not adding any feature 

## it just resample toa  larger grid

## some times it requires estimating some portionof the image that are not actually there

## like we do in image editing

## its called interpolation
# Center and level image

xfm = ndi.shift(im, shift=(-20, -20))

xfm = ndi.rotate(xfm, angle=-35, reshape=False)



# Resample image

im_dn = ndi.zoom(xfm, zoom=0.25)

im_up = ndi.zoom(xfm, zoom=4.00)



# Plot the images

fig, axes = plt.subplots(2, 1)

axes[0].imshow(im_dn)

axes[1].imshow(im_up)

# Upsample "im" by a factor of 4

up0 = ndi.zoom(im, zoom=4, order=0)

up5 = ndi.zoom(im, zoom=4, order=5)



# Print original and new shape

print('Original shape:', im.shape)

print('Upsampled shape:', up0.shape)



# Plot close-ups of the new images

fig, axes = plt.subplots(1, 2)

axes[0].imshow(up0[128:256, 128:256])

axes[1].imshow(up5[128:256, 128:256])
## we import two image

im1 = imageio.imread('../input/tcia-chest-ct-sample/chest-220.dcm')

im3 = imageio.imread('../input/tcia-chest-ct-sample/chest-222.dcm')
##find the difference

err = im1-im3
fig,axis = plt.subplots(nrows=1,ncols=3)

axis[0].imshow(err,vmin=-200,vmax=200)

axis[1].imshow(im1)

axis[2].imshow(im2)
## the difference are highlighted
## if you want absolute image difference then add np.abs

# Calculate absolute image difference

abs_err = np.abs(im1 - im3)



# Plot the difference

plt.imshow(abs_err, cmap='seismic', vmin=-200, vmax=200)
## if you want to find MAE (mean absolute error)

# Calculate mean absolute error

mean_abs_err = np.mean(np.abs(im1 - im3))

print('MAE:', mean_abs_err)
import pandas as pd

df = pd.read_csv('../input/oasis_all_volumes.csv')
df.head()
# null hypothisis : men and women brain volume are equal

# we apply a t-test
male_brain_vol=df[df.sex=="M"][['brain_vol']]

female_brain_vol=df[df.sex=="F"][['brain_vol']]
from scipy.stats import ttest_ind

result = ttest_ind(male_brain_vol,female_brain_vol)
print(result)
## the large statistics and low p value means there are significant difference between gender brain size
## but the brain and the skull size are related and the skull size is related with the body

df[['brain_vol','skull_vol']].corr()
df[['brain_vol','skull_vol']].corr()
## so lets normalize the brain size with skull size thus we are comparing with the body size because

## skull size is related with the body sizee
df['brain_norm'] = df.brain_vol/df.skull_vol

male_brain_vol=df[df.sex=="M"][['brain_norm']]

female_brain_vol=df[df.sex=="F"][['brain_norm']]
from scipy.stats import ttest_ind

result = ttest_ind(male_brain_vol,female_brain_vol)
print(result)
## now its telling different it was never dependent on the male or female

## its dependedn on the body size that means its related with other value too
## test the alzeimar with the brain vol

# Import independent two-sample t-test

from scipy.stats import ttest_ind



# Select data from "alzheimers" and "typical" groups

brain_alz = df.loc[df.alzheimers == True, 'brain_vol']

brain_typ = df.loc[df.alzheimers == False, 'brain_vol']



# Perform t-test of "alz" > "typ"

results = ttest_ind(brain_alz, brain_typ)

print('t = ', results.statistic)

print('p = ', results.pvalue)
# Import independent two-sample t-test

from scipy.stats import ttest_ind



# Select data from "alzheimers" and "typical" groups

brain_alz = df.loc[df.alzheimers == True, 'brain_vol']

brain_typ = df.loc[df.alzheimers == False, 'brain_vol']



# Perform t-test of "alz" > "typ"

results = ttest_ind(brain_alz, brain_typ)

print('t = ', results.statistic)

print('p = ', results.pvalue)



# Show boxplot of brain_vol differences

df.boxplot(column='brain_vol', by='alzheimers')

plt.show()