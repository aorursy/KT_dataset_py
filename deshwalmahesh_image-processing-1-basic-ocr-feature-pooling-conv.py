# ! pip install opencv-python # ! sign tells the system that it is a shell command but not python one
# ! pip install tesseract # OCR backed up by Goggle Engine
# ! pip install pytesseract # python wrapper for implementation of tesseract
import skimage # sibling of sklearn but for Image PRocessing
import numpy as np # we all know
import pandas as pd # evolved numpy
import matplotlib.pyplot as plt # This one is a true creator
import warnings # No one liked warnings
from keras.datasets import cifar10 # image data
import random # to perform random tasks
import cv2
# import albumentations as alb # it is a fun and knowledge library
from PIL import Image, ImageFilter # Image is you know, image and filters
import requests # request something fom the internet, it ALWAYS returns something 
from io import BytesIO # Read some file as byte file
from skimage import feature #problem with sklearn imports so if error comes, export modules explicitly
import pytesseract
from IPython import display # interesting function to show images without matplotlib
from scipy import ndimage # it contains Gaussian Filter for de noising
from skimage.morphology import reconstruction # We'll remove what's there and will bring what's not
from skimage import exposure # everyone loves the exposure
from skimage import data, img_as_float # we all know 
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim # check how similar two images are, without any ML 
PATH = '/kaggle/input/image-processing/'
# warnings.filterwarnings('ignore') # ignore all warning because "I don't care what they say, I gotta learn"
# plt.style.use('seaborn') # you can copy the style but the working of seaborn is what makes it unique ;)

def get_image_from_url(source):
    '''
    return a numpy array from the Image URL
    '''
    response = requests.get(source)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

def mse(x, y):
    '''
    Find Mean Squared Errors between X and Y Distributions
    '''
    return np.linalg.norm(x - y)
chelsea = skimage.data.chelsea() # data is given here already
print(chelsea.shape)
plt.imshow(chelsea)
plt.show()
plt.imshow(np.fliplr(chelsea))
plt.imshow(skimage.transform.rotate(chelsea,90))
plt.imshow(np.rot90(chelsea))
blue_chelsea = chelsea.copy()
mask = (blue_chelsea[:,:,0]>103) & (blue_chelsea[:,:,0]<153)  # get random pixels based on a condition
blue_chelsea[mask] = [0,0,255] # turn the mask as blue
plt.imshow(blue_chelsea)
plt.imshow(skimage.color.rgb2gray(chelsea),cmap='gray') # RGB to grayscale
plt.imshow(skimage.util.invert(chelsea)) # inverted colors
BGR_chelsea = chelsea[:,:,::-1]
plt.imshow(BGR_chelsea) # RGB to BGR
def crop_center(img,y,x):
    h,w,c = img.shape
    start_w = w//2 - (x//2)
    start_h = h//2 - (y//2)
    return img[start_h:start_h+y,start_w:start_w+w]
plt.imshow(crop_center(chelsea,300,200))
image = skimage.io.imread(PATH+'histogram.png')
gray_img = skimage.color.rgb2gray(image)

f,ax = plt.subplots(2,2,figsize=(17,11))
ax = ax.ravel() # raven the axes so that we can use directly ax[i] inside in dynamic subplot generation

ax[0].imshow(gray_img,cmap='gray')
ax[0].set_title('Grayscale Image')

ax[1].hist(gray_img.ravel(),bins=256)
ax[1].set_xlabel('Intensity Values')
ax[1].set_ylabel('Count')
ax[1].set_title('Hitogram of Grayscale Image')

ax[2].imshow(image)
ax[2].set_title('Original Image')

ax[3].hist(image.ravel(), bins = 256, color = 'teal',alpha=0.5 )
ax[3].hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.7)
ax[3].hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.7)
ax[3].hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.7)
ax[3].set_xlabel('Intensity Values')
ax[3].set_ylabel('Count')
ax[3].legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
ax[3].set_title('Histogram of RGB Image')

plt.show()
# we will change them to np.array while convolving

gaussian_blur = [[1/16,1/8,1/16],
                [1/8,1/4,1/8],
                [1/16,1/8,1/16]]

horizontal_lines = [[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]]

vertical_lines = [[-1,-2,-1],
                  [0,0,0],
                  [-1,-2,-1]]
laplacian = [[-1,-1,-1],
            [-1,8,-1],
            [-1,-1,-1]]

kernels = [gaussian_blur,horizontal_lines,vertical_lines,laplacian]
kernel_names = ['Gaussian','Horizontal', 'Vertical', 'Laplacian']

image = skimage.io.imread(PATH+'triplet.jpeg')
gray = skimage.color.rgb2gray(image)

f,ax = plt.subplots(2,3,figsize=(15,9))
ax = ax.ravel()
ax[0].imshow(image)
ax[0].set_title('Original')

ax[1].imshow(gray,cmap='gray')
ax[1].set_title('Grayscale')

i = 2
for j,kernel in enumerate(kernels):
    convolved_img = cv2.filter2D(gray, -1, np.array(kernel))
    ax[i].imshow(convolved_img,cmap='gray')
    ax[i].set_title(kernel_names[j]+' Kernel')
    
    i+=1

plt.show()
gray = skimage.color.rgb2gray(chelsea)
resized = skimage.transform.resize(gray,(300,448)) # original is 300,451
kernal = (4,4) # or block
blocked = skimage.util.view_as_blocks(resized,kernal)
# w,h should be completely divided by kernal size
blocked.shape
# reshape the image while preserving the Width and Height OR simply merge color and patches channels
reshaped = blocked.reshape(blocked.shape[0],blocked.shape[1],-1)
reshaped.shape
# MAx pooling
max_pooled = np.max(reshaped,axis=2)
avg_pooled = np.mean(reshaped,axis=2)
min_pooled = np.min(reshaped,axis=2)
med_pooled = np.median(reshaped,axis=2)

f,ax = plt.subplots(2,2,figsize=(15,8))
ax = ax.ravel()

ax[0].imshow(max_pooled,cmap='gray')
ax[0].set_title('Max Pooled')
ax[0].axis('off')

ax[1].imshow(min_pooled,cmap='gray')
ax[1].set_title('Min Pooled')
ax[1].axis('off')

ax[2].imshow(avg_pooled,cmap='gray')
ax[2].set_title('Average Pooled')
ax[2].axis('off')

ax[3].imshow(med_pooled,cmap='gray')
ax[3].set_title('Median Pooled')
ax[3].axis('off')


plt.show()
(_,_),(images,_) = cifar10.load_data() 
# (X_train,y_tain),(x_test,y_test) but we need less in quantity so we just imported x_test

images = images[:1000] # just use the first 1000 images for demo

print(f'Shape of images array is: {images.shape}')

f,ax = plt.subplots(1,3,figsize=(5,5))
ax = ax.ravel()
for i in range(3):
    ax[i].imshow(images[i])
plt.show()
images = images.reshape(-1,(32*32*3)) # -1 means it choses the best suitable i.e 1000 in our case
print(f' new shape of images is {images.shape}')
 # find the covariance matrix. i.e how much two variables change together
cov_mat = np.cov(images)

# singular value decomposition SVD (dimensionality reduction technique to find  hidden latent features)
# a plastic bag, douchebag and a weapon can have a latent feature that they are all from USA ;)
U,S,V = np.linalg.svd(cov_mat)

# dot product to get the principal components
epsilon = 0.000001 # avoid division by zero
w = np.diag(1.0/np.sqrt(S+epsilon)) # diagonal matrix
x = np.dot(w,U.T) # U transpose     
components = np.dot(U,x)

# calculate zca by using dot products of the principal components by images 
zca_images = np.dot(components,images)
zca_images.shape
f,ax = plt.subplots(1,3,figsize=(5,5))
ax = ax.ravel()
for i in range(3):
    # clip the images in some range for matplotlib else it'll throw error
    img = zca_images[i].reshape((32,32,3))
    min_,max_ = img.min(), img.max()
    ax[i].imshow((img-min_)/(max_-min_))  # clipping
plt.show()
images = images - images.mean(axis=0) # subtract the mean of whole 10000 images from each image
images = images/images.std(axis=0) # divide by the standard deviation

f,ax = plt.subplots(1,3,figsize=(5,5))
ax = ax.ravel()
for i in range(3):
    # clip the images in some range for matplotlib else it'll throw error
    img = images[i].reshape((32,32,3))
    min_,max_ = img.min(), img.max()
    ax[i].imshow((img-min_)/(max_-min_))  # clipping
plt.show()
resized = skimage.transform.resize(chelsea,(chelsea.shape[0]//2,chelsea.shape[1]//2)) 
# resize to half preserving the aspect ratio
plt.imshow(resized)
# add Gaussian Noise
sigma = 0.17 # defines the type/shape of distribution high sigma = high noise
noised = skimage.util.random_noise(resized,mode='gaussian', var=sigma**2)
plt.imshow(noised)
# if our image has Gaussian noise, this can detect and tell us the sigma. Our result is close to our sigma
sig = skimage.restoration.estimate_sigma(noised,multichannel=True,average_sigmas=True)
# we want to calculate the noise for each of the RGB channel (multichannel=True) and want avg. for all
print(sig)

gauss = ndimage.gaussian_filter(noised,sigma=0.6) # sigma can be (2,2,0)
plt.imshow(gauss)
# it makes the 'normalization' value close to the normal image by using 100 iteration by default

tv_cham = skimage.restoration.denoise_tv_chambolle(noised,multichannel=True,weight=0.12)
plt.imshow(tv_cham)
# output depends on the weight. High value of weight makes the image blurry and far from original
# this preserves the edges and works on closeness of pixels (spatial closeness) and 
# how two pixels are similar in their color channels (radiometric similarity)

bilat = skimage.restoration.denoise_bilateral(noised,multichannel=True,sigma_color=0.09,sigma_spatial=1.3)
plt.imshow(bilat)
# both the sigma as parameters are standard deviations for spatial closeness and radiometric similarity
# try to use high sigma_spatial. It'll take lot of time and the black portion at the bottom will increase
# this works on the wavelength representation of image and follows the luminosity (Y) and chroma components
# (Cb,Cr) so it is YCbCr instead of RGB format

img = skimage.restoration.denoise_wavelet(noised,multichannel=True,wavelet='db1',
                                          convert2ycbcr=True,rescale_sigma=True) # YbCr is another format like RGB

min_,max_ = img.min(), img.max()
plt.imshow((img-min_)/(max_-min_)) # clipping

# try to read the documentation about the parameters. These vary from image to image
# it takes a mean of all pixels, weighted by how similar these pixels are to the target pixel. 
# This results in greater post-filtering clarity than local means
nl_mean = skimage.restoration.denoise_nl_means(noised,multichannel=True,patch_size=6,patch_distance=13)
plt.imshow(nl_mean)
img = skimage.io.imread(PATH+'erosion.jpeg')
if len(img.shape)>2:
    img = exposure.rescale_intensity(img,(50,255))
    # img = skimage.color.rgb2gray(original_img)
    

# Erosion
erosion_seed = img.copy()
erosion_seed[1:-1,1:-1] = img.max()
mask = img
eroded_img = reconstruction(erosion_seed,mask,method='erosion')

# Plot
f,ax = plt.subplots(1,2,figsize=(13,5))
ax = ax.ravel()

ax[0].imshow(img,cmap='gray')
ax[0].set_title('Original')

min_,max_ = eroded_img.min(), eroded_img.max()
ax[1].imshow((eroded_img-min_)/(max_-min_)) # clipping
ax[1].set_title('Eroded')

plt.show()
img = skimage.io.imread(PATH+'dilation.png')

# Dilation
dilation_seed = img.copy()
dilation_seed[1:-1,1:-1] = img.min()
mask = img
dilated_img = reconstruction(dilation_seed,mask,method='dilation')

# Plot
f,ax = plt.subplots(1,2,figsize=(13,5))
ax = ax.ravel()

ax[0].imshow(img,cmap='gray')
ax[0].set_title('Original')

min_,max_ = dilated_img.min(), dilated_img.max()
ax[1].imshow((dilated_img-min_)/(max_-min_),cmap='gray') # clipping
ax[1].set_title('Dilated')

plt.show()
horse = skimage.data.horse() # grayscale image of a camera
inverted_horse = skimage.util.invert(horse) # invert the image
horse_hull = skimage.morphology.convex_hull_image(inverted_horse)

# let us super impose the hull on top of our inverted image
horse_hull_copy = skimage.img_as_float(horse_hull.copy())
horse_hull_copy[inverted_horse] = 2 # set all the pixels vales to gray except True Values


f,ax = plt.subplots(2,2,figsize=(12,10))
ax = ax.ravel()

ax[0].imshow(horse,cmap='gray')
ax[0].set_title('Original')

ax[1].imshow(inverted_horse,cmap='gray')
ax[1].set_title('Inverted')

ax[2].imshow(horse_hull,cmap='gray')
ax[2].set_title('Hull of the Horse')

ax[3].imshow(horse_hull_copy,cmap='gray')
ax[3].set_title('Super Imposed')
       
plt.show()
# open Cv has all the workings in BGR format
dog1 = cv2.imread(PATH+'single.jpeg')
dog2 = cv2.imread(PATH+'duo.jpeg')

gray1 = cv2.cvtColor(dog1,cv2.COLOR_BGR2GRAY) # working with grayscale is fast
gray2 = cv2.cvtColor(dog2,cv2.COLOR_BGR2GRAY)
sift_obj = cv2.xfeatures2d.SIFT_create() # instantiate the sift objct
kp1 = sift_obj.detect(gray1,None)
kp2 = sift_obj.detect(gray2,None) # detect the list of keypoints where mask is none
dog1_kp = cv2.drawKeypoints(dog1,kp1,outImage=None)
dog2_kp = cv2.drawKeypoints(dog2,kp2,outImage=None,flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
# second one has the keypoints with different sizes to tell which region is used to form the keypoint
# it also has gradient or direction of keypoint as line inside circle

f,ax = plt.subplots(1,2,figsize=(15,7))
ax[0].imshow(cv2.cvtColor(dog1_kp,cv2.COLOR_BGR2RGB),) # we are used to see the RGB format
ax[1].imshow(cv2.cvtColor(dog2_kp,cv2.COLOR_BGR2RGB),)
kp1,descr1 = sift_obj.compute(gray1,kp1)
print(f'Descriptors is of type {type(descr1)} and has shape {descr1.shape}') # array of numbers
dog3 = cv2.imread(PATH+'triplet.jpeg')
gray3 = cv2.cvtColor(dog3,cv2.COLOR_BGR2GRAY)
kp3,descr3 = sift_obj.detectAndCompute(gray3,None) # directly find and compute features
# Brute Force Matcher
bf_matcher = cv2.BFMatcher(cv2.NORM_L2,crossCheck=False) # match the features using L2 distance
matches = bf_matcher.match(descr1,descr3)

matched_image = cv2.drawMatches(gray1,kp1,gray3,kp3,matches[:10],gray1.copy(),flags=0)
# show the first 10 matches of 2 images keeping the first image as reference 

plt.figure(figsize=(25,20))
plt.imshow(matched_image)
dog3 = cv2.imread(PATH+'triplet.jpeg')
gray3 = cv2.cvtColor(dog3,cv2.COLOR_BGR2GRAY)

descs, desc_img = feature.daisy(gray3,step=33,radius=7,visualize=True)
# use circular Gaussian Window with radius 2 to smooth the image and calculate the histograms of gradients
# use the distance of 30 between two sampling points
# number of keypoints change based on the radius and step

fig = plt.figure(figsize=(12,6))
plt.imshow(desc_img)
pedestrian = skimage.io.imread(PATH+'single pad.jpg')

feature_descriptors, feat_image = feature.hog(pedestrian,visualize=True,multichannel=True,
                                             pixels_per_cell=(10,10),cells_per_block=(4,4),
                                              block_norm='L2-Hys')

#feat_image = skimage.exposure.rescale_intensity(feat_image,in_range=(0,50))
# normalize the pixel intensity ranges 

f,ax = plt.subplots(1,2,figsize=(15,6))
ax[0].imshow(pedestrian)
ax[1].imshow(feat_image,cmap='gray')
plt.show()
img = img_as_float(data.camera()) # import image of a camera
rows, cols = img.shape

# add random noise
noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
noise[np.random.random(size=noise.shape) > 0.5] *= -1

# add random constant to the image
img_noise = img + noise
img_const = img + abs(noise)

# get metrices MSE and SSIM
mse_none = mse(img, img)
ssim_none = ssim(img, img, data_range=img.max() - img.min())

mse_noise = mse(img, img_noise)
ssim_noise = ssim(img, img_noise,
                  data_range=img_noise.max() - img_noise.min())

mse_const = mse(img, img_const)
ssim_const = ssim(img, img_const,
                  data_range=img_const.max() - img_const.min())

# plotting
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5),
                         sharex=True, sharey=True)
ax = axes.ravel()

label = 'MSE: {:.2f},  SSIM: {:.2f}' # Label the subplots

ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(label.format(mse_none, ssim_none))
ax[0].set_title('Original image')

ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(label.format(mse_noise, ssim_noise))
ax[1].set_title('Image with noise')

ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[2].set_xlabel(label.format(mse_const, ssim_const))
ax[2].set_title('Image With constant')

plt.show()
img = Image.open(PATH+'quote.png')
print(f'Our Image is of {img.format} format with dimensions {img.size} and belongs to {type(img)}') 
# it is not a numpy array
print(pytesseract.image_to_string(PATH+'quote.png',lang='eng')) # you can directly pass the img object
display.Image(PATH+'quote.png',width=img.size[0],height=img.size[1])