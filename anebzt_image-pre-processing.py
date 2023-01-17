import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
!ls ../input/flowers/flowers
train_dir = "../input/flowers/flowers/"
classes = os.listdir(train_dir)
classes = sorted(classes)
nclasses = len(classes)
classes
for _class in classes:
    print('{} {} images'.format(_class, len(os.listdir(os.path.join(train_dir, _class)))))
from PIL import Image, ImageOps, ImageFilter
other_extensions = [".db", ".pyc", ".py"]
def open_images_pil(path, classes, dim=32):
    
    xall = []
    yall = []
    label = 0
    j = 0

    for cl in classes:
        clsdir = os.path.join(path, cl)
        for imgname in os.listdir(clsdir):
            bad_ext_found = 0
            for other_ext in other_extensions:
                if imgname.endswith(other_ext):
                    bad_ext_found = 1
                    break
            if not bad_ext_found:
                print("Opening files in {}: {}".format(cl, str(j + 1)), end="\r")
                imgpath = os.path.join(clsdir, imgname)

                #open and pre-process images
                img = Image.open(imgpath)
                img = ImageOps.fit(img, (dim, dim), Image.ANTIALIAS).convert('RGB')
                
                xall.append(img)  # Get image 
                yall.append(label)  # Get image label (folder name)
                j += 1

        j = 0
        label += 1
        print()

    n = len(xall)
    print("{} images in set".format(n))
    return xall, yall
xall, yall = open_images_pil(train_dir, classes, 256)
im = xall[0]
im
im.format, im.size, im.mode
blur = im.filter(ImageFilter.BLUR)
blur
gray = im.convert('L')
rgb = gray.convert("RGB")
gray
box = (100, 100, 400, 400)
crpd = im.crop(box)
crpd
enh = im.filter(ImageFilter.DETAIL)
enh
_xall = np.stack(xall, axis=0) # from list of len 56 of ndarrays (dim, dim, 3) to ndarray (52, dim, dim, 3)
_xall.shape
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer().fit(yall)
label = lb.transform(yall) 
label.shape
#_xall = np.asarray(xall)

for i in range(0,9): # how many imgs will show from the 3x3 grid
    plt.subplot(330 + (i+1)) # open next subplot
    plt.imshow(_xall[i + 155], cmap=plt.get_cmap('gray'))
    plt.title(yall[i + 155]);
import imageio
from skimage.transform import resize

classpath = os.path.join(train_dir, classes[0])
imgpath = os.listdir(classpath)[0] # first imagepath of first class, as example
imgpath = os.path.join(classpath, imgpath)

img = imageio.imread(imgpath)
dim = 128
img = resize(img, (dim, dim, 3))
import cv2
from cv2 import imread, cvtColor, resize, threshold, calcHist, equalizeHist
def trim_margin(img, lim):
    return img[lim:-lim, lim:-lim]
supported_dims = [16, 32, 64, 128, 256]
def img_resize(img, dims):
    if dims in supported_dims:
        return cv2.resize(img, (dims, dims))
    else:
        print("Incorrect image dimensions.\n")
        return None
other_extensions = [".db", ".pyc", ".py"]
def open_images_cv2(path, classes, dim=32):
    
    xall = []
    yall = []
    label = 0
    j = 0

    for cl in classes:
        clsdir = os.path.join(path, cl)
        for imgname in os.listdir(clsdir):
            bad_ext_found = 0
            for other_ext in other_extensions:
                if imgname.endswith(other_ext):
                    bad_ext_found = 1
                    break
            if not bad_ext_found:
                print("Opening files in {}: {}".format(cl, str(j + 1)), end="\r")
                imgpath = os.path.join(clsdir, imgname)

                #open and pre-process images
                img = imread(imgpath, cv2.IMREAD_COLOR)
                img = cvtColor(img, cv2.COLOR_BGR2RGB)
                img = trim_margin(img, int(img.shape[0] * 0.05))
                img = img_resize(img, dim)
                #img = equalize_hist(img)

                xall.append(img)  # Get image 
                yall.append(label)  # Get image label (folder name)
                j += 1

        j = 0
        label += 1
        print()

    n = len(xall)
    print("{} images in set".format(n))
    return xall, yall
xall, yall = open_images_cv2(train_dir, classes, 256)
xall = np.asarray(xall)
yall = np.asarray(yall)
image = xall[0]
plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.title(yall[0]);
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# convert the YUV image back to RGB format
imgo = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

plt.imshow(imgo)
color = ('r','g','b')
for i,col in enumerate(color):
    histr = cv2.calcHist([imgo],[i],None,[256],[0,256])
    print(max(histr))
    plt.plot(histr,color=col)
    plt.xlim([0,256])
plt.show()
# https://lmcaraig.com/image-histograms-histograms-equalization-and-histograms-comparison/

from matplotlib import ticker

bins = 256
tick_spacing = 5

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
channels_mapping = {0: 'B', 1: 'G', 2: 'R'}
for i, channels in enumerate([[0, 1], [0, 2], [1, 2]]):
    hist = cv2.calcHist([image], channels, None, [bins]*2, [0, 256]*2)

    channel_x = channels_mapping[channels[0]]
    channel_y = channels_mapping[channels[1]]

    ax = axes[i]
    ax.set_xlim([0, bins - 1])
    ax.set_ylim([0, bins - 1])

    ax.set_xlabel(f'Channel {channel_x}')
    ax.set_ylabel(f'Channel {channel_y}')
    ax.set_title(f'2D Color Histogram for {channel_x} and {channel_y}')

    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    im = ax.imshow(hist)

fig.colorbar(im, ax=axes.ravel().tolist(), orientation='orizontal')
fig.suptitle(f'2D Color Histograms with {bins} bins', fontsize=16)
plt.show()
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
gray = cvtColor(image, cv2.COLOR_RGB2GRAY)
claheImg = clahe.apply(gray)
plt.imshow(claheImg)
imgpath = "../input/files/files/unsplash/-537308-unsplash.jpg"

gray
img = cv2.medianBlur(gray,5)

ret, th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

# doesn't look very good
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(image, kernel, iterations=1)

plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(erosion)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(opening)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(closing)
from skimage.measure import compare_ssim
from skimage.transform import resize

image2 = xall[1]

(score, diff) = compare_ssim(image, image2, full=True, multichannel=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

