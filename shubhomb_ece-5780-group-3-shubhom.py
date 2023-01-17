# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import cv2

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



trainpaths = []

testpaths = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        fpath = os.path.join(dirname, filename)

        if os.path.splitext(fpath)[1] == ".tif":

            if "train" in fpath and not "mask" in fpath:

                trainpaths.append(fpath)

            elif "test" in fpath and not "mask" in fpath:

                testpaths.append(fpath)



def show_tif(im, title=None):

    if title:

        plt.title(title)

    plt.imshow(im, cmap="gray")

    plt.show()

    

def find_mask_file(trf):

    dir, name = os.path.split (trf)

    name = os.path.splitext(name)[0] + "_mask" + ".tif"

    return os.path.join(dir, name)



def mask_over_image(trf):

    a = np.array(Image.open(trf))

    b = (np.array(Image.open(find_mask_file(trf)))/255).astype(np.bool)

    assert a.shape == b.shape

    return b * a



def show_im_and_mask(trf, fs=(10,10)):

    f, axarr = plt.subplots (1,2, figsize=fs)

    axarr[0].imshow(Image.open(trf), cmap="gray")

    axarr[0].set_title(os.path.split(trf)[1])

    m = mask_over_image(trf)

    axarr[1].set_title(os.path.split(find_mask_file(trf))[1])

    axarr[1].imshow(m, cmap="gray")

    plt.show()

    

def split_into_patches(img, dims, mask, overlap=False):

    # An image is 420 x 580; #TODO: if wanted, we can make them overlap

    if img.shape[0] != 420 and img.shape[1] != 580:

        raise ValueError ("img shape %s not 420 x 580"%str(img.shape))

    patches = []

    if ((img.shape[0] / dims[0]) != (img.shape[0] // dims[0])) or ((img.shape[1] / dims[1]) != (img.shape[1] // dims[1])): 

        raise ValueError ("Patch dimension must be evenly divisible")

    numys = img.shape[0] // dims[0]

    numxs = img.shape[1] // dims[1]

    labs = np.zeros(numys * numxs, dtype=np.float32)

    for i in range(numys):

        for j in range(numxs):

            if len(img.shape) < 3: 

                patches.append(img[i * dims[0]: (i+1) * dims[0], j * dims[1]: (j+1) * dims[1]])

                labs[i * numxs + j] = np.sum(mask[i * dims[0]: (i+1) * dims[0], j * dims[1]: (j+1) * dims[1]])

            else: # 3rd dimension is feature space or RGB channels, shouldn't change the piecewise

                patches.append(img[i * dims[0]: (i+1) * dims[0], j * dims[1]: (j+1) * dims[1], :])

                labs[i * numxs + j] = np.sum(mask[i * dims[0]: (i+1) * dims[0], j * dims[1]: (j+1) * dims[1]])

    return np.stack(patches), labs

    

print (len(trainpaths), " training images")

print (len(testpaths), " test images")

train_masks = pd.read_csv("/kaggle/input/ultrasound-nerve-segmentation/train_masks.csv") 

print (train_masks.head())



# Any results you write to the current directory are saved as output.
for j in range(10):

    show_im_and_mask(trainpaths[j])

    
for j in range(10):

    imf = trainpaths[j]

    im = cv2.imread(imf)

    mask = cv2.imread(find_mask_file(imf))

    if np.sum(mask) == 0:

        continue

    mask_outline = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_outline = cv2.blur(mask, (3,3)) # blur and filter is just to find the border of the labeled region for display

    mask_outline = mask_outline * ((mask_outline < 255) & (mask_outline > 0))

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    cl1 = clahe.apply(gray)

    med = cv2.medianBlur(cl1, 3)

    f, axarr = plt.subplots(1,3, figsize=(20,20))

    axarr[0].imshow(im, cmap="gray")

    axarr[0].imshow(mask_outline, cmap="gray", alpha=0.4)

    axarr[0].set_title("Original Image")

    axarr[1].imshow(cl1, cmap="gray")

    axarr[1].set_title("CLAHE-applied")

    axarr[1].imshow(mask_outline, cmap="gray", alpha=0.4)

    axarr[2].imshow(med, cmap="gray")

    axarr[2].set_title("CLAHE followed by median filter")

    axarr[2].imshow(mask_outline, cmap="gray", alpha=0.4)



    plt.show()



    xs, ys = np.where(mask)[0], np.where(mask)[1]

    f, axarr = plt.subplots(1,3, figsize=(20,20))

    buff = 10

    

    axarr[0].imshow(im[min(xs)-buff: max(xs)+buff, min(ys)-buff:max(ys)+buff], cmap="gray")

    axarr[0].imshow(mask_outline[min(xs)-buff: max(xs)+buff, min(ys)-buff:max(ys)+buff], cmap="gray", alpha=0.4)

    axarr[0].set_title("Original Image")

    axarr[1].imshow(cl1[min(xs)-buff: max(xs)+buff, min(ys)-buff:max(ys)+buff], cmap="gray")

    axarr[1].set_title("CLAHE-applied")

    axarr[1].imshow(mask_outline[min(xs)-buff: max(xs)+buff, min(ys)-buff:max(ys)+buff], cmap="gray", alpha=0.4)

    axarr[2].imshow(med[min(xs)-buff: max(xs)+buff, min(ys)-buff:max(ys)+buff], cmap="gray")

    axarr[2].set_title("CLAHE followed by median filter")

    axarr[2].imshow(mask_outline[min(xs)-buff: max(xs)+buff, min(ys)-buff:max(ys)+buff], cmap="gray", alpha=0.4)

    plt.show()

from skimage.feature import hog, local_binary_pattern



lbp_8_1 = local_binary_pattern(med, P=8, R=1, method="ror") # rotation-invariant by default

lbp_16_2 = local_binary_pattern(med, P=16, R=2, method="ror")

features, hogim = hog(med, visualize=True)



f, axarr = plt.subplots(1,4, figsize=(20,20))

axarr[0].imshow(med, cmap="gray")

axarr[0].imshow(mask_outline, cmap="gray", alpha=0.4)

axarr[0].set_title("CLAHE-Median filtered Image")

axarr[1].imshow(lbp_8_1, cmap="gray")

axarr[1].imshow(mask_outline, cmap="gray", alpha=0.4)

axarr[1].set_title("LBP R=1, P=8")

axarr[2].imshow(lbp_16_2, cmap="gray")

axarr[2].imshow(mask_outline, cmap="gray", alpha=0.4)

axarr[2].set_title("LBP R=3, P=16")

axarr[3].imshow(hogim)

axarr[3].set_title("HOG Feature Visualization")

plt.show()

xs, ys = np.where(mask)[0], np.where(mask)[1]

plt.imshow(lbp_8_1[min(xs):max(xs), min(ys):max(ys)])

plt.imshow(mask[min(xs):max(xs), min(ys):max(ys)], alpha=0.5)

plt.title("LBPs in ROI")

plt.colorbar()

plt.show()



f, axarr = plt.subplots(2, figsize=(12, 5))

global_hist = np.histogram(lbp_8_1.flatten(),bins=256)

local_hist = np.histogram(lbp_8_1[min(xs):max(xs), min(ys):max(ys)].flatten(), bins=256)



axarr[0].hist(lbp_8_1.flatten(), bins=256)

axarr[0].grid()

axarr[0].set_title("Global Image LBP histogram")

axarr[1].hist(lbp_8_1[min(xs):max(xs), min(ys):max(ys)].flatten(), bins=256)

axarr[1].grid()

axarr[1].set_title("RoI LBP Histogram: KL %f")

plt.show()
stack, labs = split_into_patches(lbp_8_1, (70,58), np.array(Image.open(find_mask_file(trainpaths[0]))).astype(np.bool))



print ("%d stacks made" %stack.shape[0])

f, axarr = plt.subplots(6, 10, figsize=(60,40))

plt.suptitle (" A bunch of patches")

for i in range(6):

    for j in range(10):

        k = i * 10 + j 

        axarr[i,j].set_title("(" + str(i * 70) + " , " + str(j * 58) + "), %f"%labs[k])

        axarr[i,j].imshow(stack[k, :, :])

plt.show()
from sklearn.svm import SVR

from sklearn.metrics import confusion_matrix

import pandas

xTr = stack.reshape(stack.shape[0], -1)

yTr = labs

svm = SVR(kernel="rbf", gamma="scale", C=1.0)

svm.fit(xTr, labs)

preds = svm.predict(xTr)

one_hot_labs = labs > 0

one_hot_preds = preds > 0.3 # TODO: what is a good way to threshold/round continuous SVR output 

confusion = confusion_matrix(one_hot_labs, one_hot_preds)

confusion 
from sklearn.model_selection import train_test_split

import datetime



def preprocess (img_path):

    im = cv2.imread(imf)

    mask = cv2.imread(find_mask_file(imf))

    mask_outline = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_outline = cv2.blur(mask, (3,3))

    mask_outline = mask_outline * ((mask_outline < 255) & (mask_outline > 0))

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    cl1 = clahe.apply(gray)

    med = cv2.medianBlur(cl1, 3)

    return med



def generate_features(img):

    lbp_8_1 = local_binary_pattern(med, P=8, R=1, method="ror") # rotation-invariant by default

    lbp_16_2 = local_binary_pattern(med, P=16, R=2, method="ror")

    features, hogim = hog(med, visualize=True)

    return np.stack([lbp_8_1, lbp_16_2,hogim], axis=-1)

    

def select_features(img):

    return img



def prepare_set (xpaths, patch_size):

    xs, ys = [], []

    for path in xpaths:

        med = preprocess(path)

        features = generate_features(med)

        features = select_features(features)

        stack, labs = split_into_patches(features, patch_size, np.array(Image.open(find_mask_file(path))).astype(np.bool))

        stack = stack.reshape(stack.shape[0], -1)

        xs.append(stack)

        ys.append(labs)

    xs = np.array(xs)

    ys = np.array(ys)

    return xs.reshape(-1, xs.shape[2]), ys.flatten()



def train_model(model, xTr, yTr, patch_size=(70,58),xVal=None,random_seed=0):

    start = datetime.datetime.now()

    model = model.fit(xTr, yTr)

    trpreds = model.predict(xTr)

    if xVal is not None: 

        valpreds = model.predict(xVal)

    else:

        valpreds = None

    print ("Model finished training %d patches after %s"%(xTr.shape[0], str(datetime.datetime.now() - start)))

    return trpreds, valpreds





def patch_confusion(labs, predictions):

    one_hot_labs = labs > 0

    one_hot_preds = predictions > 0.5 # TODO: what is a good way to threshold/round continuous SVR output 

    confusion = confusion_matrix(one_hot_labs, one_hot_preds)

    return confusion



    

svm = SVR(kernel="linear", gamma="scale", C=5.0)

seed = 0

patch_size = (70, 58)

tr_paths, val_paths = train_test_split(trainpaths[:150], shuffle=True, test_size=0.2, random_state=seed)

xTr,yTr = prepare_set(tr_paths, patch_size)

xVal, yVal = prepare_set(val_paths, patch_size)

# xTest, yTest = prepare_set(test_paths, patch_size) not used till end

tp, vp = train_model(svm, xTr,yTr, xVal=xVal)

print ("Training confusion matrix")

print (patch_confusion(yTr, tp))

print ("Validation confusion matrix")

print (patch_confusion(yVal, vp))
