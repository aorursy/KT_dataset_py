import cv2

import numpy as np

from glob import glob

from skimage import measure

from skimage.io import imread

import matplotlib.pyplot as plt

from skimage.color import rgb2gray

from scipy.ndimage import binary_fill_holes

from sklearn.metrics import confusion_matrix

from skimage.measure import label, regionprops

from skimage.filters import sobel, threshold_otsu, gaussian

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
ultrasound = glob('../input/ultrasoundcorte/cropped/*.bmp')

masks = glob('../input/ultrasoundcorte/masks/*.bmp')
otsu = []

cinza = []

filterGaussian = []



for u, ult in enumerate(ultrasound):

    cinza.append(rgb2gray(imread(ult)))

    filterGaussian.append(gaussian(cinza[u], sigma=3))



    otsu.append(filterGaussian[u] > threshold_otsu(filterGaussian[u]))
fig, aux = plt.subplots(1,9, figsize=(30,10))

for i in range(0,9):

    aux[i].imshow(cinza[i], cmap='gray')
fig, aux = plt.subplots(1,9, figsize=(30,10))

for i in range(0,9):

    aux[i].imshow(filterGaussian[i], cmap='gray')
fig, aux = plt.subplots(1,9, figsize=(30,10))

for i in range(0,9):

    aux[i].imshow(otsu[i], cmap='gray')
largestRegion = []



for kImg in otsu:



    region = np.zeros(kImg.shape)

    

    label_image = measure.label(kImg, connectivity=kImg.ndim)

    num_labels = np.unique(label_image)



    for indice in num_labels:

        props = measure.regionprops(label_image)



        area = [reg.area for reg in props]

        largest_label_ind = np.argmax(area)

        largest_label = props[largest_label_ind].label



        region[label_image == largest_label] = indice

        

    for i in range(region.shape[0]):

        for j in range(region.shape[1]):

            if region[i,j] != 0:

                region[i,j] = 1

        

    largestRegion.append(region)
fig, aux = plt.subplots(1,9, figsize=(30,10))

for i in range(0,9):

    aux[i].imshow(largestRegion[i], cmap='gray')
def border(imagem):

    imagem[:1,:] = 0.0

    imagem[256:,:] = 0.0

    imagem[:,:1] = 0.0

    imagem[:,321:] = 0.0



    return imagem
bordaSo = []



for otsuImg in largestRegion:

    img = border(otsuImg)

    bordaSo.append(sobel(img))
fig, aux = plt.subplots(1,9, figsize=(30,10))

for i in range(0,9):

    aux[i].imshow(bordaSo[i], cmap='gray')
fill_holes = []



for sobelImg in bordaSo:

    fill_holes.append(binary_fill_holes(sobelImg)*1)
fig, aux = plt.subplots(1,9, figsize=(30,10))

for i in range(0,9):

    aux[i].imshow(fill_holes[i], cmap='gray')
mask = []



for maskImg in masks:

    mask.append(imread(maskImg))
def calc_metric(y_pred, y_true):

    cm = confusion_matrix(y_pred.ravel(),y_true.ravel())

    

    tp = cm[0,0]

    fn = cm[0,1]

    fp = cm[1,0]

    tn = cm[1,1]

    

    dice = (2.0 * tp) / ((2.0 * tp) + fp + fn)

    jaccard = (1.0 * tp) / (tp + fp + fn)

    

    #sensitivity = (1.0 * tp) / (tp + fn)

    #specificity = (1.0 * tn) / (tn + fp)

    #accuracy = (1.0 * (tn + tp)) / (tn + fp + tp + fn)

    #auc = 1 - 0.5 * (((1.0 * fp) / (fp + tn)) + ((1.0 * fn) / (fn + tp)))

    #prec = float(tp)/float(tp + fp)

    #fscore = float(2*tp)/float(2*tp + fp + fn)

    #kappa = cohen_kappa_score(y_pred, y_true)

    

    return dice, jaccard #sensitivity, specificity, accuracy, auc, kappa #, dice
def compute_iou(y_pred, y_true):

     # ytrue, ypred is a flatten vector

     y_pred = y_pred.flatten()

     y_true = y_true.flatten()

     current = confusion_matrix(y_true, y_pred, labels=[0, 1])

     # compute mean iou

     intersection = np.diag(current)

     ground_truth_set = current.sum(axis=1)

     predicted_set = current.sum(axis=0)

     union = ground_truth_set + predicted_set - intersection

     IoU = intersection / union.astype(np.float32)

     return np.mean(IoU)
iou = []

for i in range(len(fill_holes)):

    iou.append(compute_iou(fill_holes[i], mask[i]))
metrics = []



for m, imgMask in enumerate(mask):

    metrics.append(calc_metric(fill_holes[m], imgMask))
jaccard = sum(met[0] for met in metrics)

dice = sum(met[1] for met in metrics)



print('Jaccard médio -> ', jaccard/len(metrics), ', Dice médio -> ', dice/len(metrics), ' e IOU médio ->', sum(iou)/len(iou))
multi = []



for n, imgFill in enumerate(fill_holes):

    multi.append(np.multiply(cinza[n], imgFill))
fig, aux = plt.subplots(1,9, figsize=(30,10))

for i in range(0,9):

    aux[i].imshow(multi[i], cmap='gray')
def corte(img):

    

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    ymin, ymax = np.where(rows)[0][[0, -1]]

    xmin, xmax = np.where(cols)[0][[0, -1]]

    

    return img[ymin:ymax+1, xmin:xmax+1]
roi = []



for img in multi:

    

    roi.append(corte(img))
fig, aux = plt.subplots(1,9, figsize=(30,10))

for i in range(0,9):

    aux[i].imshow(roi[i], cmap='gray')
linhaList = []

colunaList = []



for imgR in roi:

    linhaList.append(imgR.shape[0])

    colunaList.append(imgR.shape[1])

    

dim1 = max(linhaList)

dim2 = max(colunaList)
def zero_padding(imagem, linha, coluna):

    a = (linha - imagem.shape[0]) // 2

    b = (coluna - imagem.shape[1]) // 2

    imagemB = np.zeros((linha + 2, coluna + 2), dtype='float32')

    imagemB[a:imagem.shape[0] + a, b:imagem.shape[1] + b] = imagem



    return imagemB
resize = []



for imgRoi in roi:

    resize.append(zero_padding(imgRoi, dim1, dim2))
fig, aux = plt.subplots(1,9, figsize=(30,10))

for i in range(0,9):

    aux[i].imshow(resize[i], cmap='gray')