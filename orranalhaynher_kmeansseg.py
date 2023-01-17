import cv2

import numpy as np

from glob import glob

from pathlib import Path

from scipy import ndimage

from skimage import measure

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from skimage.color import rgb2gray

from skimage.io import imread, imsave

from scipy.ndimage import binary_fill_holes

from sklearn.metrics import confusion_matrix

from skimage.filters import sobel, gaussian

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
ultrasound = glob('../input/ultrasoundcorte/cropped/*.bmp')

masks = glob('../input/ultrasoundcorte/masks/*.bmp')
kmeans = []

cinza = []

filterGaussian = []

n_clusters = 2



for u, ult in enumerate(ultrasound):

    cinza.append(rgb2gray(imread(ult)))

    filterGaussian.append(gaussian(cinza[u], sigma=3))

    

    X = filterGaussian[u].reshape((-1, 1))  

    k_means = KMeans(n_clusters).fit(X)

    imgKmeans = k_means.labels_

    kmeans.append(imgKmeans.reshape(filterGaussian[u].shape))

    

kmeans2 = np.copy(kmeans)
fig, aux = plt.subplots(1,9, figsize=(30,10))

for i in range(0,9):

    aux[i].imshow(kmeans2[i], cmap='gray')
troca = []



for b in kmeans2:

    linhas, colunas = b.shape

    

    if b[5,160] == 0:

        for i in range(linhas):

            for j in range(colunas):

                b[i][j] = (1 - b[i][j]) 

    troca.append(b)
fig, aux = plt.subplots(1,9, figsize=(30,10))

for i in range(0,9):

    aux[i].imshow(troca[i], cmap='gray')
largestRegion = []



for kImg in troca:



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



for largeImg in largestRegion:

    lImg = border(largeImg)

    bordaSo.append(sobel(lImg))
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
metrics = []



for m, imgMask in enumerate(mask):

    metrics.append(calc_metric(fill_holes[m], imgMask))
jaccard = sum(met[0] for met in metrics)

dice = sum(met[1] for met in metrics)



print('Jaccard médio -> ', jaccard/len(metrics), 'e Dice médio -> ', dice/len(metrics))
multi = []



for n, imgBin in enumerate(fill_holes):

    multi.append(np.multiply(cinza[n], imgBin))
def corte(img):

    

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    ymin, ymax = np.where(rows)[0][[0, -1]]

    xmin, xmax = np.where(cols)[0][[0, -1]]

    

    return img[ymin:ymax+1, xmin:xmax+1]
roi = []



for img in multi:

    roi.append(corte(img))
def zero_padding(imagem, linha, coluna):

    a = (linha - imagem.shape[0]) // 2

    b = (coluna - imagem.shape[1]) // 2

    imagemB = np.zeros((linha + 2, coluna + 2), dtype='float32')

    imagemB[a:imagem.shape[0] + a, b:imagem.shape[1] + b] = imagem



    return imagemB
linhaList = []

colunaList = []



for imgR in roi:

    linhaList.append(imgR.shape[0])

    colunaList.append(imgR.shape[1])

    

dim1 = max(linhaList)

dim2 = max(colunaList)
resize = []



for imgRoi in roi:

    resize.append(zero_padding(imgRoi, dim1, dim2))
fig, aux = plt.subplots(1,9, figsize=(30,10))

for i in range(0,9):

    aux[i].imshow(resize[i], cmap='gray')
for i in range(len(resize)):

    filename = Path(ultrasound[i]).stem

    imsave('./'+filename+'.bmp', resize[i])