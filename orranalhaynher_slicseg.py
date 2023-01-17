import cv2

import numpy as np

from glob import glob

from pathlib import Path

import matplotlib.pyplot as plt

from skimage.filters import sobel

from skimage.color import rgb2gray

from skimage.segmentation import slic

from skimage.io import imread, imsave

from scipy.ndimage import binary_fill_holes

from sklearn.metrics import confusion_matrix

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
ultrasound = glob('../input/ultrasoundcorte/cropped/*.bmp')

masks = glob('../input/ultrasoundcorte/masks/*.bmp')
sc = []

cinza = []



for u, ult in enumerate(ultrasound):

    cinza.append(rgb2gray(imread(ult)))

    seg = slic(cinza[u], n_segments=2, compactness=0.2, sigma=3)

    sc.append(seg)

    

sc2 = np.copy(sc)
plt.imshow(sc2[0], cmap='gray')
troca = []



for b in sc2:

    linhas, colunas = b.shape

    

    if b[1,160] == 1:

        for i in range(linhas):

            for j in range(colunas):

                b[i][j] = (1 - b[i][j]) 

    troca.append(b)
plt.imshow(troca[0], cmap='gray')
def border(imagem):

    imagem[:1,:] = 0.0

    imagem[496:,:] = 0.0

    imagem[:,:1] = 0.0

    imagem[:,322:] = 0.0



    return imagem
bordaSo = []



for t, trocImg in enumerate(troca):

    #t = border(trocImg)

    bordaSo.append(sobel(trocImg))
fill_holes = []



for sobelImg in bordaSo:

    fill_holes.append(binary_fill_holes(sobelImg))
binary = []



for fillImg in fill_holes:

    binary.append(fillImg*1)
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

    metrics.append(calc_metric(binary[m], imgMask))
jaccard = sum(met[0] for met in metrics)

dice = sum(met[1] for met in metrics)



print('Jaccard médio -> ', jaccard/len(metrics), 'e Dice médio -> ', dice/len(metrics))
multi = []



for n, imgBin in enumerate(binary):

    multi.append(np.multiply(cinza[n], imgBin))

    

plt.imshow(multi[0], cmap='gray')
for i in range(len(multi)):

    filename = Path(ultrasound[i]).stem

    imsave('./'+filename+'.bmp', multi[i])