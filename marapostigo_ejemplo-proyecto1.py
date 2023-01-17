# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break



# Any results you write to the current directory are saved as output.
import os



#CARGAMOS LAS IMÁGENES Y MÁSCARAS DE TRAIN

data_dir= '/kaggle/input/proy-1-segmentacion-de-imagenes-dermatoscopicas'



train_imgs_files = [os.path.join(data_dir,'train/images',f) for f in sorted(os.listdir(os.path.join(data_dir,'train/images'))) 

            if (os.path.isfile(os.path.join(data_dir,'train/images',f)) and f.endswith('.jpg'))]



train_masks_files = [os.path.join(data_dir,'train/masks',f) for f in sorted(os.listdir(os.path.join(data_dir,'train/masks'))) 

            if (os.path.isfile(os.path.join(data_dir,'train/masks',f)) and f.endswith('.png'))]



#Ordenamos para que cada imagen se corresponda con cada máscara

train_imgs_files.sort()

train_masks_files.sort()

print("Número de imágenes de train", len(train_imgs_files))

print("Número de máscaras de train", len(train_masks_files))



#CARGAMOS LAS IMÁGENES DE TEST

test_imgs_files = [os.path.join(data_dir,'test/images',f) for f in sorted(os.listdir(os.path.join(data_dir,'test/images'))) 

            if (os.path.isfile(os.path.join(data_dir,'test/images',f)) and f.endswith('.jpg'))]



test_imgs_files.sort()

print("Número de imágenes de test", len(test_imgs_files))
import matplotlib.pyplot as plt

from skimage import io



images = io.ImageCollection(train_imgs_files)

masks = io.ImageCollection(train_masks_files)



index = 1

plt.figure(figsize=(15,8))

for i in range (4):

    plt.subplot(2,4,index)

    plt.imshow(images[i])

    index+=1

    plt.title("Image %i"%(i))

    plt.subplot(2,4,index)

    plt.imshow(masks[i], cmap='gray')

    index+=1

    plt.title("Mask %i"%(i))

    

import numpy as np

from skimage import io, filters, color

from sklearn.metrics import jaccard_score

from scipy import ndimage



def skin_lesion_segmentation(img_root):

    """ SKIN_LESION_SEGMENTATION: ... 

    - - -  COMPLETAR - - - 

    """

    # El siguiente código implementa el BASELINE incluido en el challenge de

    # Kaggle. 

    # - - - MODIFICAR PARA IMPLEMENTACIÓN DE LA SOLUCIÓN PROPUESTA. - - -

    

    #PREPROCESADO (de color a escala de grises) --> Probad diferentes técnicas de preprocesado 

    image = io.imread(img_root)

    image_gray = color.rgb2gray(image)

    #SEGMENTACIÓN AUTOMÁTICA (otsu) --> Probad diferentes técnicas de segmentación 

    otsu_th = filters.threshold_otsu(image_gray)

    predicted_mask = (image_gray < otsu_th).astype('int')

    #POSTPROCESADO (eliminación de agujeros) --> Probad diferentes técnicas de postprocesado

    post_predicted_mask= ndimage.binary_fill_holes(predicted_mask)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    

    """NOTA: las diferentes técnicas que probéis se pueden combinar 

    (es decir, podéis aplicar varios preprocesados en cadena, etc)"""

    

    return post_predicted_mask
def evaluate_masks(img_roots, gt_masks_roots):

    """ EVALUATE_MASKS: Función que, dadas dos listas, una con las rutas

        a las imágenes a evaluar y otra con sus máscaras Ground-Truth (GT)

        correspondientes, determina el Mean Average Precision a diferentes

        umbrales de Intersección sobre Unión (IoU) para todo el conjunto de

        imágenes.

    """

    score = []

    for i in np.arange(np.size(img_roots)):

        predicted_mask = skin_lesion_segmentation(img_roots[i])

        gt_mask = io.imread(gt_masks_roots[i])/255     

        score.append(jaccard_score(np.ndarray.flatten(gt_mask),np.ndarray.flatten(predicted_mask)))

    mean_score = np.mean(score)

    print('Jaccard Score sobre el conjunto de imágenes proporcionado: '+str(mean_score))

    return mean_score
import numpy as np # linear algebra



# ref.: https://www.kaggle.com/stainsby/fast-tested-rle

def rle_encode(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels = img.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)

 

def rle_decode(mask_rle, shape):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (height,width) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape)
import numpy as np

import os, csv

from skimage import io

import skimage.color

#from rle import rle_encode

#from project1 import skin_lesion_segmentation





def test_prediction_csv(dir_images_name='database/test/images', csv_name='test_prediction.csv'):

    dir_images = np.sort(os.listdir(dir_images_name))

    mask_otsu_all = []

    mask_aux_all = []

    score = []

    with open(csv_name, 'w', newline='') as file:

        writer = csv.writer(file)

        writer.writerow(["ImageId", "EncodedPixels"])

        for i in np.arange(np.size(dir_images)):        

            # - - - Llamada a la función 'skin_lesion_segmentation'

            # Implementa el método propuesto para la segmentación de la lesión

            # y proporciona a su salida la máscara predicha.

            predicted_mask = skin_lesion_segmentation(dir_images_name+'/'+dir_images[i])

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

            

            # - - - Codificación RLE y escritura en fichero .csv

            encoded_pixels = rle_encode(predicted_mask)

            writer.writerow([dir_images[i][:-4], encoded_pixels])

            print('Máscara '+str(i)+' codificada.')

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import copy



#1.Con el algoritmo creado en "skin_lesion_segmentation", 

#comprobamos qué tal es nuestra nota de segmentación mediante 

img_roots = train_imgs_files.copy()

gt_masks_roots = train_masks_files.copy()



mean_score = evaluate_masks(img_roots, gt_masks_roots)
#Una vez satisfechos con el resultado, generamos el fichero para hacer la submission en Kaggle

dir_images_name = '/kaggle/input/proy-1-segmentacion-de-imagenes-dermatoscopicas/test/images'

csv_name='test_prediction_rgb2g_otsu_fill_holes.csv'

test_prediction_csv(dir_images_name, csv_name)