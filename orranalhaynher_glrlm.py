!pip install pyradiomics
import numpy as np

import pandas as pd

from SimpleITK import GetImageFromArray

from radiomics import glrlm

from skimage.io import imread_collection, imread

import matplotlib.pyplot as plt

from tensorflow.image import resize_with_crop_or_pad
def get_order_filenames(file_path):

    df = pd.read_csv(file_path)

    return df["image"]
file_path = '../input/specialist-segmentation/reading_order.csv'

image_names = get_order_filenames(file_path)
path = '../input/kmeanssegmentation/'

images = [imread(path+str(name)+'.bmp') for name in image_names]

print('The database has {} segmented images'.format(len(images)))
masksIm = []



for x in range(len(images)):

    masks = np.zeros((images[x].shape[0], images[x].shape[1]))

    for i in range(images[x].shape[0]):

        for j in range(images[x].shape[1]):

            if (images[x][i,j] == 0):

                masks[i,j] = 0

            else:

                masks[i,j] = 1

                

    masksIm.append(masks)
masksIm[0].shape
glrlmFeat = []



for m in range(len(images)):

    glrlmFeatures = glrlm.RadiomicsGLRLM(GetImageFromArray(images[m]), GetImageFromArray((masksIm[m]).astype(np.uint8)), verbose=False)



    glrlmFeatures.enableFeatureByName('ShortRunEmphasis', True)

    glrlmFeatures.enableFeatureByName('LongRunEmphasis', True)

    glrlmFeatures.enableFeatureByName('GrayLevelNonUniformity', True)

    glrlmFeatures.enableFeatureByName('RunLengthNonUniformity', True)

    glrlmFeatures.enableFeatureByName('RunPercentage', True)

    glrlmFeatures.enableFeatureByName('LowGrayLevelRunEmphasis', True)

    glrlmFeatures.enableFeatureByName('HighGrayLevelRunEmphasis', True)

    glrlmFeatures.enableFeatureByName('ShortRunLowGrayLevelEmphasis', True)

    glrlmFeatures.enableFeatureByName('ShortRunHighGrayLevelEmphasis', True)

    glrlmFeatures.enableFeatureByName('LongRunLowGrayLevelEmphasis', True)

    glrlmFeatures.enableFeatureByName('LongRunHighGrayLevelEmphasis', True) 



    glrlmFeat.append(glrlmFeatures.execute())
len(glrlmFeat)
print('Atributos extraídos da primeira imagem')

glrlmFeat[0]
glrlmFinal = []

orderedNames = ['ShortRunEmphasis','LongRunEmphasis','GrayLevelNonUniformity', 'RunLengthNonUniformity', 'RunPercentage', 'LowGrayLevelRunEmphasis', 

                'HighGrayLevelRunEmphasis', 'ShortRunLowGrayLevelEmphasis', 'ShortRunHighGrayLevelEmphasis', 'LongRunLowGrayLevelEmphasis', 

                'LongRunHighGrayLevelEmphasis']



for features in glrlmFeat:

    for names in orderedNames:

        glrlmFinal.append(features.get(names).item())



glrlmFinal = np.reshape(glrlmFinal, (203, 11))
glrlmFinal[0][0]
np.save('./glrlmKmeans', glrlmFinal)