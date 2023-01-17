import numpy as np

import glob

import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt

import os

os.listdir('../input')
df = pd.read_csv('../input/stage1_labels.csv')

df.head()
np.mean(df['cancer']), len(df)
'406cb7c9c4915e0a24059008daa2972f'

df['id'][354]
basedir = '../input/sample_images'

baseids = os.listdir(basedir)
# From https://www.kaggle.com/anokas/data-science-bowl-2017/exploratory-data-analysis

import dicom 



def get_slice_location(dcm):

    return float(dcm[0x0020, 0x1041].value)



def load_patient(patient_id):

    files = glob.glob(basedir + '/{}/*.dcm'.format(patient_id))

    imgs = {}

    for f in files:

        dcm = dicom.read_file(f)

        img = dcm.pixel_array

        img[img == -2000] = 0

        sl = get_slice_location(dcm)

        imgs[sl] = img



    # Not a very elegant way to do this

    sorted_imgs = [x[1] for x in sorted(imgs.items(), key=lambda x: x[0])]

    return sorted_imgs



def draw_patient(id_i):

    pat = load_patient(id_i)

    fig, axs = plt.subplots(11, 10, sharex='all', sharey='all', figsize=(10, 11))

    # matplotlib is drunk

    #plt.title('Sorted Slices of Patient 0a38e7597ca26f9374f8ea2770ba870d - No cancer')

    for i in range(110):

        axs[i // 10, i % 10].axis('off')

        axs[i // 10, i % 10].imshow(pat[i], cmap=plt.cm.bone)

    fig.suptitle('cancer' if df['cancer'][df['id'] == id_i].values[0] else 'no cancer')
draw_patient(baseids[0])
id_i = baseids[0]

'cancer' if df['cancer'][df['id'] == id_i].values[0] else 'no cancer'
draw_patient(baseids[0])
i = imgs.items()
i.head()