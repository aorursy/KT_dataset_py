# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

import seaborn as sns

import os

import glob

%matplotlib inline

p = sns.color_palette()



os.listdir('../input')



# Any results you write to the current directory are saved as output.
for d in os.listdir('../input/sample_images'):

    print("Patient '{}' has {} scans".format(d, len(os.listdir('../input/sample_images/' + d))))

print('----')

print('Total patients {} Total DCM files {}'.format(len(os.listdir('../input/sample_images')), 

                                                      len(glob.glob('../input/sample_images/*/*.dcm'))))
import dicom

dcm = '../input/sample_images/0a38e7597ca26f9374f8ea2770ba870d/4ec5ef19b52ec06a819181e404d37038.dcm'

print('Filename: {}'.format(dcm))

dcm = dicom.read_file(dcm)

img = dcm.pixel_array

img[img == -2000] = 0



plt.axis('off')

plt.imshow(img)

plt.show()



plt.axis('off')

plt.imshow(-img) # Invert colors with -

plt.show()

def get_slice_location(dcm):

    return float(dcm[0x0020, 0x1041].value)



# Returns a list of images for that patient_id, in ascending order of Slice Location

def load_patient(patient_id):

    files = glob.glob('../input/sample_images/{}/*.dcm'.format(patient_id))

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

pat = load_patient('0acbebb8d463b4b9ca88cf38431aac69')

f, plots = plt.subplots(21, 10, sharex='all', sharey='all', figsize=(50, 105))

for i in range(203):

    plots[i // 10, i % 10].axis('off')

    plots[i // 10, i % 10].imshow(pat[i], cmap=plt.cm.bone)