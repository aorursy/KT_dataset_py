import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import glob

%matplotlib inline

p = sns.color_palette()



os.listdir('../input')

for d in os.listdir('../input/sample_images'):

    print("Patient '{}' has {} scans".format(d, len(os.listdir('../input/sample_images/' + d))))

print('----')

print('Total patients {} Total DCM files {}'.format(len(os.listdir('../input/sample_images')), 

                                                      len(glob.glob('../input/sample_images/*/*.dcm'))))

import dicom

dcm = '../input/sample_images/0a38e7597ca26f9374f8ea2770ba870d/4ec5ef19b52ec06a819181e404d37038.dcm'

print('Filename: {}'.format(dcm))

dcm = dicom.read_file(dcm)

def dicom_to_image(filename):

    dcm = dicom.read_file(filename)

    img = dcm.pixel_array

    img[img == -2000] = 0

    return img

files = glob.glob('../input/sample_images/*/*.dcm')



f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))

for i in range(20):

    plots[i // 5, i % 5].axis('off')

    plots[i // 5, i % 5].imshow(dicom_to_image(np.random.choice(files)), cmap=plt.cm.bone)



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

pat = load_patient('0ddeb08e9c97227853422bd71a2a695e')
f, plots = plt.subplots(11, 10, sharex='all', sharey='all', figsize=(10, 11))

# matplotlib is drunk

#plt.title('Sorted Slices of Patient 0a38e7597ca26f9374f8ea2770ba870d - No cancer')

for i in range(110):

    plots[i // 10, i % 10].axis('off')

    plots[i // 10, i % 10].imshow(pat[i], cmap=plt.cm.bone)
for d in os.listdir('../input/stage1/'):

    print("Patient '{}' has {} scans".format(d, len(os.listdir('../input/stage1/' + d))))

print('----')

print('Total patients {} Total DCM files {}'.format(len(os.listdir('../input/stage1/')), 

                                                      len(glob.glob('../input/stage1/*/*.dcm'))))
