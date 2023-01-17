import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import dicom

import glob #finds all the pathnames matching a specified pattern

from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

%matplotlib inline

p = sns.color_palette()



os.listdir('../input')
PathDicom = "../input"

lstFilesDCM = []  # create an empty list

for dirName, subdirList, fileList in os.walk(PathDicom):

    for filename in fileList:

        if ".dcm" in filename.lower():  # check whether the file's DICOM

            lstFilesDCM.append(os.path.join(dirName,filename))

print(lstFilesDCM[0:5])

print(len(lstFilesDCM))
# Get ref file

RefDs = dicom.read_file(lstFilesDCM[0])



# Load dimensions based on the number of rows, columns, and slices (along the Z axis)

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))



# Load spacing values (in mm)

ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceLocation))

ConstPixelSpacing
for d in os.listdir('../input/sample_images'):

    print("Patient '{}' has {} scans".format(d, len(os.listdir('../input/sample_images/' + d))))

print('----')

print('Total patients {} Total DCM files {}'.format(len(os.listdir('../input/sample_images')), 

                                                      len(glob.glob('../input/sample_images/*/*.dcm'))))
patient_sizes = [len(os.listdir('../input/sample_images/' + d)) for d in os.listdir('../input/sample_images')]

plt.hist(patient_sizes, color=p[2])

plt.ylabel('Number of patients')

plt.xlabel('DICOM files')

plt.title('Histogram of DICOM count per patient')
sizes = [os.path.getsize(dcm)/1000000 for dcm in glob.glob('../input/sample_images/*/*.dcm')]

print('DCM file sizes: min {:.3}MB max {:.3}MB avg {:.3}MB std {:.3}MB'.format(np.min(sizes), 

                                                       np.max(sizes), np.mean(sizes), np.std(sizes)))
dcm = '../input/sample_images/0a38e7597ca26f9374f8ea2770ba870d/4ec5ef19b52ec06a819181e404d37038.dcm'

print('Filename: {}'.format(dcm))

dcm = dicom.read_file(dcm)
dcm
df_train = pd.read_csv('../input/stage1_labels.csv')

df_train.head()

print('Number of training patients: {}'.format(len(df_train)))

print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
from sklearn.metrics import log_loss

logloss = log_loss(df_train.cancer, np.zeros_like(df_train.cancer) + df_train.cancer.mean())

print('Training logloss is {}'.format(logloss)) 
sample = pd.read_csv('../input/stage1_sample_submission.csv')

#sample['cancer'] = df_train.cancer.mean()+0.001

#sample.to_csv('naive_submission.csv', index=False) # LB 0.60235 

sample['cancer'] = df_train.cancer.mean()+0.003

sample.to_csv('submission1.csv', index=False) # LB 
sample.shape
img = dcm.pixel_array

img[img == -2000] = 0



plt.axis('off')

plt.imshow(img)

plt.show()



plt.axis('off')

plt.imshow(-img) # Invert colors with -

plt.show()
def dicom_to_image(filename):

    dcm = dicom.read_file(filename)

    img = dcm.pixel_array

    img[img == -2000] = 0

    return img
files = glob.glob('../input/sample_images/*/*.dcm')



f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))

for i in range(20):

    plots[i // 5, i % 5].axis('off')

    plots[i // 5, i % 5].imshow(dicom_to_image(np.random.choice(files)))
def get_slice_location(dcm):

    return float(dcm[0x0020, 0x1041].value)



# Returns a list of images for that patient_id, in ascending order of Slice Location

def sort_patient(patient_id):

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
x=sort_patient('0a38e7597ca26f9374f8ea2770ba870d')
f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))

for i in range(20):

    plots[i // 5, i % 5].axis('off')

    plots[i // 5, i % 5].imshow(x[i])