import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, cv2

import dicom

import pandas as pd

import matplotlib.pyplot as plt



PATH_BASE = '../input'

print('{} {}'.format('Files:', os.listdir(PATH_BASE)))

EXT_CSV = 'stage1_labels.csv'

EXT_SAMPLE_IMAGES = 'sample_images'

patients = os.listdir(os.path.join(PATH_BASE, EXT_SAMPLE_IMAGES))

patients.sort()

print('# Folders in sample_images: {}'.format(len(patients)))

labels = pd.read_csv(os.path.join(PATH_BASE, EXT_CSV))

print('# Patients: {}'.format(len(labels)))

print(labels.columns)
database = {}

for i, each_patient in enumerate(patients):

    if i < 5:

        try:

            scans = os.listdir(os.path.join(PATH_BASE, EXT_SAMPLE_IMAGES, each_patient))

            print('Patient {}: {} has {} scans'.format(i+1, each_patient, len(scans)))



            # Storing all the scans for each patient in a list

            dcms = []

            for j, each_scan in enumerate(scans):

                dcms.append(dicom.read_file(os.path.join(PATH_BASE, EXT_SAMPLE_IMAGES, each_patient, each_scan)))

            dcms.sort(key = lambda z: int(z.InstanceNumber))



            # Storint the images in a 3D array

            num_slices = len(dcms)

            h_w = dcms[0].Rows

            image_3d = np.zeros(shape=[num_slices, h_w, h_w])

            for k, each_slice in enumerate(dcms):

                img = np.array(each_slice.pixel_array, dtype=np.float16)

                img[img == -2000] = 0

                img = img * each_slice.RescaleSlope + each_slice.RescaleIntercept

                image_3d[k, :, :] = img



            # Getting the label for each patient

            each_patient_label = int(labels.loc[labels['id'] == each_patient]['cancer'])



            # Storing dicom info and labels in dictionary

            patient_info = {'dcm': dcms, '3d_image': image_3d, 'label': each_patient_label}

            database[each_patient] = patient_info

        except:

            print('Error in', each_patient)
print(database[list(database.keys())[0]]['dcm'][0])
patient_names = database.keys()

for patient_name in patient_names:

    print('Patient name: {}\nNumber of slices: {}\nPixel Spacing: {}\nSlice Thickness: {}\nLabel: {}\n'.format(patient_name, database[patient_name]['3d_image'].shape[0], database[patient_name]['dcm'][0].PixelSpacing, abs(database[patient_name]['dcm'][0].SliceLocation - database[patient_name]['dcm'][1].SliceLocation), database[patient_name]['label']))
# Credits Guido Zuidhof for the resample function. I took it from his kernel.

def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

    spacing = map(float, ([abs(scan[0].SliceLocation - scan[1].SliceLocation)] + scan[0].PixelSpacing))

    spacing = np.array(list(spacing))

    org_shape = image.shape

    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    print(real_resize_factor)

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    

    return org_shape, image, spacing, new_spacing
# Credits Guido Zuidhof for the resample function. I took it from his kernel.

import scipy.ndimage

for person in database.keys():

    test_scan = database[person]['dcm']

    test_images = database[person]['3d_image']

    l, a, m, b = resample(test_images, test_scan, [1, 1, 1])

    print(l, a.shape, m, b)

    print
THRESHOLD = -700

test_image = database['00cba091fa4ad62cc3200a657aeb957e']['3d_image']

test_slice1 = test_image[50]

print('Maximum value in image: {}'.format(test_slice1.max()))

print('Minimum value in image: {}'.format(test_slice1.min()))

test_slice1 = test_slice1 > THRESHOLD

plt.imshow(test_slice1, cmap=plt.cm.bone)

plt.show()



test_slice2 = test_image[60]

test_slice2 = test_slice2 > THRESHOLD

plt.imshow(test_slice2, cmap=plt.cm.bone)

plt.show()



test_slice3 = test_image[70]

test_slice3 = test_slice3 > THRESHOLD

plt.imshow(test_slice3, cmap=plt.cm.bone)

plt.show()