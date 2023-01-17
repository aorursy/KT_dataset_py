!pip install natsort



import os

import numpy as np

import SimpleITK as sitk

import matplotlib.pyplot as plt

from natsort import natsorted
path_train = os.path.join('../input/osic-pulmonary-fibrosis-progression', 'train')

ptns_train = [os.path.join(path_train, _) for _ in os.listdir(path_train)]
ptns_train[:10]
tot_len = []



for ptn in ptns_train:

    dcmlist = os.listdir(ptn)

    tot_len.append(len(dcmlist))
plt.hist(tot_len, bins=20)

plt.show()
slice_thickness = []



for ptn in ptns_train:

    exdcm = [os.path.join(ptn, _) for _ in os.listdir(ptn)][0]

    spacing = sitk.ReadImage(exdcm).GetSpacing()

    slice_thickness.append(spacing[2])

plt.hist(slice_thickness, bins=20)

plt.show()
slice_interval = []

spacing_list = []



import pydicom



for ptn in ptns_train:

    try:

        exdcm1 = natsorted([os.path.join(ptn, _) for _ in os.listdir(ptn)])[0]

        exdcm2 = natsorted([os.path.join(ptn, _) for _ in os.listdir(ptn)])[1]

        location1 = sitk.ReadImage(exdcm1).GetMetaData('0020|0032').split('\\')[2]

        location2 = sitk.ReadImage(exdcm2).GetMetaData('0020|0032').split('\\')[2]

        spacing = sitk.ReadImage(exdcm1).GetSpacing()[0]

        spacing_list.append(spacing)

        interval = np.abs(float(location2) - float(location1))

        slice_interval.append(interval)

    except:

        print(ptn)

    slice_thickness.append(interval)



print(len(slice_interval))
plt.title("Physical size of each pixel")

plt.hist(spacing_list, bins=20)

plt.show()
plt.title("Distance between slices")

plt.hist(slice_interval, bins=20)

plt.show()
ptn1_path = os.path.join(path_train, 'ID00078637202199415319443')

ptn1_dcm = [os.path.join(path_train, 'ID00078637202199415319443', _) for _ in natsorted(os.listdir(ptn1_path))][0]

npy1 = sitk.GetArrayFromImage(sitk.ReadImage(ptn1_dcm)).squeeze()

plt.title('Patient 1')

plt.imshow(npy1, 'gray')

plt.show()



ptn2_path = os.path.join(path_train, 'ID00128637202219474716089')

ptn2_dcm = [os.path.join(path_train, 'ID00128637202219474716089', _) for _ in natsorted(os.listdir(ptn2_path))][0]

npy2 = sitk.GetArrayFromImage(sitk.ReadImage(ptn2_dcm)).squeeze()

plt.title("Patient 2")

plt.imshow(npy2, 'gray')

plt.show()
print(npy1.min(), npy2.min())
plt.title("Histogram of Patient 1")

plt.hist(npy1.flatten(), bins=20)

plt.show()



plt.title("Histogram of Patient 2")

plt.hist(npy2.flatten(), bins=20)

plt.show()
less_than_3000 = []

bigger_than_3000 = []



for ptn in ptns_train:

    exdcm = natsorted([os.path.join(ptn, _) for _ in os.listdir(ptn)])[0]

    if sitk.GetArrayFromImage(sitk.ReadImage(exdcm)).min()<-3000:

        less_than_3000.append(ptn)

    else:

        bigger_than_3000.append(ptn)

    

print("Number of DICOMS that have HU less than -3000:", len(less_than_3000))

print("Else:", len(bigger_than_3000))