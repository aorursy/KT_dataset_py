# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import pydicom

import scipy.ndimage



from skimage import measure 

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



from IPython.display import HTML

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from os import listdir



basepath = "../input/osic-pulmonary-fibrosis-progression/"

listdir(basepath)
train = pd.read_csv(basepath + "train.csv")

test = pd.read_csv(basepath + "test.csv")



def load_scans(dcm_path):

    slices = [pydicom.dcmread(dcm_path + "/" + file) for file in listdir(dcm_path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    return slices
print( train)

print( train.Patient.values)

print( train.Patient.values[0])

#../input/osic-pulmonary-fibrosis-progression/train

#../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/1.dcm

example = basepath + "train/" + train.Patient.values[0]

scans = load_scans(example)



scans[0]
print(scans[0].PatientID)

#医療機器製造会社

print(scans[0].Manufacturer)

#スライス厚さ

print(scans[0][0x0018,0x0050],"mm")



ct_image = scans[0].pixel_array

plt.imshow(ct_image)





#../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/10.dcm
import math

ax=len(scans)

x=int(math.sqrt(ax))

y=x+1



print(x,y,ax)


"""

ax=len(scans)

x=int(math.sqrt(ax))

y=x+1





fig=plt.figure(figsize=(12, 12))

for i in range(ax):

    ct_image = scans[i].pixel_array

    fig.add_subplot(x,y,i)



    plt.imshow(scans[i].pixel_array, cmap='gray')#グレースケールで出力

    

plt.show()

"""

def transform_to_hu(slices):

    images = np.stack([file.pixel_array for file in slices])

    images = images.astype(np.int16)



    # convert ouside pixel-values to air:

    # I'm using <= -1000 to be sure that other defaults are captured as well

    images[images <= -1000] = 0

    

    # convert to HU

    for n in range(len(slices)):

        

        intercept = slices[n].RescaleIntercept

        slope = slices[n].RescaleSlope

        

        if slope != 1:

            images[n] = slope * images[n].astype(np.float64)

            images[n] = images[n].astype(np.int16)

            

        images[n] += np.int16(intercept)

    

    return np.array(images, dtype=np.int16)

hu_scans = transform_to_hu(scans)
fig, ax = plt.subplots(1,4,figsize=(20,3))

ax[0].set_title("Original CT-scan")

ax[0].imshow(scans[0].pixel_array, cmap="bone")

ax[1].set_title("Pixelarray distribution");

sns.distplot(scans[0].pixel_array.flatten(), ax=ax[1]);



ax[2].set_title("CT-scan in HU")

ax[2].imshow(hu_scans[0], cmap="bone")

ax[3].set_title("HU values distribution");

sns.distplot(hu_scans[0].flatten(), ax=ax[3]);



for m in [0,2]:

    ax[m].grid(False)
def get_window_value(feature):

    if type(feature) == pydicom.multival.MultiValue:

        return np.int(feature[0])

    else:

        return np.int(feature)



pixelspacing_r = []

pixelspacing_c = []

slice_thicknesses = []

patient_id = []

patient_pth = []

row_values = []

column_values = []

window_widths = []

window_levels = []



for patient in train.Patient.values:

    patient_id.append(patient)

    example_dcm = listdir(basepath + "train/" + patient + "/")[0]

    patient_pth.append(basepath + "train/" + patient)

    dataset = pydicom.dcmread(basepath + "train/" + patient + "/" + example_dcm)

    

    window_widths.append(get_window_value(dataset.WindowWidth))

    window_levels.append(get_window_value(dataset.WindowCenter))

    

    spacing = dataset.PixelSpacing

    slice_thicknesses.append(dataset.SliceThickness)

    

    row_values.append(dataset.Rows)

    column_values.append(dataset.Columns)

    pixelspacing_r.append(spacing[0])

    pixelspacing_c.append(spacing[1])

    

scan_properties = pd.DataFrame(data=patient_id, columns=["patient"])

scan_properties.loc[:, "rows"] = row_values

scan_properties.loc[:, "columns"] = column_values

scan_properties.loc[:, "area"] = scan_properties["rows"] * scan_properties["columns"]

scan_properties.loc[:, "pixelspacing_r"] = pixelspacing_r

scan_properties.loc[:, "pixelspacing_c"] = pixelspacing_c

scan_properties.loc[:, "pixelspacing_area"] = scan_properties.pixelspacing_r * scan_properties.pixelspacing_c

scan_properties.loc[:, "slice_thickness"] = slice_thicknesses

scan_properties.loc[:, "patient_pth"] = patient_pth

scan_properties.loc[:, "window_width"] = window_widths

scan_properties.loc[:, "window_level"] = window_levels





scan_properties.head()

fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(pixelspacing_r, ax=ax[0], color="Limegreen", kde=False)

ax[0].set_title("Pixel spacing distribution \n in row direction ")

ax[0].set_ylabel("Counts in train")

ax[0].set_xlabel("mm")

sns.distplot(pixelspacing_c, ax=ax[1], color="Mediumseagreen", kde=False)

ax[1].set_title("Pixel spacing distribution \n in column direction");

ax[1].set_ylabel("Counts in train");

ax[1].set_xlabel("mm");
counts = scan_properties.groupby(["rows", "columns"]).size()

counts = counts.unstack()

counts.fillna(0, inplace=True)





fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(slice_thicknesses, color="orangered", kde=False, ax=ax[0])

ax[0].set_title("Slice thicknesses of all patients");

ax[0].set_xlabel("Slice thickness in mm")

ax[0].set_ylabel("Counts in train");



for n in counts.index.values:

    for m in counts.columns.values:

        ax[1].scatter(n, m, s=counts.loc[n,m], c="midnightblue")

ax[1].set_xlabel("rows")

ax[1].set_ylabel("columns")

ax[1].set_title("Pixel area of ct-scan per patient");
scan_properties["r_distance"] = scan_properties.pixelspacing_r * scan_properties.rows

scan_properties["c_distance"] = scan_properties.pixelspacing_c * scan_properties["columns"]

scan_properties["area_cm2"] = 0.1* scan_properties["r_distance"] * 0.1*scan_properties["c_distance"]

scan_properties["slice_volume_cm3"] = 0.1*scan_properties.slice_thickness * scan_properties.area_cm2



fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(scan_properties.area_cm2, ax=ax[0], color="purple")

sns.distplot(scan_properties.slice_volume_cm3, ax=ax[1], color="magenta")

ax[0].set_title("CT-slice area in $cm^{2}$")

ax[1].set_title("CT-slice volume in $cm^{3}$")

ax[0].set_xlabel("$cm^{2}$")

ax[1].set_xlabel("$cm^{3}$");
max_path = scan_properties[

    scan_properties.area_cm2 == scan_properties.area_cm2.max()].patient_pth.values[0]

min_path = scan_properties[

    scan_properties.area_cm2 == scan_properties.area_cm2.min()].patient_pth.values[0]



min_scans = load_scans(min_path)

min_hu_scans = transform_to_hu(min_scans)



max_scans = load_scans(max_path)

max_hu_scans = transform_to_hu(max_scans)





def set_manual_window(hu_image, custom_center, custom_width):

    w_image = hu_image.copy()

    min_value = custom_center - (custom_width/2)

    max_value = custom_center + (custom_width/2)

    w_image[w_image < min_value] = min_value

    w_image[w_image > max_value] = max_value

    return w_image





fig, ax = plt.subplots(1,2,figsize=(20,10))

ax[0].imshow(set_manual_window(min_hu_scans[np.int(len(min_hu_scans)/2)], -500, 1000), cmap="YlGnBu")

ax[1].imshow(set_manual_window(max_hu_scans[np.int(len(max_hu_scans)/2)], -500, 1000), cmap="YlGnBu");

ax[0].set_title("CT-scan with small slice area")

ax[1].set_title("CT-scan with large slice area");

for n in range(2):

    ax[n].axis("off")
fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(max_hu_scans[np.int(len(max_hu_scans)/2)].flatten(), kde=False, ax=ax[1])

ax[1].set_title("Large area image")

sns.distplot(min_hu_scans[np.int(len(min_hu_scans)/2)].flatten(), kde=False, ax=ax[0])

ax[0].set_title("Small area image")

ax[0].set_xlabel("HU values")

ax[1].set_xlabel("HU values");
max_path = scan_properties[

    scan_properties.slice_volume_cm3 == scan_properties.slice_volume_cm3.max()].patient_pth.values[0]

min_path = scan_properties[

    scan_properties.slice_volume_cm3 == scan_properties.slice_volume_cm3.min()].patient_pth.values[0]



min_scans = load_scans(min_path)

min_hu_scans = transform_to_hu(min_scans)



max_scans = load_scans(max_path)

max_hu_scans = transform_to_hu(max_scans)





fig, ax = plt.subplots(1,2,figsize=(20,10))

ax[0].imshow(set_manual_window(min_hu_scans[np.int(len(min_hu_scans)/2)], -500, 1000), cmap="YlGnBu")

ax[1].imshow(set_manual_window(max_hu_scans[np.int(len(max_hu_scans)/2)], -500, 1000), cmap="YlGnBu");

ax[0].set_title("CT-scan with small slice volume")

ax[1].set_title("CT-scan with large slice volume");

for n in range(2):

    ax[n].axis("off")
fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(max_hu_scans[np.int(len(max_hu_scans)/2)].flatten(), kde=False, ax=ax[1])

ax[1].set_title("Large slice volume")

sns.distplot(min_hu_scans[np.int(len(min_hu_scans)/2)].flatten(), kde=False, ax=ax[0])

ax[0].set_title("Small slice volume")

ax[0].set_xlabel("HU values")

ax[1].set_xlabel("HU values");

def plot_3d(image, threshold=700, color="navy"):

    

    # Position the scan upright, 

    # so the head of the patient would be at the top facing the camera

    p = image.transpose(2,1,0)

    

    verts, faces,_,_ = measure.marching_cubes_lewiner(p, threshold)



    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')



    # Fancy indexing: `verts[faces]` to generate a collection of triangles

    mesh = Poly3DCollection(verts[faces], alpha=0.5)

    mesh.set_facecolor(color)

    ax.add_collection3d(mesh)



    ax.set_xlim(0, p.shape[0])

    ax.set_ylim(0, p.shape[1])

    ax.set_zlim(0, p.shape[2])



    plt.show()

    

    

plot_3d(max_hu_scans)
old_distribution = max_hu_scans.flatten()



example = basepath + "train/" + train.Patient.values[0]

scans = load_scans(example)

hu_scans = transform_to_hu(scans)



plot_3d(hu_scans)
plt.figure(figsize=(20,5))

sns.distplot(old_distribution, label="weak 3d plot")

sns.distplot(hu_scans.flatten(), label="strong 3d plot")

plt.title("HU value distribution")

plt.legend();
print(len(max_hu_scans), len(hu_scans))
def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)



    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    

    return image, new_spacing
pix_resampled, spacing = resample(max_hu_scans, scans, [1,1,1])

print("Shape before resampling\t", hu_scans.shape)

print("Shape after resampling\t", pix_resampled.shape)
# Guidos code:



def largest_label_volume(im, bg=-1):

    vals, counts = np.unique(im, return_counts=True)



    counts = counts[vals != bg]

    vals = vals[vals != bg]



    if len(counts) > 0:

        return vals[np.argmax(counts)]

    else:

        return None

    

def segment_lung_mask(image, fill_lung_structures=True):

    

    # not actually binary, but 1 and 2. 

    # 0 is treated as background, which we do not want

    binary_image = np.array(image > -320, dtype=np.int8)+1

    labels = measure.label(binary_image)

    

    # Pick the pixel in the very corner to determine which label is air.

    #   Improvement: Pick multiple background labels from around the patient

    #   More resistant to "trays" on which the patient lays cutting the air 

    #   around the person in half

    background_label_1 = labels[0,0]

    background_label_2 = labels[0,-1]

    background_label_3 = labels[-1,0]

    background_label_4 = labels[-1,-1]

    

    #Fill the air around the person

    binary_image[background_label_1 == labels] = 2

    binary_image[background_label_2 == labels] = 2

    binary_image[background_label_3 == labels] = 2

    binary_image[background_label_4 == labels] = 2

    

    # Method of filling the lung structures (that is superior to something like 

    # morphological closing)

    if fill_lung_structures:

        # For every slice we determine the largest solid structure

        for i, axial_slice in enumerate(binary_image):

            axial_slice = axial_slice - 1

            labeling = measure.label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)

            

            if l_max is not None: #This slice contains some lung

                binary_image[i][labeling != l_max] = 1



    

    binary_image -= 1 #Make the image actual binary

    binary_image = 1-binary_image # Invert it, lungs are now 1

    

    # Remove other air pockets insided body

    labels = measure.label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None: # There are air pockets

        binary_image[labels != l_max] = 0

 

    return binary_image
binary_image = np.array((hu_scans[20]>-320), dtype=np.int8) + 1

np.unique(binary_image)
labels = measure.label(binary_image)



background_label_1 = labels[0,0]

background_label_2 = labels[0,-1]

background_label_3 = labels[-1,0]

background_label_4 = labels[-1,-1]
binary_image_2 = binary_image.copy()

binary_image_2[background_label_1 == labels] = 2

binary_image_2[background_label_2 == labels] = 2

binary_image_2[background_label_3 == labels] = 2

binary_image_2[background_label_4 == labels] = 2
fig, ax = plt.subplots(1,4,figsize=(20,5))

ax[0].imshow(binary_image, cmap="binary", interpolation='nearest')

ax[1].imshow(labels, cmap="jet", interpolation='nearest')

ax[2].imshow(binary_image_2, cmap="binary", interpolation='nearest')

ax[0].set_title("Binary image")

ax[1].set_title("Labelled image");
segmented_lungs = segment_lung_mask(hu_scans, fill_lung_structures=False)

#segmented_lungs_fill = segment_lung_mask(hu_scans, fill_lung_structures=True)
plt.imshow(segmented_lungs[20])