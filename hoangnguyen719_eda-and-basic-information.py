!conda install -c conda-forge gdcm -y

import os
from matplotlib import pyplot as plt, style
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import glob
import pydicom
import re
import scipy
from skimage import measure
from skimage.morphology import disk, closing, opening
from tqdm import tqdm
ROOT = '../input/rsna-str-pulmonary-embolism-detection/'
TRAIN_IMG = glob.glob(ROOT + 'train/*')
train = pd.read_csv(ROOT + 'train.csv')
test = pd.read_csv(ROOT + 'test.csv')
def add_dcmpath(df, data='train', root=ROOT):
    df['dcm_path'] = root + data + '/' + df.StudyInstanceUID + '/' + df.SeriesInstanceUID
add_dcmpath(train)
add_dcmpath(test)
train.head(2)
train_id_agg = train.iloc[:, :3].groupby("StudyInstanceUID").nunique()
train_id_agg.agg(['max', 'min'])
test_id_agg = test.iloc[:, :3].groupby("SeriesInstanceUID").nunique()
test_id_agg.agg(['max', 'min'])
print(train.StudyInstanceUID.nunique())
print(train.SeriesInstanceUID.nunique())
print(test.StudyInstanceUID.nunique())
print(test.SeriesInstanceUID.nunique())
qa = ['qa_motion', 'qa_contrast', 'indeterminate']
def counting(cols):
    return train.groupby(cols).size().reset_index(name='count')
counting(qa)
conclusion = ['pe_present_on_image', 'negative_exam_for_pe', 'indeterminate']
counting(conclusion)
rv_lv = ['rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']
counting(rv_lv)
positional = ['leftsided_pe', 'rightsided_pe', 'central_pe']
counting(positional)
acute_chronic = ['chronic_pe', 'acute_and_chronic_pe']
counting(acute_chronic + ['negative_exam_for_pe'])
def load_scans(path): 
    slices = [pydicom.dcmread(path + '/' + file) for file in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) # sort by the z position of the image
    return slices
example_scans = load_scans(train.dcm_path[0])
plt.style.use('default')
fig, ax = plt.subplots(1, 2, figsize=(13,5))
n = 15
for i in range(n):
    scan = example_scans[i]
    image = scan.pixel_array.flatten()
    # Transforming image data to Hounsfield unit for comparability
    # since images from different CT system can have different measurements
    rescaled_image = image * scan.RescaleSlope + scan.RescaleIntercept 
    
    sns.distplot(image, ax=ax[0])
    sns.distplot(rescaled_image, ax=ax[1])

ax[0].set_title(f'Raw image data distributions for {n} examples')
ax[1].set_title(f'Hounsfield unit distribution for {n} examples')
plt.show()
# Convert to Hounsfield unit
def to_hu(dicoms):
    images = np.stack([image.pixel_array for image in dicoms])
    images = images.astype(np.int16)
    
#     # Convert outside pixels (of circular images) to air's Hu value
#     images[images <= -1000] = 0
    
    # Convert to HU
    for i in range(len(images)):
        intercept = dicoms[i].RescaleIntercept
        slope = dicoms[i].RescaleSlope
        if slope != 1:
            images[i] = slope * images.astype(np.float64)
            images[i] = images[i].astype(np.in16)
        images[i] += np.int16(intercept)
    # Convert outside pixels (of circular images) to air's Hu value
    images[images <= -1000] = -1000
    return images.astype(np.int16)
hu_scans = to_hu(example_scans)
fig, ax = plt.subplots(2, 2, figsize=(7,7))
ax[0, 0].set_title('Original CT Scan')
ax[0, 0].axis('off')
ax[0, 0].imshow(example_scans[0].pixel_array, cmap='gray')
ax[0, 1].set_title('Original pixel array\' distribution')
sns.distplot(example_scans[0].pixel_array.flatten(), ax=ax[0, 1])

ax[1, 0].set_title('CT Scan in Hu')
ax[1, 0].axis('off')
ax[1, 0].imshow(hu_scans[0], cmap='gray')
ax[1, 1].set_title('Hu values distribution')
sns.distplot(hu_scans[0].flatten(), ax=ax[1, 1])

plt.show()
N = 200
def get_window_value(window):
    if type(window) == pydicom.multival.MultiValue:
        return np.int(window[0])
    else:
        return np.int(window)

patient_id = []
row_values= []
col_values = []
pixelspacing_r = []
pixelspacing_c = []
slice_thickness = []
patient_pth = []
window_widths = []
window_centers = []

np.random.seed(713)
ids = np.random.randint(0, len(train), N)
patients = train.loc[ids, 'SeriesInstanceUID']
for patient in tqdm(patients):
    path = train[train.SeriesInstanceUID == patient].dcm_path.values[0]
    patient_pth.append(path)
    dcom_name = os.listdir(path)[0]
    dcom_file = pydicom.dcmread(path + '/' + dcom_name)
    
    row_values.append(dcom_file.Rows)
    col_values.append(dcom_file.Columns)
    
    pixelspacing_r.append(dcom_file.PixelSpacing[0])
    pixelspacing_c.append(dcom_file.PixelSpacing[1])
    
    window_widths.append(get_window_value(dcom_file.WindowWidth))
    window_centers.append(get_window_value(dcom_file.WindowCenter))
    
    slice_thickness.append(float(dcom_file.SliceThickness))

example_prop = pd.DataFrame()
example_prop['patients'] = patients
example_prop['row_values'] = row_values
example_prop['col_values'] =  col_values
example_prop['area'] = example_prop.row_values * example_prop.col_values
example_prop['pixelspacing_r'] = pixelspacing_r
example_prop['pixelspacing_c'] = pixelspacing_c
example_prop['slice_thickness'] = slice_thickness
example_prop['patient_pth'] = patient_pth
example_prop['window_width'] = window_widths
example_prop['window_center'] = window_centers
example_prop.head()
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
sns.distplot(example_prop.pixelspacing_r, ax=ax[0, 0])
ax[0, 0].set_title('Pixel-spacing distribution\nin row-direction', size=10)
ax[0, 0].set_xlabel('mm')

sns.distplot(example_prop.pixelspacing_c, ax=ax[0, 1])
ax[0, 1].set_title('Pixel-spacing distribution\nin column-direction', size=10)
ax[0, 1].set_xlabel('mm')

reso = example_prop.groupby(["row_values", "col_values"]).size()
reso = reso.reset_index(name='counts')
sns.scatterplot(x=reso.col_values, y=reso.row_values, s=reso.counts, ax=ax[1, 0])
ax[1, 0].set_title('Pixel sizes of images\' rows and columns', size=10)
ax[1, 0].set_xlabel('Column')
ax[1, 0].set_ylabel('Row')

sns.distplot(example_prop.slice_thickness, ax=ax[1, 1], kde=False)
ax[1, 1].set_title('Slice thickness distribution', size=10)
ax[1, 1].set_xlabel('mm')

plt.tight_layout()
plt.show()
example_prop['phys_distance_r'] = example_prop.pixelspacing_r * example_prop.row_values
example_prop['phys_distance_c'] = example_prop.pixelspacing_c * example_prop.col_values
example_prop['phys_area'] = example_prop.phys_distance_r * example_prop.phys_distance_c
example_prop['phys_vol'] = example_prop.phys_area * example_prop.slice_thickness

fig, ax = plt.subplots(1, 2, figsize=(9,4))
sns.distplot(example_prop.phys_area/1e2, ax=ax[0])
ax[0].set_title('Distribution of physical distance\ncovered by a CT-slice', size=10)
ax[0].set_xlabel('cm^2')

sns.distplot(example_prop.phys_vol/1e3, ax=ax[1])
ax[1].set_title('Distribution of physical volume\ncovered by a CT-slice', size=10)
ax[1].set_xlabel('cm^3')

plt.tight_layout()
plt.show()
import pydicom
def custom_window(pixel_array, center, width):
    pixel_array = np.array(pixel_array.copy())
    lower_bound = center - width/2
    upper_bound = center + width/2
    pixel_array[pixel_array < lower_bound] = lower_bound
    pixel_array[pixel_array > upper_bound] = upper_bound
    return pixel_array

def plot_biggest_smallest(plot_area=True, center=-50, width=300): # if False then plot volume
    if plot_area:
        series = example_prop.phys_area
    else:
        series = example_prop.phys_vol
        
    biggest = example_prop[series == series.max()]
    biggest = biggest.patient_pth.values[0]
    biggest = pydicom.dcmread(biggest + '/' + os.listdir(biggest)[0])
    

    smallest = example_prop[series == series.min()]
    smallest = smallest.patient_pth.values[0]
    smallest = pydicom.dcmread(smallest + '/' + os.listdir(smallest)[0])
    
    biggest, smallest = to_hu([biggest, smallest])
    biggest_window = custom_window(biggest, center, width)
    smallest_window = custom_window(smallest, center, width)

    fig, ax = plt.subplots(2, 2, figsize=(7,7))
    title = 'area' if plot_area else 'volume'
    ax[0, 0].imshow(biggest_window, cmap='gray')
    ax[0, 0].set_title(f'CT-scan with biggest physical {title}', size=10)
    ax[0, 0].axis('off')
    
    sns.distplot(biggest.flatten(), ax=ax[1,0])

    ax[0, 1].imshow(smallest_window, cmap='gray')
    ax[0, 1].set_title(f'CT-scan with smallest physical {title}', size=10)
    ax[0, 1].axis('off')
    
    sns.distplot(smallest.flatten(), ax=ax[1,1])
    
    fig.suptitle('Image and distribution of Hu')
    plt.tight_layout()
    plt.show()
plot_biggest_smallest()
plot_biggest_smallest(False, 0, 300)
example_dcm = load_scans(example_prop.patient_pth.values[0])
example_imgs = to_hu(example_dcm)

plt.figure(figsize=(5,5))
plt.title('Distribution of pixels\' Hu')
sns.distplot(example_imgs.flatten(), norm_hist=True)
def plot_3d(image, threshold=-300, color='navy'):
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1, projection='3d')
    
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.2)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    plt.show()
def resample(image, scan, new_spacing=[1,1,1]):
    # Current spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = new_real_shape.round()
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing
new_example_imgs, new_imgs_spacing = resample(example_imgs, example_dcm)
print(example_imgs.shape)
print(new_example_imgs.shape)
def segment_lung_mask(images, threshold=-320, selem=disk(4)):
    segmented = np.zeros(images.shape)
    
    for i, image in enumerate(images):
        # Separate lung and air from the rest: lung/air=1, others=2
        binary = np.array(image > threshold, dtype=np.int8) + 1
        
        # Segment using connected component analysis
        labeling = measure.label(binary)
        # Convert all edge-labels (labels of pixels on the furthest left/right/top/bottom)
        # to others. These edge-labels should all be non-lung
        bg_labelings = np.unique(
            [labeling[0,:], labeling[-1,:], labeling[:,0], labeling[:,-1]]
        )
        for bg in bg_labelings:
            binary[labeling == bg] = 2 # now lung=1, non-lung=2
            
        # Revert: lung=1, non-lung=0
        binary -= 1
        binary = 1 - binary
        
        # Remove air-pocket inside lung
        if selem is not None:
            binary = closing(binary, selem)
        
        segmented[i] = binary * image
    
    return segmented
segmented_example_imgs = segment_lung_mask(example_imgs)
ex_idx = 90
c = -500
w = 255

fig, ax = plt.subplots(1,2, figsize=(9,4))

ax[0].imshow(custom_window(example_imgs[ex_idx], c, w),  cmap='gray')
ax[0].axis('off')
ax[0].set_title('Original image')

ax[1].imshow(custom_window(segmented_example_imgs[ex_idx], c, w), cmap='gray')
ax[1].axis('off')
ax[1].set_title('Segmented image')
segmented_example_imgs.shape
study_columns = ['StudyInstanceUID', 'SeriesInstanceUID'
                 , 'negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1'
                 ,'leftsided_pe', 'chronic_pe', 'true_filling_defect_not_pe'
                 ,'rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']
diagnosis = ['negative_exam_for_pe', 'indeterminate', 'positive']
rv_lv_ratio = ['rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']
leftsided_pe = ['leftsided_pe']
rightsided_pe = ['rightsided_pe']
central_pe = ['central_pe']
chronic_acute = ['chronic_pe', 'acute_pe', 'acute_and_chronic_pe']

train_study_level = train[study_columns + ['SOPInstanceUID']]
train_study_level = train_study_level.groupby(study_columns).agg('count').reset_index()
train_study_level.columns = study_columns + ['scan_count']
train_study_level['positive'] = np.where((train_study_level.negative_exam_for_pe == 0)
                                         & (train_study_level.indeterminate == 0), 1, 0)
train_study_level['acute_pe'] = np.where((train_study_level.chronic_pe == 0)
                                         & (train_study_level.positive == 1)
                                         & (train_study_level.acute_and_chronic_pe == 0), 1, 0)
study_count = len(train_study_level)
label_group = [diagnosis, rv_lv_ratio, leftsided_pe
               , rightsided_pe, central_pe, chronic_acute]
fig, ax = plt.subplots(len(label_group), 1, figsize=(5, 20))
min_bin = train_study_level.scan_count.min()
max_bin = train_study_level.scan_count.max()
plt.setp(ax, xlim=(min_bin, max_bin))
for i in range(len(label_group)):
    labels = label_group[i]
    for label in labels:
        count = sum(train_study_level[label])
        pct = round(count / study_count * 100, 2)
        print(label, f': {count}/{study_count} ({pct}%)')
    bins = np.linspace(min_bin, max_bin, 20)
    for label in labels:
        train_study_level[train_study_level[label]==1].hist(column='scan_count'
                                                            , grid=False, bins=bins
                                                            , ax=ax[i], label=label
                                                            , alpha=0.5
                                                            , density=True)
    ax[i].legend()
plt.show()
!pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

columns = [c for cols in label_group for c in cols]
kfolds = pd.DataFrame(columns = ['Fold', 'Size'] + columns)
MSKF = MultilabelStratifiedKFold(n_splits=15)
f = 0
for _, test_idx in MSKF.split(X=np.zeros(len(train_study_level)), y=train_study_level[columns]):
    fold = train_study_level.iloc[test_idx, :]
    size = len(fold)
    row = [f, size]
    for c in columns:
        row.append(round(fold[c].sum()/size * 100,2))
    kfolds.loc[f] = row
    f += 1

kfolds
plot_3d(segmented_example_imgs, threshold=-600)