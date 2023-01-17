!conda install -c conda-forge gdcm -y
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import pydicom
import scipy.ndimage
import gdcm

from skimage import measure 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.morphology import disk, opening, closing
from tqdm import tqdm

from IPython.display import HTML
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from os import listdir,mkdir
listdir("../input/")
#basepath = "../input/osic-pulmonary-fibrosis-progression/"
# or if you are taking part in RSNA pulmonary embolism detection:
basepath = "../input/rsna-str-pulmonary-embolism-detection/"
listdir(basepath)
train = pd.read_csv(basepath + "train.csv")
test = pd.read_csv(basepath + "test.csv")
train.shape
train.head()
if basepath == "../input/osic-pulmonary-fibrosis-progression/":
    train["dcm_path"] = basepath + "train/" + train.Patient + "/"
else:
    train["dcm_path"] = basepath + "train/" + train.StudyInstanceUID + "/" + train.SeriesInstanceUID  
def load_scans(dcm_path):
    if basepath == "../input/osic-pulmonary-fibrosis-progression/":
        # in this competition we have missing values in ImagePosition, this is why we are sorting by filename number
        files = listdir(dcm_path)
        file_nums = [np.int(file.split(".")[0]) for file in files]
        sorted_file_nums = np.sort(file_nums)[::-1]
        slices = [pydicom.dcmread(dcm_path + "/" + str(file_num) + ".dcm" ) for file_num in sorted_file_nums]
    else:
        # otherwise we sort by ImagePositionPatient (z-coordinate) or by SliceLocation
        slices = [pydicom.dcmread(dcm_path + "/" + file) for file in listdir(dcm_path)]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    return slices
example = train.dcm_path.values[0]
scans = load_scans(example)
scans[0]
HTML('<iframe width="600" height="400" src="https://www.youtube.com/embed/KZld-5W99cI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
fig, ax = plt.subplots(1,2,figsize=(20,5))
for n in range(10):
    image = scans[n].pixel_array.flatten()
    rescaled_image = image * scans[n].RescaleSlope + scans[n].RescaleIntercept
    sns.distplot(image.flatten(), ax=ax[0]);
    sns.distplot(rescaled_image.flatten(), ax=ax[1])
ax[0].set_title("Raw pixel array distributions for 10 examples")
ax[1].set_title("HU unit distributions for 10 examples");
def set_outside_scanner_to_air(raw_pixelarrays):
    # in OSIC we find outside-scanner-regions with raw-values of -2000. 
    # Let's threshold between air (0) and this default (-2000) using -1000
    raw_pixelarrays[raw_pixelarrays <= -1000] = 0
    return raw_pixelarrays
def transform_to_hu(slices):
    images = np.stack([file.pixel_array for file in slices])
    images = images.astype(np.int16)

    images = set_outside_scanner_to_air(images)
    
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
N = 1000
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

if basepath == "../input/osic-pulmonary-fibrosis-progression/":
    patients = train.Patient.unique()[0:N]
else:
    patients = train.SeriesInstanceUID.unique()[0:N]

for patient in patients:
    patient_id.append(patient)
    if basepath == "../input/osic-pulmonary-fibrosis-progression/":
        path = train[train.Patient == patient].dcm_path.values[0]
    else:
        path = train[train.SeriesInstanceUID == patient].dcm_path.values[0]
    example_dcm = listdir(path)[0]
    patient_pth.append(path)
    dataset = pydicom.dcmread(path + "/" + example_dcm)
    
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

background_water_hu_scans = max_hu_scans.copy()
def set_manual_window(hu_image, custom_center, custom_width):
    w_image = hu_image.copy()
    min_value = custom_center - (custom_width/2)
    max_value = custom_center + (custom_width/2)
    w_image[w_image < min_value] = min_value
    w_image[w_image > max_value] = max_value
    return w_image
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(set_manual_window(min_hu_scans[np.int(len(min_hu_scans)/2)], -700, 255), cmap="YlGnBu")
ax[1].imshow(set_manual_window(max_hu_scans[np.int(len(max_hu_scans)/2)], -700, 255), cmap="YlGnBu");
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
ax[0].imshow(set_manual_window(min_hu_scans[np.int(len(min_hu_scans)/2)], -700, 255), cmap="YlGnBu")
ax[1].imshow(set_manual_window(max_hu_scans[np.int(len(max_hu_scans)/2)], -700, 255), cmap="YlGnBu");
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
    mesh = Poly3DCollection(verts[faces], alpha=0.2)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
plot_3d(max_hu_scans)
old_distribution = max_hu_scans.flatten()
example = train.dcm_path.values[0]
scans = load_scans(example)
hu_scans = transform_to_hu(scans)
plot_3d(hu_scans)
plt.figure(figsize=(20,5))
sns.distplot(old_distribution, label="weak 3d plot", kde=False)
sns.distplot(hu_scans.flatten(), label="strong 3d plot", kde=False)
plt.title("HU value distribution")
plt.legend();
print(len(max_hu_scans), len(hu_scans))
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)
    
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor)
    
    # recompute the resize factor and spacing such that we match the rounded new shape above
    rounded_resize_factor = new_shape / image.shape
    rounded_new_spacing = spacing / rounded_resize_factor
    
    # zoom with resize factor
    image = scipy.ndimage.interpolation.zoom(image, rounded_resize_factor, mode='nearest')
    
    return image, rounded_new_spacing
img_resampled, spacing = resample(max_hu_scans, scans, [1,1,1])
print("Shape before resampling\t", max_hu_scans.shape)
print("Shape after resampling\t", img_resampled.shape)
plot_3d(img_resampled)
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
    
def fill_lungs(binary_image):
    image = binary_image.copy()
    # For every slice we determine the largest solid structure
    for i, axial_slice in enumerate(image):
        axial_slice = axial_slice - 1
        labeling = measure.label(axial_slice)
        l_max = largest_label_volume(labeling, bg=0)

        if l_max is not None: #This slice contains some lung
            image[i][labeling != l_max] = 1
    return image

def segment_lung_mask(image):
    segmented = np.zeros(image.shape)   
    
    for n in range(image.shape[0]):
        binary_image = np.array(image[n] > -320, dtype=np.int8)+1
        labels = measure.label(binary_image)
        
        bad_labels = np.unique([labels[0,:], labels[-1,:], labels[:,0], labels[:,-1]])
        for bad_label in bad_labels:
            binary_image[labels == bad_label] = 2
    
        #We have a lot of remaining small signals outside of the lungs that need to be removed. 
        #In our competition closing is superior to fill_lungs 
        selem = disk(2)
        binary_image = opening(binary_image, selem)
    
        binary_image -= 1 #Make the image actual binary
        binary_image = 1-binary_image # Invert it, lungs are now 1
        
        segmented[n] = binary_image.copy() * image[n]
    
    return segmented
plt.figure(figsize=(20,5))
sns.distplot(hu_scans[20], kde=False)
plt.title("Example HU value distribution");
plt.xlabel("HU-value")
plt.ylabel("count")
binary_image = np.array((hu_scans[20]>-320), dtype=np.int8) + 1
np.unique(binary_image)
labels = measure.label(binary_image)

bad_labels = np.unique([labels[0,:], labels[-1,:], labels[:,0], labels[:,-1]])
binary_image_2 = binary_image.copy()
for bad_label in bad_labels:
    binary_image_2[labels == bad_label] = 2
fig, ax = plt.subplots(1,3,figsize=(20,7))
ax[0].imshow(binary_image, cmap="binary", interpolation='nearest')
ax[1].imshow(labels, cmap="jet", interpolation='nearest')
ax[2].imshow(binary_image_2, cmap="binary", interpolation='nearest')

ax[0].set_title("Binary image")
ax[1].set_title("Labelled image");
ax[2].set_title("Binary image - background removed");
selem = disk(2)
closed_binary_2 = closing(binary_image_2, selem)

closed_binary_2 -= 1 #Make the image actual binary
closed_binary_2 = 1-closed_binary_2 # Invert it, lungs are now 1
filled_lungs_binary = fill_lungs(binary_image_2)
air_pocket_binary = closed_binary_2.copy()
# Remove other air pockets insided body
labels_2 = measure.label(air_pocket_binary, background=0)
l_max = largest_label_volume(labels_2, bg=0)
if l_max is not None: # There are air pockets
    air_pocket_binary[labels_2 != l_max] = 0
fig, ax = plt.subplots(1,3,figsize=(20,7))

ax[0].imshow(closed_binary_2, cmap="binary", interpolation='nearest')
ax[1].imshow(filled_lungs_binary, cmap="binary", interpolation='nearest')
ax[2].imshow(air_pocket_binary, cmap="binary", interpolation='nearest')


ax[0].set_title("Morphological closing");
ax[1].set_title("Guidos filling lung structures");
ax[2].set_title("Guidos air pocket removal");
segmented = segment_lung_mask(np.array([hu_scans[20]]))

fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(set_manual_window(hu_scans[20], -700, 255), cmap="Blues_r")
ax[1].imshow(set_manual_window(segmented[0], -700, 255), cmap="Blues_r");
segmented_lungs = segment_lung_mask(hu_scans)
fig, ax = plt.subplots(6,5, figsize=(20,20))
for n in range(6):
    for m in range(5):
        ax[n,m].imshow(set_manual_window(segmented_lungs[(n+1)*5+m], -700, 255), cmap="Blues_r")
plot_3d(segmented_lungs, threshold=-600)
image_sizes = scan_properties.groupby(["rows", "columns"]).size().sort_values(ascending=False)
image_sizes
plt.figure(figsize=(8,8))
for n in counts.index.values:
    for m in counts.columns.values:
        plt.scatter(n, m, s=counts.loc[n,m], c="dodgerblue", alpha=0.7)
plt.xlabel("rows")
plt.ylabel("columns")
plt.title("Pixel area of ct-scan per patient");
plt.plot(np.arange(0,1400), '-.', c="purple", label="squared")
plt.plot(888 * np.ones(1400), '-.', c="crimson", label="888 rows");
plt.legend();
class ImageObserver:
    
    def __init__(self, scan_properties, batch_size):
        self.scan_properties = scan_properties
        self.batch_size = batch_size
    
    def select_group(self, group=(512,512)):
        self.group = group
        self.name = "rows {}, columns {}".format(group[0], group[1])
        self.batch_shape = (self.batch_size, group[0], group[1])
        self.selection = self.scan_properties[
            (self.scan_properties["rows"]==group[0]) & (self.scan_properties["columns"]==group[1])
        ].copy()
        self.patient_pths = self.selection.patient_pth.unique()
    
    
    def get_loader(self):
        
        idx=0
        images = np.zeros(self.batch_shape)
        
        for path in self.patient_pths:
            
            scans = load_scans(path)
            hu_scans = transform_to_hu(scans)
            images[idx,:,:] = hu_scans[0]
            
            idx += 1
            if idx == self.batch_shape[0]:
                yield images
                images = np.zeros(self.batch_shape)
                idx = 0
        if idx > 0:
            yield images
my_choice = image_sizes.index.values[0]
print(my_choice)
to_display = 4
observer = ImageObserver(scan_properties, to_display)
observer.select_group(my_choice)
observer_iterator = observer.get_loader()
images = next(observer_iterator)
fig, ax = plt.subplots(1,to_display,figsize=(20,5))


for m in range(to_display):
    image = images[m]
    ax[m].imshow(set_manual_window(image, -500, 1000), cmap="YlGnBu")
    ax[m].set_title(observer.name)
scan_properties.shape
scan_properties.head(1)
def resize_scan(scan, new_shape):
    # read slice as 32 bit signed integers
    img = Image.fromarray(scan, mode="I")
    # do the resizing
    img = img.resize(new_shape, resample=Image.LANCZOS)
    # convert back to 16 bit integers
    resized_scan = np.array(img, dtype=np.int16)
    return resized_scan
def crop_scan(scan):
    img = Image.fromarray(scan, mode="I")
    
    left = (scan.shape[0]-512)/2
    right = (scan.shape[0]+512)/2
    top = (scan.shape[1]-512)/2
    bottom = (scan.shape[1]+512)/2

    img = img.crop((left, top, right, bottom))
    # convert back to 16 bit integers
    cropped_scan = np.array(img, dtype=np.int16)
    return cropped_scan
def crop_and_resize(scan, new_shape):
    img = Image.fromarray(scan, mode="I")
    
    left = (scan.shape[0]-512)/2
    right = (scan.shape[0]+512)/2
    top = (scan.shape[1]-512)/2
    bottom = (scan.shape[1]+512)/2
    
    img = img.crop((left, top, right, bottom))
    img = img.resize(new_shape, resample=Image.LANCZOS)
    
    cropped_resized_scan = np.array(img, dtype=np.int16)
    return cropped_resized_scan
def preprocess_to_hu_scans(scan_properties, my_shape, output_dir):
    
    for i, patient in enumerate(tqdm(scan_properties.patient.values)):
        pth = scan_properties.loc[scan_properties.patient==patient].patient_pth.values[0]
        scans = load_scans(pth)
        hu_scans = transform_to_hu(scans) 
        prepared_scans = np.zeros((hu_scans.shape[0], my_shape[0], my_shape[1]), dtype=np.int16)
        
        # if squared:
        if hu_scans.shape[1] == hu_scans.shape[2]:
            
            # if size is as desired
            if hu_scans.shape[1] == my_shape[0]:
                continue
            # else resize:
            else:
               # as we have not converted to jpeg to keep all information, we need to do a workaround
                hu_scans = hu_scans.astype(np.int32)
                for s in range(hu_scans.shape[0]): 
                    prepared_scans[s] = resize_scan(hu_scans[s,:,:], my_shape)

        # if non-squared - do a center crop to 512, 512 and then resize to desired shape
        else:
            hu_scans = hu_scans.astype(np.int32)
            for s in range(hu_scans.shape[0]):
                # if desired shape is 512x512:
                if my_shape[0]==512:
                    prepared_scans[s] = crop_scan(hu_scans[s,:,:])
                else:
                    prepared_scans[s] = crop_and_resize(hu_scans[s,:,:], my_shape)
                
        # save the prepared scans of patient:
        np.save(output_dir + "/" + patient + '_hu_scans', prepared_scans)
generate_512_512 = False

if generate_512_512:
    output_dir = "scans_512x512"
    mkdir(output_dir)
    my_shape = (512, 512)
    preprocess_to_hu_scans(scan_properties, my_shape, output_dir)
generate_224_224 = False

if generate_224_224:
    output_dir = "scans_224x224"
    mkdir(output_dir)
    my_shape = (224, 224)
    preprocess_to_hu_scans(scan_properties, my_shape, output_dir)
generate_128_128 = False

if generate_128_128:
    output_dir = "scans_128x128"
    mkdir(output_dir)
    my_shape = (128, 128)
    preprocess_to_hu_scans(scan_properties, my_shape, output_dir)
generate_64_64 = False

if generate_64_64:
    output_dir = "scans_64x64"
    mkdir(output_dir)
    my_shape = (64, 64)
    preprocess_to_hu_scans(scan_properties, my_shape, output_dir)