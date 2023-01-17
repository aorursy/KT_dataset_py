from IPython.display import HTML

HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/Sm7GX5vNQ0k?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
!conda install -c conda-forge gdcm -y
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from IPython.display import HTML



sns.set_style('darkgrid')

import pydicom

import scipy.ndimage

import gdcm

import imageio

from IPython import display





from skimage import measure 

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage.morphology import disk, opening, closing

from tqdm import tqdm



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.figure_factory as ff

from plotly.graph_objs import *

init_notebook_mode(connected=True) 

from PIL import Image



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



from os import listdir, mkdir
basepath = "../input/rsna-str-pulmonary-embolism-detection/"

listdir(basepath)
train = pd.read_csv(basepath + "train.csv")

test = pd.read_csv(basepath + "test.csv")
train.shape, test.shape
train.head().T
train.info()
test.info()
print("Number of unique Study instances are", train['StudyInstanceUID'].nunique())

print("Number of unique Series instances are", train['SeriesInstanceUID'].nunique())
print('Null values in train data:',train.isnull().sum().sum())

print('Null values in test data:',test.isnull().sum().sum())
def load_scans(dcm_path):

    files = listdir(dcm_path)

    f = [pydicom.dcmread(dcm_path + "/" + str(file)) for file in files]

    return f
example = basepath + "train/" + train.StudyInstanceUID.values[0] +'/'+ train.SeriesInstanceUID.values[0]

file_names = listdir(example)
scans = load_scans(example)
scans[0]
from IPython.display import HTML

HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/KZld-5W99cI?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
plt.figure(figsize=(12,6))

for n in range(10):

    image = scans[n].pixel_array.flatten()

    rescaled_image = image * scans[n].RescaleSlope + scans[n].RescaleIntercept

    sns.distplot(image.flatten());

plt.title("HU unit distributions for 10 examples");
def load_slice(path):

    slices = [pydicom.read_file(path + '/' + s) for s in listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



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



def resample(image, scan, new_spacing=[1,1,1]):

    spacing = np.array([float(scans_0[0].SliceThickness), 

                        float(scans_0[0].PixelSpacing[0]), 

                        float(scans_0[0].PixelSpacing[0])])





    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    

    return image, new_spacing



def make_mesh(image, threshold=-300, step_size=1):

    p = image.transpose(2,1,0)

    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)

    return verts, faces





def plt_3d(verts, faces):

    print("Drawing")

    x,y,z = zip(*verts) 

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')



    # Fancy indexing: `verts[faces]` to generate a collection of triangles

    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)

    face_color = [1, 1, 0.9]

    mesh.set_facecolor(face_color)

    ax.add_collection3d(mesh)



    ax.set_xlim(0, max(x))

    ax.set_ylim(0, max(y))

    ax.set_zlim(0, max(z))

#     ax.set_axis_bgcolor((0.7, 0.7, 0.7))

    ax.set_facecolor((0.7,0.7,0.7))

    plt.show()

sns.set_style('white')

hu_scans = transform_to_hu(scans)



fig, ax = plt.subplots(1,2,figsize=(15,4))





ax[0].set_title("CT-scan in HU")

ax[0].imshow(hu_scans[0], cmap="plasma")

ax[1].set_title("HU values distribution");

sns.distplot(hu_scans[0].flatten(), ax=ax[1],color='red', kde_kws=dict(lw=2, ls="--",color='blue'));

ax[1].grid(False)
first_patient = load_slice('../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf')

first_patient_pixels = transform_to_hu(first_patient)



def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=5):

    fig,ax = plt.subplots(rows,cols,figsize=[18,20])

    for i in range(rows*cols):

        ind = start_with + i*show_every

        ax[int(i/rows),int(i % rows)].set_title(f'slice {ind}')

        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='bone')

        ax[int(i/rows),int(i % rows)].axis('off')

    plt.show()



sample_stack(first_patient_pixels)
imageio.mimsave("/tmp/gif.gif", first_patient_pixels, duration=0.1)

display.Image(filename="/tmp/gif.gif", format='png')
first_patient_scan = '../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf'

scans_0 = load_scans(first_patient_scan)

imgs_after_resamp, spacing = resample(first_patient_pixels, scans_0, [1,1,1])

v, f = make_mesh(imgs_after_resamp, threshold = 350)

plt_3d(v, f)
im_path = []

train_path = '../input/rsna-str-pulmonary-embolism-detection/train/'

for i in listdir(train_path): 

    for j in listdir(train_path + i):

        x = i+'/'+j

        im_path.append(x)
def get_window_value(feature):

    if type(feature) == pydicom.multival.MultiValue:

        return np.int(feature[0])

    else:

        return np.int(feature)



pixelspacing_r = []

pixelspacing_c = []

slice_thicknesses = []

ids = []

id_pth = []

row_values = []

column_values = []

window_widths = []

window_levels = []

modality = []

kvp = []

table_height = []

x_ray_tube = []

exp = []

pos = []

tilt = []

bits = []

rescale_inter = []

rescale_slope = []

photometric_interpretation = []

convolution_kernel = [] 





for i in im_path:

    ids.append(i.split('/')[0]+'_'+i.split('/')[1])

    example_dcm = listdir(train_path  + i + "/")[0]

    id_pth.append(train_path + i)

    dataset = pydicom.dcmread(train_path + i + "/" + example_dcm)

    

    window_widths.append(get_window_value(dataset.WindowWidth))

    window_levels.append(get_window_value(dataset.WindowCenter))

    

    spacing = dataset.PixelSpacing

    slice_thicknesses.append(dataset.SliceThickness)

    

    row_values.append(dataset.Rows)

    column_values.append(dataset.Columns)

    pixelspacing_r.append(spacing[0])

    pixelspacing_c.append(spacing[1])

    

    modality.append(dataset.Modality)

    kvp.append(dataset.KVP)

    table_height.append(dataset.TableHeight)

    x_ray_tube.append(dataset.XRayTubeCurrent)

    exp.append(dataset.Exposure)

    pos.append(dataset.PatientPosition)

    tilt.append(dataset.GantryDetectorTilt)

    bits.append(dataset.BitsAllocated)

    rescale_inter.append(dataset.RescaleIntercept)

    rescale_slope.append(dataset.RescaleSlope)

    photometric_interpretation.append(dataset.PhotometricInterpretation)

    convolution_kernel.append(dataset.ConvolutionKernel)

    

scan_properties = pd.DataFrame(data=ids, columns=["ID"])

scan_properties.loc[:, "rows"] = row_values

scan_properties.loc[:, "columns"] = column_values

scan_properties.loc[:, "area"] = scan_properties["rows"] * scan_properties["columns"]

scan_properties.loc[:, "pixelspacing_r"] = pixelspacing_r

scan_properties.loc[:, "pixelspacing_c"] = pixelspacing_c

scan_properties.loc[:, "pixelspacing_area"] = scan_properties.pixelspacing_r * scan_properties.pixelspacing_c

scan_properties.loc[:, "slice_thickness"] = slice_thicknesses

scan_properties.loc[:, "id_pth"] = id_pth

scan_properties.loc[:, "window_width"] = window_widths

scan_properties.loc[:, "window_level"] = window_levels

scan_properties.loc[:, "modality"] = modality

scan_properties.loc[:, "kvp"] = kvp

scan_properties.loc[:, "table_height"] = table_height

scan_properties.loc[:, "x_ray_tube_current"] = x_ray_tube

scan_properties.loc[:, "exposure"] = exp 

scan_properties.loc[:,"patient_position"] = pos

scan_properties.loc[:,"detector_tilt"] = tilt

scan_properties.loc[:,"bits_allocated"] = bits

scan_properties.loc[:,"rescale_intercept"] = rescale_inter

scan_properties.loc[:, "rescale_slope"] = rescale_slope

scan_properties.loc[:, "photometric_interpretation"] = photometric_interpretation

scan_properties.loc[:, "convolution_kernel"] = convolution_kernel



scan_properties.head().T
sns.set_style('darkgrid')

fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(pixelspacing_r, ax=ax[0], color='green', kde_kws=dict(lw=3, ls="--",color='red'))

ax[0].set_title("Pixel spacing distribution \n in row direction ")

ax[0].set_ylabel("Counts in train")

ax[0].set_xlabel("mm")

sns.distplot(pixelspacing_c, ax=ax[1], color="Blue",kde_kws=dict(lw=3, ls="--",color='red'))

ax[1].set_title("Pixel spacing distribution \n in column direction");

ax[1].set_ylabel("Counts in train");

ax[1].set_xlabel("mm");
scan_properties["r_distance"] = scan_properties.pixelspacing_r * scan_properties.rows

scan_properties["c_distance"] = scan_properties.pixelspacing_c * scan_properties["columns"]

scan_properties["area_cm2"] = 0.1* scan_properties["r_distance"] * 0.1*scan_properties["c_distance"]

scan_properties["slice_volume_cm3"] = 0.1*scan_properties.slice_thickness * scan_properties.area_cm2
fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(scan_properties.area_cm2, ax=ax[0], color="Limegreen",kde_kws=dict(lw=3, ls="--",color='red'))

sns.distplot(scan_properties.slice_volume_cm3, ax=ax[1], color="Mediumseagreen",kde_kws=dict(lw=3, ls="--",color='red'))

ax[0].set_title("CT-slice area in $cm^{2}$")

ax[1].set_title("CT-slice volume in $cm^{3}$")

ax[0].set_xlabel("$cm^{2}$")

ax[1].set_xlabel("$cm^{3}$");
scan_properties.head(3).T
scan_properties.describe().T
scan_properties.dtypes
scan_properties.to_csv('Pulmonary_Embolism_CT_scans_data.csv',index=False)
scan_cols = scan_properties.copy()

scan_cols.drop(['rows','columns','area','detector_tilt','bits_allocated','rescale_slope'],axis=1,inplace=True)



corr = scan_cols.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(12, 12))

    ax = sns.heatmap(corr,mask=mask,square=True,linewidths=.8,cmap="viridis",annot=True)
y_cols = ['pixelspacing_r','pixelspacing_c','pixelspacing_area']

x_cols = 'slice_volume_cm3'



plt.figure(figsize=(18,6))

for i in range(len(y_cols)):

    plt.subplot(1, 3, i+1)

    sns.scatterplot(x=x_cols,y=y_cols[i],data=scan_cols,hue='slice_thickness',size='slice_thickness')
cols = train.copy()

cols.drop(['StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID'],axis=1,inplace=True)

columns = cols.columns
fig, ax = plt.subplots(7,2,figsize=(16,28))

for i,col in enumerate(columns): 

    plt.subplot(7,2,i+1)

    sns.countplot(cols[col],palette='hot')   
corr = cols.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(12, 12))

    ax = sns.heatmap(corr,mask=mask,square=True,linewidths=.8,cmap="summer",annot=True)