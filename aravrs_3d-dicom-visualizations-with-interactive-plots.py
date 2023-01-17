import os

import plotly

import pydicom

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from glob import glob

import scipy.ndimage

from skimage import measure

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.figure_factory import create_trisurf

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
### Helper functions





def load_scan(path, reverse=True):

    slices = [pydicom.read_file(path + "/" + s) for s in os.listdir(path)]

    slices.sort(key=lambda x: int(x.InstanceNumber), reverse=reverse)



    try:

        slice_thickness = np.abs(

            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]

        )

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)



    for s in slices:

        s.SliceThickness = slice_thickness



    return slices





def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans])

    image = image.astype(np.int16)

    image[image == -2000] = 0



    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope



    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)



    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)





def resample(image, scan, new_spacing=[1, 1, 1]):

    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))

    spacing = np.array(list(spacing))



    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor



    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing





def make_mesh(image, threshold=-300, step_size=1):

    p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)

    return verts, faces





def largest_label_volume(im, bg=-1):

    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]

    vals = vals[vals != bg]

    if len(counts) > 0:

        return vals[np.argmax(counts)]

    else:

        return None





def segment_lung_mask(image, fill_lung_structures=True):

    binary_image = np.array(image >= -700, dtype=np.int8) + 1

    labels = measure.label(binary_image)

    background_label = labels[0, 0, 0]

    binary_image[background_label == labels] = 2



    if fill_lung_structures:

        for i, axial_slice in enumerate(binary_image):

            axial_slice = axial_slice - 1

            labeling = measure.label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)



            if l_max is not None:

                binary_image[i][labeling != l_max] = 1

    binary_image -= 1

    binary_image = 1 - binary_image



    labels = measure.label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None:

        binary_image[labels != l_max] = 0



    return binary_image
TRAIN_DIR = "../input/osic-pulmonary-fibrosis-progression/train/"

sub_folder_list = []

for x in os.listdir(TRAIN_DIR):

    if os.path.isdir(TRAIN_DIR + '/' + x):

        sub_folder_list.append(x)



n_dicom_dict = {"Patient":[],"n_dicom":[]}

for x in sub_folder_list:

    g = glob(TRAIN_DIR+x + '/*.dcm')

    n_dicom_dict["n_dicom"].append(len(g))

    n_dicom_dict["Patient"].append(x)



dicom_df = pd.DataFrame(n_dicom_dict)

dicom_df.sort_values(['n_dicom'], inplace=True)
print("Minimum number of dicom files:", min(dicom_df['n_dicom']))

print("Maximum number of dicom files:", max(dicom_df['n_dicom']))



plt.figure(figsize=(20,10))

sns.distplot(dicom_df['n_dicom'], bins=20, color="#a55eea")

plt.title('Number of dicom files per patient');
data_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00180637202240177410333/"

print(f"{data_path.split('/')[-3].upper()} - {data_path.split('/')[-2]}")

g = glob(data_path + "/*.dcm")

print(f"Total of {len(g)} DICOM images.")



patient = load_scan(data_path, False)

print(f"Slice Thickness: {patient[0].SliceThickness}")

print(f"Pixel Spacing (row, col): ({patient[0].PixelSpacing[0]}, {patient[0].PixelSpacing[1]})")



imgs = get_pixels_hu(patient)
def plot_3d(data_path, reverse=False):

    print(f"{data_path.split('/')[-3].upper()} - {data_path.split('/')[-2]}")

    g = glob(data_path + "/*.dcm")

    print(f"Total of {len(g)} DICOM images.")



    patient = load_scan(data_path, reverse)

    print(f"Slice Thickness: {patient[0].SliceThickness}")

    print(f"Pixel Spacing (row, col): ({patient[0].PixelSpacing[0]}, {patient[0].PixelSpacing[1]})")



    imgs = get_pixels_hu(patient)

    print(f"Shape resampling: {imgs.shape}", end="")

    imgs_after_resamp, spacing = resample(imgs, patient, [1, 1, 1])

    print(f" -> {imgs_after_resamp.shape}")



    v1, f1 = make_mesh(imgs_after_resamp, 350, 2)



    segmented_lungs = segment_lung_mask(imgs_after_resamp, fill_lung_structures=False)

    segmented_lungs_fill = segment_lung_mask(imgs_after_resamp, fill_lung_structures=True)

    internal_structures = segmented_lungs_fill - segmented_lungs

    p = internal_structures.transpose(2, 1, 0)

    v2, f2, _, _ = measure.marching_cubes_lewiner(p)



    ### PLOTS

    fig = plt.figure(figsize=(20, 10))

    bg = np.array((30, 39, 46))/255.0

    

    # Ext

    print(".", end="")

    x, y, z = zip(*v1)

    ax1 = fig.add_subplot(121, projection="3d")

    mesh = Poly3DCollection(v1[f1], alpha=0.8)

    face_color = (1, 1, 0.9)

    mesh.set_facecolor(face_color)

    ax1.add_collection3d(mesh)

    ax1.set_xlim(0, max(x))

    ax1.set_ylim(0, max(y))

    ax1.set_zlim(0, max(z))

    ax1.w_xaxis.set_pane_color((*bg, 1))

    ax1.w_yaxis.set_pane_color((*bg, 1))

    ax1.w_zaxis.set_pane_color((*bg, 1))



    # Int

    print(".", end="")

    x, y, z = zip(*v2)

    ax2 = fig.add_subplot(122, projection="3d")

    mesh = Poly3DCollection(v2[f2], alpha=0.8)

    face_color = np.array((255, 107, 107))/255.0

    mesh.set_facecolor(face_color)

    ax2.add_collection3d(mesh)

    ax2.set_xlim(0, max(x))

    ax2.set_ylim(0, max(y))

    ax2.set_zlim(0, max(z))

    ax2.w_xaxis.set_pane_color((*bg, 1))

    ax2.w_yaxis.set_pane_color((*bg, 1))

    ax2.w_zaxis.set_pane_color((*bg, 1))



    print(".", end="")

    fig.tight_layout()

    plt.show()
plot_3d(data_path)
### 3D interactive ploting helper

def plotly_3d(verts, faces, ext=True):

    x, y, z = zip(*verts)



    fig = create_trisurf(

        x=x,

        y=y,

        z=z,

        plot_edges=False,

        show_colorbar=False,

        showbackground=False,

        colormap=["rgb(236, 236, 212)", "rgb(236, 236, 212)"] if ext else ["rgb(255, 107, 107)", "rgb(255, 107, 107)"],

        simplices=faces,

        backgroundcolor="rgb(30, 39, 46)",

        gridcolor="rgb(30, 39, 46)",

        title="<b>Interactive Visualization</b>",

    )

    fig.layout.template = "plotly_dark"  # for dark theme 

    fig.show()
### Ploting functions





def plot3d_interactive_ext(data_path, reverse=False):

    print(f"{data_path.split('/')[-3].upper()} - {data_path.split('/')[-2]}")

    g = glob(data_path + "/*.dcm")

    patient = load_scan(data_path, reverse)

    imgs = get_pixels_hu(patient)

    imgs_after_resamp, spacing = resample(imgs, patient, [1, 1, 1])



    v, f = make_mesh(imgs_after_resamp, 350, 2)

    plotly_3d(v, f)





def plot3d_interactive_int(data_path, reverse=False):

    print(f"{data_path.split('/')[-3].upper()} - {data_path.split('/')[-2]}")

    g = glob(data_path + "/*.dcm")

    patient = load_scan(data_path, reverse)

    imgs = get_pixels_hu(patient)

    imgs_after_resamp, spacing = resample(imgs, patient, [1, 1, 1])



    segmented_lungs = segment_lung_mask(imgs_after_resamp, fill_lung_structures=False)

    segmented_lungs_fill = segment_lung_mask(imgs_after_resamp, fill_lung_structures=True)

    internal_structures = segmented_lungs_fill - segmented_lungs



    p = internal_structures.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes_lewiner(p)

    plotly_3d(verts, faces, ext=False)
plot3d_interactive_ext(data_path)
plot3d_interactive_int(data_path)
plot_3d("../input/osic-pulmonary-fibrosis-progression/train/ID00104637202208063407045/")
plot_3d("../input/osic-pulmonary-fibrosis-progression/train/ID00104637202208063407045/")
plot_3d("../input/osic-pulmonary-fibrosis-progression/train/ID00035637202182204917484/")
plot_3d("../input/osic-pulmonary-fibrosis-progression/train/ID00291637202279398396106/", True)
plot_3d("../input/osic-pulmonary-fibrosis-progression/train/ID00042637202184406822975/", True)
plot_3d("../input/osic-pulmonary-fibrosis-progression/train/ID00422637202311677017371/", True)
plot_3d("../input/osic-pulmonary-fibrosis-progression/train/ID00309637202282195513787/", True)
plot3d_interactive_ext("../input/osic-pulmonary-fibrosis-progression/train/ID00078637202199415319443/", True)