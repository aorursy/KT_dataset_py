import os

from collections import defaultdict

import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import figure_factory as FF



import scipy.ndimage

from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



import random

import pydicom



INPUT_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression'



trainset = pd.read_csv(f'{INPUT_DIR}/train.csv')

testset = pd.read_csv(f'{INPUT_DIR}/test.csv')

sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')



def load_scan(path):

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

        

    return slices



def dicom_file(idx_num, patient_id=None):

    if patient_id:

        return load_scan(dicom_dict[patient_id][0])

    return load_scan(dicom_dict[p_id[idx_num]][0])



def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slbice_number] = image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)





def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)



    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    

    return image, new_spacing



def load_scan(path):

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

        

    return slices



def dicom_file(idx_num, patient_id=None):

    if patient_id:

        return load_scan(dicom_dict[patient_id][0])

    return load_scan(dicom_dict[p_id[idx_num]][0])



def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slbice_number] = image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)



def make_mesh(image, threshold):

    p = image.transpose(2, 1, 0)

    

    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)

    return verts, faces



def static_3d(image, threshold=-300, angle=0):

    

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    

    verts, faces = make_mesh(image, threshold)

    x, y, z = zip(*verts)

    

    mesh = Poly3DCollection(verts[faces], alpha=0.1)

    face_color = [0.5, 0.5, 1]

    mesh.set_facecolor(face_color)

    

    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))

    ax.set_ylim(0, max(y))

    ax.set_zlim(0, max(z))

    return ax,



DICOM_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression/train'



dicom_dict = defaultdict(list)





for dirname in os.listdir(DICOM_DIR):

    path = os.path.join(DICOM_DIR, dirname)

    dicom_dict[dirname].append(path)

    

p_id = sorted(trainset['Patient'].unique())



test2 = dicom_file(40)



test2_hu = get_pixels_hu(test2)



resampled_test2_hu, spacing = resample(test2_hu, test2)



from scipy.ndimage import rotate



sub = resampled_test2_hu[::5,::5,::5]



from matplotlib import animation



THRESHOLD = -300



fig,_ = plt.subplots(figsize=(11,11))

ax = fig.add_subplot(111, projection='3d')



def make_mesh(image, threshold):

    p = image.transpose(2, 1, 0)

    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)

    return verts, faces



def static_3d(image,ax, angle=0):

    global x

    global y

    global z

    

    verts, faces = make_mesh(image, THRESHOLD)

    x, y, z = zip(*verts)

    

    mesh = Poly3DCollection(verts[faces], alpha=0.1)

    face_color = [0.5, 0.5, 1]

    mesh.set_facecolor(face_color)

    

    #ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))

    ax.set_ylim(0, max(y))

    ax.set_zlim(0, max(z))

    return ax







def init_time_line(image=sub, axi=ax):

    axi = static_3d(image, axi)

    return axi



def animate_all(angle,axi=ax):

    

    image = (rotate(sub, angle=angle, axes=(1,2), reshape=False,cval=-1024) )

    verts, faces = make_mesh(image, THRESHOLD)   

    

    axi.clear()

    mesh = Poly3DCollection(verts[faces], alpha=0.1)

    face_color = [0.5, 0.5, 1]

    mesh.set_facecolor(face_color)

    axi.add_collection3d(mesh)

    ax.set_xlim(0, max(x))

    ax.set_ylim(0, max(y))

    ax.set_zlim(0, max(z))

    return axi



frames = np.arange(0,360, 6)



anim = animation.FuncAnimation(fig, animate_all, init_func=init_time_line, frames=frames, interval=41, blit=False)



from matplotlib import animation, rc

rc('animation', html='jshtml')

anim