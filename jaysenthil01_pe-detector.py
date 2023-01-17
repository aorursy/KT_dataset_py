### Imports
import os
import scipy
import pandas as pd
# import pydicom
import matplotlib.pyplot as plt
import numpy as np
import vtk
from vtk.util import numpy_support
import cv2
import SimpleITK as sitk
!pip install git+https://github.com/JoHof/lungmask
from lungmask import mask    
from skimage.morphology import convex_hull_image
from skimage.transform import resize
import h5py
#!conda install -c conda-forge gdcm -y
# Preprocess
# load basepath
NUM_DS = 7279
basepath = "../input/rsna-str-pulmonary-embolism-detection/"
# load CSV labels
train = pd.read_csv(basepath + "train.csv")
test = pd.read_csv(basepath + "test.csv")
# create new column
train["dcm_path"] = basepath + "train/" + train.StudyInstanceUID + "/" + train.SeriesInstanceUID;
reader = vtk.vtkDICOMImageReader()
def load_scans_VTK(PathDicom):
    reader.SetDirectoryName(PathDicom)
    reader.Update()

    # Load dimensions using `GetDataExtent`
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    # Load spacing values
    ConstPixelSpacing = reader.GetPixelSpacing()

    # Get the 'vtkImageData' object from the reader
    imageData = reader.GetOutput()
    # Get the 'vtkPointData' object from the 'vtkImageData' object
    pointData = imageData.GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays()==1)
    # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)

    # Convert the `vtkArray` to a NumPy array
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape((512, 512,-1), order='F')
    ArrayDicom[~(ArrayDicom==0).all((2,1))]
    return ArrayDicom
# Set outside circle to air.
def set_outside_scanner_to_air(raw_pixelarrays):
    # in OSIC we find outside-scanner-regions with raw-values of -2000. 
    # Let's threshold between air (0) and this default (-2000) using -1000
    raw_pixelarrays[raw_pixelarrays <= -1000] = 0
    return raw_pixelarrays
def flood_fill_hull(image):    
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull
# initial model is exam level, so get one row for every exam
ndTrain = train.drop_duplicates(subset='StudyInstanceUID', keep="first")
ndTrain['positive_exam_for_pe'] = ~ndTrain['negative_exam_for_pe'].astype('bool') & ~ndTrain['indeterminate'].astype('bool')
nnTrainLabels = [ndTrain['negative_exam_for_pe'], ndTrain['positive_exam_for_pe'].astype('int64'), ndTrain['indeterminate']]
nnTrainLabels = pd.concat(nnTrainLabels, axis=1)
nnTrainLabels = nnTrainLabels.to_numpy()
print(nnTrainLabels.shape)
# save to HDF5 later on
# create new h5
f = h5py.File('/kaggle/working/s1-traindata.h5', 'w')
dset_images = f.create_dataset("images", (NUM_DS, 64, 224, 224), chunks=(1,64, 224,224))
dset_labels = f.create_dataset("labels", (NUM_DS, 3))
dset_labels = nnTrainLabels
# save all the images after removing lung and bwconvhull 7280 NUM_DS+1
for i in range(0, NUM_DS+1):
    tempLoaded = load_scans_VTK(ndTrain.dcm_path.values[i])
    tempLoaded = set_outside_scanner_to_air(tempLoaded)
    segmentation = mask.apply(sitk.GetImageFromArray(np.transpose(tempLoaded,(2,1, 0))))
    # orient
    tempLoaded = np.rot90(tempLoaded)
    segmentation = np.flip(segmentation,1)
    segmentation = np.transpose(segmentation,(1,2,0))
    # Remove 2s
    segmentation[segmentation >= 1] = 1
    # convex hull
    segmentation,h = flood_fill_hull(segmentation)
    # mask * data
    tempLHData = segmentation * tempLoaded
    lungIndices = np.where(tempLHData.any(axis=(0,1)))[0]
    tempLHData = tempLHData[:,:,lungIndices]
    #print(tempLHData.shape)
    tempLHData = resize(tempLHData, (224, 224, 64), anti_aliasing=False)
    #print(tempLHData.shape)
    tempLHData = np.transpose(tempLHData,(2,0,1))
    #print(tempLHData.shape)
    #tempLHData = np.expand_dims(tempLHData, 3)
    #print(tempLHData.shape)
    #tempLHData = np.concatenate([tempLHData, tempLHData, tempLHData], axis=3)
    #print(tempLHData.shape)
    dset_images[i,:,:,:] = tempLHData
    if i%50 == 0:
        print('===================== done with:', i, '=====================')
f.close()