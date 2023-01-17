import glob
from skimage import measure

%matplotlib inline

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
def myfunc(e):
    return e[-6:] 
patient_list = train.Patient.unique()
data_path = '../input/osic-pulmonary-fibrosis-progression/train/'

patient = pd.DataFrame()
pid = []
count = []
path = []
for pat in patient_list:
    
    data_list = glob.glob(data_path + pat + '/*.dcm')
    data_list.sort(key=myfunc)
    pid.append(pat)
    path.append(data_list)
    count.append(len(data_list))
    
patient['pid']=pid
patient['path']= path
patient['count']=count
patient.head()
patient.path[0]
slices = [pydicom.read_file(s) for s in patient.path[0]] #lets read metadeta of pydiacom file
print('The total no of ct scan associated with 1st patient',len(slices))
print(slices[3])
slices.sort(key = lambda x: int(x.InstanceNumber)) # order the slice serially

slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
print(slice_thickness)
#lets create a function for above task
def load_scan(path):
    slices = [pydicom.read_file(path +'/'+ s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices 

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
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

#lets take alook at patient
patient = load_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430')
imgs = get_pixels_hu(patient)
plt.hist(imgs.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()


plt.imshow(imgs[29], cmap=plt.cm.gray)
plt.show()

print("Slice Thickness: %f" % patient[0].SliceThickness)
print("Pixel Spacing (row, col): (%f, %f) "% (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing
pix_resampled, spacing = resample(imgs, patient, [1,1,1])
print("Shape before resampling\t", imgs.shape)
print("Shape after resampling\t", pix_resampled.shape)
plt.hist(pix_resampled.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()


plt.imshow(imgs[29], cmap=plt.cm.gray)
plt.show()
def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    
    p = image.transpose(2,1,0) #get image in order(h,w,c)
    
    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
plot_3d(pix_resampled, 400)

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image