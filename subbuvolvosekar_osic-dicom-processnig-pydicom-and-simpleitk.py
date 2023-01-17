# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline



import numpy as np

import pydicom

import os

import matplotlib.pyplot as plt

from glob import glob

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scipy.ndimage

from skimage import morphology

from skimage import measure

from skimage.transform import resize

from sklearn.cluster import KMeans

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly.tools import FigureFactory as FF

from plotly.graph_objs import *

import plotly.express as px

init_notebook_mode(connected=True)
data_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/'

train_data = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')

patient_list = train_data.Patient.unique()

print("Number of Patients :",len(patient_list) )
patient = pd.DataFrame()

pid = []

count = []

path = []

for pat in patient_list: 

    g = glob(data_path +pat +'/*.dcm')

    pid.append(pat)

    path.append(g)

    count.append(len(g))

patient['pid'] = pid

patient['scan_count'] = count

patient['path'] = path

fig = px.bar(patient, x = 'pid', y = 'scan_count')

fig.show()
#      

# Loop over the image files and store everything into a list.

# 



def load_scan(path):

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 1

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

        

    image += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)
# ID00007637202177411956430

patient = load_scan(data_path+'ID00007637202177411956430/')

imgs = get_pixels_hu(patient)

out_path = '/kaggle/working/'

id = 0

np.save(out_path + "fullimages_%d.npy" % (id), imgs)
plt.imshow(imgs[25], cmap=plt.cm.bone)
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

    binary_image = np.array(image >= -700, dtype=np.int8)+1

    labels = measure.label(binary_image)

 

    # Pick the pixel in the very corner to determine which label is air.

    # Improvement: Pick multiple background labels from around the patient

    # More resistant to “trays” on which the patient lays cutting the air around the person in half

    background_label = labels[0,0,0]

 

    # Fill the air around the person

    binary_image[background_label == labels] = 2

 

    # Method of filling the lung structures (that is superior to 

    # something like morphological closing)

    if fill_lung_structures:

        # For every slice we determine the largest solid structure

        for i, axial_slice in enumerate(binary_image):

            axial_slice = axial_slice-1

            labeling = measure.label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)

 

            if l_max is not None: #This slice contains some lung

                binary_image[i][labeling != l_max] = 1

    binary_image -= 1 #Make the image actual binary

    binary_image = 1-binary_image # Invert it, lungs are now 1

 

    # Remove other air pockets inside body

    labels = measure.label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None: # There are air pockets

        binary_image[labels != l_max] = 0

 

    return binary_image
# get masks 

import copy

segmented_lungs = segment_lung_mask(imgs, fill_lung_structures=False)

segmented_lungs_fill = segment_lung_mask(imgs,fill_lung_structures=True)

internal_structures = segmented_lungs_fill - segmented_lungs

# isolate lung from chest

copied_pixels = copy.deepcopy(imgs)

for i, mask in enumerate(segmented_lungs_fill): 

    get_high_vals = mask == 0

    copied_pixels[i][get_high_vals] = 0

seg_lung_pixels = copied_pixels

# sanity check

plt.figure(figsize = (20,15))

plt.imshow(seg_lung_pixels[25], cmap=plt.cm.bone)
file_used=out_path+"fullimages_%d.npy" % id

imgs_to_process = np.load(file_used).astype(np.float64) 

plt.figure(figsize = (15,12))

plt.hist(imgs_to_process.flatten(), bins=50, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()
id = 0

imgs_to_process = np.load(out_path+'fullimages_{}.npy'.format(id))



def sample_stack(stack, rows=5, cols=5, start_with=1, show_every=1):

    fig,ax = plt.subplots(rows,cols,figsize=[12,12])

    for i in range(rows*cols):

        ind = start_with + i*show_every

        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)

        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')

        ax[int(i/rows),int(i % rows)].axis('off')

    plt.show()



sample_stack(imgs_to_process)
print("Slice Thickness: %f" % patient[0].SliceThickness)

print("Pixel Spacing (row, col): (%f, %f) " % (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))
id = 0

imgs_to_process = np.load(out_path+'fullimages_{}.npy'.format(id))

def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))

    spacing = np.array(list(spacing))



    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    

    return image, new_spacing



print("Shape before resampling\t", imgs_to_process.shape)

imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])

print("Shape after resampling\t", imgs_after_resamp.shape)
def make_mesh(image, threshold=-300, step_size=1):



    print("Transposing surface")

    p = image.transpose(2,1,0)

    

    print("Calculating surface")

    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True) 

    return verts, faces



def plotly_3d(verts, faces):

    x,y,z = zip(*verts) 

    

    print("Drawing") 

    

    # Make the colormap single color since the axes are positional not intensity. 

#    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']

    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']

    

    fig = FF.create_trisurf(x=x,

                        y=y, 

                        z=z, 

                        plot_edges=False,

                        colormap=colormap,

                        simplices=faces,

                        backgroundcolor='rgb(64, 64, 64)',

                        title="Interactive Visualization")

    iplot(fig)



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

    #ax.set_axis_bgcolor((0.7, 0.7, 0.7))

    plt.show()

v, f = make_mesh(imgs_after_resamp, 350)

plt_3d(v, f)
#Standardize the pixel values

def make_lungmask(img, display=False):

    row_size= img.shape[0]

    col_size = img.shape[1]

    

    mean = np.mean(img)

    std = np.std(img)

    img = img-mean

    img = img/std

    # Find the average pixel value near the lungs

    # to renormalize washed out images

    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 

    mean = np.mean(middle)  

    max = np.max(img)

    min = np.min(img)

    # To improve threshold finding, I'm moving the 

    # underflow and overflow on the pixel spectrum

    img[img==max]=mean

    img[img==min]=mean

    #

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)

    #

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))

    centers = sorted(kmeans.cluster_centers_.flatten())

    threshold = np.mean(centers)

    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image



    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  

    # We don't want to accidentally clip the lung.



    eroded = morphology.erosion(thresh_img,np.ones([3,3]))

    dilation = morphology.dilation(eroded,np.ones([8,8]))



    labels = measure.label(dilation) # Different labels are displayed in different colors

    label_vals = np.unique(labels)

    regions = measure.regionprops(labels)

    good_labels = []

    for prop in regions:

        B = prop.bbox

        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:

            good_labels.append(prop.label)

    mask = np.ndarray([row_size,col_size],dtype=np.int8)

    mask[:] = 0



    #

    #  After just the lungs are left, we do another large dilation

    #  in order to fill in and out the lung mask 

    #

    for N in good_labels:

        mask = mask + np.where(labels==N,1,0)

    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation



    if (display):

        fig, ax = plt.subplots(3, 2, figsize=[12, 12])

        ax[0, 0].set_title("Original")

        ax[0, 0].imshow(img, cmap='gray')

        ax[0, 0].axis('off')

        ax[0, 1].set_title("Threshold")

        ax[0, 1].imshow(thresh_img, cmap='gray')

        ax[0, 1].axis('off')

        ax[1, 0].set_title("After Erosion and Dilation")

        ax[1, 0].imshow(dilation, cmap='gray')

        ax[1, 0].axis('off')

        ax[1, 1].set_title("Color Labels")

        ax[1, 1].imshow(labels)

        ax[1, 1].axis('off')

        ax[2, 0].set_title("Final Mask")

        ax[2, 0].imshow(mask, cmap='gray')

        ax[2, 0].axis('off')

        ax[2, 1].set_title("Apply Mask on Original")

        ax[2, 1].imshow(mask*img, cmap='gray')

        ax[2, 1].axis('off')

        

        plt.show()

    return mask*img
img = imgs_after_resamp[260]

make_lungmask(img, display=True)
masked_lung = []



for img in imgs_after_resamp:

    masked_lung.append(make_lungmask(img))



sample_stack(masked_lung, show_every=10)
def sitk_show(img, title=None, margin=0.05, dpi=40 ):

    nda = SimpleITK.GetArrayFromImage(img)

    spacing = img.GetSpacing()

    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi

    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])



    plt.set_cmap("gray")

    ax.imshow(nda,extent=extent,interpolation=None)

    

    if title:

        plt.title(title)

    

    plt.show()
import SimpleITK
reader = SimpleITK.ImageSeriesReader()

filenamesDICOM = reader.GetGDCMSeriesFileNames(data_path+'ID00007637202177411956430/')

reader.SetFileNames(filenamesDICOM)

imgOriginal = reader.Execute()
idxSlice = 25

imgOriginal = imgOriginal[:,:,idxSlice]

sitk_show(imgOriginal)
imgSmooth = SimpleITK.CurvatureFlow(image1=imgOriginal,

                                    timeStep=0.125,

                                    numberOfIterations=5)

sitk_show(imgSmooth)