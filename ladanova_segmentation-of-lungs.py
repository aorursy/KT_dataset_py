import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import figure_factory as FF



import scipy.ndimage

from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pydicom

import os

import scipy.ndimage

import matplotlib.pyplot as plt



from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



from IPython.display import HTML
# Some constants 

path_input = '../input/osic-pulmonary-fibrosis-progression/train/'

patients = os.listdir(path_input)

patients.sort()
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
first_patient = load_scan(path_input + patients[1])



print(type(first_patient[8].pixel_array))
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
imgs = get_pixels_hu(first_patient)

plt.hist(imgs.flatten(), bins=50, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()
#Standardize the pixel values

from sklearn.cluster import KMeans

from scipy import ndimage



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

        if ( B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/6 and B[2]<col_size/6*5):

            good_labels.append(prop.label)

    mask = np.ndarray([row_size,col_size],dtype=np.int8)

    mask[:] = 0



    #

    #  After just the lungs are left, we do another large dilation

    #  in order to fill in and out the lung mask 

    #

    for N in good_labels:

        mask = mask + np.where(labels==N,1,0)

#     mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation



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
make_lungmask(first_patient[50].pixel_array, display=True)
import matplotlib.animation as animation



fig = plt.figure()



ims = []

for scan in first_patient:

    im = plt.imshow(scan.pixel_array, animated=True, cmap="Greys")

    plt.axis("off")

    ims.append([im])



ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,

                                repeat_delay=1000)



HTML(ani.to_jshtml())
fig, ax = plt.subplots(figsize=(5,5))



ims = []

for scan in first_patient:

    im = ax.imshow(make_lungmask(scan.pixel_array), animated=True, cmap="gray")

    ax.axis("off")

    ims.append([im])



ani = animation.ArtistAnimation(fig, ims, interval=150, blit=False,

                                repeat_delay=1000)



HTML(ani.to_jshtml())