import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
import os
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
from plotly import *

file_name = '../input/dicom-data/series-000001/image-000002.dcm'
dcm = pydicom.dcmread(file_name)
var=dcm.pixel_array
plt.imshow(var, cmap=plt.cm.bone)
type(var)
var
datainputpath = "../input/dicom-data/series-000001/"
dataoutputpath = working_path = "./"
g = glob(datainputpath + '/*.dcm')
def loadingscan(path):
    slic = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
    slic.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slicethicknes = np.abs(slic[0].ImagePositionPatient[2] - slic[1].ImagePositionPatient[2])
    except:
        slicethicknes = np.abs(slic[0].SliceLocation - slic[1].SliceLocation)
        
    for s in slic:
        s.SliceThickness = slicethicknes
        
    return slic
def get_hu(scans):
    imagez = np.stack([s.pixel_array for s in scans])
    imagez = imagez.astype(np.int16)
    imagez[imagez == -2000] = 0
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    if slope != 1:
        imagez = slope * imagez.astype(np.float64)
        imagez = imagez.astype(np.int16)
        
    imagez += np.int16(intercept)
    
    return imagez
id=0
patient= loadingscan(datainputpath)
patient[0]
type(patient[0])
img = get_hu(patient)
img
np.save(dataoutputpath + "fullimages_%d.npy" % (id), img)
file_used=dataoutputpath+"fullimages_%d.npy" % id
imgstoprocess = np.load(file_used).astype(np.float64) 

plt.hist(imgstoprocess.flatten(), bins=50, color='c')
plt.xlabel("HounsfieldUnits(HU)")
plt.ylabel("Frequency")
plt.show()
id = 0
imgstoprocess = np.load(dataoutputpath+'fullimages_{}.npy'.format(id))

def samplstack(stack, rows=5, cols=5, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()
samplstack(imgstoprocess)
print("SliceThickness: %f" % patient[0].SliceThickness)
print("Pixelspacing (row, col): (%f, %f) " % (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))
id = 0
imgs_to_process = np.load(dataoutputpath+'fullimages_{}.npy'.format(id))
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

print("Shapebeforeresampling\t", imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
print("Shapeafterresampling\t", imgs_after_resamp.shape)
import plotly
def makemesh(image, threshold, step_size=1):
    p = image.transpose(2,1,0)
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces
def plotin3d(verts, faces):
    x,y,z = zip(*verts) 
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    fig =plotly.figure_factory.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    iplot(fig)
def plt_3d(verts, faces):
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.get_fc()
    plt.show()
v,f=makemesh(imgs_to_process,1800)
plotin3d(v,f)

def lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    img[img==max]=mean
    img[img==min]=mean
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) 
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10]))

    if (display):
        fig, ax = plt.subplots(1,2, figsize=[12, 12])
        ax[0].set_title("Original")
        ax[0].imshow(img, cmap='gray')
        ax[0].axis('off')
        ax[1].set_title("Threshold")
        ax[1].imshow(thresh_img, cmap='gray')
        ax[1].axis('off')
        
        plt.show()
    return mask*img
img = imgs_after_resamp[85]
lungmask(img, display=True)
