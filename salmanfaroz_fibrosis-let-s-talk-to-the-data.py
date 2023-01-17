import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import pandas as pd
import numpy as np
from colorama import Fore, Back, Style 
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.formula.api import ols
import plotly.graph_objs as gobj
import argparse
import cv2
import cv2 as cv
import pydicom as dicom
from pydicom.filereader import dcmread
import pydicom
import re
from PIL import Image
from IPython.display import Image as show_gif
from PIL import Image
from IPython.display import Image as show_gif
import scipy.misc
import matplotlib
from skimage import exposure
import numpy as np
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
import plotly.figure_factory as ff
from plotly.graph_objs import *
init_notebook_mode(connected=True) 
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)

%matplotlib inline

train=pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
train.head(5)
train.isnull().mean()
hist_data =[train["Age"].values]
group_labels = ['Age'] 

fig = ff.create_distplot(hist_data, group_labels,)
#fig.update_layout(title_text='Age Distribution plot')

fig.show()
fig = px.box(train, x="Sex", y="Age", points="all",)
#fig.update_layout(
  #  title_text="Gender wise Age Spread - Male = 1224 Female =325")
print(train["Sex"].value_counts())
fig.show()
train["SmokingStatus"].value_counts()
smok=train[train["SmokingStatus"]=="Ex-smoker"]["FVC"]
not_smok=train[train["SmokingStatus"]=="Never smoked"]["FVC"]
cr_smoke=train[train["SmokingStatus"]=="Currently smokes"]["FVC"]
hist_data = [smok,not_smok,cr_smoke]

group_labels = ['EX-Smoke', 'Never Smoke',"Currently Smoke"]

fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

fig.show()
fig = px.violin(train, y="Percent", x="Sex", color="SmokingStatus", box=True,
          hover_data=train.columns)

fig.show()

dt = train.groupby(by="Patient")["Weeks"].count().reset_index()
train["time"] = 0

for patient, times in zip(dt["Patient"], dt["Weeks"]):
    train.loc[train["Patient"] == patient, 'time'] = range(1, times+1)
df = px.data.gapminder().query("continent != 'Asia'") # remove Asia for visibility
fig = px.line(train, x="Weeks", y="FVC", color="SmokingStatus",
              line_group="Patient",hover_name="time")
fig.show()
dataset = dcmread("../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/23.dcm")
fig = px.imshow(dataset.pixel_array, color_continuous_scale='plasma')
fig.update_layout(coloraxis_showscale=False)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()
from PIL import Image

def img2gif(id_num):
    tr=train.iloc[id_num,:]
    d=tr.Patient
    smoke=tr.SmokingStatus
    age=tr.Age
    gender=tr.Sex
    inputdir = '../input/osic-pulmonary-fibrosis-progression/train/'+ d
    outdir = './'

    test_list = [ f for f in  os.listdir(inputdir)]
    tt=[]

    for f in test_list[:]: 
        ds = pydicom.read_file(inputdir +"/"+ f) 
        img = ds.pixel_array
        img=exposure.equalize_adapthist(img)

        plt.axis('off')

        plt.imsave(outdir + f.replace('.dcm','.png'),img,cmap="plasma")

        #cv2.imwrite(outdir + f.replace('.dcm','.png'),img) 
        tt.append(outdir + f.replace('.dcm','.png'))
    tt.sort(key=lambda f: int(re.sub('\D', '', f)))
    im_cnt=len(tt)
    for i in tt:
        im_gray = cv2.imread(i)
        kernel = np.ones((1,1), np.uint8)
        erosion = cv2.erode(im_gray, kernel, iterations = 1)

        dilation = cv2.dilate(erosion, kernel, iterations = 1)
        cv2.imwrite(i,dilation)

    new_im=[]
    for file in tt:
        new_frame = Image.open(file)
        new_im.append(new_frame)
    new_im[0].save("./"+'gif_ok.gif', format='GIF',append_images=new_im[:],save_all=True,duration=400, loop=0)
    return im_cnt,d,smoke,age,gender
im_cnt,id_num,smoke,age, gender=img2gif(1)



print("Image count : ",im_cnt,"\nPatient id : ",id_num,"\nSmokingStatus : ",smoke, "\nAge : ",age,"\nGender : ",gender)
show_gif(filename="gif_ok.gif", format='png', width=400, height=400)

# Let's define our kernel size
kernel = np.ones((5,5), np.uint8)
image=dataset.pixel_array
image=exposure.equalize_adapthist(image)
plt.figure(figsize = (65,35))
plt.axis('off')

plt.subplot(341)

# Now we erode
erosion = cv2.erode(image, kernel, iterations = 1)
plt.axis('off')
plt.title("Erosion", fontsize=50)

plt.imshow(erosion)

plt.subplot(342, frameon=False)

kernel = np.ones((5,5), np.uint8)
dilation = cv2.dilate(image, kernel, iterations = 1)
plt.axis('off')
plt.title("Dilation", fontsize=50)

plt.imshow(dilation)

plt.subplot(343, frameon=False)

# Opening - Good for removing noise
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
plt.axis('off')
plt.title("Opening", fontsize=50)

plt.imshow(opening)

plt.subplot(344, frameon=False)

# Closing - Good for removing noise
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
plt.title("Closing", fontsize=50)
plt.axis('off')

plt.imshow(closing)


srted=train.sort_values(by="FVC")
hr=srted.iloc[-1,0]

data_path = "../input/osic-pulmonary-fibrosis-progression/train/"+hr
output_path = working_path = "../input/output/"
g = glob(data_path + '/*.dcm')

# Print out the first 5 file names to verify we're in the right folder.
print ("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
print ('\n'.join(g[:5]))
# Loop over the image files and store everything into a list.

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
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

    image = image.astype(np.int16)

    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

id=0
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)
np.save("fullimages_0.npy", imgs)
file_used="fullimages_0.npy"
imgs_to_process = np.load(file_used).astype(np.float64) 
plt.figure(figsize=(20,6))
plt.hist(imgs_to_process.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()


imgs_to_process = np.load('fullimages_0.npy')

def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=6):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.axis('off')
    plt.show()

sample_stack(imgs_to_process)
print ("Slice Thickness: %f" % patient[0].SliceThickness)
print ("Pixel Spacing (row, col): (%f, %f) " % (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))

imgs_to_process = np.load('fullimages_0.npy')
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

print ("Shape before resampling\t", imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
print ("Shape after resampling\t", imgs_after_resamp.shape)
def make_mesh(image, threshold=-300, step_size=1):

    print ("Transposing surface")
    p = image.transpose(2,1,0)
    
    print ("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces

def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    print ("Drawing")
    
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
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.show()
v, f = make_mesh(imgs_after_resamp, 730, 2)
plotly_3d(v, f)

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
img = imgs_after_resamp[230]
make_lungmask(img, display=True)
masked_lung = []

for img in imgs_after_resamp:
    masked_lung.append(make_lungmask(img))

sample_stack(masked_lung, show_every=10)
#to remove all png siles in working dir
import os

filelist = [ f for f in os.listdir("./") if f.endswith(".png") ]
for f in filelist:
    os.remove(os.path.join("./", f))