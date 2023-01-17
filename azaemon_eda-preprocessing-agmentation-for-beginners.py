import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import cv2

from matplotlib import pyplot as plt

# Set Color Palettes for the notebook (https://color.adobe.com/)
colors_nude = ['#FFE61A','#B2125F','#FF007B','#14B4CC','#099CB3']
sns.palplot(sns.color_palette(colors_nude))

# Set Style
sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)

train_data = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test_data = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

print(train_data.head())
print()
print("The number of total records in the Train Data: ",len(train_data))
print(train_data.isna().sum())
print()
print('Here, we can see there is no missing data in any of the columns.')
train_data.describe(percentiles=[.20,.40,.60,.80])
print("The number of unique patients: ",len(train_data["Patient"].unique()))
fig = px.sunburst(data_frame=train_data,
                  path=['Age', 'Sex', 'SmokingStatus'],
                  color='Sex',
                  maxdepth=-1,
                  title='Sunburst Chart')

fig.update_traces(textinfo='label+percent parent')
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
fig.show()
sns.distplot(train_data['Age'],rug=True)

print("This plot shows most of the patients are from 65 to 75 years old.")
# Occurance of landmark_id in decreasing order(Top categories)
temp = pd.DataFrame(train_data.Age.value_counts().head(10))
temp.reset_index(inplace=True)
temp.columns = ['Age','Number of Patients']

# Plot the most frequent landmark_ids
plt.figure(figsize = (9, 10))
plt.title('The most common ages of the patients')
sns.set_color_codes("deep")
sns.barplot(x="Age", y="Number of Patients", data=temp,
            label="Count")
plt.show()
df= train_data
df= df.drop_duplicates(subset='Patient',keep='first') 


temp = pd.DataFrame(df.Sex.value_counts().head(10))
temp.reset_index(inplace=True)
temp.columns = ['Sex','Number of Patients']


# Plot the most frequent landmark_ids
plt.figure(figsize = (9, 10))
plt.title('Sex distribution of the patients')
sns.set_color_codes("deep")
sns.barplot(x="Sex", y="Number of Patients", data=temp,
            label="Count")
plt.show()

print("This plot shows the number of male patients are about 3.5 times higher than female patients.")
sns.catplot(x="Patient", y="Age", hue="Sex", kind="swarm", data=df);
print("From this plot we can see that the oldest patient is Male and the youngest patient is a Female")
df= train_data
df= df.drop_duplicates(subset='Patient',keep='first') 


temp = pd.DataFrame(df.SmokingStatus.value_counts().head(10))
temp.reset_index(inplace=True)
temp.columns = ['Smoking Status','Number of Patients']


# Plot the most frequent landmark_ids
plt.figure(figsize = (9, 10))
plt.title('Smoking Status distribution of the patients')
sns.set_color_codes("deep")
sns.barplot(x="Smoking Status", y="Number of Patients", data=temp,
            label="Count")
plt.show()
sns.catplot(x="Patient", y="Age", hue="SmokingStatus", kind="swarm", data=df)
sns.catplot(x="SmokingStatus", y="Age", hue="Sex", kind="swarm", data=df);
print("This plot shows that most of the Male patients are Ex-smoker and most the Frmale patients have Never Smoked")
# Figure
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))

a = sns.distplot(train_data["FVC"], ax=ax1, hist=True)
b = sns.distplot(train_data["Percent"], ax=ax2, hist=True)

a.set_title("FVC Distribution", fontsize=16)
b.set_title("Percent Distribution", fontsize=16);
sns.catplot(x="SmokingStatus", y="FVC", hue="Sex", kind="box", data=df);
sns.catplot(x="SmokingStatus", y="Percent", hue="Sex", kind="box", data=df);
import pydicom
from ipywidgets.widgets import * 
import ipywidgets as widgets

import re
from PIL import Image
from IPython.display import Image as show_gif
import scipy.misc
import matplotlib
from skimage import measure
from skimage import morphology
from skimage.transform import resize
from sklearn.cluster import KMeans
def load_scan(path):
    slices = [pydicom.dcmread(path + '/' + s) for s in               
              os.listdir(path)]
    slices = [s for s in slices if 'SliceLocation' in s]
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
    # Set outside-of-scan pixels to 0
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

def dicom_animation(x):
    plt.figure(figsize = (15,5))
    plt.imshow(patient_pixels[x],cmap='gray')
    return x
path = '../input/osic-pulmonary-fibrosis-progression/train/ID00010637202177584971671'
patient_dicom = load_scan(path)
patient_pixels = get_pixels_hu(patient_dicom)

print('There are {} images in this scan.'.format(len(patient_pixels)))
print()
print('Pateint: ', path.split('/')[-1])
interact(dicom_animation, x=(0, len(patient_pixels)-1))
path= '../input/osic-pulmonary-fibrosis-progression/train/'
patients = os.listdir(path)
image_counts = []
for p in patients:
    number_of_images= len(os.listdir(path+p))
    #print(p, number_of_images)
    image_counts.append(number_of_images)

data={'Patients':patients,'Number of Scans':image_counts}

plt.figure(figsize = (20, 8))
p=sns.barplot(x='Patients', y='Number of Scans',data=data,palette="Blues_d")
plt.xlabel('Patient ID', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.title("Number of CT Scans per Patient", fontsize = 15)
path='../input/osic-pulmonary-fibrosis-progression/train/ID00019637202178323708467/1.dcm'
image= pydicom.dcmread(path)
print(image)
def create_gif(number_of_CT = 87):
    """Picks a patient at random and creates a GIF with their CT scans."""
    
    # Select one of the patients
    # patient = "ID00007637202177411956430"
    patient = patients[image_counts.index(number_of_CT)]
    
    print('Patient: ',patient)
    
    # === READ IN .dcm FILES ===
    patient_dir = "../input/osic-pulmonary-fibrosis-progression/train/" + patient
    datasets = []

    # First Order the files in the dataset
    files = []
    for dcm in list(os.listdir(patient_dir)):
        files.append(dcm) 
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Read in the Dataset from the Patient path
    for dcm in files:
        path = patient_dir + "/" + dcm
        datasets.append(pydicom.dcmread(path))
        
        
    # === SAVE AS .png ===
    # Create directory to save the png files
    if os.path.isdir(f"png_{patient}") == False:
        os.mkdir(f"png_{patient}")

    # Save images to PNG
    for i in range(len(datasets)):
        img = datasets[i].pixel_array
        matplotlib.image.imsave(f'png_{patient}/img_{i}.png', img)
        
        
    # === CREATE GIF ===
    # First Order the files in the dataset (again)
    files = []
    for png in list(os.listdir(f"../working/png_{patient}")):
        files.append(png) 
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Create the frames
    frames = []

    # Create frames
    for file in files:
    #     print("../working/png_images/" + name)
        new_frame = Image.open(f"../working/png_{patient}/" + file)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(f'gif_{patient}.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=200, loop=0)
create_gif()
show_gif(filename="./gif_ID00199637202248141386743.gif", format='png', width=400, height=400)
def split_lung_parenchyma(target,size,thr):
    img=cv2.imdecode(np.fromfile(target,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    try:
        img_thr= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,size,thr).astype(np.uint8)
    except:
        img_thr= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,999,thr).astype(np.uint8)
    img_thr=255-img_thr
    img_test=measure.label(img_thr, connectivity = 1)
    props = measure.regionprops(img_test)
    img_test.max()
    areas=[prop.area for prop in props]
    ind_max_area=np.argmax(areas)+1
    del_array = np.zeros(img_test.max()+1)
    del_array[ind_max_area]=1
    del_mask=del_array[img_test]
    img_new = img_thr*del_mask
    mask_fill=fill_water(img_new)
    img_new[mask_fill==1]=255
    img_out=img*~img_new.astype(bool)
    return img_out

def fill_water(img):
    copyimg = img.copy()
    copyimg.astype(np.float32)
    height, width = img.shape
    img_exp=np.zeros((height+20,width+20))
    height_exp, width_exp = img_exp.shape
    img_exp[10:-10, 10:-10]=copyimg
    mask1 = np.zeros([height+22, width+22],np.uint8)   
    mask2 = mask1.copy()
    mask3 = mask1.copy()
    mask4 = mask1.copy()
    cv2.floodFill(np.float32(img_exp), mask1, (0, 0), 1) 
    cv2.floodFill(np.float32(img_exp), mask2, (height_exp-1, width_exp-1), 1) 
    cv2.floodFill(np.float32(img_exp), mask3, (height_exp-1, 0), 1) 
    cv2.floodFill(np.float32(img_exp), mask4, (0, width_exp-1), 1)
    mask = mask1 | mask2 | mask3 | mask4
    output = mask[1:-1, 1:-1][10:-10, 10:-10]
    return output
fig= plt.figure(figsize=(16,6))
index= './png_ID00199637202248141386743/img_30.png'
a= fig.add_subplot(1,2,1)
a.set_title('Original Image')
plt.imshow(plt.imread(index))
plt.grid(None)

a= fig.add_subplot(1,2,2)
a.set_title('Masked Image')
img_split=split_lung_parenchyma('./png_ID00199637202248141386743/img_30.png',15599,-66)
plt.imshow(img_split)
plt.grid(None)

plt.show()

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
    
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    
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


    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    
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
img_split=make_lungmask(cv2.imdecode(np.fromfile(index,dtype=np.uint8),cv2.IMREAD_GRAYSCALE), display=True)
from skimage import measure, morphology, segmentation
import scipy.ndimage as ndimage

def generate_markers(image):
    #Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    #Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    #Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    
    return marker_internal, marker_external, marker_watershed

#Show some example markers from the middle   
path = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00010637202177584971671'
patient_dicom = load_scan(path)
patient_pixels = get_pixels_hu(patient_dicom)

test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(patient_pixels[50])

fig= plt.figure(figsize=(12,6))
a= fig.add_subplot(1,3,1)
a.set_title('Internal Marker')
plt.imshow(test_patient_internal, cmap='gray')
plt.grid(None)

a= fig.add_subplot(1,3,2)
a.set_title('External Marker')
plt.imshow(test_patient_external, cmap='gray')
plt.grid(None)

a= fig.add_subplot(1,3,3)
a.set_title('External Marker')
plt.imshow(test_patient_watershed, cmap='gray')
plt.grid(None)

plt.show()
def seperate_lungs(image):
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)
    
    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    
    #Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)
    
    #Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)
    
    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    #Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)
    
    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)
    
    #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000*np.ones((512, 512)))
    
    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed

#Some Testcode:
test_segmented, test_lungfilter, test_outline, test_watershed, test_sobel_gradient, test_marker_internal, test_marker_external, test_marker_watershed = seperate_lungs(patient_pixels[50])

print ("Sobel Gradient")
plt.imshow(test_sobel_gradient, cmap='gray')
plt.grid(None)
plt.show()
print ("Watershed Image")
plt.imshow(test_watershed, cmap='gray')
plt.grid(None)
plt.show()

print ("Outline after reinclusion")
plt.imshow(test_outline, cmap='gray')
plt.grid(None)
plt.show()
print ("Lungfilter after closing")
plt.imshow(test_lungfilter, cmap='gray')
plt.grid(None)
plt.show()
print ("Segmented Lung")
plt.imshow(test_segmented, cmap='gray')
plt.grid(None)
plt.show()
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate
)
albumentation_list =  [
    HorizontalFlip(p=0.5),
    RandomContrast(limit=0.5, p=1),
    RandomGamma(gamma_limit=(80, 120), p=1),
    RandomBrightness(limit=0.5, p=0.5),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1, 
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8), 
    ChannelShuffle(p=1),
    ElasticTransform(p=1,border_mode=cv2.BORDER_REFLECT_101,alpha_affine=60)
]


chosen_image= plt.imread('/kaggle/working/png_ID00199637202248141386743/img_30.png')
img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image = chosen_image)['image']
    img_matrix_list.append(img)
    
img_matrix_list.insert(0,chosen_image)    

titles_list = ["Original","Horizontal Flip","Random Contrast","Random Gamma","RandomBrightness",
               "Shift Scale Rotate","Channel Shuffle", "Elastic Transform"]
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title="Data Augmentation"):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=2, ncols=ncols, squeeze=True)
    fig.suptitle(main_title, fontsize = 30)
    #fig.subplots_adjust(wspace=0.3)
    #fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].grid(None)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
plot_multiple_img(img_matrix_list, titles_list, ncols = 4)
def strong_aug(p=1):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        #HueSaturationValue(p=0.3),
    ], p=p)
aug = strong_aug(p=1)
img = aug(image = chosen_image)['image']
plt.imshow(img)
plt.grid(None)