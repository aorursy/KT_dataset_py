# Load the required packages:

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Tools

from glob import glob

import os

from tqdm import tqdm_notebook

import pydicom

# Plotting

import matplotlib.pyplot as plt

import seaborn as sns

# Image procesing

from scipy import ndimage

import scipy.misc

from skimage import morphology

from skimage.segmentation import slic

from skimage import measure

from skimage.transform import resize, warp

from skimage import exposure

# Some machine learning as a treat

from sklearn.preprocessing import MaxAbsScaler

from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold
PATH = '../input/rsna-intracranial-hemorrhage-detection/' # Set up the path.

# Load the stage 1 file

train_csv = pd.read_csv(f'{PATH}stage_1_train.csv')

# Create a path to the train image location:

image_path = os.path.join(PATH, 'stage_1_train_images') + os.sep 

print(image_path)
# Check out this kernel: https://www.kaggle.com/currypurin/simple-eda 

# This is a really nice preprocessing of ID and labels :)

train_csv['Image ID'] = train_csv['ID'].apply(lambda x: x.split('_')[1]) 

train_csv['Sub-type'] = train_csv['ID'].apply(lambda x: x.split('_')[2]) 

train_csv = pd.pivot_table(train_csv, index='Image ID', columns='Sub-type')
train_csv.head()
# find hemorrhage images:

hem_img = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[12:16] # We will look only at a few

plt.figure(figsize=(15,10))

for n, ii in enumerate(hem_img.index):

    plt.subplot(2, 4, n + 1)

    img = pydicom.read_file(image_path + 'ID_' + ii + '.dcm').pixel_array # Read the pixel values

    tmp = hem_img.loc[ii]

    plt.title(tmp.unstack().columns[tmp.unstack().values.ravel() == 1][-1]) # Hacky way to give it a title... 

    plt.imshow(img, cmap='bone')

    plt.subplot(2, 4, n + 5)

    plt.hist(img.ravel())

def image_to_hu(image_path, image_id):

    ''' 

    Minimally adapted from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/

    '''

    dicom = pydicom.read_file(image_path + 'ID_' + image_id + '.dcm')

    image = dicom.pixel_array.astype(np.float64)

         

    # Convert to Hounsfield units (HU)

    intercept = dicom.RescaleIntercept

    slope = dicom.RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.float64)

        

    image += np.float64(intercept)

    

    image[image < -1024] = -1024 # Setting values smaller than air, to air.

    # Values smaller than -1024, are probably just outside the scanner.

    return image, dicom
# find hemorrhage images:

hem_img = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[12:16] # We will look only at a few

plt.figure(figsize=(15,10))

for n, img_id in enumerate(hem_img.index):

    plt.subplot(2, 4, n + 1)

    img, _ = image_to_hu(image_path, img_id)

    tmp = hem_img.loc[img_id]

    plt.title(tmp.unstack().columns[tmp.unstack().values.ravel() == 1][-1]) # Hacky way to give it a title... 

    plt.imshow(img, cmap='bone')

    plt.subplot(2, 4, n + 5)

    plt.hist(img.ravel())

def image_windowed(image, custom_center=50, custom_width=130, out_side_val=False):

    '''

    Important thing to note in this function: The image migth be changed in place!

    '''

    # see: https://www.kaggle.com/allunia/rsna-ih-detection-eda-baseline

    min_value = custom_center - (custom_width/2)

    max_value = custom_center + (custom_width/2)

    

    # Including another value for values way outside the range, to (hopefully) make segmentation processes easier. 

    out_value_min = custom_center - custom_width

    out_value_max = custom_center + custom_width

    

    if out_side_val:

        image[np.logical_and(image < min_value, image > out_value_min)] = min_value

        image[np.logical_and(image > max_value, image < out_value_max)] = max_value

        image[image < out_value_min] = out_value_min

        image[image > out_value_max] = out_value_max

    

    else:

        image[image < min_value] = min_value

        image[image > max_value] = max_value

    

    return image
# find hemorrhage images:

hem_img = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[12:16] # We will look only at a few

plt.figure(figsize=(15,10))

for n, img_id in enumerate(hem_img.index):

    plt.subplot(2, 4, n + 1)

    img, _ = image_to_hu(image_path, img_id)

    img = image_windowed(img, out_side_val=False)

    tmp = hem_img.loc[img_id]

    plt.title(tmp.unstack().columns[tmp.unstack().values.ravel() == 1][-1]) # Hacky way to give it a title... 

    plt.imshow(img, cmap='bone')

    plt.subplot(2, 4, n + 5)

    plt.hist(img.ravel())

def image_resample(image, dicom_header, new_spacing=[1,1]):

    # Code from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/

    # Adapted to work for pixels.

    spacing = map(float, dicom_header.PixelSpacing)

    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    

    return image
tmp = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[np.random.randint(

    train_csv.iloc[train_csv['Label']['any'].values == 1].shape[0])].name



hu_img, dicom_header = image_to_hu(image_path, tmp)

resamp_img = image_resample(hu_img, dicom_header)



# Window images, for visualization

hu_img = image_windowed(hu_img)

resamp_img = image_windowed(resamp_img)



plt.figure(figsize=(7.5, 5))

plt.subplot(121)

plt.imshow(hu_img, cmap='bone')

plt.axis('off')

plt.title(f'Orig shape\n{hu_img.shape}')

plt.subplot(122)

plt.imshow(resamp_img, cmap='bone')

plt.title(f'New shape\n{resamp_img.shape}');

plt.axis('off');
def image_background_segmentation(image_path, image_id, WW=40, WL=80, display=False):

    img, dcm_head = image_to_hu(image_path, image_id)

    img = image_resample(img, dcm_head)

    img_out = img.copy()

    # use values outside the window as well, helps with segmentation

    img = image_windowed(img, custom_center=WW, custom_width=WL, out_side_val=True)

    

    # Calculate the outside values by hand (again)

    lB = WW - WL

    uB = WW + WL

    

    # Keep only values inside of the window

    background_seperation = np.logical_and(img > lB, img < uB)

    

    # Get largest connected component:

    # From https://github.com/nilearn/nilearn/blob/master/nilearn/_utils/ndimage.py

    background_seperation = morphology.dilation(background_seperation,  np.ones((5, 5)))

    labels, label_nb = scipy.ndimage.label(background_seperation)

    

    label_count = np.bincount(labels.ravel().astype(np.int))

    # discard the 0 label

    label_count[0] = 0

    mask = labels == label_count.argmax()

    

    # Fill holes in the mask

    mask = morphology.dilation(mask, np.ones((5, 5))) # dilate the mask for less fuzy edges

    mask = scipy.ndimage.morphology.binary_fill_holes(mask)

    mask = morphology.dilation(mask, np.ones((3, 3))) # dilate the mask again



    if display:

        plt.figure(figsize=(15,2.5))

        plt.subplot(141)

        plt.imshow(img, cmap='bone')

        plt.title('Original Images')

        plt.axis('off')



        plt.subplot(142)

        plt.imshow(background_seperation)

        plt.title('Segmentation')

        plt.axis('off')



        plt.subplot(143)

        plt.imshow(mask)

        plt.title('Mask')

        plt.axis('off')



        plt.subplot(144)

        plt.imshow(mask * img, cmap='bone')

        plt.title('Image * Mask')

        plt.suptitle(image_id)

        plt.axis('off')



    return mask * img_out
for ii in range(5):

    tmp = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[np.random.randint(

        train_csv.iloc[train_csv['Label']['any'].values == 1].shape[0])].name

    masked_image = image_background_segmentation(image_path, tmp, display=True)
def image_crop(image):

    # Based on this stack overflow post: https://stackoverflow.com/questions/26310873/how-do-i-crop-an-image-on-a-white-background-with-python

    mask = image == 0



    # Find the bounding box of those pixels

    coords = np.array(np.nonzero(~mask))

    top_left = np.min(coords, axis=1)

    bottom_right = np.max(coords, axis=1)



    out = image[top_left[0]:bottom_right[0],

                top_left[1]:bottom_right[1]]

    

    return out
plt.figure(figsize=(7.5,5))

for ii in range(3):

    tmp = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[np.random.randint(

        train_csv.iloc[train_csv['Label']['any'].values == 1].shape[0])].name

    masked_image = image_background_segmentation(image_path, tmp, False)

    masked_image = image_windowed(masked_image)

    cropped_image = image_crop(masked_image)

    plt.subplot(1, 3, ii + 1)

    plt.imshow(cropped_image, cmap='bone')

    plt.title(f'Image Shape:\n{cropped_image.shape}')

    plt.axis('off')
def image_pad(image, new_height, new_width):

    # based on https://stackoverflow.com/questions/26310873/how-do-i-crop-an-image-on-a-white-background-with-python

    height, width = image.shape



    # make canvas

    im_bg = np.zeros((new_height, new_width))



    # Your work: Compute where it should be

    pad_left = int( (new_width - width) / 2)

    pad_top = int( (new_height - height) / 2)



    im_bg[pad_top:pad_top + height,

          pad_left:pad_left + width] = image



    return im_bg
plt.figure(figsize=(7.5, 5))

for ii in range(3):

    tmp = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[np.random.randint(

        train_csv.iloc[train_csv['Label']['any'].values == 1].shape[0])].name

    masked_image = image_background_segmentation(image_path, tmp, False)

    masked_image = image_windowed(masked_image)

    cropped_image = image_crop(masked_image)

    padded_image = image_pad(cropped_image, 256, 256)

    plt.subplot(1, 3, ii + 1)

    plt.imshow(padded_image, cmap='bone')

    plt.title(f'Image Shape:\n{padded_image.shape}')

    plt.axis('off')
plt.figure(figsize=(7.5, 5))

for ii in range(3):

    tmp = train_csv.iloc[train_csv['Label']['any'].values == 1].iloc[np.random.randint(

        train_csv.iloc[train_csv['Label']['any'].values == 1].shape[0])].name

    masked_image = image_background_segmentation(image_path, tmp, False)

    masked_image = image_windowed(masked_image)

    cropped_image = image_crop(masked_image)

    padded_image = image_pad(cropped_image, 256, 256)

    padded_image = MaxAbsScaler().fit_transform(padded_image.reshape(-1, 1)).reshape([256, 256])

    plt.subplot(2, 3, ii + 1)

    plt.imshow(padded_image, cmap='bone')

    plt.title(f'Image Shape:\n{padded_image.shape}')

    plt.axis('off')

    plt.subplot(2, 3, ii + 4)

    plt.hist(padded_image.ravel())
hist_bins = np.array([ 0.,  5.,  10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.,

       65., 70., 75., 80., 85., 90., 95., 100.])
def extract_histogram(image_path, image_id, hist_bins):

    # hu_img, dicom_header = get_pixels_hu(image_path, image_id)

    # windowed_img = set_manual_window(hu_img.copy(), custom_center=40, custom_width=80)

    try:

        masked_image = image_background_segmentation(image_path, image_id, False)

        masked_image = image_windowed(masked_image)

        cropped_image = image_crop(masked_image)



        val, _ = np.histogram(cropped_image.flatten(), bins=hist_bins)

        tmp = val[1:-1] # Remove the first and last bin, as they are probably noisy

        tmp = (tmp - np.mean(tmp)) / np.std(tmp) # z-score

    except:

        tmp=np.zeros(18)

    return tmp 
xTrain = np.zeros((6000, 18))

yTrain = np.hstack([np.zeros((3000)), np.ones((3000))])
no_hem = train_csv.iloc[train_csv['Label']['any'].values == 0].index.values

hem = train_csv.iloc[train_csv['Label']['any'].values == 1].index.values



for ii, tmp in tqdm_notebook(enumerate(np.random.choice(no_hem, 3000, replace=False))):

    xTrain[ii, :] = extract_histogram(image_path, tmp, hist_bins)



for ii, tmp in tqdm_notebook(enumerate(np.random.choice(hem, 3000, replace=False))):

    xTrain[ii+3000, :] = extract_histogram(image_path, tmp, hist_bins)

    

xTrain[np.isnan(xTrain)] = 0 # somehow, there are some nans in there
from scipy.stats import ttest_ind

plt.figure(figsize=(15,10))

for n, vec in enumerate(xTrain.T):

    plt.subplot(3,6, n + 1)

    sns.distplot(vec[yTrain==0])

    sns.distplot(vec[yTrain==1])

    t, p = ttest_ind(vec[yTrain==0], vec[yTrain==1])

    plt.title(f'HU bin {hist_bins[n+1]:2.0f}, t={t:4.2f}')



plt.axis('tight');
from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold



SKF = StratifiedKFold(5)



logReg_score = cross_val_score(LogisticRegressionCV(cv=5), xTrain, yTrain, cv = SKF)

rfClf_score = cross_val_score(RandomForestClassifier(n_estimators=100), xTrain, yTrain, cv = SKF)



print(f'Logistic Regression Accuracy: {np.mean(logReg_score):4.3f} +/- {np.std(logReg_score):4.3f}')

print(f'Random Forest Accuracy: {np.mean(rfClf_score):4.3f} +/- {np.std(rfClf_score):4.3f}')