from IPython.display import YouTubeVideo



YouTubeVideo('1IBvrOBQ268', width=800, height=600)
# data reading and manipulation libraries

import numpy as np, pandas as pd



# navigation and directory management libraries

import os, glob, pydicom, imageio



#data visualization libraries

import matplotlib.pyplot as plt, seaborn as sns,  plotly.express as px

from IPython import display



# Machine Learning Libraries

import tensorflow as tf

import keras
PATH = "../input/rsna-str-pulmonary-embolism-detection/"



train_df = pd.read_csv(PATH + "train.csv")

test_df = pd.read_csv(PATH + "test.csv")



TRAIN_PATH = PATH + "train/"

TEST_PATH = PATH + "test/"

sub = pd.read_csv(PATH + "sample_submission.csv")

train_image_file_paths = glob.glob(TRAIN_PATH + '/*/*/*.dcm')

test_image_file_paths = glob.glob(TEST_PATH + '/*/*/*.dcm')



print(f'Train dataframe shape  :{train_df.shape}')

print(f'Test dataframe shape   :{test_df.shape}')



print(f'Number of train images : {len(train_image_file_paths)}')

print(f'Number of test images  : {len(test_image_file_paths)}')
def read_dicom(file_path, show = False, cmap = 'gray'):

    im = pydicom.read_file(file_path)

    image_unscaled = im.pixel_array

    image_rescaled = im.pixel_array * im.RescaleSlope + im.RescaleIntercept

    

    image_rescaled[image_rescaled <-1500] = 0

    

    if show:

        f, axarr = plt.subplots(1,2)

        axarr[0].imshow(image_unscaled, cmap = cmap)

        axarr[0].axis(False)

        axarr[0].set_title('no_rescale')

        

        axarr[1].imshow(image_rescaled, cmap = cmap)

        axarr[1].axis(False)

        axarr[1].set_title('windowed')

    return image_rescaled





image = read_dicom(train_image_file_paths[2200], show = True)

image.dtype
train_df.info()
train_df.head(30)
test_df.info()
test_df.head()
sub.info()
from tqdm.notebook import tqdm

prediction_counts = {}

for idx in tqdm(range(sub.shape[0])):

    if len(sub['id'][idx][13:]) > 1:

        key = sub['id'][idx][13:]

    else:

        key = 'pe_present_on_image'

    prediction_counts[key] = prediction_counts.get(key, 0) + 1

print(f'Total row count in submission: {sub.shape[0]}')

prediction_counts
N_img = 146853

N_exams = 650

N_exam_level_features = 9



total_rows_submission = N_img + (N_exams * N_exam_level_features)

print(f'Total row count in submission: {total_rows_submission}')
def plot_bar(col):

    ds = train_df[col].value_counts().reset_index()

    ds.columns = ['value', 'number']

    fig = px.bar(ds, x='value', y="number", orientation='v',title='Distribution of train set for ' + col, width=500, height=400 )

    fig.show()



plot_bar('pe_present_on_image')
dcm_file =  pydicom.read_file("../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf/00ac73cfc372.dcm")

print(dcm_file.file_meta)
dcm_file
image = dcm_file.pixel_array

print(f'Image Size: {image.shape}')
fig, ax = plt.subplots(2,1,figsize=(20,10))

for file in train_image_file_paths[0:10]:

    dataset = pydicom.read_file(file)

    image = dataset.pixel_array.flatten()

    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept

    sns.distplot(image.flatten(), ax=ax[0]);

    sns.distplot(rescaled_image.flatten(), ax=ax[1])

ax[0].set_title("Raw pixel array distributions for 10 examples");
counter  = 0

rows = 3

cols = 5

fig = plt.figure(figsize=(25,15))

for i in range(1, rows*cols+1):

    img = read_dicom(train_image_file_paths[counter + i])

    fig.add_subplot(rows, cols, i)

    plt.imshow(img, cmap='gray')

    plt.title(f'[{img.min()} {img.max()}]')

    plt.axis(False)

    fig.add_subplot

counter += rows*cols
selected_exam = 10

EXAM_IDs = os.listdir(TRAIN_PATH)

SERIES = os.listdir(TRAIN_PATH + '/' + EXAM_IDs[selected_exam])

files = os.listdir(TRAIN_PATH + '/' + EXAM_IDs[selected_exam] + '/' + SERIES[0])

single_experiment_files = [TRAIN_PATH + '/' + EXAM_IDs[selected_exam] + '/' + SERIES[0] + '/' + file for file in files]

counter  = 0

rows = 3

cols = 5

fig = plt.figure(figsize=(25,15))

for i in range(1, rows*cols+1):

    fig.add_subplot(rows, cols, i)

    plt.imshow(read_dicom(single_experiment_files[counter + i]), cmap='gray')

    plt.axis(False)

    fig.add_subplot

counter += rows*cols
def load_slice(paths):

    slices = [pydicom.read_file(path) for path in paths]

#     labels = [train_df[train_df.SOPInstanceUID == path[-16:-4]].pe_present_on_image.values for path in patient_image_paths]

#     labels = np.array(labels).squeeze()

    slices.sort(key = lambda x: int(x.InstanceNumber), reverse = False)

#     labels.sort(key = lambda x: int(x.InstanceNumber), reverse = False)

    

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)    

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def transform_to_hu(slices):

    images = np.stack([file.pixel_array for file in slices])

    images = images.astype(np.int16)



    # convert ouside pixel-values to air:

    # I'm using <= -1000 to be sure that other defaults are captured as well

    images[images <= -1000] = 0

    

    # convert to HU

    for n in range(len(slices)):    

        intercept = slices[n].RescaleIntercept

        slope = slices[n].RescaleSlope

        if slope != 1:

            images[n] = slope * images[n].astype(np.float64)

            images[n] = images[n].astype(np.int16)      

        images[n] += np.int16(intercept)

    return np.array(images, dtype=np.int16)
stacked_dicoms = load_slice(single_experiment_files)

stacked_patient_pixels = transform_to_hu(stacked_dicoms)



def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):

    fig,ax = plt.subplots(rows,cols,figsize=[20,22])

    for i in range(rows*cols):

        ind = start_with + i*show_every

        ax[int(i/rows),int(i % rows)].set_title(f'slice {ind}')

        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')

        ax[int(i/rows),int(i % rows)].axis('off')

    plt.show()



print(f'Total Number of Slices: {len(stacked_patient_pixels)}')

sample_stack(stacked_patient_pixels, 

             show_every = int((len(stacked_patient_pixels)-10)/36))
imageio.mimsave(f'stacked_{EXAM_IDs[selected_exam]}.gif', stacked_patient_pixels, duration=0.1)

display.Image(f'stacked_{EXAM_IDs[selected_exam]}.gif', format='png')