# Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import random

import math

import matplotlib

from termcolor import colored

import os

from os import listdir

from os.path import join, getsize

import glob

import cv2



#plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



# 

from skimage import measure

from skimage.morphology import disk, opening, closing

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Dropout



from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dropout





# Magic function to display In-Notebook display

%matplotlib inline



# Setting seabon style

sns.set(style='darkgrid', palette='Set2')



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')



# Settings for pretty nice plots

plt.style.use('fivethirtyeight')

plt.show()



# pydicom

import pydicom



# Print versions of libraries

print(f"Numpy version : Numpy {np.__version__}")

print(f"Pandas version : Pandas {pd.__version__}")

print(f"Matplotlib version : Matplotlib {matplotlib.__version__}")

print(f"Seaborn version : Seaborn {sns.__version__}")

print(f"Tensorflow version : Tensorflow {tf.__version__}")
def seed_everything(seed=100):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(101)
# List files available

base_dir = "../input/osic-pulmonary-fibrosis-progression/"
# List files available

list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))
# Train & Test set shape

train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv', encoding = 'latin-1')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv', encoding = 'latin-1')

submission_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv', encoding = 'latin-1')



print(colored('Training data set shape.......... : ','yellow'),train_df.shape)

print(colored('Test data set shape...............: ','red'),test_df.shape)

print(colored('Submission data set shape.........: ','blue'),submission_df.shape)
# print top 5 rows of train set

train_df.head()
# print top 5 rows of test set

test_df.head()
# Null values and Data types

print(colored('Train Set !!', 'yellow'))

print(colored('------------', 'yellow'))

print(train_df.info())



print('\n')



print(colored('Test Set !!','red'))

print(colored('-----------','red'))

print(test_df.info())
# Null values and Data types

print(colored('Train Set !!', 'yellow'))

print(train_df.describe())
# Total missing values for each feature

print(colored('Missing values in Train Set !!', 'yellow'))

print(train_df.isnull().sum())



print("\n")



print(colored('Missing values in Test Set !!', 'red'))

print(test_df.isnull().sum())
train_df.groupby( ['Sex','SmokingStatus'] )['FVC'].agg( ['mean','std','count'] ).sort_values(by=['Sex','count'],ascending=False)
# Total number of Patient in the dataset(train+test)



print(colored("Total Patients in Train set... : ", 'yellow'),train_df['Patient'].count())

print(colored("Total Patients in Test set.... : ", 'red'),test_df['Patient'].count())

print("\n")

print(colored("Unique Patients in Train set...: ", 'yellow'),train_df['Patient'].nunique())

print(colored("Unique Patients in Test set....: ", 'red'),test_df['Patient'].nunique())
print(colored("Few most repeated Patients in Train set: ", 'yellow'))

print(train_df['Patient'].value_counts().head())



print("\n")



print(colored("Few most repeated Patients in Test set: ", 'red'))

print(test_df['Patient'].value_counts().head())
train_df_unique = train_df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates().reset_index()

print(colored("Shape of unique patient data set : ",'yellow'),train_df_unique.shape)

train_df_unique.head()
patient_feq = train_df.groupby(['Patient'])['Patient'].count()

patient_feq = pd.DataFrame({'Patient':patient_feq.index, 'Frequency':patient_feq.values})



# Merge two dataframes based on patient's ids.

train_df_unique = pd.merge(train_df_unique,patient_feq,how='inner',on='Patient')
train_df_unique.sort_values(by='Frequency', ascending=False).head()
fig = px.bar(train_df_unique, x='Patient',y ='Frequency',color='Frequency')

fig.update_layout(xaxis={'categoryorder':'total ascending'},title='Frequency of each patient')

fig.update_xaxes(showticklabels=False)

fig.show()
# Creating unique patient lists 

# (here patient == dictory and files == CT Scan)

train_dir = '../input/osic-pulmonary-fibrosis-progression/train/'



patient_ids = os.listdir(train_dir)

patient_ids = sorted(patient_ids)



#Creating a new blank dataframe

CtScan = pd.DataFrame(columns=['Patient','CtScanCount'])





for patient_id in patient_ids:

    # count number of images in each folder

    cnt = len(os.listdir(train_dir + patient_id))

    # insert patient id and ct scan count in dataframe

    CtScan.loc[len(CtScan)] = [patient_id,cnt]

    



# Merge two dataframes based on patient's ids.

patient_df = pd.merge(train_df_unique,CtScan,how='inner',on='Patient')



# Reset index

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html

patient_df = patient_df.reset_index(drop=True)



# Print new dataframe

patient_df.head()

print(colored("CT Scans numbers in Train set ","yellow"))

print(colored("Maximum number of CT Scans for a patient.... : ","blue"),patient_df['CtScanCount'].max())

print(colored("Minimum number of CT Scans for a patient.... : ","blue"),patient_df['CtScanCount'].min())

print(colored("Average number of CT Scans per patient...... : ","blue"),round(patient_df['CtScanCount'].mean(),3))

print(colored("Total number of CT Scans of all patients.... : ","blue"),patient_df['CtScanCount'].sum())

print(colored("Median of CT Scans counts................... : ","blue"),patient_df['CtScanCount'].median())
# Creating unique patient lists 

# (here patient == dictory and files == CT Scan)

test_dir = '../input/osic-pulmonary-fibrosis-progression/test/'



test_patient_ids = os.listdir(test_dir)

test_patient_ids = sorted(test_patient_ids)



#Creating a new blank dataframe

TestCtScan = pd.DataFrame(columns=['Patient','CtScanCount'])



for patient_id in test_patient_ids:

    # count number of images in each folder

    cnt = len(os.listdir(test_dir + patient_id))

    # insert patient id and ct scan count in dataframe

    TestCtScan.loc[len(TestCtScan)] = [patient_id,cnt]

    



# Merge two dataframes based on patient's ids.

test_patient_df = pd.merge(test_df,TestCtScan,how='inner',on='Patient').reset_index()



# Print new dataframe

test_patient_df.head()
print(colored("CT Scans numbers in Test set ","red"))

print(colored("Maximum number of CT Scans for a patient... : ","green"),test_patient_df['CtScanCount'].max())

print(colored("Minimum number of CT Scans for a patient... : ","green"),test_patient_df['CtScanCount'].min())

print(colored("Average number of CT Scans per patient..... : ","green"),test_patient_df['CtScanCount'].mean())

print(colored("Total number of CT Scans of all patients... : ","green"),test_patient_df['CtScanCount'].sum())
train_df['Weeks'].iplot(kind='hist',

                        bins=100, xTitle='Weeks', yTitle='Frequency', 

                        linecolor='white',opacity=0.7,

                        color='rgb(0, 200, 200)', theme='pearl',

                        bargap=0.01, title='Distribution of Weeks')
patient_df['Age'].iplot(kind='hist',

                        bins=10, xTitle='Age', yTitle='Frequency', 

                        linecolor='white',opacity=0.7,

                        color='rgb(0, 100, 200)', theme='pearl',

                        bargap=0.01, title='Distribution of Age column')
print(colored("Gender wise distribution of patients :","blue"))

print(patient_df['Sex'].value_counts())
sex_count = patient_df["Sex"].value_counts()

sex_labels = patient_df["Sex"].unique()



fig = px.pie(patient_df, values=sex_count, names=sex_labels, hover_name=sex_labels)

fig.show()
plt.figure(figsize=(16, 6))



sns.kdeplot(patient_df[patient_df['Sex'] == 'Male']['Age'], label = 'Male',shade=True)

sns.kdeplot(patient_df[patient_df['Sex'] == 'Female']['Age'], label = 'Female',shade=True)



plt.xlabel('Age (years)'); 

plt.ylabel('Density'); 

plt.title('Distribution of Ages');
print(colored('Total Smoking counts', 'red'))

print(patient_df['SmokingStatus'].value_counts())



print("\n")

print(colored("Male Smoking counts",'blue'))

print(patient_df[patient_df['Sex']=='Male']['SmokingStatus'].value_counts())



print("\n")

print(colored("Female Smoking counts",'green'))

print(patient_df[patient_df['Sex']=='Female']['SmokingStatus'].value_counts())
plt.figure(figsize=(16, 6))



sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently smokes', shade=True)



# Labeling of plot

plt.xlabel('Age (years)'); 

plt.ylabel('Density'); 

plt.title('Distribution of Ages');
plt.figure(figsize=(10,8))

sns.countplot(x='SmokingStatus', data=patient_df, hue='Sex')

plt.title('Gender split by SmokingStatus', fontsize=16)

plt.show()
print(colored("Maximum value of FVC... :",'blue'),colored(train_df['FVC'].max(),'blue'))

print(colored("Minimum value of FVC... :",'green'),colored(train_df['FVC'].min(),'green'))



print("\n")



# Distribution of FVC

print(colored("Distribution of FVC","yellow"))

print(colored(train_df['FVC'].value_counts(normalize=False, ascending=False, bins=62).head(),"yellow"))
train_df['FVC'].iplot(kind='hist',

                      xTitle='Lung Capacity(ml)', 

                      yTitle='Frequency', 

                      linecolor='black', 

                      bargap=0.2,

                      title='Distribution of the FVC in the training set')
fig = px.violin(train_df, y='FVC', x='SmokingStatus', 

                box=True, color='Sex', points="all", hover_data=train_df.columns, title="FVC of various Smoking Status")

fig.show()
fig = px.scatter(train_df, x="Age", y="FVC", color='Sex', title='FVC values for Patient Age')

fig.show()
train_df[train_df['FVC'] > 5000].sort_values(by='FVC', ascending=False)
fig = go.Figure()

fig = px.scatter(train_df, x="Weeks", y="FVC", color='SmokingStatus')

fig.show()
patient = train_df[(train_df['Age'] == train_df['Age'].max()) | (train_df['Age'] == train_df['Age'].min())]

fig = px.line(patient, x="Weeks", y="FVC", color='Age',line_group="Sex", hover_name="SmokingStatus")

fig.show()
print(colored("Maximum value of Percent... :",'blue'),colored(train_df['Percent'].max(),'blue'))

print(colored("Minimum value of Percent... :",'green'),colored(train_df['Percent'].min(),'green'))



print("\n")



# Distribution of Percent

print(colored("Distribution of Percent","yellow"))

print(colored(train_df['Percent'].value_counts(normalize=False, ascending=False, bins=62).head(),"yellow"))
train_df['Percent'].iplot(kind='hist',

                      xTitle='Percent', 

                      yTitle='Frequency', 

                      linecolor='black', 

                      bargap=0.2,

                      title='Distribution of Percent in the training set')
fig = px.violin(train_df, y='Percent', x='SmokingStatus', 

                box=True, color='Sex', points="all", hover_data=train_df.columns, title="Percent of various Smoking Status")

fig.show()
patient = train_df[(train_df['Age'] == train_df['Age'].max()) | (train_df['Age'] == train_df['Age'].min())]

fig = px.line(patient, x="Weeks", y="Percent", color='Age',line_group="Sex", hover_name="SmokingStatus")



patient = train_df[(train_df['Age'] == train_df['Age'].max()) | (train_df['Age'] == train_df['Age'].min())]

fig = px.line(patient, x="Weeks", y="Percent", color='Age',line_group="Sex", hover_name="SmokingStatus")



fig.show()
fig = px.scatter(train_df, x="Age", y="Percent", color="SmokingStatus", marginal_y="violin",

           marginal_x="box", trendline="ols", template="simple_white")

fig.show()
fig = px.scatter(train_df, x="FVC", y="Percent", color='SmokingStatus', size='Age', 

                 hover_name='SmokingStatus',hover_data=['Weeks'])

fig.show()
corrmat = train_df.corr() 

fig = px.imshow(corrmat, x=corrmat.columns, y=corrmat.columns)

fig.update_xaxes(side="top")

fig.show()
## Patients & their CT Scans in Training Images Folder



file_len = folder_len = 0

files = []



for dirpath, dirnames, filenames in os.walk(train_dir):

    file_len += len(filenames)

    folder_len += len(dirnames)

    files.append(len(filenames))



print("Training folder contains", f'{file_len:,}', "CT scans for all patients.") 

print('Training folder have only',f'{folder_len:,}', "unique patients.")



print("\n")



print('Each patient have', f'{round(np.mean(files)):,}', 'average number of CT scans.')

print('Maximum images per patient', f'{round(np.max(files)):,}')

print('Minimum images per patient', f'{round(np.min(files)):,}')
# https://www.kaggle.com/schlerp/getting-to-know-dicom-and-the-data



def show_dcm_info(file_path):

    #print(colored("Filename.........:",'yellow'),file_path)

    #print()

    print(colored("File Path...........:",'blue'), file_path)

    

    dataset = pydicom.dcmread(file_path)



    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    

    print(colored("Patient's name......:",'blue'), display_name)

    print(colored("Patient id..........:",'blue'), dataset.PatientID)

    print(colored("Patient's Sex.......:",'blue'), dataset.PatientSex)

    print(colored("Modality............:",'blue'), dataset.Modality)

    print(colored("Body Part Examined..:",'blue'), dataset.BodyPartExamined)

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print(colored("Image size..........:",'blue')," {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print(colored("Pixel spacing.......:",'blue'),dataset.PixelSpacing)

            dataset.PixelSpacing = [1, 1]

        plt.figure(figsize=(10, 10))

        plt.imshow(dataset.pixel_array, cmap='gray')

        plt.show()
for file_path in glob.glob(train_dir + '*/*.dcm'):

    show_dcm_info(file_path)

    break # Comment this out to see all
show_dcm_info(train_dir + 'ID00027637202179689871102/11.dcm')
patient_dir = train_dir + "ID00123637202217151272140"



print("total images for patient ID00123637202217151272140: ", len(os.listdir(patient_dir)))



# view first (columns*rows) images in order

fig=plt.figure(figsize=(16, 16))

columns = 4

rows = 5

imglist = os.listdir(patient_dir)

for i in range(1, columns*rows +1):

    filename = patient_dir + "/" + str(i) + ".dcm"

    ds = pydicom.dcmread(filename)

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap='gray')

plt.show()
# view first (columns*rows) images in order

fig=plt.figure(figsize=(16, 16))

columns = 4

rows = 5

imglist = os.listdir(patient_dir)

for i in range(1, columns*rows +1):

    filename = patient_dir + "/" + str(i) + ".dcm"

    ds = pydicom.dcmread(filename)

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap='jet')

    #plt.imshow(cv2.cvtColor(ds.pixel_array, cv2.COLOR_BGR2RGB))

plt.show()
# Ref : 

# https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial

# https://www.kaggle.com/akh64bit/full-preprocessing-tutorial

# https://www.researchgate.net/post/How_can_I_convert_pixel_intensity_values_to_housefield_CT_number



# Load the scans in given folder path

def load_scan(path):

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

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
first_patient = load_scan(train_dir + patient_ids[0])

first_patient_pixels = get_pixels_hu(first_patient)



plt.figure(figsize=(10, 10))

plt.hist(first_patient_pixels.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()



# Show some slice in the middle

plt.figure(figsize=(10, 10))

plt.imshow(first_patient_pixels[15], cmap=plt.cm.gray)

plt.show()
first_patient_scan = load_scan(train_dir + patient_ids[0])
first_patient_scan[0]
def set_lungwin(img, hu=[-1200., 600.]):

    lungwin = np.array(hu)

    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])

    newimg[newimg < 0] = 0

    newimg[newimg > 1] = 1

    newimg = (newimg * 255).astype('uint8')

    return newimg
first_patient_scan_array = set_lungwin(get_pixels_hu(first_patient_scan))
import imageio

from IPython.display import Image



imageio.mimsave("/tmp/gif.gif", first_patient_scan_array, duration=0.00001)

Image(filename="/tmp/gif.gif", format='png')
fig, ax = plt.subplots(1,2,figsize=(20,5))

for n in range(10):

    image = first_patient_scan[n].pixel_array.flatten()

    rescaled_image = image * first_patient_scan[n].RescaleSlope + first_patient_scan[n].RescaleIntercept

    sns.distplot(image.flatten(), ax=ax[0]);

    sns.distplot(rescaled_image.flatten(), ax=ax[1])

ax[0].set_title("Raw pixel array distributions for 10 examples")

ax[1].set_title("HU unit distributions for 10 examples")
fig, ax = plt.subplots(1,4,figsize=(20,3))

ax[0].set_title("Original CT-scan")

ax[0].imshow(first_patient_scan[0].pixel_array, cmap="bone")

ax[1].set_title("Pixelarray distribution");

sns.distplot(first_patient_scan[0].pixel_array.flatten(), ax=ax[1]);



ax[2].set_title("CT-scan in HU")

ax[2].imshow(first_patient_pixels[0], cmap="bone")

ax[3].set_title("HU values distribution");

sns.distplot(first_patient_pixels[0].flatten(), ax=ax[3]);



for m in [0,2]:

    ax[m].grid(False)
def segment_lung_mask(image):

    segmented = np.zeros(image.shape)   

    

    for n in range(image.shape[0]):

        binary_image = np.array(image[n] > -320, dtype=np.int8)+1

        labels = measure.label(binary_image)

        

        background_label_1 = labels[0,0]

        background_label_2 = labels[0,-1]

        background_label_3 = labels[-1,0]

        background_label_4 = labels[-1,-1]

    

        #Fill the air around the person

        binary_image[background_label_1 == labels] = 2

        binary_image[background_label_2 == labels] = 2

        binary_image[background_label_3 == labels] = 2

        binary_image[background_label_4 == labels] = 2

    

        #We have a lot of remaining small signals outside of the lungs that need to be removed. 

        #In our competition closing is superior to fill_lungs 

        selem = disk(4)

        binary_image = closing(binary_image, selem)

    

        binary_image -= 1 #Make the image actual binary

        binary_image = 1-binary_image # Invert it, lungs are now 1

        

        segmented[n] = binary_image.copy() * image[n]

    

    return segmented
segmented = segment_lung_mask(np.array([first_patient_pixels[20]]))



fig, ax = plt.subplots(1,2,figsize=(20,10))

ax[0].imshow(first_patient_pixels[20], cmap="Blues_r")

ax[1].imshow(segmented[0], cmap="Blues_r")
segmented_lungs = segment_lung_mask(first_patient_pixels)
segmented_lungs.shape
fig, ax = plt.subplots(6,5, figsize=(20,20))

for n in range(6):

    for m in range(5):

        ax[n,m].imshow(segmented_lungs[n*5+m], cmap="Blues_r")
