import os

from os import listdir

import pandas as pd

import numpy as np

import glob

import tqdm

from typing import Dict

import matplotlib.pyplot as plt

%matplotlib inline



!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



from colorama import Fore, Back, Style



import seaborn as sns

sns.set(style="whitegrid")



import pydicom



import warnings

warnings.filterwarnings('ignore')

3

plt.style.use('fivethirtyeight')

plt.show()
list(os.listdir('../input/osic-pulmonary-fibrosis-progression'))
IMAGE_PATH = "../input/osic-pulmonary-fibrosis-progression/"



train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')



print(Fore.YELLOW + 'Training data shape: ',Style.RESET_ALL,train_df.shape)

train_df.head(5)
train_df.groupby(['SmokingStatus']).count()['Sex'].to_frame()
print(Fore.YELLOW + 'Train Set !!',Style.RESET_ALL)

print(train_df.info())

print('-------------')

print(Fore.BLUE + 'Test Set !!',Style.RESET_ALL)

print(test_df.info())
train_df.isnull().sum()
test_df.isnull().sum()
print(Fore.YELLOW +"Total Patients in Train set: ",Style.RESET_ALL,train_df['Patient'].count())

print(Fore.BLUE +"Total Patients in Test set: ",Style.RESET_ALL,test_df['Patient'].count())
print(Fore.YELLOW + "The total patient ids are",Style.RESET_ALL,f"{train_df['Patient'].count()},", Fore.BLUE + "from those the unique ids are", Style.RESET_ALL, f"{train_df['Patient'].value_counts().shape[0]}.")
train_patient_ids = set(train_df['Patient'].unique())

test_patient_ids = set(test_df['Patient'].unique())



train_patient_ids.intersection(test_patient_ids)
columns = train_df.keys()

columns = list(columns)

print(columns)
train_df['Patient'].value_counts().max()
test_df['Patient'].value_counts().max()
np.quantile(train_df['Patient'].value_counts(), 0.75) - np.quantile(test_df['Patient'].value_counts(), 0.25)
print(np.quantile(train_df['Patient'].value_counts(), 0.95))

print(np.quantile(test_df['Patient'].value_counts(), 0.95))
files = folders = 0



path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"



for _, dirnames, filenames in os.walk(path):

  # ^ this idiom means "we won't be using this value"

    files += len(filenames)

    folders += len(dirnames)

#print(Fore.YELLOW +"Total Patients in Train set: ",Style.RESET_ALL,train_df['Patient'].count())

print(Fore.YELLOW +f'{files:,}',Style.RESET_ALL,"files/images, " + Fore.BLUE + f'{folders:,}',Style.RESET_ALL ,'folders/patients')
files = []

for _, dirnames, filenames in os.walk(path):

  # ^ this idiom means "we won't be using this value"

    files.append(len(filenames))



print(Fore.YELLOW +f'{round(np.mean(files)):,}',Style.RESET_ALL,'average files/images per patient')

print(Fore.BLUE +f'{round(np.max(files)):,}',Style.RESET_ALL, 'max files/images per patient')

print(Fore.GREEN +f'{round(np.min(files)):,}',Style.RESET_ALL,'min files/images per patient')
patient_df = train_df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates()

patient_df.head()
train_dir = '../input/osic-pulmonary-fibrosis-progression/train/'

test_dir = '../input/osic-pulmonary-fibrosis-progression/test/'



patient_ids = os.listdir(train_dir)

patient_ids = sorted(patient_ids)



no_of_instances = []

age = []

sex = []

smoking_status = []



for patient_id in patient_ids:

    patient_info = train_df[train_df['Patient'] == patient_id].reset_index()

    no_of_instances.append(len(os.listdir(train_dir + patient_id)))

    age.append(patient_info['Age'][0])

    sex.append(patient_info['Sex'][0])

    smoking_status.append(patient_info['SmokingStatus'][0])

   

patient_df = pd.DataFrame(list(zip(patient_ids, no_of_instances, age, sex, smoking_status)), 

                                 columns =['Patient', 'no_of_instances', 'Age', 'Sex', 'SmokingStatus'])

print(patient_df.info())

patient_df.head()
patient_df['SmokingStatus'].value_counts()
patient_df['SmokingStatus'].value_counts().iplot(kind='bar',

                                              yTitle='Counts', 

                                              linecolor='black', 

                                              opacity=0.7,

                                              color='blue',

                                              theme='pearl',

                                              bargap=0.5,

                                              gridcolor='white',

                                              title='Distribution of the SmokingStatus column in the Unique Patient Set')
train_df['Weeks'].value_counts().head()
train_df['Weeks'].value_counts().iplot(kind='barh',

                                      xTitle='Counts(Weeks)', 

                                      linecolor='black', 

                                      opacity=0.7,

                                      color='#FB8072',

                                      theme='pearl',

                                      bargap=0.2,

                                      gridcolor='white',

                                      title='Distribution of the Weeks in the training set')
train_df['Weeks'].iplot(kind='hist',

                              xTitle='Weeks', 

                              yTitle='Counts',

                              linecolor='black', 

                              opacity=0.7,

                              color='#FB8072',

                              theme='pearl',

                              bargap=0.2,

                              gridcolor='white',

                              title='Distribution of the Weeks in the training set')
fig = px.scatter(train_df, x="Weeks", y="Age", color='Sex')

fig.show()
train_df['FVC'].value_counts()
train_df['FVC'].iplot(kind='hist',

                      xTitle='Lung Capacity(ml)', 

                      linecolor='black', 

                      opacity=0.8,

                      color='#FB8072',

                      bargap=0.5,

                      gridcolor='white',

                      title='Distribution of the FVC in the training set')
fig = px.scatter(train_df, x="FVC", y="Percent", color='Age')

fig.show()
fig = px.scatter(train_df, x="FVC", y="Age", color='Sex')

fig.show()
fig = px.scatter(train_df, x="FVC", y="Weeks", color='SmokingStatus')

fig.show()
patient = train_df[train_df.Patient == 'ID00228637202259965313869']

fig = px.line(patient, x="Weeks", y="FVC", color='SmokingStatus')

fig.show()
train_df['Percent'].value_counts()
train_df['Percent'].iplot(kind='hist',bins=30,color='blue',xTitle='Percent distribution',yTitle='Count')
df = train_df

fig = px.violin(df, y='Percent', x='SmokingStatus', box=True, color='Sex', points="all",

          hover_data=train_df.columns)

fig.show()
plt.figure(figsize=(16, 6))

ax = sns.violinplot(x = train_df['SmokingStatus'], y = train_df['Percent'], palette = 'Reds')

ax.set_xlabel(xlabel = 'Smoking Habit', fontsize = 15)

ax.set_ylabel(ylabel = 'Percent', fontsize = 15)

ax.set_title(label = 'Distribution of Smoking Status Over Percentage', fontsize = 20)

plt.show()
fig = px.scatter(train_df, x="Age", y="Percent", color='SmokingStatus')

fig.show()
patient = train_df[train_df.Patient == 'ID00228637202259965313869']

fig = px.line(patient, x="Weeks", y="Percent", color='SmokingStatus')

fig.show()
patient_df['Age'].iplot(kind='hist',bins=30,color='red',xTitle='Ages of distribution',yTitle='Count')
patient_df['SmokingStatus'].value_counts()
plt.figure(figsize=(16, 6))

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)

sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently smokes', shade=True)



# Labeling of plot

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
plt.figure(figsize=(16, 6))

ax = sns.violinplot(x = patient_df['SmokingStatus'], y = patient_df['Age'], palette = 'Reds')

ax.set_xlabel(xlabel = 'Smoking habit', fontsize = 15)

ax.set_ylabel(ylabel = 'Age', fontsize = 15)

ax.set_title(label = 'Distribution of Smokers over Age', fontsize = 20)

plt.show()
plt.figure(figsize=(16, 6))

sns.kdeplot(patient_df.loc[patient_df['Sex'] == 'Male', 'Age'], label = 'Male',shade=True)

sns.kdeplot(patient_df.loc[patient_df['Sex'] == 'Female', 'Age'], label = 'Female',shade=True)

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
patient_df['Sex'].value_counts()
patient_df['Sex'].value_counts().iplot(kind='bar',

                                          yTitle='Count', 

                                          linecolor='black', 

                                          opacity=0.7,

                                          color='blue',

                                          theme='pearl',

                                          bargap=0.8,

                                          gridcolor='white',

                                          title='Distribution of the Sex column in Patient Dataframe')
plt.figure(figsize=(16, 6))

a = sns.countplot(data=patient_df, x='SmokingStatus', hue='Sex')



for p in a.patches:

    a.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')



plt.title('Gender split by SmokingStatus', fontsize=16)

sns.despine(left=True, bottom=True);
fig = px.box(patient_df, x="Sex", y="Age", points="all")

fig.show()
ids_train = train_df.Patient.values

ids_test = test_df.Patient.values

ids_train_set = set(ids_train)

print(Fore.YELLOW + "There are",Style.RESET_ALL,f'{len(ids_train_set)}', Fore.BLUE + 'unique Patient IDs',Style.RESET_ALL,'in the training set')

ids_test_set = set(ids_test)

print(Fore.YELLOW + "There are", Style.RESET_ALL, f'{len(ids_test_set)}', Fore.BLUE + 'unique Patient IDs',Style.RESET_ALL,'in the test set')



patient_overlap = list(ids_train_set.intersection(ids_test_set))

n_overlap = len(patient_overlap)

print(Fore.YELLOW + "There are", Style.RESET_ALL, f'{n_overlap}', Fore.BLUE + 'Patient IDs',Style.RESET_ALL, 'in both the training and test sets')

print('')

print(Fore.CYAN + 'These patients are in both the training and test datasets:', Style.RESET_ALL)

print(f'{patient_overlap}')
corrmat = train_df.corr() 

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corrmat, ax = ax, cmap = 'RdYlBu_r', linewidths = 0.5) 
print(Fore.YELLOW + 'Train .dcm number of images:',Style.RESET_ALL, len(list(os.listdir('../input/osic-pulmonary-fibrosis-progression/train'))), '\n' +

      Fore.BLUE + 'Test .dcm number of images:',Style.RESET_ALL, len(list(os.listdir('../input/osic-pulmonary-fibrosis-progression/test'))), '\n' +

      '--------------------------------', '\n' +

      'There is the same number of images as in train/ test .csv datasets')
def plot_pixel_array(dataset, figsize=(5,5)):

    plt.figure(figsize=figsize)

    plt.grid(False)

    plt.imshow(dataset.pixel_array, cmap='gray') # cmap=plt.cm.bone)

    plt.show()
def show_dcm_info(dataset):

    print(Fore.YELLOW + "Filename.........:",Style.RESET_ALL,file_path)

    print()



    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print(Fore.BLUE + "Patient's name......:",Style.RESET_ALL, display_name)

    print(Fore.BLUE + "Patient id..........:",Style.RESET_ALL, dataset.PatientID)

    print(Fore.BLUE + "Patient's Sex.......:",Style.RESET_ALL, dataset.PatientSex)

    print(Fore.YELLOW + "Modality............:",Style.RESET_ALL, dataset.Modality)

    print(Fore.GREEN + "Body Part Examined..:",Style.RESET_ALL, dataset.BodyPartExamined)

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print(Fore.BLUE + "Image size.......:",Style.RESET_ALL," {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print(Fore.YELLOW + "Pixel spacing....:",Style.RESET_ALL,dataset.PixelSpacing)

            dataset.PixelSpacing = [1, 1]

        plt.figure(figsize=(10, 10))

        plt.imshow(dataset.pixel_array, cmap='gray')

        plt.show()

for file_path in glob.glob('../input/osic-pulmonary-fibrosis-progression/train/*/*.dcm'):

    dataset = pydicom.dcmread(file_path)

    show_dcm_info(dataset)

    break
imdir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140"

print("total images for patient ID00123637202217151272140: ", len(os.listdir(imdir)))



fig=plt.figure(figsize=(12, 12))

columns = 4

rows = 5

imglist = os.listdir(imdir)

for i in range(1, columns*rows +1):

    filename = imdir + "/" + str(i) + ".dcm"

    ds = pydicom.dcmread(filename)

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap='gray')

plt.show()
imdir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140"

print("total images for patient ID00123637202217151272140: ", len(os.listdir(imdir)))



# view first (columns*rows) images in order

fig=plt.figure(figsize=(12, 12))

columns = 4

rows = 5

imglist = os.listdir(imdir)

for i in range(1, columns*rows +1):

    filename = imdir + "/" + str(i) + ".dcm"

    ds = pydicom.dcmread(filename)

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap='jet')

plt.show()
apply_resample = False



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
def set_lungwin(img, hu=[-1200., 600.]):

    lungwin = np.array(hu)

    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])

    newimg[newimg < 0] = 0

    newimg[newimg > 1] = 1

    newimg = (newimg * 255).astype('uint8')

    return newimg
scans = load_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')

scan_array = set_lungwin(get_pixels_hu(scans))
from scipy.ndimage.interpolation import zoom



def resample(imgs, spacing, new_spacing):

    new_shape = np.round(imgs.shape * spacing / new_spacing)

    true_spacing = spacing * imgs.shape / new_shape

    resize_factor = new_shape / imgs.shape

    imgs = zoom(imgs, resize_factor, mode='nearest')

    return imgs, true_spacing, new_shape



spacing_z = (scans[-1].ImagePositionPatient[2] - scans[0].ImagePositionPatient[2]) / len(scans)



if apply_resample:

    scan_array_resample = resample(scan_array, np.array(np.array([spacing_z, *scans[0].PixelSpacing])), np.array([1.,1.,1.]))[0]
import imageio

from IPython.display import Image



imageio.mimsave("/tmp/gif.gif", scan_array, duration=0.0001)

Image(filename="/tmp/gif.gif", format='png')