from IPython.display import HTML

HTML('<center><iframe width="560" height="315" src="https://www.youtube.com/embed/AfK9LPNj-Zo" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
import os

from os import listdir

import pandas as pd

import numpy as np

import glob

import tqdm

from typing import Dict

import matplotlib.pyplot as plt

%matplotlib inline



#plotly

!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



#color

from colorama import Fore, Back, Style



import seaborn as sns

sns.set(style="whitegrid")



#pydicom

import pydicom



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')



# Settings for pretty nice plots

plt.style.use('fivethirtyeight')

plt.show()
# List files available

list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))
IMAGE_PATH = "../input/osic-pulmonary-fibrosis-progressiont/"



train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')



print(Fore.YELLOW + 'Training data shape: ',Style.RESET_ALL,train_df.shape)

train_df.head(5)
train_df.groupby(['SmokingStatus']).count()['Sex'].to_frame()
# Null values and Data types

print(Fore.YELLOW + 'Train Set !!',Style.RESET_ALL)

print(train_df.info())

print('-------------')

print(Fore.BLUE + 'Test Set !!',Style.RESET_ALL)

print(test_df.info())
### Missing values



train_df.isnull().sum()
test_df.isnull().sum()
# Total number of Patient in the dataset(train+test)



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
# Creating unique patient lists and their properties. 

train_dir = '../input/osic-pulmonary-fibrosis-progression/train/'

test_dir = '../input/osic-pulmonary-fibrosis-progression/test/'



patient_ids = os.listdir(train_dir)

patient_ids = sorted(patient_ids)



#Creating new rows

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



#Creating the dataframe for the patient info    

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
# Extract patient id's for the training set

ids_train = train_df.Patient.values

# Extract patient id's for the validation set

ids_test = test_df.Patient.values

#print(Fore.YELLOW +"Total Patients in Train set: ",Style.RESET_ALL,train_df['Patient'].count())

# Create a "set" datastructure of the training set id's to identify unique id's

ids_train_set = set(ids_train)

print(Fore.YELLOW + "There are",Style.RESET_ALL,f'{len(ids_train_set)}', Fore.BLUE + 'unique Patient IDs',Style.RESET_ALL,'in the training set')

# Create a "set" datastructure of the validation set id's to identify unique id's

ids_test_set = set(ids_test)

print(Fore.YELLOW + "There are", Style.RESET_ALL, f'{len(ids_test_set)}', Fore.BLUE + 'unique Patient IDs',Style.RESET_ALL,'in the test set')



# Identify patient overlap by looking at the intersection between the sets

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
# https://www.kaggle.com/schlerp/getting-to-know-dicom-and-the-data

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

    break # Comment this out to see all
# https://www.kaggle.com/yeayates21/osic-simple-image-eda



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

    plt.imshow(ds.pixel_array, cmap='gray')

plt.show()
# https://www.kaggle.com/yeayates21/osic-simple-image-eda



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
# Resample to 1mm (An optional step, it may not be relevant to this competition because of the large slice thickness on the z axis)



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
import matplotlib.animation as animation



fig = plt.figure()



ims = []

for image in scan_array:

    im = plt.imshow(image, animated=True, cmap="Greys")

    plt.axis("off")

    ims.append([im])



ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,

                                repeat_delay=1000)
HTML(ani.to_jshtml())
HTML(ani.to_html5_video())
def extract_dicom_meta_data(filename: str) -> Dict:

    # Load image

    

    image_data = pydicom.read_file(filename)

    img=np.array(image_data.pixel_array).flatten()

    row = {

        'Patient': image_data.PatientID,

        'body_part_examined': image_data.BodyPartExamined,

        'image_position_patient': image_data.ImagePositionPatient,

        'image_orientation_patient': image_data.ImageOrientationPatient,

        'photometric_interpretation': image_data.PhotometricInterpretation,

        'rows': image_data.Rows,

        'columns': image_data.Columns,

        'pixel_spacing': image_data.PixelSpacing,

        'window_center': image_data.WindowCenter,

        'window_width': image_data.WindowWidth,

        'modality': image_data.Modality,

        'StudyInstanceUID': image_data.StudyInstanceUID,

        'SeriesInstanceUID': image_data.StudyInstanceUID,

        'StudyID': image_data.StudyInstanceUID, 

        'SamplesPerPixel': image_data.SamplesPerPixel,

        'BitsAllocated': image_data.BitsAllocated,

        'BitsStored': image_data.BitsStored,

        'HighBit': image_data.HighBit,

        'PixelRepresentation': image_data.PixelRepresentation,

        'RescaleIntercept': image_data.RescaleIntercept,

        'RescaleSlope': image_data.RescaleSlope,

        'img_min': np.min(img),

        'img_max': np.max(img),

        'img_mean': np.mean(img),

        'img_std': np.std(img)}



    return row
train_image_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/train'

train_image_files = glob.glob(os.path.join(train_image_path, '*', '*.dcm'))



meta_data_df = []

for filename in tqdm.tqdm(train_image_files):

    try:

        meta_data_df.append(extract_dicom_meta_data(filename))

    except Exception as e:

        continue
# Convert to a pd.DataFrame from dict

meta_data_df = pd.DataFrame.from_dict(meta_data_df)

meta_data_df.head()
# source: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/154658

folder='train'

PATH='../input/osic-pulmonary-fibrosis-progression/'



last_index = 2



column_names = ['image_name', 'dcm_ImageOrientationPatient', 

                'dcm_ImagePositionPatient', 'dcm_PatientID',

                'dcm_PatientName', 'dcm_PatientSex'

                'dcm_rows', 'dcm_columns']



def extract_DICOM_attributes(folder):

    patients_folder = list(os.listdir(os.path.join(PATH, folder)))

    df = pd.DataFrame()

    

    i = 0

    

    for patient_id in patients_folder:

   

        img_path = os.path.join(PATH, folder, patient_id)

        

        print(img_path)

        

        images = list(os.listdir(img_path))

        

        #df = pd.DataFrame()



        for image in images:

            image_name = image.split(".")[0]



            dicom_file_path = os.path.join(img_path,image)

            dicom_file_dataset = pydicom.read_file(dicom_file_path)

                

            '''

            print(dicom_file_dataset.dir("pat"))

            print(dicom_file_dataset.data_element("ImageOrientationPatient"))

            print(dicom_file_dataset.data_element("ImagePositionPatient"))

            print(dicom_file_dataset.data_element("PatientID"))

            print(dicom_file_dataset.data_element("PatientName"))

            print(dicom_file_dataset.data_element("PatientSex"))

            '''

            

            imageOrientationPatient = dicom_file_dataset.ImageOrientationPatient

            #imagePositionPatient = dicom_file_dataset.ImagePositionPatient

            patientID = dicom_file_dataset.PatientID

            patientName = dicom_file_dataset.PatientName

            patientSex = dicom_file_dataset.PatientSex

        

            rows = dicom_file_dataset.Rows

            cols = dicom_file_dataset.Columns

            

            #print(rows)

            #print(columns)

            

            temp_dict = {'image_name': image_name, 

                                    'dcm_ImageOrientationPatient': imageOrientationPatient,

                                    #'dcm_ImagePositionPatient':imagePositionPatient,

                                    'dcm_PatientID': patientID, 

                                    'dcm_PatientName': patientName,

                                    'dcm_PatientSex': patientSex,

                                    'dcm_rows': rows,

                                    'dcm_columns': cols}





            df = df.append([temp_dict])

            

        i += 1

        

        if i == last_index:

            break

            

    return df
extract_DICOM_attributes('train')
import pandas_profiling as pdp
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
profile_train_df = pdp.ProfileReport(train_df)
profile_train_df
profile_test_df = pdp.ProfileReport(test_df)
profile_test_df
!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index

!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index
import os

import cv2

import pydicom

import pandas as pd

import numpy as np 

import tensorflow as tf 

import matplotlib.pyplot as plt 

import random

from tqdm.notebook import tqdm 

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_absolute_error

from tensorflow_addons.optimizers import RectifiedAdam

from tensorflow.keras import Model

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from tensorflow.keras.optimizers import Nadam

import seaborn as sns

from PIL import Image



def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(42)
config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv') 
def get_tab(df):

    vector = [(df.Age.values[0] - 30) / 30] 

    

    if df.Sex.values[0] == 'male':

       vector.append(0)

    else:

       vector.append(1)

    

    if df.SmokingStatus.values[0] == 'Never smoked':

        vector.extend([0,0])

    elif df.SmokingStatus.values[0] == 'Ex-smoker':

        vector.extend([1,1])

    elif df.SmokingStatus.values[0] == 'Currently smokes':

        vector.extend([0,1])

    else:

        vector.extend([1,0])

    return np.array(vector) 
A = {} 

TAB = {} 

P = [] 

for i, p in tqdm(enumerate(train.Patient.unique())):

    sub = train.loc[train.Patient == p, :] 

    fvc = sub.FVC.values

    weeks = sub.Weeks.values

    c = np.vstack([weeks, np.ones(len(weeks))]).T

    a, b = np.linalg.lstsq(c, fvc)[0]

    

    A[p] = a

    TAB[p] = get_tab(sub)

    P.append(p)
def get_img(path):

    d = pydicom.dcmread(path)

    return cv2.resize(d.pixel_array / 2**11, (512, 512))
from tensorflow.keras.utils import Sequence



class IGenerator(Sequence):

    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

    def __init__(self, keys, a, tab, batch_size=32):

        self.keys = [k for k in keys if k not in self.BAD_ID]

        self.a = a

        self.tab = tab

        self.batch_size = batch_size

        

        self.train_data = {}

        for p in train.Patient.values:

            self.train_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

    

    def __len__(self):

        return 1000

    

    def __getitem__(self, idx):

        x = []

        a, tab = [], [] 

        keys = np.random.choice(self.keys, size = self.batch_size)

        for k in keys:

            try:

                i = np.random.choice(self.train_data[k], size=1)[0]

                img = get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')

                x.append(img)

                a.append(self.a[k])

                tab.append(self.tab[k])

            except:

                print(k, i)

       

        x,a,tab = np.array(x), np.array(a), np.array(tab)

        x = np.expand_dims(x, axis=-1)

        return [x, tab] , a
from tensorflow.keras.layers import (

    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 

    LeakyReLU, Concatenate 

)

import efficientnet.tfkeras as efn



def get_efficientnet(model, shape):

    models_dict = {

        'b0': efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),

        'b1': efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),

        'b2': efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),

        'b3': efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),

        'b4': efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),

        'b5': efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),

        'b6': efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),

        'b7': efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)

    }

    return models_dict[model]



def build_model(shape=(512, 512, 1), model_class=None):

    inp = Input(shape=shape)

    base = get_efficientnet(model_class, shape)

    x = base(inp)

    x = GlobalAveragePooling2D()(x)

    inp2 = Input(shape=(4,))

    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)

    x = Concatenate()([x, x2]) 

    x = Dropout(0.4)(x) 

    x = Dense(1)(x)

    model = Model([inp, inp2] , x)

    

    weights = [w for w in os.listdir('../input/osic-model-weights') if model_class in w][0]

    model.load_weights('../input/osic-model-weights/' + weights)

    return model



model_classes = ['b5'] #['b0','b1','b2','b3',b4','b5','b6','b7']

models = [build_model(shape=(512, 512, 1), model_class=m) for m in model_classes]

print('Number of models: ' + str(len(models)))
from sklearn.model_selection import train_test_split 



tr_p, vl_p = train_test_split(P, 

                              shuffle=True, 

                              train_size= 0.8) 
sns.distplot(list(A.values()));
def score(fvc_true, fvc_pred, sigma):

    sigma_clip = np.maximum(sigma, 70) # changed from 70, trie 66.7 too

    delta = np.abs(fvc_true - fvc_pred)

    delta = np.minimum(delta, 1000)

    sq2 = np.sqrt(2)

    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)

    return np.mean(metric)
subs = []

for model in models:

    metric = []

    for q in tqdm(range(1, 10)):

        m = []

        for p in vl_p:

            x = [] 

            tab = [] 



            if p in ['ID00011637202177653955184', 'ID00052637202186188008618']:

                continue



            ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

            for i in ldir:

                if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:

                    x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/{i}')) 

                    tab.append(get_tab(train.loc[train.Patient == p, :])) 

            if len(x) < 1:

                continue

            tab = np.array(tab) 



            x = np.expand_dims(x, axis=-1) 

            _a = model.predict([x, tab]) 

            a = np.quantile(_a, q / 10)



            percent_true = train.Percent.values[train.Patient == p]

            fvc_true = train.FVC.values[train.Patient == p]

            weeks_true = train.Weeks.values[train.Patient == p]



            fvc = a * (weeks_true - weeks_true[0]) + fvc_true[0]

            percent = percent_true[0] - a * abs(weeks_true - weeks_true[0])

            m.append(score(fvc_true, fvc, percent))

        print(np.mean(m))

        metric.append(np.mean(m))



    q = (np.argmin(metric) + 1)/ 10



    sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv') 

    test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv') 

    A_test, B_test, P_test,W, FVC= {}, {}, {},{},{} 

    STD, WEEK = {}, {} 

    for p in test.Patient.unique():

        x = [] 

        tab = [] 

        ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/')

        for i in ldir:

            if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:

                x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/{i}')) 

                tab.append(get_tab(test.loc[test.Patient == p, :])) 

        if len(x) <= 1:

            continue

        tab = np.array(tab) 



        x = np.expand_dims(x, axis=-1) 

        _a = model.predict([x, tab]) 

        a = np.quantile(_a, q)

        A_test[p] = a

        B_test[p] = test.FVC.values[test.Patient == p] - a*test.Weeks.values[test.Patient == p]

        P_test[p] = test.Percent.values[test.Patient == p] 

        WEEK[p] = test.Weeks.values[test.Patient == p]



    for k in sub.Patient_Week.values:

        p, w = k.split('_')

        w = int(w) 



        fvc = A_test[p] * w + B_test[p]

        sub.loc[sub.Patient_Week == k, 'FVC'] = fvc

        sub.loc[sub.Patient_Week == k, 'Confidence'] = (

            P_test[p] - A_test[p] * abs(WEEK[p] - w) 

    ) 



    _sub = sub[["Patient_Week","FVC","Confidence"]].copy()

    subs.append(_sub)
N = len(subs)

sub = subs[0].copy() # ref

sub["FVC"] = 0

sub["Confidence"] = 0

for i in range(N):

    sub["FVC"] += subs[0]["FVC"] * (1/N)

    sub["Confidence"] += subs[0]["Confidence"] * (1/N)
sub.head()
sub[["Patient_Week","FVC","Confidence"]].to_csv("submission_img.csv", index=False)
img_sub = sub[["Patient_Week","FVC","Confidence"]].copy()
ROOT = "../input/osic-pulmonary-fibrosis-progression"

BATCH_SIZE=128



tr = pd.read_csv(f"{ROOT}/train.csv")

tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])

chunk = pd.read_csv(f"{ROOT}/test.csv")



print("add infos")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")

sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]

sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")
tr['WHERE'] = 'train'

chunk['WHERE'] = 'val'

sub['WHERE'] = 'test'

data = tr.append([chunk, sub])
print(tr.shape, chunk.shape, sub.shape, data.shape)

print(tr.Patient.nunique(), chunk.Patient.nunique(), sub.Patient.nunique(), 

      data.Patient.nunique())

#
data['min_week'] = data['Weeks']

data.loc[data.WHERE=='test','min_week'] = np.nan

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
base = data.loc[data.Weeks == data.min_week]

base = base[['Patient','FVC']].copy()

base.columns = ['Patient','min_FVC']

base['nb'] = 1

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

base = base[base.nb==1]

base.drop('nb', axis=1, inplace=True)
data = data.merge(base, on='Patient', how='left')

data['base_week'] = data['Weeks'] - data['min_week']

del base
COLS = ['Sex','SmokingStatus'] #,'Age'

FE = []

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        data[mod] = (data[col] == mod).astype(int)
#

data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )

data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )

data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )

data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )

FE += ['age','percent','week','BASE']
tr = data.loc[data.WHERE=='train']

chunk = data.loc[data.WHERE=='val']

sub = data.loc[data.WHERE=='test']

del data
tr.shape, chunk.shape, sub.shape
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")



def score(y_true, y_pred):

    tf.dtypes.cast(y_true, tf.float32)

    tf.dtypes.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    

    #sigma_clip = sigma + C1

    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true[:, 0] - fvc_pred)

    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )

    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)

    return K.mean(metric)



def qloss(y_true, y_pred):

    # Pinball loss for multiple quantiles

    qs = [0.2, 0.50, 0.8]

    q = tf.constant(np.array([qs]), dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    return K.mean(v)



def mloss(_lambda):

    def loss(y_true, y_pred):

        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)

    return loss



def make_model(nh):

    z = L.Input((nh,), name="Patient")

    x = L.Dense(100, activation="relu", name="d1")(z)

    x = L.Dense(100, activation="relu", name="d2")(x)

    #x = L.Dense(100, activation="relu", name="d3")(x)

    p1 = L.Dense(3, activation="linear", name="p1")(x)

    p2 = L.Dense(3, activation="relu", name="p2")(x)

    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 

                     name="preds")([p1, p2])

    

    model = M.Model(z, preds, name="CNN")

    #model.compile(loss=qloss, optimizer="adam", metrics=[score])

    model.compile(loss=mloss(0.8), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])

    return model
y = tr['FVC'].values

z = tr[FE].values

ze = sub[FE].values

nh = z.shape[1]

pe = np.zeros((ze.shape[0], 3))

pred = np.zeros((z.shape[0], 3))
net = make_model(nh)

print(net.summary())

print(net.count_params())
NFOLD = 5 # originally 5

kf = KFold(n_splits=NFOLD)
%%time

cnt = 0

EPOCHS = 800

for tr_idx, val_idx in kf.split(z):

    cnt += 1

    print(f"FOLD {cnt}")

    net = make_model(nh)

    net.fit(z[tr_idx], y[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS, 

            validation_data=(z[val_idx], y[val_idx]), verbose=0) #

    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))

    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))

    print("predict val...")

    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)

    print("predict test...")

    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD
sigma_opt = mean_absolute_error(y, pred[:, 1])

unc = pred[:,2] - pred[:, 0]

sigma_mean = np.mean(unc)

print(sigma_opt, sigma_mean)
idxs = np.random.randint(0, y.shape[0], 100)

plt.plot(y[idxs], label="ground truth")

plt.plot(pred[idxs, 0], label="q25")

plt.plot(pred[idxs, 1], label="q50")

plt.plot(pred[idxs, 2], label="q75")

plt.legend(loc="best")

plt.show()
print(unc.min(), unc.mean(), unc.max(), (unc>=0).mean())
plt.hist(unc)

plt.title("uncertainty in prediction")

plt.show()
sub.head()
# PREDICTION

sub['FVC1'] = 1.*pe[:, 1]

sub['Confidence1'] = pe[:, 2] - pe[:, 0]

subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()

subm.loc[~subm.FVC1.isnull()].head(10)
subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']

if sigma_mean<70:

    subm['Confidence'] = sigma_opt

else:

    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']
subm.head()
subm.describe().T
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(otest)):

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1
subm[["Patient_Week","FVC","Confidence"]].to_csv("submission_regression.csv", index=False)
reg_sub = subm[["Patient_Week","FVC","Confidence"]].copy()
df1 = img_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)

df2 = reg_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)
df = df1[['Patient_Week']].copy()

df['FVC'] = 0.25*df1['FVC'] + 0.75*df2['FVC']

df['Confidence'] = 0.26*df1['Confidence'] + 0.74*df2['Confidence']

df.head()
df.to_csv('submission.csv', index=False)