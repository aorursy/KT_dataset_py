# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Define data path

path="../input/siim-isic-melanoma-classification/"

train=pd.read_csv(path+'train.csv')

test=pd.read_csv(path+'test.csv')
#Training data

display('Training data shape:',train.shape)

display(train.head())
# Test data

display('Test data shape:',test.shape)

display(test.head())
train.groupby(['benign_malignant']).count()['sex'].to_frame()
train.groupby(['age_approx']).count()['benign_malignant'].to_frame()
train.groupby(['anatom_site_general_challenge']).count()['sex'].to_frame()
train.groupby(['age_approx']).count()['anatom_site_general_challenge'].to_frame()
display('Train Set')

display(train.info())

display('-------------')

display('Test Set')

display(test.info())
display("Total patients ids are :", train['patient_id'].count())

display('Unique ids are:',train['patient_id'].value_counts().shape[0])
col=train.keys()

col=list(col)

display(col)
display(train['target'].value_counts(normalize=True))

display(sns.countplot(train['target']))
display(train['sex'].value_counts(normalize=True))

display(sns.countplot(train['sex']))
display(train['benign_malignant'].value_counts(normalize=True))

display(sns.countplot(train['benign_malignant']))
display(train['anatom_site_general_challenge'].value_counts(normalize=True).sort_values())

display(sns.countplot(train['anatom_site_general_challenge']))
display(train['diagnosis'].value_counts())

sns.countplot(train['diagnosis'])
z=train.groupby(['target','sex'])['benign_malignant'].count().to_frame().reset_index()

z.style.background_gradient(cmap='Blues')
sns.catplot(x='target',y='benign_malignant',hue='sex',data=z,kind='bar')

plt.ylabel('Count')

plt.xlabel('benign:0 vs malignant:1')
z=train.groupby(['anatom_site_general_challenge','sex','target'])['benign_malignant'].count().to_frame().reset_index()

z.style.background_gradient(cmap='Blues')
z1=train.groupby(['sex','anatom_site_general_challenge'])['benign_malignant'].count().to_frame().reset_index()

z1.style.background_gradient(cmap='Reds')

sns.catplot(x='anatom_site_general_challenge',y='benign_malignant', hue='sex',data=z1,kind='bar')

plt.gcf().set_size_inches(10,8)

plt.xlabel('location of imaged site')

plt.xticks(rotation=45,fontsize='10', horizontalalignment='right')

plt.ylabel('count of melanoma cases')
z2=train.groupby(['sex','anatom_site_general_challenge','target'])['benign_malignant'].count().to_frame().reset_index()

z2.style.background_gradient(cmap='Reds')

sns.catplot(x='anatom_site_general_challenge',y='benign_malignant', hue='target',data=z2,kind='bar')

plt.gcf().set_size_inches(10,8)

plt.xlabel('location of imaged site')

plt.xticks(rotation=45,fontsize='10', horizontalalignment='right')

plt.ylabel('count of melanoma cases')
z3=train.groupby(['anatom_site_general_challenge','sex','target','diagnosis'])['benign_malignant'].count().to_frame().reset_index()

z3.style.background_gradient(cmap='Blues')
sns.catplot(x='diagnosis',y='benign_malignant', hue='sex',data=z3,kind='bar')

plt.gcf().set_size_inches(10,8)

plt.xlabel('Diagnosis')

plt.xticks(rotation=45,fontsize='10', horizontalalignment='right')

plt.ylabel('count of melanoma cases')
sns.catplot(x='anatom_site_general_challenge',y='benign_malignant', hue='diagnosis',data=z3,kind='bar')

plt.gcf().set_size_inches(10,8)

plt.xlabel('location of imaged site')

plt.xticks(rotation=45,fontsize='10', horizontalalignment='right')

plt.ylabel('count of melanoma cases')
# KDE plot of age that were diagnosed as benign

sns.kdeplot(train.loc[train['target'] == 0, 'age_approx'], label = 'Benign',shade=True)



# KDE plot of age that were diagnosed as malignant

sns.kdeplot(train.loc[train['target'] == 1, 'age_approx'], label = 'Malignant',shade=True)



# Labeling of plot

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
# KDE plot of age that were diagnosed as benign

sns.kdeplot(train.loc[train['sex'] == 'male', 'age_approx'], label = 'Male',shade=True)



# KDE plot of age that were diagnosed as malignant

sns.kdeplot(train.loc[train['sex'] == 'female', 'age_approx'], label = 'Female',shade=True)



# Labeling of plot

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
# Extract patient id's for the training set

ids_train = train['patient_id'].values

# Extract patient id's for the validation set

ids_test = test['patient_id'].values



# Create a "set" datastructure of the training set id's to identify unique id's

ids_train_set = set(ids_train)

display(f'There are {len(ids_train_set)} unique Patient IDs in the training set')

# Create a "set" datastructure of the validation set id's to identify unique id's

ids_test_set = set(ids_test)

display(f'There are {len(ids_test_set)} unique Patient IDs in the training set')



# Identify patient overlap by looking at the intersection between the sets

patient_overlap = list(ids_train_set.intersection(ids_test_set))

n_overlap = len(patient_overlap)

display(f'There are {n_overlap} Patient IDs in both the training and test sets')

display('')

display(f'These patients are in both the training and test datasets:')

display(f'{patient_overlap}')

images = train['image_name'].values



# Extract 9 random images from it

random_images = [np.random.choice(images+'.jpg') for i in range(9)]



# Location of the image dir

img_dir = path+'/jpeg/train'



print('Display Random Images')



# Adjust the size of your images

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(img_dir, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()   
benign = train[train['benign_malignant']=='benign']

malignant = train[train['benign_malignant']=='malignant']
images = benign['image_name'].values



# Extract 9 random images from it

random_images = [np.random.choice(images+'.jpg') for i in range(9)]



# Location of the image dir

img_dir = path+'/jpeg/train'



print('Display benign Images')



# Adjust the size of your images

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(img_dir, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout() 
images = malignant['image_name'].values



# Extract 9 random images from it

random_images = [np.random.choice(images+'.jpg') for i in range(9)]



# Location of the image dir

img_dir = path+'/jpeg/train'



print('Display Malignant Images')



# Adjust the size of your images

plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(img_dir, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout() 
f = plt.figure(figsize=(16,8))

f.add_subplot(1,2, 1)



sample_img = benign['image_name'][3]+'.jpg'

raw_image = plt.imread(os.path.join(img_dir, sample_img))

plt.imshow(raw_image, cmap='gray')

plt.colorbar()

plt.title('Benign Image')

print(f"Image dimensions:  {raw_image.shape[0],raw_image.shape[1]}")

print(f"Maximum pixel value : {raw_image.max():.1f} ; Minimum pixel value:{raw_image.min():.1f}")

print(f"Mean value of the pixels : {raw_image.mean():.1f} ; Standard deviation : {raw_image.std():.1f}")



f.add_subplot(1,2, 2)



#_ = plt.hist(raw_image.ravel(),bins = 256, color = 'orange',)

_ = plt.hist(raw_image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)

_ = plt.hist(raw_image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)

_ = plt.hist(raw_image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)

_ = plt.xlabel('Intensity Value')

_ = plt.ylabel('Count')

_ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])

plt.show()
f = plt.figure(figsize=(16,8))

f.add_subplot(1,2, 1)



sample_img = malignant['image_name'][235]+'.jpg'

raw_image = plt.imread(os.path.join(img_dir, sample_img))

plt.imshow(raw_image, cmap='gray')

plt.colorbar()

plt.title('Malignant Image')

print(f"Image dimensions:  {raw_image.shape[0],raw_image.shape[1]}")

print(f"Maximum pixel value : {raw_image.max():.1f} ; Minimum pixel value:{raw_image.min():.1f}")

print(f"Mean value of the pixels : {raw_image.mean():.1f} ; Standard deviation : {raw_image.std():.1f}")



f.add_subplot(1,2, 2)



#_ = plt.hist(raw_image.ravel(),bins = 256, color = 'orange',)

_ = plt.hist(raw_image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)

_ = plt.hist(raw_image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)

_ = plt.hist(raw_image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)

_ = plt.xlabel('Intensity Value')

_ = plt.ylabel('Count')

_ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])

plt.show()
!pip install pydicom
import pydicom

print(pydicom.__version__)


def show_dcm_info(dataset):

    print("Filename.........:", file_path)

    print("Storage type.....:", dataset.SOPClassUID)

    print()



    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print("Patient's name......:", display_name)

    print("Patient id..........:", dataset.PatientID)

    print("Patient's Age.......:", dataset.PatientAge)

    print("Patient's Sex.......:", dataset.PatientSex)

    print("Modality............:", dataset.Modality)

    print("Body Part Examined..:", dataset.BodyPartExamined)

   

    

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing....:", dataset.PixelSpacing)
def plot_pixel_array(dataset, figsize=(5,5)):

    plt.figure(figsize=figsize)

    plt.grid(False)

    plt.imshow(dataset.pixel_array)

    plt.show()

    

i = 1

num_to_plot = 5

for file_name in os.listdir('../input/siim-isic-melanoma-classification/train/'):

        file_path = os.path.join('../input/siim-isic-melanoma-classification/train/',file_name)

        dataset = pydicom.dcmread(file_path)

        show_dcm_info(dataset)

        plot_pixel_array(dataset)

    

        if i >= num_to_plot:

            break

    

        i += 1
# source: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/154658

folder='train'

PATH='../input/siim-isic-melanoma-classification/'



def extract_DICOM_attributes(folder):

    images = list(os.listdir(os.path.join(PATH, folder)))

    df = pd.DataFrame()

    for image in images:

        image_name = image.split(".")[0]

        dicom_file_path = os.path.join(PATH,folder,image)

        dicom_file_dataset = pydicom.read_file(dicom_file_path)

        study_date = dicom_file_dataset.StudyDate

        modality = dicom_file_dataset.Modality

        age = dicom_file_dataset.PatientAge

        sex = dicom_file_dataset.PatientSex

        body_part_examined = dicom_file_dataset.BodyPartExamined

        patient_orientation = dicom_file_dataset.PatientOrientation

        photometric_interpretation = dicom_file_dataset.PhotometricInterpretation

        rows = dicom_file_dataset.Rows

        columns = dicom_file_dataset.Columns



        df = df.append(pd.DataFrame({'image_name': image_name, 

                        'dcm_modality': modality,'dcm_study_date':study_date, 'dcm_age': age, 'dcm_sex': sex,

                        'dcm_body_part_examined': body_part_examined,'dcm_patient_orientation': patient_orientation,

                        'dcm_photometric_interpretation': photometric_interpretation,

                        'dcm_rows': rows, 'dcm_columns': columns}, index=[0]))

    return df

extract_DICOM_attributes('train')