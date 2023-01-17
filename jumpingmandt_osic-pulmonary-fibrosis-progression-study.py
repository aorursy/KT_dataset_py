# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#visualisation
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import glob
import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation

#plotly
!pip install chart_studio
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


#read the .dcm file
import pydicom

from scipy.stats import probplot, mode

#color
from colorama import Fore, Back, Style

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Check the list of files or folders in the data source
list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))
# input the data
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

print(Fore.RED + 'Training data shape: ',Style.RESET_ALL,train_df.shape)
print(Fore.BLUE + 'Test data shape: ',Style.RESET_ALL,test_df.shape)
# preview of the train dataframe
train_df.head(5)
# preview of the test dataframe
test_df.head(5)
# Show the list of columns
columns = train_df.keys()
columns = list(columns)
print(Fore.BLUE + "List of columns in the train_df",Fore.RED + "", columns)
# check if there is missing data in the dataframe
# check the null part in the whole data set, red part is missing data, blue is non-null
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
train_df.isnull().sum()
# check the null part in the whole data set, red part is missing data, blue is non-null
sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
test_df.isnull().sum()
# check the type of dataframe
train_df.info()
# Check the unique number of patients' ids in train dataframe

print(Fore.WHITE + "In the train_df,",Fore.RED + "the total patient ids are",Style.RESET_ALL,f"{train_df['Patient'].count()},"
      , Fore.BLUE + "from those the unique ids are", Style.RESET_ALL, f"{train_df['Patient'].value_counts().shape[0]}.")
# Check the unique number of patients' ids in test dataframe

print(Fore.WHITE + "In the test_df,",Fore.RED + "the total patient ids are",Style.RESET_ALL,f"{test_df['Patient'].count()},"
      , Fore.BLUE + "from those the unique ids are", Style.RESET_ALL, f"{test_df['Patient'].value_counts().shape[0]}.")
# compare the test patients' ids and train patients' ids

train_patient_ids = set(train_df['Patient'].unique())
test_patient_ids = set(test_df['Patient'].unique())

# get the intersection of test and train datasets
test_patient_ids.intersection(train_patient_ids)
# The histogram of patients' samples (how many samples are in the same patient) 

plt.figure(figsize=(6,4))
train_df['Patient'].value_counts().hist(alpha=0.5,color='green',label='Samples for the same patient')
plt.legend()

plt.xlabel('Samples for the same patient')
plt.ylabel('Patients')
# Let's verify the features for the same patient, for example the patient with id = ID00419637202311204720264

train_df[train_df['Patient'] == 'ID00419637202311204720264']
# Let's verify the features for the same patient, for example the patient with id = ID00421637202311550012437

train_df[train_df['Patient'] == 'ID00421637202311550012437']
# Let's verify the features for the same patient, for example the patient with id = ID00422637202311677017371

train_df[train_df['Patient'] == 'ID00422637202311677017371']
# Let's verify the features for the same patient, for example the patient with id = ID00423637202312137826377

train_df[train_df['Patient'] == 'ID00423637202312137826377']
# Let's verify the features for the same patient, for example the patient with id = ID00426637202313170790466

train_df[train_df['Patient'] == 'ID00426637202313170790466']
# verify the relation between the 'FVC' and the 'Percent' for the same patient, for example the first patient with id = ID00123637202217151272140

sns.regplot(x="FVC", y="Percent", data=train_df[train_df['Patient'] == 'ID00123637202217151272140'])
# get the initial FVC at the week = 0, for example the first patient with id = ID00123637202217151272140
# FVC is integer, so it is necessary to use function .round()

initial_FVC = ((train_df[train_df['Patient'] == 'ID00123637202217151272140']['FVC'].iloc[0]) / (train_df[train_df['Patient']== 'ID00123637202217151272140']['Percent'].iloc[0])*100).round() 

print('The FVC of the patient with id = ID00123637202217151272140 is',Fore.BLUE + '', initial_FVC)
# get the initial FVC at the week = 0, for example the first patient with id = ID00009637202177434476278
# FVC is integer, so it is necessary to use function .round()

initial_FVC = ((train_df[train_df['Patient'] == 'ID00009637202177434476278']['FVC'].iloc[0]) / (train_df[train_df['Patient']== 'ID00009637202177434476278']['Percent'].iloc[0])*100).round() 

print('The FVC of the patient with id = ID00009637202177434476278 is',Fore.BLUE + '', initial_FVC)
# Create individual patient dataframe
patient_df = train_df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates()
patient_df.head()
# create a new row for each patient about the initial FVC at the week = 0
# # iterating the columns 
i = 0
Init_FVC = []
for row in patient_df.index: 
    ID = patient_df['Patient'].loc[row]
    temp_FVC = ((train_df[train_df['Patient'] == ID]['FVC'].iloc[0]) / (train_df[train_df['Patient']== ID]['Percent'].iloc[0])*100).round() 
    Init_FVC.append(temp_FVC)
    print(i,ID) 
    i = i+1

# add the initial FVC inside the patient_df 
patient_df['FVC'] = Init_FVC
patient_df.head()
# The corresponding Weeks is 0
patient_df['Weeks'] = 0
patient_df.head()
# The corresponding Percent is 100%
patient_df['Percent'] = 100
patient_df.head()
# check the sex elemnts in histogram
# The Histogram of sex
patient_df['Sex'].value_counts().iplot(kind='bar',yTitle='Counts',xTitle = 'Sex',linecolor='black',opacity=0.7,color='green',theme='pearl',bargap=0.5,
                                       gridcolor='white',title='Distribution of the Sex column in the Unique Patient Set')
# check the SmokingStatus elemnts in histogram
# The Histogram of SmokingStatus
patient_df['SmokingStatus'].value_counts().iplot(kind='bar',yTitle='Counts',xTitle = 'SmokingStatus',linecolor='black',opacity=0.7,color='red',theme='pearl',bargap=0.5,
                                       gridcolor='white',title='Distribution of the SmokingStatus column in the Unique Patient Set')
# check the age distribution in histogram
# The Histogram of age
plt.figure(figsize=(6,4))
patient_df['Age'].hist(alpha=0.5,color='blue',label='Age', bins = 30)
plt.legend()

plt.xlabel('Age')
plt.ylabel('Count')
# check the FVC distribution in histogram
# The Histogram of FVC
plt.figure(figsize=(6,4))
patient_df['FVC'].hist(alpha=0.5,color='brown',label='FVC(Week=0)', bins = 30)
plt.legend()

plt.xlabel('FVC(Week=0)')
plt.ylabel('Count')
# convert the "Ex-smoker" = 1, "Never smoked" = 0, "Currently smokes" = 2 in the "SmokingStatus"
Smoking_list = []
for i in np.arange(len(patient_df)):
    status = patient_df['SmokingStatus'].iloc[i]
    if status == 'Ex-smoker':
        Smoking_list.append(1)
    elif status == 'Never smoked':
        Smoking_list.append(0)
    else:
        Smoking_list.append(2)

patient_df['SmokingStatus'] = Smoking_list
patient_df.info()
# convert the "Male" = 1, "Female" = 0 in the "Sex"
Sex_list = []
for i in np.arange(len(patient_df)):
    gender = patient_df['Sex'].iloc[i]
    if gender == 'Male':
        Sex_list.append(1)
    else:
        Sex_list.append(0)

patient_df['Sex'] = Sex_list
#check the information of file '1.dcm' for the patient with id = ID00228637202259965313869
file_path = '../input/osic-pulmonary-fibrosis-progression/train/ID00228637202259965313869/1.dcm'
dicom_file = pydicom.dcmread(file_path)

print(f'Patient: ID00228637202259965313869 Image: 1.dcm Dataset\n{"." * 56}\n\n{dicom_file}')
#check the information of file '1.dcm' for the patient with id = ID00228637202259965313869
file_path = '../input/osic-pulmonary-fibrosis-progression/train/ID00228637202259965313869/2.dcm'
dicom_file = pydicom.dcmread(file_path)

print(f'Patient: ID00228637202259965313869 Image: 2.dcm Dataset\n{"." * 56}\n\n{dicom_file}')
#check the information of file '1.dcm' for the patient with id = ID00062637202188654068490
file_path = '../input/osic-pulmonary-fibrosis-progression/train/ID00062637202188654068490/1.dcm'
dicom_file = pydicom.dcmread(file_path)

print(f'Patient: ID00422637202311677017371 Image: 1.dcm Dataset\n{"." * 56}\n\n{dicom_file}')
#check the information of file '1.dcm' for the patient with id = ID00009637202177434476278
file_path = '../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/1.dcm'
dicom_file = pydicom.dcmread(file_path)
dicom_file.dir()
# get the relation between the 'SliceThickness', 'SingleCollimationWidth' and the difference of 'ImagePosition Patient of Z' or 'SliceLocation'
patient_name = 'ID00228637202259965313869'
patient_directory = sorted(os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}')
                               , key=(lambda f: int(f.split('.')[0])))
print (len(patient_directory))
# get the relation between the 'SliceThickness', 'SingleCollimationWidth' and the difference of 'ImagePosition Patient of Z' or 'SliceLocation'
patient_name = 'ID00228637202259965313869'
patient_directory = sorted(os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}')
                               , key=(lambda f: int(f.split('.')[0])))
for name in patient_directory:
    eachslice = pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}/{name}')
    print (eachslice.ImagePositionPatient[2], '  ',eachslice.SliceLocation,'   ', eachslice.SliceThickness, '   ', eachslice.SingleCollimationWidth,'    ', eachslice.TableSpeed)
# get the relation between the 'SliceThickness', 'SingleCollimationWidth' and the difference of 'ImagePosition Patient of Z' or 'SliceLocation'
patient_name = 'ID00422637202311677017371'
patient_directory = sorted(os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}')
                               , key=(lambda f: int(f.split('.')[0])))
for name in patient_directory:
    eachslice = pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}/{name}')
    print (eachslice.ImagePositionPatient[2], '  ',eachslice.SliceLocation,'   ', eachslice.SliceThickness, '   ', eachslice.SingleCollimationWidth,'    ', eachslice.TableSpeed)
patient_name = 'ID00007637202177411956430'
patient_directory = sorted(os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}')
                               , key=(lambda f: int(f.split('.')[0])))
for name in patient_directory:
    eachslice = pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}/{name}')
    print (eachslice.ImagePositionPatient[2], '  ',eachslice.SliceLocation,'   ', eachslice.SliceThickness, '   ')
# the first patient in the test file
patient_name = 'ID00419637202311204720264'
patient_directory = sorted(os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{patient_name}')
                               , key=(lambda f: int(f.split('.')[0])))
print (len(patient_directory))

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
patient_name = 'ID00419637202311204720264'
train_file_path = (f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}/10.dcm')
test_file_path = (f'../input/osic-pulmonary-fibrosis-progression/test/{patient_name}/10.dcm')

train_dataset = pydicom.dcmread(train_file_path)
test_dataset = pydicom.dcmread(test_file_path)



f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = (12, 12))

ax1.imshow(train_dataset.pixel_array, cmap='gray')
ax1.set_title("Train file")
ax1.axis('off')

ax2.imshow(test_dataset.pixel_array, cmap='gray')
ax2.set_title("Test file")
ax2.axis('off')

plt.show()
patient_name = 'ID00007637202177411956430'
patient_directory = sorted(os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}')
                               , key=(lambda f: int(f.split('.')[0])))
for name in patient_directory:
    eachslice = pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}/{name}')
    print (eachslice.ImagePositionPatient[2], '  ', eachslice.SliceThickness)

patient_df2 = patient_df.copy()

def get_metadata(patient_name):
    
    patient_directory = sorted(os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}')
                               , key=(lambda f: int(f.split('.')[0])))
    first_slice = pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}/{patient_directory[0]}')

    
    features = ['Manufacturer','TotalCollimationWidth','SingleCollimationWidth','TableSpeed','KVP','Columns','Rows','DistanceSourceToDetector',
               'DistanceSourceToPatient','GeneratorPower','HighBit','PixelRepresentation','SliceLocation','SliceThickness','TableHeight',
               'RevolutionTime','PatientPosition','XRayTubeCurrent']
    for feature in features:
        if feature in first_slice.dir():
             patient_df2.loc[patient_df2['Patient'] == patient_name, feature] = first_slice.get(feature)
    
    
    patient_df2.loc[patient_df2['Patient'] == patient_name, 'PixelSpacing'] = first_slice.PixelSpacing[0]
    
    
    if 'ImagePositionPatient' in first_slice.dir():
        patient_df2.loc[patient_df2['Patient'] == patient_name, 'ImagePositionPatient_X'] = first_slice.ImagePositionPatient[0]
        patient_df2.loc[patient_df2['Patient'] == patient_name, 'ImagePositionPatient_Y'] = first_slice.ImagePositionPatient[1]
        patient_df2.loc[patient_df2['Patient'] == patient_name, 'ImagePositionPatient_Z'] = first_slice.ImagePositionPatient[2]
for patient in patient_df2['Patient']:
    get_metadata(patient)

patient_df2
# check the Manufacturer distribution in histogram
fig = px.histogram(patient_df2, x="Manufacturer")
fig.show()
# check the SingleCollimationWidth distribution in histogram
fig = px.scatter(patient_df2, x="SingleCollimationWidth")
fig.show()
# check the TotalCollimationWidth distribution in histogram
fig = px.scatter(patient_df2, x="TotalCollimationWidth")
fig.show()
# check the SliceThickness distribution in histogram
fig = px.scatter(patient_df2, x="SliceThickness")
fig.show()
# check the RevolutionTime distribution in histogram
fig = px.histogram(patient_df2, x="RevolutionTime")
fig.show()

# check the TableSpeed distribution in scatter
fig = px.scatter(patient_df2, x="TableSpeed")
fig.show()
# check the PatientPosition distribution in scatter
fig = px.scatter(patient_df2, x="PatientPosition")
fig.show()

# check the SliceLocation distribution in scatter
fig = px.scatter(patient_df2, x="SliceLocation")
fig.show()
# https://www.kaggle.com/aadhavvignesh/lung-segmentation-by-marker-controlled-watershed
def load_scan(path):
    """
    Loads scans from a folder and into a list.
    
    Parameters: path (Folder path)
    
    Returns: slices (List of slices)
    """
    
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
# https://www.kaggle.com/aadhavvignesh/lung-segmentation-by-marker-controlled-watershed
def get_pixels_hu(scans):
    """
    Converts raw images to Hounsfield Units (HU).
    
    Parameters: scans (Raw images)
    
    Returns: image (NumPy array)
    """
    
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Since the scanning equipment is cylindrical in nature and image output is square,
    # we set the out-of-scan pixels to 0
    image[image == -2000] = 0
    
    
    # HU = m*P + b
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)
INPUT_FOLDER = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/'

patients = os.listdir(INPUT_FOLDER)
patients.sort()
test_patient_scans = load_scan(INPUT_FOLDER + patients[24])
test_patient_images = get_pixels_hu(test_patient_scans)
test_patient_scans[0]
plt.imshow(test_patient_images[0], cmap='gray')
plt.title("Original Slice")
plt.show()
# https://www.kaggle.com/aadhavvignesh/lung-segmentation-by-marker-controlled-watershed
def generate_markers(image):
    """
    Generates markers for a given image.
    
    Parameters: image
    
    Returns: Internal Marker, External Marker, Watershed Marker
    """
    
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
    
    # Creation of the External Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    
    # Creation of the Watershed Marker
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    
    return marker_internal, marker_external, marker_watershed
# https://www.kaggle.com/aadhavvignesh/lung-segmentation-by-marker-controlled-watershed
test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(test_patient_images[12])

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))

ax1.imshow(test_patient_internal, cmap='gray')
ax1.set_title("Internal Marker")
ax1.axis('off')

ax2.imshow(test_patient_external, cmap='gray')
ax2.set_title("External Marker")
ax2.axis('off')

ax3.imshow(test_patient_watershed, cmap='gray')
ax3.set_title("Watershed Marker")
ax3.axis('off')

plt.show()
test_patient_internal
test_patient_internal.shape
test_patient_internal_array = np.ravel(test_patient_internal)
test_patient_internal_array.shape
number_white_pixel = 0
for x in test_patient_internal_array:
    if x == True:
        number_white_pixel = number_white_pixel+1
print ('The number of white pixels is ', number_white_pixel)
print ('The percentage of white pixels in the dark image is {:0.2f}'.format( number_white_pixel/262144*100), '%')
# the pixel spacing of patient ID00062637202188654068490
print (Fore.BLUE + "The pixel spacing of patient ID00062637202188654068490 is", Style.RESET_ALL,test_patient_scans[0].PixelSpacing[0])
# The area of chest in this slice

print (Fore.RED + "The area of chest in the slice 12.dcm of patient ID00062637202188654068490 is", Style.RESET_ALL,(test_patient_scans[0].PixelSpacing[0])*(test_patient_scans[0].PixelSpacing[0])*number_white_pixel)
# only return the internal area
def only_internal(image):
    """
    Generates markers for a given image.
    
    Parameters: image
    
    Returns: Internal Marker, External Marker, Watershed Marker
    """
    
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
    
    return marker_internal
# The volume of chest of the patient ID00062637202188654068490
for image in test_patient_images:
    image_pixel = only_internal(image)
    test_patient_internal_array = np.ravel(test_patient_internal)
