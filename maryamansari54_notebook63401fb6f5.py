# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
cufflinks.set_config_file(world_readable=True,theme='pearl')

#color
from colorama import Fore, Back, Style
import seaborn as sns
sns.set(style='whitegrid')

#supress warnings
import warnings
warnings.filterwarnings('ignore')

#settings for pretty nice plots
plt.style.use('fivethirtyeight')
plt.show()


#List files available
list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))
IMAGE_PATH = "../input/osic-pulmonary-fibrosis-progression/"
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
print(Fore.YELLOW + 'Training data shape: ',Style.RESET_ALL,train_df.shape)
train_df.head(5)
train_df.groupby(['SmokingStatus']).count()['Sex'].to_frame()
#Null values and Data types
print(Fore.YELLOW + 'Train set !!',Style.RESET_ALL)
print(train_df.info())
print('-----------------')
print(Fore.BLUE + 'Test set !!',Style.RESET_ALL)
print(test_df.info())
train_df.isnull().sum()
test_df.isnull().sum()
# Total number of patients in the dataset(train+test)
print(Fore.YELLOW + "Total Patients in Train Set: ",Style.RESET_ALL,train_df['Patient'].count())
print(Fore.BLUE + "Total Patients in Test set: ",Style.RESET_ALL,test_df['Patient'].count())
print(Fore.YELLOW + "The total patient IDs are ",Style.RESET_ALL,f"{train_df['Patient'].count()},",
      Fore.BLUE + "from those the unique IDs are",Style.RESET_ALL,f"{train_df['Patient'].value_counts().shape[0]}.")
train_patient_ids = set(train_df['Patient'].unique())
test_patient_ids = set(test_df['Patient'].unique())

train_patient_ids.intersection(test_patient_ids)
columns = train_df.keys()
columns = list(columns)
print(columns)
train_df['Patient'].value_counts().max()
test_df['Patient'].value_counts().max()
files = folders = 0
path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"

for _, dirnames, filenames in os.walk(path):
    # ^ this idion means "we won't be using this value"
    files += len(filenames)
    folders += len(dirnames)
#print (Fore.YELLOW + "Total Patients in Train set: ",Style.RESET_ALL,train_df['Patient'].count())
print(Fore.YELLOW + f'{files:,}',Style.RESET_ALL,"files/images " + Fore.BLUE + f'{folders:,}',Style.RESET_ALL,'folder/patients')
files = []
for _,dirnames,filenames in os.walk(path):
    files.append(len(filenames))
print(Fore.YELLOW +f'{round(np.mean(files)):,}',Style.RESET_ALL,'average files/images per patient')
print(Fore.BLUE + f'{round(np.max(files)):,}',Style.RESET_ALL,'max files/images per patient')
print(Fore.GREEN + f'{round(np.min(files)):,}',Style.RESET_ALL,'min files/images per patient')
patient_df = train_df[['Patient','Age','Sex','SmokingStatus']].drop_duplicates()
patient_df.head()
#Creating unique patient lists and their properties
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
patient_df = pd.DataFrame(list(zip(patient_ids,no_of_instances,age,sex,smoking_status)),
                         columns = ['Patient','no_of_instances','Age','Sex','SmokingStatus'])
print(patient_df.info())
patient_df.head()

patient_df['SmokingStatus'].value_counts()
patient_df['SmokingStatus'].value_counts().iplot(kind='bar',yTitle='Counts',linecolor='black',opacity=0.7,color='blue',theme='pearl',bargap=0.5,gridcolor='white',title='Distribution of the SmokingStatus column in the Unique Patient Set')
train_df['Weeks'].value_counts().iplot(kind = 'barh',xTitle = 'Counts(Weeks)',linecolor='black',opacity=0.7,color='#FB8072',theme='pearl',bargap=0.2,gridcolor='white',title='Distribution of the Weeks in the training set')
train_df['Weeks'].iplot(kind='hist',
                       xTitle='Weeks',
                       yTitle='Counts',
                       linecolor='black',
                       opacity=0.7,
                       color='#FB8072',
                       theme='PEARL',
                       bargap=0.2,
                       gridcolor='white',
                       title='Distribution of the Weeks in the training set')
fig = px.scatter(train_df,x="Weeks",y="Age",color='Sex')
fig.show()
train_df['FVC'].value_counts()
train_df['FVC'].iplot(kind='hist',
                     xTitle='Lung Capacity(mL)',
                     linecolor='black',
                     opacity=0.8,
                     color='#FB8072',
                     bargap=0.5,
                     gridcolor='white',
                     title='Distribution of the FVC in the training set')
fig=px.scatter(train_df,x="FVC",y="Percent",color='Age')
fig.show()
fig = px.scatter(train_df,x="FVC",y="Age",color='Sex')
fig.show()
fig = px.scatter(train_df,x="FVC",y="Weeks",color='SmokingStatus')
fig.show()
patient = train_df[train_df.Patient == 'ID00228637202259965313869']
fig = px.line(patient,x="Weeks",y="FVC",color='SmokingStatus')
fig.show()
train_df['Percent'].value_counts()
train_df['Percent'].iplot(kind='hist',bins=30,color='blue',xTitle='Percent distribution',yTitle='Count')
df = train_df
fig = px.violin(df,y='Percent',x='SmokingStatus',box=True,color='Sex',points="all",
               hover_data=train_df.columns)
fig.show()
plt.figure(figsize=(16,6))
ax = sns.violinplot(x = train_df['SmokingStatus'],y=train_df['Percent'],palette = 'Reds')
ax.set_xlabel(xlabel='Smoking Habit',fontsize=15)
ax.set_ylabel(ylabel='Percent',fontsize=15)
ax.set_title(label='Distribution of Smoking Status Over Percentage',fontsize=20)
plt.show()
fig = px.scatter(train_df,x="Age",y="Percent",color='SmokingStatus')
fig.show()
patient = train_df[train_df.Patient=='ID00228637202259965313869']
fig = px.line(patient,x="Weeks",y="Percent",color='SmokingStatus')
fig.show()
patient_df['Age'].iplot(kind='hist',bins=30,color='red',xTitle='Ages of distribution',yTitle='Count')
patient_df['SmokingStatus'].value_counts()
plt.figure(figsize=(16,6))
sns.kdeplot(patient_df.loc[patient_df['SmokingStatus']=='Ex-smoker','Age'],label='Ex-smoker',shade=True)
sns.kdeplot(patient_df.loc[patient_df['SmokingStatus']=='Never smoked','Age'],label='Never smoked',shade=True)
sns.kdeplot(patient_df.loc[patient_df['SmokingStatus']=='Currently smokes','Age'],label='Currently smokes',shade=True)
#labeling plot
plt.xlabel('Age(years)');plt.ylabel('Density');plt.title('Distribution of Ages');

plt.figure(figsize=(16,6))
ax = sns.violinplot(x = patient_df['SmokingStatus'],y=train_df['Age'],palette = 'Reds')
ax.set_xlabel(xlabel='Smoking Habit',fontsize=15)
ax.set_ylabel(ylabel='Age',fontsize=15)
ax.set_title(label='Distribution of Smoking Status Over Age',fontsize=20)
plt.show()
plt.figure(figsize=(16,6))
sns.kdeplot(patient_df.loc[patient_df['Sex']=='Male','Age'],label='Male',shade=True)
sns.kdeplot(patient_df.loc[patient_df['Sex']=='Female','Age'],label='Female',shade=True)
plt.xlabel('Age (years)');plt.ylabel('Density');plt.title('Distribution of Ages');
patient_df['Sex'].value_counts()
patient_df['Sex'].value_counts().iplot(kind='bar',yTitle='Count',linecolor='black',opacity=0.7,color='blue',theme='pearl',bargap=0.8,gridcolor='white',title='Distribution of the Sex column in Patient Dataframe')
plt.figure(figsize=(16,6))
a = sns.countplot(data=patient_df,x='SmokingStatus',hue='Sex')
for p in a.patches:
    a.annotate(format(p.get_height(),','),
               (p.get_x() + p.get_width()/2.,
               p.get_height()),ha='center',va='center',xytext=(0,4),textcoords='offset points')
plt.title('Gender split by SmokingStatus',fontsize=16)
sns.despine(left=True,bottom=True)

fig=px.box(patient_df,x="Sex",y="Age",points="all")
fig.show()
train_patient_ids = set(train_df['Patient'].unique())
test_patient_ids = set(test_df['Patient'].unique())
train_patient_ids.intersection(test_patient_ids)

print(Fore.YELLOW + "There are ",Style.RESET_ALL,f"{train_df['Patient'].value_counts().shape[0]}",
      Fore.BLUE + "unique Patient IDs",Fore.WHITE + "in the training set.")
print(Fore.YELLOW + "There are ",Style.RESET_ALL,f"{test_df['Patient'].value_counts().shape[0]}",
      Fore.BLUE + "unique Patient IDs",Fore.WHITE + "in the test set.")
print(Fore.YELLOW + "There are ",Style.RESET_ALL,f"{len(train_patient_ids.intersection(test_patient_ids))}",
     Fore.BLUE,"Patient IDs",Fore.WHITE + "in both the training and test sets")
print(Fore.CYAN + "These patients are in both the training and test data sets:")
print(Fore.WHITE,Style.RESET_ALL,f"{train_patient_ids.intersection(test_patient_ids)}")


corrmat = train_df.corr()
f,ax = plt.subplots(figsize = (9,8))
sns.heatmap(corrmat,ax=ax,cmap='RdYlBu_r',linewidths=0.5)
print(Fore.YELLOW + 'Train .dcm number of images:',Style.RESET_ALL,len(list(os.listdir('../input/osic-pulmonary-fibrosis-progression/train'))), '\n' +
Fore.BLUE + 'Test .dcm number of images:',Style.RESET_ALL,len(list(os.listdir('../input/osic-pulmonary-fibrosis-progression/test'))), '\n' +
'-----------------------------------','\n' +
'There is the same number of images as in train/test .csv datasets')
def plot_pixel_array(dataset,figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.grid(False)
    plt.imshow(dataset.pixel_array,cmap='gray')
    plt.show()
import pydicom
print(__doc__)
filename = '../input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/137.dcm'
dataset = pydicom.dcmread(filename)
#Normal mode:
print()
print("Filename.........:", filename)
pat_name = dataset.PatientName
display_name = pat_name.family_name
print("Patient's name...:", display_name)
print("Patient id.......:", dataset.PatientID)
print("Modality.........:", dataset.Modality)
print("Body Part Examined.........:", dataset.BodyPartExamined)
if 'PixelData' in dataset:
    rows = int(dataset.Rows)
    cols = int(dataset.Columns)
    print("Image size.........: {rows:d} x {cols:d}, {size:d} bytes".format(rows=rows,cols=cols,size=len(dataset.PixelData)))
    if 'PixelSpacing' in dataset:
        print("Pixel spacing.........",dataset.PixelSpacing)
#use .get() if not sure the item exists, and want a default value if missing
print("Slice location.........",dataset.get('SliceLocation',"(missing)"))
#plot the image using matplotlib
plt.imshow(dataset.pixel_array,cmap=plt.cm.bone)
plt.show()

imdir = "../input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140"
print("total images for patient ID00123637202217151272140:",len(os.listdir(imdir)))
#view first (columns*rows) images in order
fig=plt.figure(figsize=(12,12))
columns=4
rows=5
imlist=os.listdir(imdir)
for i in range(1,columns*rows+1):
    filename = imdir + "/" + str(i) + ".dcm"
    ds = pydicom.dcmread(filename)
    fig.add_subplot(rows,columns,i)
    plt.imshow(ds.pixel_array,cmap='gray')
imdir = "../input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140"
print("total images for patient ID00123637202217151272140:",len(os.listdir(imdir)))
#view first (columns*rows) images in order
fig=plt.figure(figsize=(12,12))
columns=4
rows=5
imlist=os.listdir(imdir)
for i in range(1,columns*rows+1):
    filename = imdir + "/" + str(i) + ".dcm"
    ds = pydicom.dcmread(filename)
    fig.add_subplot(rows,columns,i)
    plt.imshow(ds.pixel_array,cmap='jet')
def load_scan(path):
    slices=[pydicom.read_file(path+'/'+s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation-slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices
def get_pixels_hu(scans):
    image=np.stack([s.pixel_array for s in scans])
    #convert to int16 (from sometimes int16),
    #should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    #set outside-of-scan pixels to 1
    #the intercept is usually -1024, so air is approximately 0
    image[image == -2000]=0
    #convert to hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    if slope != 1:
        image = slope*image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image,dtype=np.int16)
def set_lungwin(img,hu=[-1200.,600.]):
    lungwin = np.array(hu)
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg
id=0
scans=load_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140')
scan_array = set_lungwin(get_pixels_hu(scans))
import imageio
from IPython.display import Image
imageio.mimsave("/tmp/gif.gif",scan_array,duration=0.0001)
Image(filename="/tmp/gif.gif",format='png')
import matplotlib.animation as animation
from IPython.display import HTML
fig = plt.figure()
ims = []
for image in scan_array:
    im = plt.imshow(image,animated=True,cmap="Greys")
    plt.axis("off")
    ims.append([im])
ani = animation.ArtistAnimation(fig,ims,interval=100,blit=False,repeat_delay=1000)
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

#convert to a pd.DataFrame from dict
meta_data_df = pd.DataFrame.from_dict(meta_data_df)
#meta_data_df.head()
display(meta_data_df)
data_dir = '../input/osic-pulmonary-fibrosis-progression/train/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
for patient in patients[:20]:
    try:
        path = data_dir + patient
        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: int(x.InstanceNumber))
        print(len(slices),slices[0].pixel_array.shape)
    
    except Exception as e:
        # again, some patients are not labeled, but JIC we still want the error if something
        # else is wrong with our code
        print(str(e))
import cv2
import math
def chunks(l,n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from 1."""
    for i in range(0,len(l),n):
        yield l[i:i+n]
def mean(l):
    return sum(l)/len(l)
IMG_PX_SIZE = 30
HM_SLICES = 15
data_dir = '../input/osic-pulmonary-fibrosis-progression/train/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
new_slices = []


for patient in patients[:100]:
    try:
        path = data_dir + patient
        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: int(x.InstanceNumber))
        new_slices = []

        slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

        chunk_sizes = math.ceil(len(slices) / HM_SLICES)


        for slice_chunk in chunks(slices, chunk_sizes):
            slice_chunk = list(map(mean, zip(*slice_chunk)))
            new_slices.append(slice_chunk)

        if len(new_slices) < HM_SLICES:
            while len(new_slices) < HM_SLICES:
                new_slices.append(new_slices[-1])

        if len(new_slices) > HM_SLICES:
            while len(new_slices) > HM_SLICES:
                new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
                del new_slices[HM_SLICES]
                new_slices[HM_SLICES-1] = new_val

        print(len(slices), len(new_slices))
    
    except Exception as e:
        # again, some patients are not labeled, but JIC we still want the error if something
        # else is wrong with our code
        print(str(e))
import pandas_profiling as pdp
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

profile_train_df = pdp.ProfileReport(train_df)
profile_train_df
profile_test_df = pdp.ProfileReport(test_df)
profile_test_df