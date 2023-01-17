import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os 
%matplotlib inline
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
from os import listdir, mkdir
import pydicom
import scipy.ndimage
import pydicom as dcm
import imageio
import tqdm as tqdm
import glob
from PIL import Image
from IPython.display import HTML

train = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")
test = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")
sub = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/sample_submission.csv")

train.shape , test.shape, sub.shape 
train.head()
train.nunique()
files = glob.glob('../input/rsna-str-pulmonary-embolism-detection/train/*/*/*.dcm')
features = []
for i in train.columns:
    features.append(i)
features = features[3:]
print(features)
print(len(features))

fig,ax = plt.subplots(3,5,figsize=(15,15))
sns.countplot(train['pe_present_on_image'],ax=ax[0][0],)
sns.countplot(train['negative_exam_for_pe'],ax=ax[0][1])
sns.countplot(train['qa_motion'],ax=ax[0][2])
sns.countplot(train['qa_contrast'],ax=ax[0][3])
sns.countplot(train['flow_artifact'],ax=ax[1][0])
sns.countplot(train['rv_lv_ratio_gte_1'],ax=ax[1][1])
sns.countplot(train['rv_lv_ratio_lt_1'],ax=ax[1][2])
sns.countplot(train['leftsided_pe'],ax=ax[1][3])
sns.countplot(train['chronic_pe'],ax=ax[1][4])
sns.countplot(train['true_filling_defect_not_pe'],ax=ax[2][0])
sns.countplot(train['rightsided_pe'],ax=ax[2][1])
sns.countplot(train['acute_and_chronic_pe'],ax=ax[2][2])
sns.countplot(train['central_pe'],ax=ax[2][3])
sns.countplot(train['indeterminate'],ax=ax[2][4])

train.info()
test.head()
print(f" Total predictions to be Done in test set are {test.shape[0]} samples")
print('Null values in train data:',train.isnull().sum().sum())
print('Null values in test data:',test.isnull().sum().sum())
sample = train.iloc[0]
sample
train["acute_pe"] = -1
train.drop("acute_pe",axis=1,inplace=True)
'''
def acute_pe_type(df):
    if df["chronic_pe"] == 0 and df["acute_and_chronic_pe "] == 0:
        df["acute_pe"] =1
    return df
'''
import plotly.express as px

from matplotlib import animation, rc
from plotly.subplots import make_subplots
import plotly.graph_objs as go
rc('animation', html='jshtml')

np.random.seed(42)
x = train.drop(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], axis=1).sum(axis=0).sort_values().reset_index()
x.columns = ['Labels', 'Records']
fig = px.bar(
    x, 
    x='Records', 
    y='Labels', 
    orientation='h', 
    title='Lables with Non-Zero Entries', 
    height=600, 
    width=800
)
fig.show()
data = train.drop(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], axis=1).astype(bool).sum(axis=1).reset_index()
data.columns = ['row', 'counts']
data = data.groupby(['counts'])['row'].count().reset_index()
fig = px.pie(
    data, 
    values=100 * data['row']/len(train), 
    names="counts", 
    title='Percentage Activations of Samples', 
    width=800, 
    height=500
)
fig.show()
corr = train.corr()
f,ax=plt.subplots(figsize=(10,10))
ax = sns.heatmap(corr,cmap="afmhot")#annot=True
def load_scans(dcm_path):
    files = listdir(dcm_path)
    f = [pydicom.dcmread(dcm_path + "/" + str(file)) for file in files]
    return f
basepath = "../input/rsna-str-pulmonary-embolism-detection/"
example = basepath + "train/" + train.StudyInstanceUID.values[0] +'/'+ train.SeriesInstanceUID.values[0]
file_names = os.listdir(example)
scans = load_scans(example)

print("Some Meta-Deta Information")
scans[1]
fig,ax = plt.subplots(figsize=(14,10))
ax.imshow(dcm.dcmread("../input/rsna-str-pulmonary-embolism-detection/train/6897fa9de148/2bfbb7fd2e8b/be0b7524ffb4.dcm").pixel_array);
#6897fa9de148	2bfbb7fd2e8b	41220fda34a3	
print("Sample DICOM Image CT Scan")
test_image = dcm.dcmread("../input/rsna-str-pulmonary-embolism-detection/train/4833c9b6a5d0/57e3e3c5f910/f4fdc88f2ace.dcm").pixel_array
print('Image shape: ', test_image.shape)
f, plots = plt.subplots(6, 6, sharex='col', sharey='row', figsize=(17, 17))
for i in range(36):
    plots[i // 6, i % 6].axis('off')
    plots[i // 6, i % 6].imshow(dcm.dcmread(np.random.choice(files[:5000])).pixel_array)
### Source: https://www.kaggle.com/allunia/pulmonary-fibrosis-dicom-preprocessing
def load_slice(path):
    slices = [dcm.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
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
sns.set_style('white')
hu_scans = transform_to_hu(scans)

fig, ax = plt.subplots(1,2,figsize=(15,4))


ax[0].set_title("CT-scan in HU")
ax[0].imshow(hu_scans[0], cmap="plasma")
#ax[1].set_title("HU values distribution");
sns.distplot(hu_scans[0].flatten(), ax=ax[1],color='red', kde_kws=dict(lw=2, ls="--",color='blue'));
ax[1].grid(False)
plt.figure(figsize=(12,6))
for n in range(20):
    image = scans[n].pixel_array.flatten()
    rescaled_image = image * scans[n].RescaleSlope + scans[n].RescaleIntercept
    sns.distplot(image.flatten());
plt.title("HU unit distributions for 20 examples");
sample_patient = load_slice('../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf')
sample_patient_pixels = transform_to_hu(sample_patient)

def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=5):
    fig,ax = plt.subplots(rows,cols,figsize=[18,20])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title(f'slice {ind}')
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='bone')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

sample_stack(sample_patient_pixels)
from IPython import display

imageio.mimsave("/tmp/gif.gif", sample_patient_pixels, duration=0.1)
display.Image(filename="/tmp/gif.gif", format='png')
def getvalue(feature):
    if type(feature) == pydicom.multival.MultiValue:
        return np.int(feature[0])
    else:
        return np.int(feature)

im_path = []
train_path = '../input/rsna-str-pulmonary-embolism-detection/train/'
for i in listdir(train_path): 
    for j in listdir(train_path + i):
        x = i+'/'+j
        im_path.append(x)

right_pixelspacing = []
center_pixelspacing = []
slice_thicknesses = []
ids = []
id_pth = []
row_values = []
column_values = []
window_widths = []
window_levels = []

for i in im_path:
    ids.append(i.split('/')[0]+'_'+i.split('/')[1])
    example_dcm = listdir(train_path  + i + "/")[0]
    id_pth.append(train_path + i)
    dataset = pydicom.dcmread(train_path + i + "/" + example_dcm)
    
    window_widths.append(getvalue(dataset.WindowWidth))
    window_levels.append(getvalue(dataset.WindowCenter))
    
    spacing = dataset.PixelSpacing
    slice_thicknesses.append(dataset.SliceThickness)
    
    row_values.append(dataset.Rows)
    column_values.append(dataset.Columns)
    right_pixelspacing.append(spacing[0])
    center_pixelspacing.append(spacing[1])

    
dicaom_meta = pd.DataFrame(data=ids, columns=["ID"])
dicaom_meta.loc[:, "rows"] = row_values
dicaom_meta.loc[:, "columns"] = column_values
dicaom_meta.loc[:, "area"] = dicaom_meta["rows"] * dicaom_meta["columns"]
dicaom_meta.loc[:, "right_pixel_space"] = right_pixelspacing
dicaom_meta.loc[:, "center_pixel_space"] = center_pixelspacing
dicaom_meta.loc[:, "pixelspacing_area"] = dicaom_meta.center_pixel_space * dicaom_meta.right_pixel_space
dicaom_meta.loc[:, "slice_thickness"] = slice_thicknesses
dicaom_meta.loc[:, "id_pth"] = id_pth
dicaom_meta.loc[:, "window_width"] = window_widths
dicaom_meta.loc[:, "window_level"] = window_levels
dicaom_meta.to_csv("meta_data_dcm.csv",index=False)
dicaom_meta.head()
