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

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import pydicom
import scipy.ndimage

from skimage import measure 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from IPython.display import HTML

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from os import listdir
basepath = "../input/osic-pulmonary-fibrosis-progression/"
listdir(basepath)

train = pd.read_csv(basepath + "train.csv")
test = pd.read_csv(basepath + "test.csv")

def load_scans(dcm_path):
    slices = [pydicom.dcmread(dcm_path + "/" + file) for file in listdir(dcm_path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    return slices
example = basepath + "train/" + train.Patient.values[0]
scans = load_scans(example)
train.isnull().values.any() 
train.describe()
test.describe()
train.head()
scans[0]
fig, ax = plt.subplots(1,2,figsize=(20,5))
for n in range(10):
    image = scans[n].pixel_array.flatten()
    rescaled_image = image * scans[n].RescaleSlope + scans[n].RescaleIntercept
    sns.distplot(image.flatten(), ax=ax[0]);
    sns.distplot(rescaled_image.flatten(), ax=ax[1])
ax[0].set_title("Raw pixel array distributions for 10 examples")
ax[1].set_title("HU unit distributions for 10 examples");
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
hu_scans = transform_to_hu(scans)

fig, ax = plt.subplots(1,4,figsize=(20,3))
ax[0].set_title("Original CT-scan")
ax[0].imshow(scans[0].pixel_array, cmap="bone")
ax[1].set_title("Pixelarray distribution");
sns.distplot(scans[0].pixel_array.flatten(), ax=ax[1]);

ax[2].set_title("CT-scan in HU")
ax[2].imshow(hu_scans[0], cmap="bone")
ax[3].set_title("HU values distribution");
sns.distplot(hu_scans[0].flatten(), ax=ax[3]);

for m in [0,2]:
    ax[m].grid(False)
pixelspacing_w = []
pixelspacing_h = []
slice_thicknesses = []
patient_id = []
patient_pth = []

for patient in train.Patient.values:
    patient_id.append(patient)
    example_dcm = listdir(basepath + "train/" + patient + "/")[0]
    patient_pth.append(basepath + "train/" + patient)
    dataset = pydicom.dcmread(basepath + "train/" + patient + "/" + example_dcm)
    spacing = dataset.PixelSpacing
    slice_thicknesses.append(dataset.SliceThickness)
    pixelspacing_w.append(spacing[0])
    pixelspacing_h.append(spacing[1])
fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(pixelspacing_w, ax=ax[0], color="Limegreen", kde=False)
ax[0].set_title("Pixel spacing width \n distribution")
ax[0].set_ylabel("Counts in train")
ax[0].set_xlabel("width in mm")
sns.distplot(pixelspacing_h, ax=ax[1], color="Mediumseagreen", kde=False)
ax[1].set_title("Pixel spacing height \n distribution");
ax[1].set_ylabel("Counts in train");
ax[1].set_xlabel("height in mm");
plt.figure(figsize=(10,5))
sns.distplot(slice_thicknesses, color="orangered", kde=False)
plt.title("Slice thicknesses of all patients");
plt.xlabel("Slice thickness in mm")
plt.ylabel("Counts in train");
my_attribute = pixelspacing_w
min_idx = np.argsort(my_attribute)[0]
max_idx = np.argsort(my_attribute)[-1]

patient_min = patient_pth[min_idx]
patient_max = patient_pth[max_idx]

min_scans = load_scans(patient_min)
min_hu_scans = transform_to_hu(min_scans)

max_scans = load_scans(patient_max)
max_hu_scans = transform_to_hu(max_scans)
def plot_3d(image, threshold=700, color="navy"):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces,_,_ = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.5)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
plot_3d(max_hu_scans)
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing
pix_resampled, spacing = resample(max_hu_scans, scans, [1,1,1])
print("Shape before resampling\t", hu_scans.shape)
print("Shape after resampling\t", pix_resampled.shape)
def segment_tissue(image, threshold=-300, fill_lung_structures=True):
    
    labelled_image = np.array(image > threshold, dtype=np.int8)+1
    labels = measure.label(labelled_image)
    
    
    #Fill the air around the person
    background_label = labels[0,0]
    labelled_image[background_label == labels] = 2
    
    labelled_image -= 1 #Make the image actual binary
    labelled_image = 1-labelled_image # Invert it, lungs are now 1
    
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(labelled_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                labelled_image[i][labeling != l_max] = 1
    
    # Remove other air pockets insided body
    labels = measure.label(labelled_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        labelled_image[labels != l_max] = 0
    
    return labelled_image
segmented_lungs = segment_tissue(max_hu_scans, fill_lung_structures=False)
segmented_lungs_fill = segment_tissue(max_hu_scans, fill_lung_structures=True)
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(segmented_lungs[20], cmap="bone_r")
ax[1].imshow(segmented_lungs_fill[20], cmap="bone_r")
for n in range(2):
    ax[n].grid(False)
plot_3d(segmented_lungs_fill, threshold=0, color="crimson")
plot_3d(segmented_lungs_fill-segmented_lungs, threshold=0, color="crimson")


















#!conda install -c conda-forge gdcm -y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import os
from sympy import symbols
from sympy import expand
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from skimage import morphology
from skimage import measure
from skimage.transform import resize
import tensorflow as tf
from sklearn.cluster import KMeans
import matplotlib.patches as patches
import PIL
train_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train_csv.info()
unique_ids=train_csv['Patient'].unique()
week_x=[]
fvc_y=[]
percent=[]
for id in unique_ids:
    week=np.array(train_csv[train_csv['Patient']==id]['Weeks'])
    fvc=np.array(train_csv[train_csv['Patient']==id]['FVC'])
    per=np.array(train_csv[train_csv['Patient']==id]['Percent'])
    week_x.append(week)
    fvc_y.append(fvc)
    percent.append(per)
    
unique_train=pd.DataFrame(train_csv['Patient'].unique(),columns=['Patient'])
unique_train['week_x']=week_x
unique_train['fvc_y']=fvc_y
unique_train['percent']=percent
unique_train.head()
X=unique_train['week_x'][0]
Y=unique_train['fvc_y'][0]
def using_poly_reg(X,Y,degree=3):
    poly_features=PolynomialFeatures(degree=degree,include_bias=False)
    x_poly=poly_features.fit_transform(X[:,np.newaxis])

    lin_reg=LinearRegression()
    lin_reg.fit(x_poly,Y)

    x_test=np.arange(-12,133)[:,np.newaxis]
    x_test_poly=poly_features.fit_transform(x_test)
    plt.plot(x_test,lin_reg.predict(x_test_poly))
    plt.plot(X,Y)
    #plt.ylim(0,6400)
    #plt.xlim(-12,133)
    plt.grid(True)
using_poly_reg(X,Y,degree=3)  #here we can customize the degree of the polynomial so it is better
                              #by the way if degree=len(X)-1 then it is same as interpolation
train_csv['healthy_person_FVC']=(train_csv['FVC']/(train_csv['Percent']/100)).round()
train_csv
healthy_fvc_info=train_csv.groupby(['Age','Sex','SmokingStatus'])['healthy_person_FVC'].mean().round()
plt.plot(healthy_fvc_info[:,'Male','Ex-smoker'],label='male ex smoker')
plt.plot(healthy_fvc_info[:,'Male','Never smoked'],label='male never smoked')
plt.plot(healthy_fvc_info[:,'Male','Currently smokes'],label='male currently smokes')

plt.plot(healthy_fvc_info[:,'Female','Ex-smoker'],label='female ex smoker')
plt.plot(healthy_fvc_info[:,'Female','Never smoked'],label='female never smoked')
plt.plot(healthy_fvc_info[:,'Female','Currently smokes'],label='female currently smokes')

plt.title('healthy fvc related to age,sex and smoking status')
plt.legend()
plt.grid(True)
def RForestRegressor(x,y):
    reg=RandomForestRegressor(n_estimators=50)
    reg.fit(x[:,np.newaxis],y)
    x_test=np.arange(0,100)
    y_test=reg.predict(x_test[:,np.newaxis])
    plt.plot(x_test,y_test,label='predicted')
    plt.plot(x,y,label='real')
    plt.grid(True)
    plt.legend()
#x=np.array(healthy_fvc_info[:,'Male','Ex-smoker'].index)
#y=np.array(healthy_fvc_info[:,'Male','Ex-smoker'].values)
x=np.array(healthy_fvc_info[:,'Male','Never smoked'].index)
y=np.array(healthy_fvc_info[:,'Male','Never smoked'].values)

RForestRegressor(x,y)

X=unique_train['week_x'][0]
Y=unique_train['fvc_y'][0]
RForestRegressor(X,Y)
age=train_csv.groupby('Patient')['Age'].unique()
for item in age:
    if len(item)==1:
        continue
    else:
        print(item.index)
SS=train_csv.groupby('Patient')['SmokingStatus'].unique()
for item in SS:
    if len(item)==1:
        continue
    else:
        print(item)
sex=[]
for id in unique_train['Patient']:
    sex.append(train_csv[train_csv['Patient']==id]['Sex'].unique()[0])
    
unique_train['sex']=sex
age=[]
for id in unique_train['Patient']:
    age.append(train_csv[train_csv['Patient']==id]['Age'].unique()[0])
    
unique_train['age']=age
ss=[]
for id in unique_train['Patient']:
    ss.append(train_csv[train_csv['Patient']==id]['SmokingStatus'].unique()[0])
    
unique_train['smoking-status']=ss
unique_train
'''from sklearn.cluster import KMeans

lung=pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/26.dcm')
image=lung.pixel_array
X = image.reshape(-1,1)
#X=image

#good_init=np.array([[-2048],[-1000],[892],[-177],[190]])
#kmeans = KMeans(n_clusters=8,init=good_init,n_init=1).fit(X)

kmeans = KMeans(n_clusters=6).fit(X)

segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)
plt.imshow(segmented_img)'''

def fitter(img):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/4):int(col_size/4*3),int(row_size/4):int(row_size/4*3)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    return kmeans

lung=pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/26.dcm')
image=lung.pixel_array

kmeans=fitter(image)
def make_lungmask(img,kmeans,display=False):
    image=img
    row_size= img.shape[0]
    col_size = img.shape[1]
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/4):int(col_size/4*3),int(row_size/4):int(row_size/4*3)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)

    
    #
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([5,5]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    #for prop in regions:
    #    b = prop.bbox
    #    if (abs((b[2]+b[0])/2-(row_size/2))<100) and ( (abs((b[3]+b[1])/2-(col_size/4))<110) or (abs((b[3]+b[1])/2-(col_size/4)*3)<110) ):
    #        good_labels.append(prop.label)
    

    for prop in regions:
        b = prop.bbox
        lung_row=abs((b[2]+b[0])/2-(row_size/2))
        left_lung_col=abs((b[3]+b[1])/2-(col_size/4))
        right_lung_col=abs((b[3]+b[1])/2-(col_size/4)*3)
        
        if lung_row<100 and (left_lung_col<110 or right_lung_col<110):
            good_labels.append(prop.label)
            
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([8,8])) # one last dilation
    
    
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
        
        
    air=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i][j]==1:
                air.append(image[i][j])
    if len(air)==0:
        air_percent=0.0
    else:
        air_percent=abs((sum(air)/len(air))/10).round(4)
    return mask,air_percent
id='ID00011637202177653955184'
path='../input/osic-pulmonary-fibrosis-progression/train/'+id+'/' 
filenames=os.listdir(path) 
fileno=int(len(filenames)/2)
img=pydicom.dcmread(path+str(fileno)+'.dcm')
#make_lungmask(img,kmeans,display=True)
fig=plt.figure(figsize=(20,20)) 
col=14
row=14 
i=1 
air_percent_dict={}
for id in train_csv['Patient'].unique(): 
    path='../input/osic-pulmonary-fibrosis-progression/train/'+id+'/' 
    filenames=os.listdir(path) 
    fileno=int(len(filenames)/2)
    try:
        lung=pydicom.dcmread(path+str(fileno)+'.dcm') 
        image=lung.pixel_array 
        fig.add_subplot(row,col,i) 
        mask,air_percent=make_lungmask(image,kmeans,display=False)
        air_percent_dict[id]=air_percent
        plt.title(air_percent)
        plt.imshow(mask,cmap='gray')
        plt.grid(False)
        plt.axis(False)
    except: 
        print(id) 
    i=i+1 
    if i==177: 
        break
train=train_csv
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()#sex
train.iloc[:,5]=lb.fit_transform(train.iloc[:,5])
lb2=LabelEncoder()#ss
train.iloc[:,6]=lb2.fit_transform(train.iloc[:,6])
lung_percent=[]
for id in train['Patient']:
    lung_percent.append(float(air_percent_dict[id]))
train['lung percent']=lung_percent
train
test=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
test.iloc[:,5]=lb.transform(test.iloc[:,5])
test.iloc[:,6]=lb2.transform(test.iloc[:,6])
air_percent_dict={}
for id in test['Patient'].unique(): 
    path='../input/osic-pulmonary-fibrosis-progression/test/'+id+'/' 
    filenames=os.listdir(path) 
    fileno=int(len(filenames)/2)
    try:
        lung=pydicom.dcmread(path+str(fileno)+'.dcm') 
        image=lung.pixel_array  
        mask,air_percent=make_lungmask(image,kmeans,display=False)
        air_percent_dict[id]=air_percent
    except: 
        print(id)
lung_percent_test=[]
for id in test['Patient']:
    lung_percent_test.append(float(air_percent_dict[id]))
test['lung percent']=lung_percent_test
test
train.to_csv('changed_train.csv',index=False)
def healthy_fvc_predictor(age,sex,smoking_status):
    x=np.array(healthy_fvc_info[:,sex,smoking_status].index)
    y=np.array(healthy_fvc_info[:,sex,smoking_status].values)
    reg=RandomForestRegressor(n_estimators=50)
    reg.fit(x[:,np.newaxis],y)
    return reg.predict([[age]])
def RForestRegressor(x,y):
    reg=RandomForestRegressor(n_estimators=400)
    reg.fit(np.array(x),np.array(y))
    return reg

x=train[['Weeks','Percent','SmokingStatus','Sex','Age']]
#x=train[['Percent','lung percent','Weeks','Sex']]
y=train['FVC']
percent_reg=RForestRegressor(x,y)



test_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
def plot_fi(forest,X):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature : %s (%f)" % (f + 1, np.array(X.columns)[indices[f]], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],color="g", yerr=std[indices])
    plt.xticks(range(X.shape[1]),np.array(X.columns)[indices])
    plt.xlim([-1, X.shape[1]])
    plt.show()
    
plot_fi(percent_reg,x)
test_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
test_csv.iloc[:,5]=lb.transform(test_csv.iloc[:,5])
test_csv.iloc[:,6]=lb2.transform(test_csv.iloc[:,6])
weeks=np.arange(-12,134)
result={}
for id in test_csv['Patient'].unique():
    percent=np.array(test_csv[test_csv['Patient']==id]['Percent'])
    sex=np.array(test_csv[test_csv['Patient']==id]['Sex'])
    age=np.array(test_csv[test_csv['Patient']==id]['Age'])
    ss=np.array(test_csv[test_csv['Patient']==id]['SmokingStatus'])
    percent=np.repeat(percent,len(weeks))
    sex=np.repeat(sex,len(weeks))
    age=np.repeat(age,len(weeks))
    ss=np.repeat(ss,len(weeks))
    x=np.concatenate([weeks[:,np.newaxis],percent[:,np.newaxis],ss[:,np.newaxis],sex[:,np.newaxis],age[:,np.newaxis]],axis=1)
    outcome=percent_reg.predict(x)
    result[id]=outcome
ans_df_list=[]
for i,id in enumerate(result):
    ID=np.repeat(id,len(weeks))
    ans=np.concatenate([ID[:,np.newaxis],weeks[:,np.newaxis],result[id][:,np.newaxis]],axis=1)
    ans=pd.DataFrame(ans)
    ans_df_list.append(ans)
submit=pd.concat(ans_df_list,ignore_index=True)
submit.columns=['Patient','Weeks','FVC']
submit
test_csv=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
confidence_dict={}
for id in submit['Patient'].unique():
    real=float(test_csv[test_csv['Patient']==id]['FVC'])
    predicted=float(submit[(submit['Patient']==id) & (submit['Weeks'].astype(int)==int(test_csv[test_csv['Patient']==id]['Weeks']))]['FVC'])
    confidence_dict[id]=abs(real-predicted)
confidence=[]
for i in range(len(submit)):
    confidence.append(confidence_dict[submit.iloc[i,0]])
submit['Confidence']=confidence
submit['Patient']=submit['Patient']+'_'+(submit['Weeks'].astype(str))
submit.drop(['Weeks'],axis=1,inplace=True)
submit.columns=['Patient_Week','FVC','Confidence']
submit['FVC']=submit['FVC'].astype(float)
submit
submit.to_csv('submission.csv',index=False)
