# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 

from skimage import transform

# For Lee Filter - Speckle reduction function
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

# Closing
from skimage.morphology import closing, disk

# Gaussian filter
from skimage.filters import gaussian

# Median filter
from skimage.filters.rank import median

# Thresholding
from skimage.filters import threshold_otsu

# Lable regions
from skimage.measure import label, regionprops

# Derive statistical features
from scipy import stats

# Extract Texture Features - GLMC
from skimage.feature import greycomatrix, greycoprops

# Model evaluation
# Evaluate using Leave One Out Cross Validation
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

# K -Fold
from sklearn.model_selection import KFold

# Logistic regression
from sklearn.linear_model import LogisticRegression

# SVM
from sklearn.svm import SVC




# scaling
def scale_img(dframe):
    final_img = []
    
    for i, r in dframe.iterrows():
        # reshaping images to 75x75
        band_1 = np.array(r['band_1']).reshape(75, 75)
        band_2 = np.array(r['band_2']).reshape(75, 75)
        
        # Rescale to 0-255 px (local scaling : min-max)
        b1 = 255*(band_1 - band_1.min()) / (band_1.max() - band_1.min())
        b2 = 255*(band_2 - band_2.min()) / (band_2.max() - band_2.min())
        
        # Rescale to 0-255 px (global scaling(min,max) = (-50,40))
        b3 = 255*(band_1 - (-50)) / (40 - (-50))
        b4 = 255*(band_2 - (-50)) / (40 - (-50))
        
        # center image
        p=12 # boundry pixels
        final_img.append((b1[p:(75-p),p:(75-p)], b2[p:(75-p),p:(75-p)],
                          b3[p:(75-p),p:(75-p)], b4[p:(75-p),p:(75-p)]))
    
    
    return np.array(final_img,dtype=np.uint8)

# Lee Filter - Speckle reduction
def lee_filter(img, size=3):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = variance(img)
    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    img_output = 255*(img_output - img_output.min()) / (img_output.max() - img_output.min())
    return np.array(img_output,dtype=np.uint8)

# function to add, substract and multiply pixels (equivalent to open cv functions)
def img_add(image, val = 0):
    img = np.float16(image) + np.float16(val)
    img[img>255]=255
    img[img<0]=0
    return np.uint8(np.round(img))

def img_sub(image, val = 0):
    img = np.float16(image) - np.float16(val)
    img[img>255]=255
    img[img<0]=0
    return np.uint8(np.round(img))

def img_mult(image, val = 1):
    img = np.float16(image) * np.float16(val)
    img[img>255]=255
    img[img<0]=0
    return np.uint8(np.round(img))

def img_mean(image, val = 1):
    img = (np.float16(image) + np.float16(val))/2
    img[img>255]=255
    img[img<0]=0
    return np.uint8(np.round(img))


# Function to return features for an image
def shape_features(img_HH,img_HV,glob_img_HH,glob_img_HV):
    # center image
    #p=12 # boundry pixels
    #img_HH = img_HH[p:(75-p),p:(75-p)] #[y1:y2,x1:x2]
    #img_HV = img_HV[p:(75-p),p:(75-p)]
    #glob_img_HH = glob_img_HH[p:(75-p),p:(75-p)]
    #glob_img_HV = glob_img_HV[p:(75-p),p:(75-p)]
    
    img = img_HH.copy()
    # Apply Lee filter - remove speckle
    img = lee_filter(img)
    
    # Apply closing to remove noise
    img = closing(img, disk(1))
    
    # Improve contrast
    img = img_add(img_mult(img,1.2),-150)
    
    # Smooth image
    img = gaussian(img, sigma=0.8)
    
    # reduce noise
    img = median(img, disk(1))
    
    # apply automatic threshold otsu
    thresh = threshold_otsu(img)
    binary = img > thresh + 10 # setting higher threshold
    
    # select region with maximum area to extract features
    label_image = label(binary)
    max_reg = max(regionprops(label_image), key=lambda region: region.area)
    
    # get count of regions
    count_reg = len(regionprops(label_image))
    
    # extract max region image boundary
    minr, minc, maxr, maxc = max_reg.bbox
    
    # stats of pixels
    pixels_HH = glob_img_HH[minr:maxr,minc:maxc][binary[minr:maxr,minc:maxc]]
    pixels_HV = glob_img_HV[minr:maxr,minc:maxc][binary[minr:maxr,minc:maxc]]
    
    stats_HH = stats.describe(pixels_HH)
    stats_HV = stats.describe(pixels_HH)
    
    # GLCM - texture features
    glcm_HH = greycomatrix(glob_img_HH[minr:maxr,minc:maxc], [3], [0], 256, symmetric=True, normed=True)
    glcm_HV = greycomatrix(glob_img_HV[minr:maxr,minc:maxc], [3], [0], 256, symmetric=True, normed=True)
    
    # return features
    return np.array([count_reg,
                     glob_img_HH.mean(),
                     glob_img_HH.min(),
                     glob_img_HH.max(),
                     glob_img_HH.std(),
                     glob_img_HV.mean(),
                     glob_img_HV.min(),
                     glob_img_HV.max(),
                     glob_img_HV.std(),
                     max_reg.area,
                     max_reg.convex_area,
                     max_reg.eccentricity,
                     max_reg.equivalent_diameter,
                     max_reg.extent,
                     max_reg.filled_area,
                     max_reg.major_axis_length,
                     max_reg.minor_axis_length,
                     max_reg.orientation,
                     max_reg.perimeter,
                     max_reg.solidity,
                     pixels_HH.mean(),
                     pixels_HH.min(),
                     pixels_HH.max(),
                     pixels_HH.std(),
                     pixels_HH.var(),
                     stats_HH.kurtosis,
                     stats_HH.skewness,
                     pixels_HV.mean(),
                     pixels_HV.min(),
                     pixels_HV.max(),
                     pixels_HV.std(),
                     pixels_HV.var(),
                     stats_HV.kurtosis,
                     stats_HV.skewness,
                     greycoprops(glcm_HH, 'dissimilarity')[0, 0],
                     greycoprops(glcm_HH, 'correlation')[0, 0],
                     greycoprops(glcm_HV, 'dissimilarity')[0, 0],
                     greycoprops(glcm_HV, 'correlation')[0, 0]
                    ])
    
# load dataset
df = pd.read_json('../input/train.json')

# Apply local scaling on the images
X = scale_img(df)

# Dependant variable
y = np.array(df['is_iceberg'])
X[0][0].shape
# Extraxt features fomr the images(only HH)
X_extracted = []
for x in X:
    X_extracted.append(shape_features(img_HH=x[0],img_HV=x[1],glob_img_HH=x[2],glob_img_HV=x[3]))
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

model = VGG16(weights='imagenet', include_top=False)

vgg16_ip = []

for x in X:
    img = np.dstack((x[2],x[3],img_sub(x[2],x[3])))
    img = transform.resize(img, (244,244),mode='reflect')
    img = preprocess_input(img)
    vgg16_ip.append(img)
vgg16_ip = np.array(vgg16_ip)
pred = model.predict(vgg16_ip)

vgg16_features = []
for p in pred:
    vgg16_features.append(p.flatten())
vgg16_features = np.array(vgg16_features)

del model,vgg16_ip,p,pred
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

model = VGG19(weights='imagenet', include_top=False)

vgg19_ip = []

for x in X:
    img = np.dstack((x[2],x[3],img_sub(x[2],x[3])))
    img = transform.resize(img, (244,244),mode='reflect')
    img = preprocess_input(img)
    vgg19_ip.append(img)
vgg19_ip = np.array(vgg19_ip)
pred = model.predict(vgg19_ip)

vgg19_features = []
for p in pred:
    vgg19_features.append(p.flatten())
vgg19_features = np.array(vgg19_features)

del model,vgg19_ip,p,pred
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

model = ResNet50(weights='imagenet', include_top=False)

resnet_ip = []

for x in X:
    img = np.dstack((x[2],x[3],img_sub(x[2],x[3])))
    img = transform.resize(img, (244,244),mode='reflect')
    img = preprocess_input(img)
    resnet_ip.append(img)
resnet_ip = np.array(resnet_ip)
pred = model.predict(resnet_ip)

resnet_features = []
for p in pred:
    resnet_features.append(p.flatten())
resnet_features = np.array(resnet_features)

del model,resnet_ip,p,pred
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

model = InceptionV3(weights='imagenet', include_top=False)

incept_ip = []

for x in X:
    img = np.dstack((x[2],x[3],img_sub(x[2],x[3])))
    img = transform.resize(img, (244,244),mode='reflect')
    img = preprocess_input(img)
    incept_ip.append(img)
incept_ip = np.array(incept_ip)
pred = model.predict(incept_ip)

incept_features = []
for p in pred:
    incept_features.append(p.flatten())
incept_features = np.array(incept_features)

del model,incept_ip,p,pred
# PCA Analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
pca = PCA(whiten=True)
sc = StandardScaler()
ip = sc.fit_transform(vgg16_features)                        
pca.fit(ip)

fig = plt.figure(1,figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('n_components')
ax.set_ylabel('explained variance')
ax.set_ylim(0,10)
ax.set_xlim(0,100)
ax.plot(pca.explained_variance_)
print("Evaluated components: ", pca.n_components_ )
print("Explained variances: ", pca.explained_variance_)
print(pd.Series(pca.explained_variance_ >= 2).value_counts())
pca = PCA(whiten=True)
sc = StandardScaler()
ip = sc.fit_transform(vgg19_features)                        
pca.fit(ip)

fig = plt.figure(1,figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('n_components')
ax.set_ylabel('explained variance')
ax.set_ylim(0,10)
ax.set_xlim(0,100)
ax.plot(pca.explained_variance_)
print("Evaluated components: ", pca.n_components_ )
print("Explained variances: ", pca.explained_variance_)
print(pd.Series(pca.explained_variance_ >= 2).value_counts())
pca = PCA(whiten=True)
sc = StandardScaler()
ip = sc.fit_transform(resnet_features)                        
pca.fit(ip)

fig = plt.figure(1,figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('n_components')
ax.set_ylabel('explained variance')
ax.set_ylim(0,10)
ax.set_xlim(0,100)
ax.plot(pca.explained_variance_)
print("Evaluated components: ", pca.n_components_ )
print("Explained variances: ", pca.explained_variance_)
print(pd.Series(pca.explained_variance_ >= 3).value_counts())
pca = PCA(whiten=True)
sc = StandardScaler()
ip = sc.fit_transform(incept_features)                        
pca.fit(ip)

fig = plt.figure(1,figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('n_components')
ax.set_ylabel('explained variance')
ax.set_ylim(0,200)
ax.set_xlim(0,100)
ax.plot(pca.explained_variance_)
print("Evaluated components: ", pca.n_components_ )
print("Explained variances: ", pca.explained_variance_)
print(pd.Series(pca.explained_variance_ >= 30).value_counts())
# Save top 50 PCA components
np.savetxt("vgg16_features.csv", Pipeline([('scaling', StandardScaler()), 
                                          ('pca', PCA(n_components=40,whiten=True))]).fit_transform(vgg16_features),
           delimiter=",")

np.savetxt("vgg19_features.csv", Pipeline([('scaling', StandardScaler()), 
                                          ('pca', PCA(n_components=40,whiten=True))]).fit_transform(vgg19_features),
           delimiter=",")

np.savetxt("resnet_features.csv", Pipeline([('scaling', StandardScaler()), 
                                          ('pca', PCA(n_components=40,whiten=True))]).fit_transform(resnet_features),
           delimiter=",")

np.savetxt("incept_features.csv", Pipeline([('scaling', StandardScaler()),
                                            ('pca', PCA(n_components=40,whiten=True))]).fit_transform(incept_features),
           delimiter=",")