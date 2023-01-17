# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plotting
import seaborn as sns # statistical data visualization
import plotly.express as px
import plotly.graph_objects as go # interactive plots

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train_df.head()
unique_patient_df = train_df.drop(['Weeks', 'FVC', 'Percent'], axis=1).drop_duplicates().reset_index(drop=True)
unique_patient_df['# visits'] = [train_df['Patient'].value_counts().loc[pid] for pid in unique_patient_df['Patient']]

print('Number of data points: ' + str(len(unique_patient_df)))
print('----------------------')

for col in unique_patient_df.columns:
    print('{} : {} unique values, {} missing.'.format(col, 
                                                          str(len(unique_patient_df[col].unique())), 
                                                          str(unique_patient_df[col].isna().sum())))
unique_patient_df.head()
train_df['Expected FVC'] = train_df['FVC'] + (train_df['Percent']/100)*train_df['FVC']


import statistics

volume_path = []
for patient in unique_patient_df['Patient']:
    visits = train_df.loc[train_df['Patient'] == patient]   
    volumes = visits['FVC'].to_numpy()
    descent = 100*(statistics.mean(volumes[-3:])/volumes[0])
    volume_path.append([patient,descent])
    
sc = pd.DataFrame(volume_path,columns=['Patient','Final_FVC'])
    
unique_patient_df['Final_FVC'] = sc['Final_FVC']

unique_patient_df.tail()
import pydicom
from glob import glob
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.filters import threshold_otsu, median
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import clear_border
from scipy.stats import describe
def load_single_slice(path):
    slice = pydicom.dcmread(path) 
    return slice


slice_no = 20
images = []


for patient_id in unique_patient_df['Patient']:
    patient_dir = '../input/osic-pulmonary-fibrosis-progression/train/' + patient_id + '/'
    
    fname = str(slice_no) + ".dcm"
    path = patient_dir + fname
    if (os.path.exists(path)):
        patient_img = load_single_slice(path)
        images.append(patient_img)
        
    

import time

import matplotlib.pyplot as plt
import numpy as np


from sklearn import datasets
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d

from collections import Counter

# #############################################################################
# Learn the dictionary of images

no_clusters = 10

print('Learning the dictionary... ')
rng = np.random.RandomState(0)
kmeans = MiniBatchKMeans(n_clusters=no_clusters, random_state=rng, verbose=False)
patch_size = (10,10)

buffer = []
t0 = time.time()
no_patches = 200

# cycle over the whole dataset 3 times

for _ in range(3):
    index = 0
    for img in images:
        #
        # abandoning images that fail extraction
        #
        try:
            imgf = (img.pixel_array).astype(np.float64)
            data = extract_patches_2d(imgf, patch_size, max_patches=no_patches,random_state=rng)
            data = np.reshape(data, (len(data), -1))
            buffer.append(data)
            index += 1
            if (index % 10 == 0):
                data = np.concatenate(buffer, axis=0)
                data -= np.mean(data, axis=0)
                data /= np.std(data, axis=0)
                kmeans.partial_fit(data)
                buffer = []
                #print("Partial fit of %4i out of %i" % (index, 6 * len(images))) 
                
        except Exception as e: print(e)
           

dt = time.time() - t0
print('done in %.2fs.' % dt)

labels = kmeans.labels_
cluster_count = Counter(kmeans.labels_)

print(cluster_count)

# #############################################################################
# Plot the results


plt.figure(figsize=(4.2, 4))
for i, patch in enumerate(kmeans.cluster_centers_):
    plt.subplot(9, 9, i + 1)
    plt.imshow(patch.reshape(patch_size), cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())


plt.suptitle('Patches of CT slice\nTrain time %.1fs on %d patches' % (dt, 8 * len(images)), fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
patient_images = []
patient_clusters = []

for patient_id in unique_patient_df['Patient']:
    patient_dir = '../input/osic-pulmonary-fibrosis-progression/train/' + patient_id + '/'
    
    fname = str(slice_no) + ".dcm"
    path = patient_dir + fname
    if (os.path.exists(path)):
        patient_img = load_single_slice(path)
        patient_images.append([patient_id,patient_img])
           
            
            
for (patient_id,img) in patient_images:
    cluster_list = []
    row=[patient_id]
    #
    try:
        imgf = (img.pixel_array).astype(np.float64)
        data = extract_patches_2d(imgf, patch_size, max_patches=300,random_state=rng)
        data = np.reshape(data, (len(data), -1))
        closest = list(kmeans.predict(data)) 
                
    except Exception as e: print(e) 
     
    cls = [0]*no_clusters
    for i in range(no_clusters):
        cls[i] = closest.count(i)       
    row = row + cls    
    patient_clusters.append(row)
    
hd = ['Patient'] + [str(x) for x in range(no_clusters)]

cluster_df = pd.DataFrame(patient_clusters,columns=hd)

cluster_df.head()
  
           
cluster_df = pd.merge(cluster_df,unique_patient_df,on='Patient')

cluster_df = cluster_df.drop(['Sex','SmokingStatus','# visits'],axis=1)

cluster_df.head()




X = cluster_df.drop(['Patient','Final_FVC'],axis=1)
y = cluster_df['Final_FVC']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(df)