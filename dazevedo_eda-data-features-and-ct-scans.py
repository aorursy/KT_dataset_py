# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from collections import Counter
import pydicom
import os
from skimage import morphology
from skimage import measure
from skimage.filters import threshold_otsu, median
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import clear_border
import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        if 'ID00419637202311204720264' in dirname:
#            print(os.path.join(dirname, filename))
            

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')

train_df.head()
test_df.head()
#Patient

patients_dist = Counter(train_df['Patient'].values).most_common()

plt.figure(figsize=(10, 6))
plt.bar(range(len(patients_dist)), list(dict(patients_dist).values()))

plt.xlabel('Ids Patient')
plt.ylabel('Observations per Patient')
plt.show()

info_df = pd.DataFrame(columns=['Patient','Age','Sex', 'SmokingStatus'])

for ind, row in train_df.groupby('Patient'):
    new_row = {'Patient': ind, 'Age': row.iloc[0]['Age'], 'Sex': row.iloc[0]['Sex'], 'SmokingStatus': row.iloc[0]['SmokingStatus']}
    info_df.loc[len(info_df)] = new_row
    
    
info_df
#Age

plt.figure(figsize = (10, 7))
plt.hist(info_df['Age'], 10)
plt.xlabel('Ages')
plt.ylabel('Frequency')
plt.show()
#SmokingStatus

plt.figure(figsize = (10, 7))
info_df['SmokingStatus'].value_counts().plot(kind='bar');
plt.show()
#Gender

plt.figure(figsize = (10, 7))
info_df['Sex'].value_counts().plot(kind='bar');
plt.show()
train_df

plt.figure(figsize=(12, 8))

for ind, patient_data in train_df.groupby('Patient'):
    plt.plot(patient_data['Weeks'], patient_data['FVC'], '.', label=ind)

plt.xlabel('Weeks')
plt.ylabel('FVC')
plt.show()

plt.figure(figsize=(12, 8))

c = ['blue', 'red', 'green']
i = 0

for ind_smoke, status_data in train_df.groupby('SmokingStatus'):
    counter = 0
    print(ind_smoke,':', c[i], 'lines')
    
    for ind, patient_data in status_data.groupby('Patient'):
        plt.plot(range(len(patient_data)), patient_data['FVC'], 'o-', label=ind_smoke, color=c[i])

        counter += 1

        if counter == 10:
            break
    i += 1

plt.xlabel('Time')
plt.ylabel('FVC')
#plt.legend()
plt.show()
c = ['blue', 'red', 'green']
i = 0

first_FVC = {}
for ind_smoke, status_data in train_df.groupby('SmokingStatus'):
    
    first_FVC[ind_smoke] = []
    for ind, patient_data in status_data.groupby('Patient'):
        first_FVC[ind_smoke].append(patient_data.iloc[0]['FVC'])

plt.figure(figsize=(12, 8))

plt.hist(first_FVC['Ex-smoker'], bins = 15, label='Ex-smoker')
plt.hist(first_FVC['Never smoked'], bins = 15, label='Never smoked')
plt.hist(first_FVC['Currently smokes'], bins = 5, label='Currently smokes')


plt.legend()
plt.xlabel('FVC')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(12, 8))

for ind, patient_data in train_df.groupby('Patient'):
    plt.plot(patient_data['FVC'], patient_data['Percent'], '.', label=ind)

plt.xlabel('FVC')
plt.ylabel('Percent')
plt.show()

ct_patient = 'ID00007637202177411956430'
#ct_patient = 'ID00015637202177877247924'

img_paths = []
for dirname, _, filenames in os.walk('/kaggle/input/osic-pulmonary-fibrosis-progression/train'):
    if ct_patient in dirname:
        filenames.sort(key = lambda x: int(x.split('.')[0]))
        for filename in filenames:
            img_paths.append(os.path.join(dirname, filename))

#img_paths
#First image

ds = pydicom.dcmread(img_paths[0])

plt.figure(figsize = (8,8))
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
#Sequence of images
from IPython.display import clear_output


for i in img_paths:
    plt.figure(figsize = (8,8))
    ds = pydicom.dcmread(i)
    
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.pause(0.1)
    clear_output(wait=True)


#from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

patient = load_scan('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')
imgs = get_pixels_hu(patient)
index_example = 20

plt.imshow(imgs[index_example], cmap='gray')
plt.title('HU Image')
plt.show()

ds = pydicom.dcmread(img_paths[index_example])
plt.imshow(ds.pixel_array, cmap='gray')
plt.title('Original Image')
plt.show()

new_img = imgs[index_example] < -500

plt.imshow(new_img, cmap='gray')
plt.title('Applying HU threshold')
plt.show()


lungs = median(clear_border(new_img))
lungs = morphology.binary_closing(lungs, selem=morphology.disk(7))
mask = binary_fill_holes(lungs)


def compute_lung_volume(ds, mask):
    return np.sum(mask) * float(ds.SliceThickness) * ds.PixelSpacing[0] * ds.PixelSpacing[1]


print("Lung Volume: ", compute_lung_volume(ds, mask))

plt.imshow(mask, cmap='gray')
plt.title('Applying HU threshold and Masking the outside region')
plt.show()

#oversampling, assuming that FVC follows a Linear Regression according to the Week value

for i, d in train_df.groupby('Patient'):
    
    week_data = d['Weeks']
    
    
    xx = [c for c in range(min(week_data), max(week_data)) if c not in week_data]
    
    reg = LinearRegression().fit(week_data.values.reshape(-1,1), d['FVC'].values)
    res = reg.predict(np.array(xx).reshape(-1,1))
    
    for j in range(len(xx)):
        train_df.loc[len(train_df)] = [d.iloc[0]['Patient'], xx[j], res[j], d.iloc[0]['Percent'], d.iloc[0]['Age'], d.iloc[0]['Sex'], d.iloc[0]['SmokingStatus']]

#Transforming Training Set
X_train = pd.concat([train_df, test_df], axis = 0, ignore_index = True)

y_train = X_train['FVC']
X_train = X_train.drop(['FVC', 'Percent'], axis = 1)

X_train.reset_index(inplace=True, drop=True)


#Transforming Test Set
X_test = pd.DataFrame(columns = ['Patient', 'Weeks', 'Age', 'Sex', 'SmokingStatus'])

for ind, row in test_df.iterrows():
    
    for i in range(-12, 133+1):
        new_row = [row.Patient, i, row.Age, row.Sex, row.SmokingStatus]
        X_test.loc[len(X_test)] = new_row

categorical_vars = ['Patient', 'Sex', 'SmokingStatus']
int_vars = ['Weeks', 'Age']


#Categorize features

encoder = OneHotEncoder(categories = 'auto', handle_unknown = 'ignore', sparse = False)
encoder.fit(X_train[categorical_vars])

X_train = pd.concat([X_train.drop(categorical_vars, axis = 1), pd.DataFrame(encoder.transform(X_train[categorical_vars]), 
                                           columns = encoder.get_feature_names())], axis=1, sort=False)
X_test = pd.concat([X_test.drop(categorical_vars, axis = 1), pd.DataFrame(encoder.transform(X_test[categorical_vars]),
                                         columns = encoder.get_feature_names())], axis=1, sort=False)


X_test.Weeks = pd.to_numeric(X_test.Weeks)
X_test.Age = pd.to_numeric(X_test.Age)



#Normalization

sc = StandardScaler()
sc.fit(X_train[int_vars])
X_train[int_vars] = sc.transform(X_train[int_vars])
X_test[int_vars] = sc.transform(X_test[int_vars])

display(X_train.head())
display(X_test.head())
#Train the model

parameters = {
    'max_depth': [5, 10],
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.005]
}

#Cross Validation
grid_search = GridSearchCV(
    estimator=xgboost.XGBRegressor(random_state = i),
    param_grid=parameters,
    n_jobs = 5,
    cv = 5
)

grid_search.fit(X_train, y_train)
print("Best Parms", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
best_model = grid_search.best_estimator_    
