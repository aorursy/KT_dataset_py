# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# loading data

train_data = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')

test_data = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')
train_data.head()
test_data
train_data.info()
train_data.isna().sum()
train_data.columns
train_data = train_data.drop(columns= ['Patient'], axis = 1)

train_data.head()
# for getting unique values from data

for i in list(train_data.select_dtypes(include= np.object).columns):

    print(train_data[i].unique(), '\n')
plt.style.use('seaborn')

train_data.select_dtypes(include= [np.int64, np.float64]).hist(figsize= (10,10))

plt.tight_layout()

plt.show()
sns.pairplot(train_data[list(train_data.columns)[:-1]], hue= 'Sex')

plt.show()
train_cols = list(train_data.columns)

train_cols.remove('Sex')

train_cols
sns.pairplot(train_data[train_cols], hue= 'SmokingStatus')

plt.show()
# get dummies for training data 

train_data_final = pd.get_dummies(train_data, columns=['Sex', 'SmokingStatus'], drop_first= True)

train_data_final.head()
X_data = train_data_final.drop(columns= 'FVC', axis = 1)

y_data = train_data_final['FVC']
X = X_data.iloc[:, ].values

y = y_data.iloc[:, ].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
X_train[:, :3]
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train[:, :3] = sc.fit_transform(X_train[:, :3])

X_test[:, :3] = sc.transform(X_test[:, :3])
X_train
import xgboost as xgb

xgb_model = xgb.XGBRFRegressor()

xgb_model.fit(X_train, y_train)

print('training score: {}'.format(xgb_model.score(X_train, y_train)))

print('testing score: {}'.format(xgb_model.score(X_test, y_test)))
import pydicom
# check how many images in out train and test folders

import os

print('training image folders : {}'.format(len(list(os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train')))))

print('training image folders : {}'.format(len(list(os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/test')))))
# to check individual patient DICOMs

img_dir = '../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362'

print('Patient ID : ID00012637202177665765362 , images found: {}'.format(len(list(os.listdir(img_dir)))))



# visualizations of DICOM



fig = plt.figure(figsize=(12, 12))

columns = 4

rows = 5

for i in range(1, columns*rows +1):

    filename = img_dir + "/" + str(i) + ".dcm"

    ds = pydicom.dcmread(filename)

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap='gray')

plt.tight_layout()    

plt.show()
# to check individual patient DICOMs

img_dir = '../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362'

print('Patient ID : ID00012637202177665765362 , images found: {}'.format(len(list(os.listdir(img_dir)))))



# visualizations of DICOM

# official documentation for cmap colors : https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html





fig = plt.figure(figsize=(12, 12))

columns = 4

rows = 5

for i in range(1, columns*rows +1):

    filename = img_dir + "/" + str(i) + ".dcm"

    ds = pydicom.dcmread(filename)

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap='terrain')

plt.tight_layout()    

plt.show()
# credits : https://www.kaggle.com/piantic/osic-pulmonary-fibrosis-progression-basic-eda



def plot_pixel_array(dataset, figsize=(5,5)):

    plt.figure(figsize=figsize)

    plt.grid(False)

    plt.imshow(dataset.pixel_array, cmap='gray') # cmap=plt.cm.bone)

    plt.show()

    

def show_dcm_info(dataset):

    print("Filename.........:", file_path)



    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print("Patient's name......:", display_name)

    

    print(dataset.data_element("ImageOrientationPatient"))

    print(dataset.data_element("ImagePositionPatient"))

    print(dataset.data_element("PatientID"))

    print(dataset.data_element("PatientName"))

    print(dataset.data_element("PatientSex"))

   

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing....:", dataset.PixelSpacing)    

            

#------------------



i = 1

num_to_plot = 2

for folder_name in os.listdir('../input/osic-pulmonary-fibrosis-progression/train/'):

        patient_path = os.path.join('../input/osic-pulmonary-fibrosis-progression/train/',folder_name)

        

        for i in range(1, num_to_plot+1):     

            file_path = os.path.join(patient_path, str(i) + '.dcm')



            dataset = pydicom.dcmread(file_path)

            show_dcm_info(dataset)

            plot_pixel_array(dataset)



        break

    
# checking each indiviual training images length

img_dir = '../input/osic-pulmonary-fibrosis-progression/train/'

print('training image folders : {}'.format(len(list(os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train')))))



for i in list(os.listdir(img_dir)):

    print('patient ID: {}, length is :{}'.format(i, len(list(os.listdir(img_dir + i)))))


