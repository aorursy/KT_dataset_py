# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Input, Conv2D, MaxPooling2D,GlobalMaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Activation, MaxPool2D, AvgPool2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications import DenseNet121, VGG19, ResNet50
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from IPython.display import display, Image
import matplotlib.pyplot as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.utils import shuffle
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')
valid_df = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_dataset_Summary.csv')

print('The training dataset has rows : ', format(train_df.shape[0]))
print('The training dataset has cols : ', format(train_df.shape[1]))
missing_vals = train_df.isnull().sum()
train_df.dropna(how = 'all')
train_df.isnull().sum()
train_data = train_df[train_df['Dataset_type'] == 'TRAIN']
test_data = train_df[train_df['Dataset_type'] == 'TEST']
assert train_data.shape[0] + test_data.shape[0] == train_df.shape[0]
print(f"Shape of train data : {train_data.shape}")
print(f"Shape of test data : {test_data.shape}")
train_fill = train_data.fillna('unknown')
test_fill = test_data.fillna('unknown')
test_img_dir = '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'
train_img_dir = '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'
final_train_data = train_data[(train_data['Label'] == 'Normal') | 
                              ((train_data['Label'] == 'Pnemonia') & (train_data['Label_2_Virus_category'] == 'COVID-19'))]


# Create a target attribute where value = positive if 'Pnemonia + COVID-19' or value = negative if 'Normal'
final_train_data['target'] = ['negative' if holder == 'Normal' else 'positive' for holder in final_train_data['Label']]

final_train_data = shuffle(final_train_data, random_state=1)

final_validation_data = final_train_data.iloc[1000:, :]
final_train_data = final_train_data.iloc[:1000, :]

print(f"Final train data shape : {final_train_data.shape}")


def DistanceMatrix_cpu(boundary_x, boundary_y, internal_points_x, internal_points_y):
    dist = []
    dist_x = (boundary_x[:,npy.newaxis] - internal_points_x[npy.newaxis,:])**2

    dist_y = (boundary_x[:,npy.newaxis] - internal_points_y[npy.newaxis,:])**2

    return np.sqrt(dist_x+dist_y)

from numpy import inf
import numpy as npy
from timeit import default_timer as timer
import numpy as np

def gen_circle(img):

        Boundary_Points = 1000
        
        y_in, x_in = npy.where(img != 0)
        
        if y_in.shape[0] == 0:
          return None

        Circ_Bound = np.linspace(0, 2*np.pi, Boundary_Points); 
        candidate_circle = 0
        highest_size = 0
 
        print('found candidate circle')

        vv = npy.size(y_in);
        ww = npy.size(x_in);
        zz = npy.zeros(ww);
        
        #Create circle boundary
        R = img.shape[0] // 2
        x = int(img.shape[0] // 2)
        y = int(img.shape[1] // 2)

        #Find all points within circle by first getting indices of all points in the image
        y_in_circle, x_in_circle = y_in, x_in

        normalized = np.asarray(npy.column_stack([x_in_circle, y_in_circle]))
    
        Circ_Bound_x = R * np.cos(Circ_Bound) + x
        Circ_Bound_y = R * np.sin(Circ_Bound) + y    

        DM_data_Sample = DistanceMatrix_cpu((Circ_Bound_x), (Circ_Bound_y), normalized[:,0], normalized[:,1]);

        IM = 1./DM_data_Sample; 
        Boundary_Values_Sample = np.sum(IM, axis=1);
        Boundary_Values_Normalized = Boundary_Values_Sample

        return Boundary_Values_Normalized
import cv2
from skimage import io 
from skimage.transform import rotate, AffineTransform, warp

dataset = {
    'Normal' : [],
    'Pnemonia' : []
    
}

covidCount = 0
normalCount = 0

subset = {
    'Normal' : [],
    'Pnemonia' : []
    
}
for index, row in final_train_data.iterrows():
    fn = row['X_ray_image_name']
    label = row['Label']
    
    path = train_img_dir + '/' + fn
    
    #read image as grayscale
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    #resize images to 600x600 
    #anything greater than it will cause our algorithm to run too slow and cause the notebook to crash due to memory bandwidtch
    path = train_img_dir + '/' + fn
    width = 600 
    height = 600
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    if img is None:
        continue
    
    subset[label].append(img)

    if label == 'Pnemonia':
        temp_data = [img]
        covidCount += 1

        temp_data = []
        
        r_image = rotate(img, angle=45) # angle value is positive for anticlockwise rotation 
        r_image1 = rotate(img, angle=-45) #angle value is negative for clockwise rotation

        hflipped_image= np.fliplr(img) #fliplr reverse the order of columns of pixels in matrix
        vflipped_image= np.flipud(img)

        temp_data.append(r_image)
        temp_data.append(r_image1)
        temp_data.append(hflipped_image)
        temp_data.append(vflipped_image)
    else:
        normalCount +=1
        if normalCount >= 200:
            continue
            
        temp_data = [img]
        
    for item in temp_data:
        b = gen_circle(item)
        dataset[label].append(b)

    if covidCount >= 200 and normalCount >= 200:
        break
print(c)

outfile = 'full_dataset.npz'
np.savez(outfile, **dataset)
try:
    dataset
except:
    dataset = np.load('../input/dataset-covid/dataset_preprocessed.npz')
import matplotlib.pyplot as plt
import scipy.stats as stats

idx= 16

for obj in dataset:
        item = dataset[obj][idx]
        
        plt.plot(item, label='Label: {}, raw signal'.format(obj))
        plt.legend()

        plt.show()
        plt.plot(item[120:145], label='Label: {}, First peak'.format(obj))
        plt.legend()
        plt.show()
        plt.plot(item[220:245], label='Label: {}, Second peak'.format(obj))
        plt.legend()

        plt.show()

data_modified = {
    'Normal' : [],
    'Pnemonia' : [],
}

for shape in dataset:
    
    t = dataset[shape]
    for item in t:
        data_modified[shape].append(item[125:145])
train_x = []
test_x = []
train_y=[]
test_y=[]

i = 0
for shape in data_modified:
    #select only the first 140 elements to make both labels have same number of objects
    data = data_modified[shape][:140]

    if len(data) == 0:
        continue
        
    print('shape', shape, len(data))

    data = np.asarray(data) 
    data /= np.linalg.norm(data)
    # data = data[:,0]
    #data = np.abs(np.apply_along_axis(np.fft.fft, 1, data))
    random_range = np.arange(data.shape[0])
    np.random.shuffle(random_range)
    train_range = int(random_range.shape[0] *.7)
    
    if i== 0:
        train_y = [i] * data[random_range[:train_range]].shape[0]
        test_y = [i] * data[random_range[train_range:]].shape[0]
        train_x = data[random_range[:train_range]]
        test_x = data[random_range[train_range:]]
    else:
        train_y = np.concatenate((train_y, [i] * data[random_range[:train_range]].shape[0]), axis=0)
        test_y = np.concatenate((test_y, ([i] * data[random_range[train_range:]].shape[0])), axis=0)

        train_x = np.concatenate((train_x, data[random_range[:train_range]]), axis=0)

        test_x = np.concatenate((test_x, data[random_range[train_range:]]), axis=0)
    i+=1
train_x.shape
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(np.nan_to_num(train_x), train_y)
pred_i = neigh.predict(np.nan_to_num(test_x))
neigh.score(np.nan_to_num(test_x), test_y)
from sklearn.metrics import accuracy_score
from sklearn import svm

clf = svm.SVC(decision_function_shape='ovo', probability=True)
clf.fit(np.nan_to_num(train_x), train_y)

predicted = clf.predict(np.nan_to_num(test_x))

# get the accuracy
accuracy_score(test_y, predicted)