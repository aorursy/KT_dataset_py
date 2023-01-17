# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        '''
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from keras.models import load_model
data = pd.read_csv('/kaggle/input/data/Data_Entry_2017.csv')
data.loc[data['Patient ID']==8270]
data.head(10)
data['Finding Labels'].values
def counting_labels(column, df):
    count_list= []
    for label in df['Finding Labels'].values:
        if column in label:
            count_list.append(1)
        else:
            count_list.append(0)
    return count_list
        
real_df= pd.DataFrame({'image': data['Image Index'].values,
                  'Cardiomegaly' : counting_labels('Cardiomegaly', data),
                  'Emphysema' : counting_labels('Emphysema', data), 
                  'Effusion' : counting_labels('Effusion', data), 
                  'Hernia' : counting_labels('Hernia', data), 
                  'Infiltration' : counting_labels('Infiltration', data), 
                  'Mass' : counting_labels('Mass', data), 
                  'Nodule' : counting_labels('Nodule', data), 
                  'Atelectasis' : counting_labels('Atelectasis', data),
                  'Pneumothorax' : counting_labels('Pneumothorax', data),
                  'Pleural_Thickening' : counting_labels('Pleural_Thickening', data), 
                  'Pneumonia' : counting_labels('Pneumonia', data), 
                  'Fibrosis' : counting_labels('Fibrosis', data), 
                  'Edema' : counting_labels('Edema', data), 
                  'Consolidation': counting_labels('Consolidation', data),
                       'Patient_Id':  data['Patient ID']
                  
})
real_df.loc[real_df['Patient_Id']== 29855]
real_df.to_csv('/kaggle/working/train_test.csv', index = False)
real_df
csv_dir = '/kaggle/working/'
df = pd.read_csv(csv_dir + 'train_test.csv')
df
percent = 0.7
train_percent = int(len(df)*percent)
train_percent
sample_data = df.head(10000)
sample_data
percent = 0.7
percent_2 = 1 - percent
train_percent = int(len(sample_data)*percent)
test_percent = int(len(sample_data)*percent_2)
train_percent
test_percent
train_data = sample_data.head(train_percent)
val_data = sample_data.tail(test_percent)
train_data
val_data
train_data.to_csv(csv_dir + 'train.csv')
val_data.to_csv(csv_dir + 'test.csv')
def check_for_leakage(df1, df2, patient_col):
    '''
    Return True if there are same patients in both df1 and df2
    '''
    
    df1_patients_unique = set(df1[patient_col].values)
    print(df1_patients_unique)
    df2_patients_unique = set(df2[patient_col].values)
    print(df2_patients_unique)
    
    patients_in_both_groups = list(df1_patients_unique.intersection(df2_patients_unique))
    
    leakage = len(patients_in_both_groups) > 0
    
    return leakage
print('Leakage between traing and valid: {}'.format(check_for_leakage(train_data, val_data, 'Patient_Id')))
def remove_leakage(df1, df2, patient_col):
    '''
    Return True if there are same patients in both df1 and df2
    '''
    
    df1_patients_unique = set(df1[patient_col].values)
    #print(df1_patients_unique)
    df2_patients_unique = set(df2[patient_col].values)
    #print(df2_patients_unique)
    
    patients_in_both_groups = list(df1_patients_unique.intersection(df2_patients_unique))
    
    #leakage = len(patients_in_both_groups) > 0
    for patient in patients_in_both_groups:
        df1= df1[df1[patient_col] != patient]
        df2= df2[df2[patient_col] != patient]
        
    return df1, df2
train_data, val_data = remove_leakage(train_data, val_data ,'Patient_Id')
print('Leakage between traing and valid: {}'.format(check_for_leakage(train_data, val_data, 'Patient_Id')))
train_data
val_data
labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']
def get_train_generator(df, image_dir, x_col, y_col, shuffle=True, batch_size=32, seed=1, target_w=320, target_h=320):
    image_generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization= True)
    
    generator = image_generator.flow_from_dataframe(
    dataframe = df,
    directory = image_dir,
    x_col = x_col,
    y_col = y_col,
    class_mode='raw',
    batch_size = batch_size,
    shuffle = shuffle,
    seed = seed,
    targe_size = (target_w, target_h))
    
    return generator
def get_test_and_valid_generator(valid_df, train_df, image_dir, x_col, y_col, sample_size = 100, batch_size = 32, seed = 1, target_w = 320, target_h = 320):
    
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df,
        directory = image_dir,
        x_col= 'image',
        y_col = labels, 
        class_mode='raw',
        batch_size = sample_size,
        shuffle = True,
        target_size = (target_w, target_h)
    )
    
    batch = raw_train_generator.next()
    data_sample = batch[0]
    
    image_generator = ImageDataGenerator(
        featurewise_center = True,
        featurewise_std_normalization = True
    )
    
    image_generator.fit(data_sample)
    
    valid_generator = image_generator.flow_from_dataframe(
        dataframe = val_data,
        directory = image_dir,
        x_col = x_col,
        y_col = y_col,
        class_mode = 'raw',
        batch_size = batch_size,
        shuffle = False,
        seed = seed,
        target_size = (target_w, target_h)
        
    )
    
    return valid_generator
IMAGE_DIR = '../input/data/images'
import os
import shutil
#os.makedirs('/kaggle/working/images/')
list1 = os.listdir('../input/data/images_001/images')
current = IMAGE_DIR + '_00' + str(1) + '/images'
list12 = os.listdir(current)
list12
for i in train_data:
    print(i)
list1 = train_data.image.values
for i in range(10):
    print(list1[i])
len(list1)
current = IMAGE_DIR + '_00' + str(1) + '/images/' + list1[0]
print(current)
## The Cell to move images to a new folder
destination = '/kaggle/working/images/'
for i in range(2,3):
    current_dir = IMAGE_DIR + '_00' + str(i) + '/images'
    list12 = os.listdir(current)
    for j in list12: ## here the problem was we had to add the second folder checkcheck manually instead of it being automatic, first it was list12
        current_file = current_dir + '/' + j
        shutil.copy(current_file, destination)
files = os.listdir('/kaggle/working/images')
len(files)
print(IMAGE_DIR)

flag = False
checkcheck= os.listdir(IMAGE_DIR + '_002/images/')
if '00001075_003.png' in checkcheck:
    flag = True
    

checkcheck
IMAGE_DIR = "/kaggle/working/images"
train_generator = get_train_generator(train_data, IMAGE_DIR, "image", labels)
valid_generator= get_test_and_valid_generator(val_data, train_data, IMAGE_DIR, "image", labels)
x, y = train_generator.__getitem__(0)
plt.imshow(x[0]);
## The images have been successfully transfered for the training into the working directory. They have been preprocess using ImageDataGenerator and ready to be fed into 
## the model after class imbalance and Weighted Loss Solution
