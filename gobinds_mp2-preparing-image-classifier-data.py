# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
     #   print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from PIL import Image
from pathlib import Path
from matplotlib.patches import Rectangle
from scipy.io import loadmat


#define paths to use later
devkit_path = Path('../input/stanford-cars-dataset/car_devkit/devkit')
train_path = Path('../input/stanford-cars-dataset/cars_train/cars_train')
test_path = Path('../input/stanford-cars-dataset/cars_test/cars_test')

#use loadmat to upload matlab file data
cars_meta = loadmat('../input/stanford-cars-dataset/car_devkit/devkit/cars_meta.mat')
cars_train_annos = loadmat('../input/stanford-cars-dataset/car_devkit/devkit/cars_train_annos.mat')
cars_test_annos = loadmat('../input/stanford-cars-dataset/car_devkit/devkit/cars_test_annos.mat')


#here I will loop through cars_meta to see how many and what types of cars there are in the image files.
cars = []
for car in cars_meta['class_names'][0]:
    cars.append(car)

models = pd.DataFrame(cars, columns=['models']) #create new df with all the models of the cars.
models
#this creates a new dataframe to that will incude all of the images' file path, image box dimensions, and, class.
frame = [[i.flat[0] for i in line] for line in cars_train_annos['annotations'][0]]
columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
df_train = pd.DataFrame(frame, columns=columns)
df_train['class'] = df_train['class']-1 # -1 to account for index starting on zero.
df_train['fname'] = [train_path/f for f in df_train['fname']] #  Appending Path
df_train.head()
# this will merge the model names and filenames from car_train_annos together.
df_train = df_train.merge(models, left_on='class', right_index=True)
df_train = df_train.sort_index() 
df_train.head()
#here I attempt to save all of the images in the file to its brand folder and rename their filenames to the name of the car model.
    
import zipfile
import os.path

zf = zipfile.ZipFile('zipfile_write.zip', mode='w') #create zip file to download easier

#needed to create an exception handler as it would not work without it.
try:
    for i in df_train.index:  #loop through index
        try:
            name = df_train['models'][i] #name of each model
            print(name)
            file_path = df_train['fname'][i] 
            file_name = os.path.basename(df_train['fname'][i]) #file_name is the base filename, so if the path was FOLDER/image.jpg it'll be image.jpg
            make = name.split(" ")[0] #save the make as everything before the first space.
            print(make)
            zf.write(file_path, os.path.join(make, file_name), zipfile.ZIP_DEFLATED ) #write file to zip while also placing them in make folder
        except Exception as exc:
            print(str(exc))
            pass
finally:
    print('closing')
    zf.close()
    
    
