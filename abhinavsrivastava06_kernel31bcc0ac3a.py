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

import hashlib
from hashlib import md5
import os
import random
import shutil as sh

import matplotlib.pyplot as plt

import fastai
from fastai.vision import *
df1 = pd.read_csv("/kaggle/input/challengedata/Trainingcopy.csv")

df1.head()
df1['path1'] = 'Image-'

df1['jpg'] = '.jpg'

df1['Image'] =  df1['path1'] + df1['file'].astype(str) + df1['jpg']
df1 = df1.drop(['file','path1','jpg'], axis = 1)

df1 = df1[['Image','label']]

label_count = df1.groupby('label')['Image'].nunique()

print("Class label and Image count\n",label_count)
###### Change of working directory
os.chdir("/kaggle/input/asdfgh/Training_Images")
os.getcwd()
file_list = os.listdir()
print("no of files: ", len(file_list))
####### Identify duplicate images ##########

#original_fname = []

#duplicates = []

duplicate_fname = []

hash_keys =dict()

from tqdm import tqdm


for index, filename in tqdm(enumerate(os.listdir(".")), desc = 'Files' , total = len(file_list), position= 0):
#    original_fname.append((index,filename))
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hash_keys:
            hash_keys[filehash] = index 
        else:
            duplicate_fname.append(filename)
            #duplicates.append((index,hash_keys[filehash]))  #uncomment to know the duplicate image index and original image index
#############################################
print("Number of duplicate images:",len(duplicate_fname))
from distutils.dir_util import copy_tree
fromDirectory = "/kaggle/input/asdfgh/Training_Images"
toDirectory = "/kaggle/working/Images/Training_Images"
copy_tree(fromDirectory,toDirectory)
print("Files copied: ",len(os.listdir("/kaggle/working/Images/Training_Images")))

####### Remove Duplcaites image files from the folder#######

source_path = "/kaggle/working/Images/Training_Images/"

j=0

while j < len(duplicate_fname):
    os.remove(source_path + duplicate_fname[j])
    j = j +1 

print("Non-Duplciate files remaining for training: ",len(os.listdir(source_path)))
############################################################

###### Change of working directory #######
os.chdir("/kaggle/working/Images/Training_Images")

########### Remove duplicate entries from label dataframe ########

new_df = df1[~df1.Image.isin(duplicate_fname)]

print("Non-Duplciate entries in the label set file: ",len(new_df))
############################################################

########### New DIRs for the class label vise data #########

os.mkdir('/kaggle/working/Images/Training_Set')
os.mkdir('/kaggle/working/Images/Training_Set/class0')
os.mkdir('/kaggle/working/Images/Training_Set/class1')
os.mkdir('/kaggle/working/Images/Training_Set/class2')
os.mkdir('/kaggle/working/Images/Training_Set/class3')
os.mkdir('/kaggle/working/Images/Training_Set/class4')
######### Extract image file names for each label ########

image_class0_df = new_df[new_df.label ==0]
image_class1_df = new_df[new_df.label ==1]
image_class2_df = new_df[new_df.label ==2]
image_class3_df = new_df[new_df.label ==3]
image_class4_df = new_df[new_df.label ==4]


image_class0_list = image_class0_df['Image'].to_list()
image_class1_list = image_class1_df['Image'].to_list()
image_class2_list = image_class2_df['Image'].to_list()
image_class3_list = image_class3_df['Image'].to_list()
image_class4_list = image_class4_df['Image'].to_list()

############################################################

########### Splitting the data for training and validation sets ###########
def split_image_files(source_dir, image_list, training_dir):
    
    training_set_length = int(len(image_list))
    random.sample(image_list,len(image_list))
    training_set = image_list[0:training_set_length]
    

    for file_name in training_set:
        temp_training_set = source_dir + file_name
        final_training_set = training_dir + file_name
        sh.copyfile(temp_training_set,final_training_set)


source_dir = '/kaggle/working/Images/Training_Images/'

class0_training_dir = '/kaggle/working/Images/Training_Set/class0/'
class1_training_dir = '/kaggle/working/Images/Training_Set/class1/'
class2_training_dir = '/kaggle/working/Images/Training_Set/class2/'
class3_training_dir = '/kaggle/working/Images/Training_Set/class3/'
class4_training_dir = '/kaggle/working/Images/Training_Set/class4/'


split_image_files(source_dir,image_class0_list,class0_training_dir)
split_image_files(source_dir,image_class1_list,class1_training_dir)
split_image_files(source_dir,image_class2_list,class2_training_dir)
split_image_files(source_dir,image_class3_list,class3_training_dir)
split_image_files(source_dir,image_class4_list,class4_training_dir)

###############################################################################
###### Checking if the images are distributed ########
for i in range(0,5):
    print("Number of training images for class",i," :",len(os.listdir('/kaggle/working/Images/Training_Set/class'+str(i)+'/')))
#     print("Number of validation images for class",i," :",len(os.listdir('/kaggle/working/Images/Validation_Set/class'+str(i)+'/')))
path = "/kaggle/working/Images/Training_Set/"
##### Generating the data for model ######
np.random.seed(42)

data2 = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2,
                                  ds_tfms=get_transforms(), size= 224, num_workers= 6).normalize()
print("Class labels:",data2.classes)
data2.c
len(data2.train_ds), len(data2.valid_ds)
print("Sample images in a batch:")
data2.show_batch(rows=4, figsize=(7, 8))
model_resnet18 = cnn_learner(data2, models.resnet18, metrics = [error_rate,accuracy], pretrained = True)
model_resnet18.summary()
model_resnet18.fit_one_cycle(10)
cm1 = ClassificationInterpretation.from_learner(model_resnet18)
cm1.plot_confusion_matrix()
plt.title("Confusion Matrix: Model_resnet18")
cm1.most_confused()
model_resnet18.unfreeze()
model_resnet18.lr_find()
from fastai.widgets import *
classlist = ['class0', 'class1', 'class2','class3','class4']
ClassConfusion(cm1, classlist, is_ordered=True)
ds, idxs = DatasetFormatter().from_toplosses(model_resnet18)
ImageCleaner(ds, idxs, path)
df = pd.read_csv(path+'cleaned.csv', header='infer')
df
####### Retraining the model with fixed learning rate interval #######
model_resnet18.unfreeze()
model_resnet18.fit_one_cycle(10, max_lr=slice(1e-5,1e-3))
model_resnet18.lr_find()
model_resnet18.recorder.plot()
plt.title("Loss Vs Learning Rate: Model_resnet18, after fixed learning rate")
###### Model with resnet18 arch but no pretrained weights #######
model_resnet18_np = cnn_learner(data2, models.resnet18, metrics = [error_rate,accuracy], pretrained = False)
model_resnet18_np.fit_one_cycle(10)
model_resnet18_np.fit_one_cycle(10)
model_resnet18_np.lr_find()
model_resnet18_np.recorder.plot()
plt.title("Loss Vs Learning Rate: Model_resnet18_np, after fixed learning rate")
model_resnet34_np = cnn_learner(data2, models.resnet34, metrics = [error_rate,accuracy], pretrained = False)
model_resnet34_np.fit_one_cycle(20)
########## Now with resnet34 model ######
model_resnet34 = cnn_learner(data2, models.resnet34, metrics = [error_rate,accuracy], pretrained = True)
model_resnet34.fit_one_cycle(15)
cm2 = ClassificationInterpretation.from_learner(model_resnet34)
cm2.plot_confusion_matrix()
plt.title("Confusion Matrix: Model_resnet34")
model_resnet34.unfreeze()
model_resnet34.lr_find()
model_resnet34.recorder.plot()
plt.title("Loss Vs Learning Rate: Model_resnet34")
######### Retraining the learner Model_resnet34 with fixed learning rate interval #######
model_resnet34.fit_one_cycle(15, max_lr=slice(1e-6,1e-3))
model_resnet50 = cnn_learner(data2, models.resnet50, metrics = [error_rate,accuracy], pretrained = True)
model_resnet50.fit_one_cycle(15)
cm3 = ClassificationInterpretation.from_learner(model_resnet50)
cm3.plot_confusion_matrix()
plt.title("Confusion Matrix: Model_resnet50")
model_resnet50.unfreeze()
model_resnet50.lr_find()
model_resnet50.recorder.plot()
plt.title("Loss Vs Learning Rate: Model_resnet50")
######### Retraining the learner Model_resnet50 with fixed learning rate interval #######
model_resnet50.fit_one_cycle(15, max_lr=slice(1e-6,1e-4))
model_resnet50.unfreeze()
model_resnet50.lr_find()
model_resnet50.recorder.plot()
plt.title("Loss Vs Learning Rate: Model_resnet50")
###### Model with VGG16 arch but no pretrained weights #######
model_vgg16_bn = cnn_learner(data2, models.vgg16_bn, metrics = [error_rate,accuracy], pretrained = False)
model_vgg16_bn.fit_one_cycle(25)
###### Model with VGG19 arch but no pretrained weights #######
model_vgg19_bn = cnn_learner(data2, models.vgg19_bn, metrics = [error_rate,accuracy], pretrained = True)
model_vgg19_bn.fit_one_cycle(15)