# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import shutil
os.mknod("dummyfile.txt")
DIR = '../input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA'

len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
DIR = '../input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL'

len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
DIR = '../input/chest-xray-pneumonia/chest_xray/chest_xray/val/PNEUMONIA'

len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
DIR = '../input/chest-xray-pneumonia/chest_xray/chest_xray/val/NORMAL'

len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
DIR = '../input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA'

len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
DIR = '../input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL'

len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
original_dataset_dir = '../input/chest-xray-pneumonia/chest_xray/chest_xray'



base_dir ='./chest_xray_postprocessed'

os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)

val_dir = os.path.join(base_dir, 'val')

os.mkdir(val_dir)

test_dir = os.path.join(base_dir, 'test')

os.mkdir(test_dir)



train_pneumonia_dir = os.path.join(train_dir, 'pneumonia')

os.mkdir(train_pneumonia_dir)



train_normal_dir = os.path.join(train_dir, 'normal')

os.mkdir(train_normal_dir)



val_pneumonia_dir = os.path.join(val_dir, 'pneumonia')

os.mkdir(val_pneumonia_dir)



val_normal_dir = os.path.join(val_dir, 'normal')

os.mkdir(val_normal_dir)



test_pneumonia_dir = os.path.join(test_dir, 'pneumonia')

os.mkdir(test_pneumonia_dir)



test_normal_dir = os.path.join(test_dir, 'normal')

os.mkdir(test_normal_dir)
os.listdir(train_dir)
import glob



paths_pneumonia = glob.glob('../input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/*')

paths_normal = glob.glob('../input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/*')
print(len(paths_pneumonia))

print(len(paths_normal))
paths_pneumonia[0]
fname_pneumonia = [x.split('/')[-1] for x in paths_pneumonia]

fname_pneumonia[:5]
fname_normal = [x.split('/')[-1] for x in paths_normal]

fname_normal[:5]
def train_val_data_split(pneumonia_list, normal_list, validation_split=0.1):

    

    n_pneumonia = len(pneumonia_list)

    n_normal = len(normal_list)

    

    shuffle_idx_pneumonia = [x for x in range(n_pneumonia)]

    shuffle_idx_normal = [x for x in range(n_normal)]

    

    num_train_pneumonia = int((1-validation_split)*n_pneumonia)

    num_val_pneumonia = n_pneumonia - num_train_pneumonia

    

    num_train_normal = int((1-validation_split)*n_normal)

    num_val_normal = n_normal - num_train_normal

    

    for i in range(num_train_pneumonia):

        src = os.path.join('../input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/', pneumonia_list[i])

        dst = os.path.join(train_pneumonia_dir, pneumonia_list[i])

        shutil.copyfile(src,dst)

        

    for i in range(num_train_pneumonia,n_pneumonia):

        src = os.path.join('../input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/', pneumonia_list[i])

        dst = os.path.join(val_pneumonia_dir, pneumonia_list[i])

        shutil.copyfile(src,dst)

        

    for i in range(num_train_normal):

        src = os.path.join('../input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/', normal_list[i])

        dst = os.path.join(train_normal_dir, normal_list[i])

        shutil.copyfile(src,dst)

        

    for i in range(num_train_normal,n_normal):

        src = os.path.join('../input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/', normal_list[i])

        dst = os.path.join(val_normal_dir, normal_list[i])

        shutil.copyfile(src,dst)

        

        

        
train_val_data_split(fname_pneumonia, fname_normal, validation_split=0.1)
def test_data_copy():

    

    test_data_pneumonia = glob.glob('../input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/*')

    test_data_normal = glob.glob('../input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/*')

    

    test_fname_pneumonia = [x.split('/')[-1] for x in test_data_pneumonia]

    test_fname_normal = [x.split('/')[-1] for x in test_data_normal]

    

    n_pneumonia = len(test_fname_pneumonia)

    n_normal = len(test_fname_normal)

    

    for i in range(n_pneumonia):

        src = os.path.join('../input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/', test_fname_pneumonia[i])

        dst = os.path.join(test_pneumonia_dir, test_fname_pneumonia[i])

        shutil.copyfile(src,dst)

        

    for i in range(n_normal):

        src = os.path.join('../input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/', test_fname_normal[i])

        dst = os.path.join(test_normal_dir, test_fname_normal[i])

        shutil.copyfile(src,dst)
test_data_copy()
import zipfile



def zipdir(path, ziph):

    # ziph is zipfile handle

    for root, dirs, files in os.walk(path):

        for file in files:

            ziph.write(os.path.join(root, file))
zipf = zipfile.ZipFile('chest_xray_postprocessed.zip', 'w', zipfile.ZIP_DEFLATED)

zipdir('./', zipf)

zipf.close()
!ls -lah ./chest_xray_postprocessed.zip
shutil.rmtree('./chest_xray_postprocessed')