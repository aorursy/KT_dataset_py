import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input/"))
label_data = pd.read_csv('../input/train.csv')

label_data=label_data.loc[:30]

label_data.head()
label_names=label_data.iloc[:,0]

label_names.head()
expand_labels=pd.get_dummies(label_data.Id)

expand_labels.head()
expand_label_bool=expand_labels.astype('bool')

frames=[label_names,expand_label_bool]

dataset=pd.concat(frames,axis=1)

dataset.head()
#Create separate folders for each label

#to get names for each folder

folder_names=label_data.Id.unique()

folder_names
import shutil



copy_from='../input/train'

copy_to='../working/temp/'

if os.path.isdir(copy_to):

    shutil.rmtree(copy_to)

    os.makedirs(copy_to)

else:

    os.makedirs(copy_to)

os.listdir(copy_to)
for name in folder_names:

    #create folder

    os.mkdir(copy_to+name)

os.listdir(copy_to)
import shutil



for name in folder_names:

    files=dataset.Image[dataset[name]==True]

    for file in files:

        path_from='../input/train/train/'+file

        path_to='../working/temp/'+name+'/'+file

        shutil.copyfile(path_from, path_to) 
os.listdir('../working/temp/')
os.listdir('../working/temp/new_whale//')