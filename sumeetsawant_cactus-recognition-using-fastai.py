import pandas as pd 

import numpy as np 

import cv2

from zipfile import ZipFile

from fastai import * 

from fastai.vision import * 
# Unzipping foldee



def unzip_folder(path=None,folder_name='train',extract_to=None):

    """

    Input: path(str):path to the folder you need to unzip 

           folder_name (str): name of the folder to unzip eg train or test 

           extract_to (str)  : path to the extracted folder defaults to current dir 

    

    Output : None 

    

    Function source : https://www.geeksforgeeks.org/working-zip-files-python/

    

    """

    # opening the zip file in READ mode 

    with ZipFile(path+folder_name+'.zip', 'r') as zip: 

    # extracting all the files 

        print('Extracting all the files now from '+folder_name+'...') 

        zip.extractall(path=extract_to) 

        print('Done!') 
unzip_folder(path='/kaggle/input/aerial-cactus-identification/',folder_name='train')





unzip_folder(path='/kaggle/input/aerial-cactus-identification/',folder_name='test')

# Opening an image using cv2
df=pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')
df.head()
# Data set has skewed labels 



df['has_cactus'].value_counts()
# No nulls in the data lables 

df.isnull().sum()
# Predicting Cactus Using FastAI and transfer learning 



src = (ImageList.from_df(df,path='./train')

      .split_by_rand_pct(0.1)

      .label_from_df(cols='has_cactus')

      )



tfms = get_transforms(do_flip=True,flip_vert=True) 



size=224



data = src.transform(tfms=tfms, size=size).databunch(bs=16).normalize(imagenet_stats)





data.show_batch(rows=3, figsize=(7,6))

#learn = cnn_learner(data, models.resnet34, metrics=accuracy)

#learn.fit_one_cycle(5)
#learn_Resent50 = cnn_learner(data, models.resnet50, metrics=accuracy)

#learn_Resent50.fit_one_cycle(5)
#learn_den121 = cnn_learner(data, models.densenet121, metrics=accuracy)#
#lr=3e-2

#learn_den121.fit_one_cycle(8,slice(lr))
lean_den161=cnn_learner(data, models.densenet161, metrics=accuracy)
lean_den161.fit_one_cycle(7)
#learn_den121.unfreeze()
lean_den161.unfreeze()
lean_den161.recorder.plot()
lean_den161.fit_one_cycle(2, max_lr=slice(1e-5))
path = '/kaggle/working/test'

items = os.listdir(path) #this gives me a list of both files and folders in dir

items = [item for item in items if os.path.isfile(os.path.join(path, item))]
file=[]

for root, dirs, files in os.walk(path):

    for filename in files:

        file.append(filename)
len(file)
submission_df=pd.DataFrame({'id': file, 'has_cactus': 1})
i=0

for image in submission_df.id:

    img=open_image(path+'/'+image)

    pred_class,pred_idx,outputs = lean_den161.predict(img)

    

    

    #print(pred_class)

    submission_df.iloc[i,1]=pred_class

    i=i+1
submission_df.head()
submission_df.to_csv('submission.csv', header=True, index=False)