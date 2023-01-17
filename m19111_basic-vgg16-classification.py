#importing standard library

from keras.applications.vgg16 import VGG16, preprocess_input,decode_predictions

from keras_preprocessing.image import load_img,img_to_array

import numpy as np

import glob

import os

import pandas as pd

import cv2

from shutil import move

from PIL import Image
import os

for dirname, _, filenames in os.walk('/kaggle/input/final_data'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

#creating folder in destination folder named 'catogories'

os.mkdir('/kaggle/working/final_data/')

os.chdir('/kaggle/working/final_data/')

dir='/kaggle/working/final_data/'
#creating new folder, as random_pics aree in read only folder

from distutils.dir_util import copy_tree



# copy subdirectory example

fromDirectory = "/kaggle/input/final_data"

toDirectory = "/kaggle/working/final_data/"



copy_tree(fromDirectory, toDirectory)
#we have 3 .png images. Converting them into .jpg and removing .ong files

im = Image.open("3722422954_4e310ac2d2_o.png")

rgb_im = im.convert('RGB')

rgb_im.save('3722422954_4e310ac2d2_o.jpg')



im = Image.open("8397706906_fe4d96a3a9_o.png")

rgb_im = im.convert('RGB')

rgb_im.save('8397706906_fe4d96a3a9_o.jpg')





im = Image.open("31974731330_f3dfe2c393_o.png")

rgb_im = im.convert('RGB')

rgb_im.save('31974731330_f3dfe2c393_o.jpg')



os.remove('3722422954_4e310ac2d2_o.png')

os.remove('8397706906_fe4d96a3a9_o.png')

os.remove('31974731330_f3dfe2c393_o.png')
#Adding the name all images in files object

files=glob.glob("*.jpg")
#preprocessing images for VGG16 model

def preprocess(im):

    im=img_to_array(im)

    im=np.expand_dims(im,axis=0)

    im=preprocess_input(im)

    return im[0]



imag1=[]

for i in files:

    im=load_img(i,target_size=(224,224,3))

    imag1.append(preprocess(im))
#converting filesname into dataframe

name=pd.DataFrame(files,columns=['file'])

#Converting pixels into numpy array

pixels=(np.array(imag1))

pixels.shape
#defining VGG16

model=VGG16()


base_model=VGG16(include_top=True,pooling='avg',input_shape=(224,224,3),weights = 'imagenet')
#Predict model for the images

pred = base_model.predict(pixels)

#converting predictions into dataframe

pred_name=decode_predictions(pred,top=1)

prediti=pd.DataFrame(pred_name,columns=['Predictions'])

prediti.shape
#creating dataframe cby concat

data4=pd.concat([prediti, name], axis=1)
data4.head()
#Doing basic funtions into dataframe

data4 = data4.astype(str) 



data4[['Class_ID','folder','Probability']] = data4.Predictions.str.split(",",expand=True) 

data4['folder']=data4.folder.str.replace("[({':]", "")
#Changing file and folder name according to requirement

data4['folder_name']=data4['folder'] 

data4["folder_name"]= data4["folder_name"].str.replace("acoustic_guitar", "acoustic",case = False) 

data4["folder_name"]= data4["folder_name"].str.replace("Arabian_camel", "arabian",case = False) 

data4["folder_name"]= data4["folder_name"].str.replace("German_shepherd", "german",case = False)

data4["folder_name"]= data4["folder_name"].str.replace("Chihuahua", "chihuahua",case = False)
#csv file

data5=data4[['file','folder_name']]

data5.to_csv('submission.csv',index=False)
#creating new dataframe for creating folder

df=data4[['file','folder_name']]

data5.shape
#dividing images into subfolders



all_images = files

os.mkdir('/kaggle/working/final_data/categories')

co = 0

for image in all_images:

    print(image)

    folder_name = df[df['file'] == image]['folder_name']

    folder_name = list(folder_name)[0]

    if not os.path.exists(os.path.join('categories', folder_name)):

        os.mkdir(os.path.join('categories', folder_name))



    path_from = os.path.join(dir, image)

    path_to = os.path.join('categories', folder_name, image)



    move(path_from, path_to)

    print('Moved {} to {}'.format(image, path_to))

    co += 1



print('Moved {} images.'.format(co))