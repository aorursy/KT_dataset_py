import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import cv2

import pydicom

import os, glob

import warnings

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

warnings.filterwarnings("ignore")
colorcodes = ["#00876c"

,"#4c9c85"

,"#78b19f"

,"#a0c6b9"

,"#c8dbd5"

,"#f1f1f1"

,"#f1cfce"

,"#eeadad"

,"#e88b8d"

,"#df676e"

,"#d43d51"]





colorcodes = ["#003f5c"

,"#2f4b7c"

,"#665191"

,"#a05195"

,"#d45087"

,"#f95d6a"

,"#ff7c43"

,"#ffa600"]
print(os.listdir("/kaggle/input/rsna-intracranial-hemorrhage-detection/"))

print(os.listdir("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_test_images")[0])

init_dir_path = "/kaggle/input/rsna-intracranial-hemorrhage-detection/"
def window_image(img, window_center,window_width, intercept, slope):

    '''

        img : dicom.pixel array

        window center : Window Center/Level center of the CT numbers

        window width : It is the range of the CT numbers that image contains

        intercept : It helps to specify the linear transformation of CT images

        slope : rescale scope is also used to specify the linear transformation of images

    '''

    img = (img*slope +intercept) #linear transformation

    img_min = window_center - window_width//2 #lower grey level

    img_max = window_center + window_width//2 #upper grey level

    img[img<img_min] = img_min

    img[img>img_max] = img_max

    return img 



def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
data = pd.read_csv(init_dir_path+"stage_1_train.csv")

smpl = pd.read_csv(init_dir_path+"stage_1_sample_submission.csv")
data_filter = pd.DataFrame(data.ID.str.split("_").tolist(), columns=["ID","Number","Class"])

data_filter.ID = data_filter.ID +"_"+data_filter.Number

data_filter["Label"] = data.Label



data_filter = data_filter.drop('Number', 1)
data_filter.sample(10)
data_dict_true = {}

data_dict_false = {}

classes = list(set(data_filter.Class))

for name in classes:

    data_dict_true[name] = np.array((data_filter.Class == name)&(data_filter.Label==1)).astype(int).sum()

    data_dict_false[name] = np.array((data_filter.Class == name)&(data_filter.Label==0)).astype(int).sum()
print("Total Images : ",len(list(set(data_filter.ID))))

print("Total Classes : ",len(classes))

print("Total Labels : ",2)
print(data_dict_true)

plt.figure(figsize=(10,7))

plt.bar(list(data_dict_true.keys()),list(data_dict_true.values()), label="Label | 1", color=colorcodes[0])

plt.grid()

plt.title("True Class Dstribution")

plt.ylabel("Count")

plt.xlabel("class")

plt.legend()

_=plt.xticks(rotation=75)
print(data_dict_true)

print(data_dict_false)

plt.figure(figsize=(10,7))

plt.bar(list(data_dict_false.keys()),list(data_dict_false.values()),color=colorcodes[6], label="Label | 0")

plt.bar(list(data_dict_true.keys()),list(data_dict_true.values()),color=colorcodes[0],bottom=list(data_dict_false.values()), label="Label | 1")

plt.grid()

plt.title("Class Dstribution")

plt.ylabel("Count")

plt.xlabel("class")

plt.legend()

_=plt.xticks(rotation=75)
ds = pydicom.dcmread("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/"+"ID_0000ca2f6.dcm")
image = ds.pixel_array

window_center , window_width, intercept, slope = get_windowing(ds)

image_windowed = window_image(image, window_center, window_width, intercept, slope)
plt.figure(figsize=(25,15))

plt.subplot(1,2,1)

plt.title("Bone Windowing")

plt.axis("off")

plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

plt.subplot(1,2,2)

plt.title("Brain Windowing")

plt.axis("off")

_=plt.imshow(image_windowed, cmap=plt.cm.bone)
ds
plt.figure(figsize=(25,10))

iterIdx=1

for cls in classes:

    idx = data_filter.Class[(data_filter.Class == cls)&(data_filter.Label==1)].index[260]

#     print("Class : ", cls, "  ,Index : ",idx,"  ,ID : ", data_filter.ID[idx])

    

    ds_class_wise = pydicom.dcmread(init_dir_path+"stage_1_train_images/"+data_filter.ID[idx]+".dcm")

    image = ds_class_wise.pixel_array

    window_center , window_width, intercept, slope = get_windowing(ds_class_wise)

    image_windowed = window_image(image, window_center, window_width, intercept, slope)    

    plt.subplot(1,6,iterIdx)

    plt.title(cls)

    plt.axis("off")

    plt.imshow(image_windowed, cmap=plt.cm.bone)

#     plt.show()

    iterIdx+=1
plt.figure(figsize=(25,10))

subdural_img_ids = data_filter[(data_filter.Class=="subdural") & (data_filter.Label==1)]['ID'].iloc[0:5]

imgs_list = {}

plt_idx_d=1

for name in subdural_img_ids:

    ds_class_wise = pydicom.dcmread(init_dir_path+"stage_1_train_images/"+name+".dcm")

    image = ds_class_wise.pixel_array

    window_center , window_width, intercept, slope = get_windowing(ds_class_wise)

    image_windowed = window_image(image, window_center, window_width, intercept, slope)

    plt.title("Subdural Image distributions")

    sns.distplot(image_windowed.flatten(), hist=False, label=name)

    imgs_list[name]=image_windowed



plt.figure(figsize=(25,10))

for name in imgs_list:

    plt.subplot(1,5,plt_idx_d)

    plt.title(name)

    plt.imshow(imgs_list[name], cmap=plt.cm.bone)

    plt.axis('off')

    

    plt_idx_d+=1
plt_idx = 1

plt.figure(figsize=(25,15))



        

for cls in classes:

    img_name = data_filter[(data_filter.Class == cls)&(data_filter.Label==1)]['ID'][100:106]

    plt.subplot(3,2,plt_idx)

    plt.title(cls)

    for name in img_name:

        ds_class_wise = pydicom.dcmread(init_dir_path+"stage_1_train_images/"+name+".dcm")

        image = ds_class_wise.pixel_array

        window_center , window_width, intercept, slope = get_windowing(ds_class_wise)

        image_windowed = window_image(image, window_center, window_width, intercept, slope) 

        minmax = MinMaxScaler(feature_range=(0,1))

        out = minmax.fit_transform(image_windowed)

        sns.distplot(out.flatten(), hist=False, label=name)

    plt_idx+=1