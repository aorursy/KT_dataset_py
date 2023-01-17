import matplotlib.pyplot as plt
import cv2
screenshot_dir="/kaggle/input/gwd-screenshots/"
img=cv2.imread(screenshot_dir+"image1.PNG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig,ax=plt.subplots(1,1,figsize=(16,8))
ax.set_axis_off()
ax.imshow(img)
img=cv2.imread(screenshot_dir+"image2.PNG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig,ax=plt.subplots(1,1,figsize=(16,8))
ax.set_axis_off()
ax.imshow(img)
img=cv2.imread(screenshot_dir+"image3_1.PNG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig,ax=plt.subplots(1,1,figsize=(16,8))
ax.set_axis_off()
ax.imshow(img)
img=cv2.imread(screenshot_dir+"image3_2.PNG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig,ax=plt.subplots(1,1,figsize=(16,8))
ax.set_axis_off()
ax.imshow(img)
img=cv2.imread(screenshot_dir+"image4.PNG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig,ax=plt.subplots(1,1,figsize=(16,8))
ax.set_axis_off()
ax.imshow(img)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image

import cv2

import torch
import torchvision
from torchvision import transforms

import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import random

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import ipywidgets as widgets
from ipywidgets import interact, interactive
from IPython.display import display
import plotly.graph_objects as go
os.listdir("/kaggle/input/global-wheat-detection")

train_dir = "/kaggle/input/global-wheat-detection/train"
test_dir = "/kaggle/input/global-wheat-detection/test"

df_train=pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")
df_train['x0'] = df_train['bbox'].map(lambda x: x[1:-1].split(",")[0]).astype(float)
df_train['y0'] = df_train['bbox'].map(lambda x: x[1:-1].split(",")[1]).astype(float)
df_train['w'] = df_train['bbox'].map(lambda x: x[1:-1].split(",")[2]).astype(float)
df_train['h'] = df_train['bbox'].map(lambda x: x[1:-1].split(",")[3]).astype(float)
df_train['x1'] = df_train['x0'] + df_train['w']
df_train['y1'] = df_train['y0'] + df_train['h']
list_image_ids = list(df_train['image_id'].unique())
dict_bbox = {}
dict_labels = {}
for img_id in list_image_ids:
    dict_bbox[img_id] = df_train.loc[df_train['image_id']==img_id,['x0','y0','x1','y1']].astype(np.int32).values
    dict_labels[img_id] = np.ones((len(dict_bbox[img_id]),1),dtype=np.int32)
def show_image(disp_image):
    fig,ax = plt.subplots(1,1,figsize=(16,8))
    ax.set_axis_off()
    ax.imshow(disp_image)
def load_image(image_id,img_dir=train_dir):
    ret_img = cv2.imread(img_dir+"//"+image_id+".jpg")
    ret_img = cv2.cvtColor(ret_img, cv2.COLOR_BGR2RGB)
    return ret_img
def load_image_bbox(disp_image,bboxes):
    ret_image = disp_image.copy()
    for box in bboxes:
        x0,y0,x1,y1 = box[0],box[1],box[2],box[3]
        cv2.rectangle(ret_image,(x0,y0),(x1,y1),(255,0,0),3)
    return ret_image
imageid_dropdown = widgets.Dropdown(description="ImageID",value=list_image_ids[0],options=list_image_ids)
x_widget = widgets.IntSlider(min=-180, max=180, step=1, value=0, continuous_update=False, description='Hue')
y_widget = widgets.IntSlider(min=0, max=100, step=1, value=0, continuous_update=False, description='Saturation')
z_widget = widgets.IntSlider(min=0, max=100, step=1, value=0, continuous_update=False, description='Value')

def interactive_hsv(img_id, x,y,z):
    sel_image_id = img_id
    sel_image = load_image(sel_image_id)
    pascal_voc_bbox = dict_bbox[sel_image_id]
    labels = dict_labels[sel_image_id]
    alb_transformations = [albumentations.HueSaturationValue(always_apply=True, hue_shift_limit=x, sat_shift_limit=y, val_shift_limit=z)]
    alb_bbox_params = albumentations.BboxParams(format='pascal_voc',label_fields=['labels'])
    aug = albumentations.Compose(alb_transformations, bbox_params = alb_bbox_params)
    augmented_image = aug(image = sel_image, bboxes=pascal_voc_bbox, labels=labels)
    #show_image(augmented_image['image'])
    show_image(load_image_bbox(augmented_image['image'],augmented_image['bboxes']))

#widgets.interact_manual(interactive_hsv,img_id=imageid_dropdown, x=x_widget, y=y_widget, z=z_widget)
widgets.interact_manual(interactive_hsv,img_id=imageid_dropdown, x=x_widget, y=y_widget, z=z_widget)
imageid_dropdown = widgets.Dropdown(description="ImageID",value=list_image_ids[0],options=list_image_ids)
gblurlimit_widget = widgets.IntSlider(min=3, max=101, step=2, value=3, continuous_update=False, description='GBlurSize')

def interactive_gaussianblur(img_id, gblurlimit):
    sel_image_id = img_id
    sel_image = load_image(sel_image_id)
    pascal_voc_bbox = dict_bbox[sel_image_id]
    labels = dict_labels[sel_image_id]
    alb_transformations = [albumentations.GaussianBlur(blur_limit=gblurlimit,always_apply=False,p=1.0)]
    alb_bbox_params = albumentations.BboxParams(format='pascal_voc',label_fields=['labels'])
    aug = albumentations.Compose(alb_transformations, bbox_params = alb_bbox_params)
    augmented_image = aug(image = sel_image, bboxes=pascal_voc_bbox, labels=labels)
    #show_image(augmented_image['image'])
    show_image(load_image_bbox(augmented_image['image'],augmented_image['bboxes']))

widgets.interact_manual(interactive_gaussianblur,img_id=imageid_dropdown, gblurlimit=gblurlimit_widget)
imageid_dropdown = widgets.Dropdown(description="ImageID",value=list_image_ids[0],options=list_image_ids)
numholes_widget = widgets.IntSlider(min=1, max=16, step=1, value=8, continuous_update=False, description='num_holes')
maxhsize_widget = widgets.IntSlider(min=10, max=1000, step=10, value=8, continuous_update=False, description='max_h_size')
maxwsize_widget = widgets.IntSlider(min=10, max=1000, step=10, value=8, continuous_update=False, description='max_h_size')

def interactive_cutout(img_id, numholes, maxhsize, maxwsize):
    sel_image_id = img_id
    sel_image = load_image(sel_image_id)
    pascal_voc_bbox = dict_bbox[sel_image_id]
    labels = dict_labels[sel_image_id]
    alb_transformations = [albumentations.Cutout(num_holes=numholes,max_h_size=maxhsize, max_w_size=maxwsize, fill_value=0, always_apply=False,p=1.0)]
    alb_bbox_params = albumentations.BboxParams(format='pascal_voc',label_fields=['labels'])
    aug = albumentations.Compose(alb_transformations, bbox_params = alb_bbox_params)
    augmented_image = aug(image = sel_image, bboxes=pascal_voc_bbox, labels=labels)
    #show_image(augmented_image['image'])
    show_image(load_image_bbox(augmented_image['image'],augmented_image['bboxes']))

widgets.interact_manual(interactive_cutout,img_id=imageid_dropdown,numholes=numholes_widget, maxhsize=maxhsize_widget, maxwsize=maxwsize_widget)
def generate_custom_cutout(new_image,boxes,n_holes=20,max_h_size=20,max_w_size=20,height=1024,width=1024):
    
    box_coord = []
    #box_coord=list(boxes)
    for box in boxes:
        x0,y0,x1,y1 = box[0],box[1],box[2],box[3]
        box_coord.append([x0,y0,x1,y1])

    
    temp_image = new_image.copy()
    final_list_coord = box_coord.copy()
    i=0
    for hole_ in range(n_holes):
        x = random.randint(0, 1024)
        y = random.randint(0, 1024)
        # Generating diagonally opposite co-ordinates
        y1 = np.clip(y-max_h_size // 2, 0, height)
        x1 = np.clip(x-max_w_size // 2, 0, width)
        y2 = np.clip(y1 + max_h_size, 0, height)
        x2 = np.clip(x1 + max_w_size, 0, width)
        mask = np.ones((1024,1024,3),np.int32)
        mask[y1:y2,x1:x2,:]=0
        invert_mask = 1- mask
        temp_image = temp_image * mask
        j=0
        for box in box_coord:
            j=j+1

            x0,y0,x1,y1 = box[0], box[1], box[2], box[3]
            # finding intersection
            img1 = invert_mask

            img2 = np.zeros((1024,1024,3))
            img2[y0:y1,x0:x1,:]=1


            intersection = np.logical_and(img1, img2)
            instersection_area = np.sum(intersection)

            if instersection_area>0:
                if box in final_list_coord:
                    final_list_coord.remove(box)

    ret_image={}
    ret_image['image']=temp_image
    ret_image['bboxes']=final_list_coord
    
    return ret_image
imageid_dropdown = widgets.Dropdown(description="ImageID",value=list_image_ids[0],options=list_image_ids)
numholes_widget = widgets.IntSlider(min=1, max=16, step=1, value=8, continuous_update=False, description='num_holes')
maxhsize_widget = widgets.IntSlider(min=10, max=1000, step=10, value=8, continuous_update=False, description='max_h_size')
maxwsize_widget = widgets.IntSlider(min=10, max=1000, step=10, value=8, continuous_update=False, description='max_h_size')

def interactive_custom_cutout(img_id, numholes, maxhsize, maxwsize):
    sel_image_id = img_id
    sel_image = load_image(sel_image_id)
    pascal_voc_bbox = dict_bbox[sel_image_id]
    labels = dict_labels[sel_image_id]
    
    augmented_image = generate_custom_cutout(sel_image,pascal_voc_bbox,n_holes=numholes,max_h_size=maxhsize,max_w_size=maxwsize,height=1024,width=1024)
    show_image(load_image_bbox(augmented_image['image'],augmented_image['bboxes']))

widgets.interact_manual(interactive_custom_cutout,img_id=imageid_dropdown,numholes=numholes_widget, maxhsize=maxhsize_widget, maxwsize=maxwsize_widget)
class mosaic:
    def __init__(self, image_dir,dict_boundingbox):
        self.dir = image_dir
        self.dict_bb = dict_boundingbox
    
    def load_input_image(self,image_id):
        ret_img = cv2.imread(self.dir+"//"+image_id+".jpg")
        ret_img = cv2.cvtColor(ret_img, cv2.COLOR_BGR2RGB)
        return ret_img
    
    
    def select_bboxes(self,img_x0,img_y0,img_x1,img_y1,h,w,list_bbox):
        bboxes =[]
        for box in list_bbox:
            bboxes.append([box[0],box[1],box[2],box[3]])
        
        mask_disp = np.zeros((h,w))
        mask_img = np.zeros((h,w))
        mask_disp[img_y0:img_y1,img_x0:img_x1]=1
        mask_img[img_y0:img_y1,img_x0:img_x1]=1
        
        res_bbox = [] 
        for box in bboxes:
            x0,y0,x1,y1 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            area_box = (y1-y0)*(x1-x0)

            mask_box = np.zeros((h,w))
            mask_box[y0:y1,x0:x1]=1

            intersection = np.logical_and(mask_img, mask_box)
            intersection_area = np.sum(intersection)
            if(int(intersection_area) == int(area_box)):
                res_bbox.append(box)
        return res_bbox
    
    def generate_mosaic(self,img_id1,img_id2,img_id3,img_id4):
        # for mosaic
        padding = 100
        blank_img = np.zeros((1024,1024,3))
        w_set,h_set = 1024,1024
        
        img_1,bbox_1 = self.load_input_image(img_id1),self.dict_bb[img_id1]
        
        img_2,bbox_2 = self.load_input_image(img_id2),self.dict_bb[img_id2]
        
        img_3,bbox_3 = self.load_input_image(img_id3),self.dict_bb[img_id3]
        
        img_4,bbox_4 = self.load_input_image(img_id4),self.dict_bb[img_id4]

        #Generating center coordinates
        xc = random.randint(0+padding, 1024-padding)
        yc = random.randint(0+padding, 1024-padding)
        

        # first block (top-left)
        blank_img[0:yc,0:xc,:] = img_1[0:yc,0:xc,:]
        bbox_sel_1 = self.select_bboxes(0,0,xc,yc,h_set,w_set,bbox_1)

        # second block (bottom-left)
        blank_img[yc:h_set,0:xc,:] = img_2[yc:h_set,0:xc,:]
        bbox_sel_2 = self.select_bboxes(0,yc,xc,h_set,h_set,w_set,bbox_2)

        # third corner (bottom-right)
        blank_img[yc:h_set,xc:w_set,:] = img_3[yc:h_set,xc:w_set,:]
        bbox_sel_3 = self.select_bboxes(xc,yc,w_set,h_set,h_set,w_set,bbox_3)

        # fourth corner (top-right)
        blank_img[0:yc,xc:w_set,:] = img_4[0:yc,xc:w_set,:]
        bbox_sel_4 = self.select_bboxes(xc,0,w_set,yc,h_set,w_set,bbox_4)
        ret_image = {}
        ret_image['bboxes']=bbox_sel_1+bbox_sel_2+bbox_sel_3+bbox_sel_4
        for box in ret_image['bboxes']:
            cv2.rectangle(blank_img,(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
        
        ret_image['image']=blank_img.astype(np.int32)
        
        return ret_image
imageid1_dropdown = widgets.Dropdown(description="ImageID1",value=list_image_ids[0],options=list_image_ids)
imageid2_dropdown = widgets.Dropdown(description="ImageID2",value=list_image_ids[0],options=list_image_ids)
imageid3_dropdown = widgets.Dropdown(description="ImageID3",value=list_image_ids[0],options=list_image_ids)
imageid4_dropdown = widgets.Dropdown(description="ImageID4",value=list_image_ids[0],options=list_image_ids)


def interactive_custom_cutout(img_id1, img_id2, img_id3, img_id4):
    
    directory_of_images = "/kaggle/input/global-wheat-detection/train"
    aug_obj = mosaic(directory_of_images,dict_bbox)
    augmented_image = aug_obj.generate_mosaic(img_id1,img_id2,img_id3,img_id4)
    #augmented_image = generate_custom_cutout(sel_image,pascal_voc_bbox,n_holes=numholes,max_h_size=maxhsize,max_w_size=maxwsize,height=1024,width=1024)
    show_image(load_image_bbox(augmented_image['image'],augmented_image['bboxes']))

widgets.interact_manual(interactive_custom_cutout,img_id1=imageid1_dropdown, img_id2=imageid2_dropdown, img_id3=imageid3_dropdown, img_id4=imageid4_dropdown)