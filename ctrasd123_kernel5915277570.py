## This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#from efficientnet_pytorch import EfficientNet



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import OrderedDict



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import sys

package_path = '../input/efficientnetpytorch/'

sys.path.append(package_path)



from efficientnet_pytorch import EfficientNet

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import print_function, absolute_import

import os

import sys

import time

import datetime

import argparse

import os.path as osp

import numpy as np

import random

from PIL import Image

import tqdm

import cv2

import csv

import math

import torchvision as tv

import torchvision

import torch.nn.functional as F

import torch.optim as optim

import torch

import torch.nn as nn

import torch.backends.cudnn as cudnn

from sklearn.metrics import f1_score

from torch.utils.data import DataLoader

from torch.autograd import Variable

from torch.optim import lr_scheduler

from tqdm import tqdm

from torch.utils.data import Dataset

import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
name_file='../input/aptos2019-blindness-detection/test.csv'

csv_file=csv.reader(open(name_file,'r'))

content=[]

for line in csv_file:

    #print(line[0]) #打印文件每一行的信息

    content.append(line[0]+'.png')

    content=content[1:]
from torch.nn.parameter import Parameter



def gem(x, p=3, eps=1e-6):

    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):

        super(GeM,self).__init__()

        self.p = Parameter(torch.ones(1)*p)

        self.eps = eps

    def forward(self, x):

        return gem(x, p=self.p, eps=self.eps)       

    def __repr__(self):

        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class Baseline_single_gem(nn.Module):

    def __init__(self, num_classes, loss_type="single BCE", **kwargs):

        super(Baseline_single_gem, self).__init__()

        self.loss_type = loss_type

        #resnet50 = torchvision.models.densenet169(pretrained=False)

        resnet50 = torchvision.models.densenet201(pretrained=False)

        self.base = nn.Sequential(*list(resnet50.children())[:-1])

        #self.feature_dim = 512*2

        self.feature_dim = 1920        

        #self.feature_dim=1664

        # self.reduce_conv = nn.Conv2d(self.feature_dim * 4, self.feature_dim, 1)

        if self.loss_type == "single BCE":

            #self.ap = nn.AdaptiveAvgPool2d(1)

            self.ap = GeM()

            self.classifiers = nn.Linear(in_features=self.feature_dim, out_features=num_classes)

            self.sigmoid = nn.Sigmoid()

            self.dropout=nn.Dropout(0.5)

            self.cal_score=nn.Linear(in_features=num_classes, out_features=1)

        # for m in self.modules():

        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

        #         nn.init.kaiming_normal(m.weight, mode='fan_in', nonlinearity='relu')

        #         # import scipy.stats as stats

        #         # stddev = m.stddev if hasattr(m, 'stddev') else 0.1

        #         # X = stats.truncnorm(-2, 2, scale=stddev)

        #         # values = torch.Tensor(X.rvs(m.weight.data.numel()))

        #         # values = values.view(m.weight.data.size())

        #         # m.weight.data.copy_(values)

        #     elif isinstance(m, nn.BatchNorm2d):

        #         m.weight.data.fill_(1)

        #         m.bias.data.zero_()

    def freeze_base(self):

        for p in self.base.parameters():

            p.requires_grad = False



    def unfreeze_all(self):

        for p in self.parameters():

            p.requires_grad = True

    def forward(self, x1):

        x = self.base(x1)

        #print(x.shape)

        map_feature=torch.tensor(x)

        # x = self.reduce_conv(x)

        if self.loss_type == "single BCE":

            x = self.ap(x)

            #print(x.shape)

            x = self.dropout(x)

            x = x.view(x.size(0), -1)

            feat_m=torch.tensor(x)

            ys = self.classifiers(x)

            #y = self.sigmoid(y)

        return ys

class Baseline_single(nn.Module):

    def __init__(self, num_classes, loss_type="single BCE", **kwargs):

        super(Baseline_single, self).__init__()

        self.loss_type = loss_type

        #resnet50 = torchvision.models.densenet169(pretrained=False)

        resnet50 = torchvision.models.densenet201(pretrained=False)

        self.base = nn.Sequential(*list(resnet50.children())[:-1])

        #self.feature_dim = 512*2

        self.feature_dim = 1920        

        #self.feature_dim=1664

        # self.reduce_conv = nn.Conv2d(self.feature_dim * 4, self.feature_dim, 1)

        if self.loss_type == "single BCE":

            self.ap = nn.AdaptiveAvgPool2d(1)

            #self.ap = GeM()

            self.classifiers = nn.Linear(in_features=self.feature_dim, out_features=num_classes)

            self.sigmoid = nn.Sigmoid()

            self.dropout=nn.Dropout(0.5)

            self.cal_score=nn.Linear(in_features=num_classes, out_features=1)

        # for m in self.modules():

        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

        #         nn.init.kaiming_normal(m.weight, mode='fan_in', nonlinearity='relu')

        #         # import scipy.stats as stats

        #         # stddev = m.stddev if hasattr(m, 'stddev') else 0.1

        #         # X = stats.truncnorm(-2, 2, scale=stddev)

        #         # values = torch.Tensor(X.rvs(m.weight.data.numel()))

        #         # values = values.view(m.weight.data.size())

        #         # m.weight.data.copy_(values)

        #     elif isinstance(m, nn.BatchNorm2d):

        #         m.weight.data.fill_(1)

        #         m.bias.data.zero_()

    def freeze_base(self):

        for p in self.base.parameters():

            p.requires_grad = False



    def unfreeze_all(self):

        for p in self.parameters():

            p.requires_grad = True

    def forward(self, x1):

        x = self.base(x1)

        #print(x.shape)

        map_feature=torch.tensor(x)

        # x = self.reduce_conv(x)

        if self.loss_type == "single BCE":

            x = self.ap(x)

            #print(x.shape)

            x = self.dropout(x)

            x = x.view(x.size(0), -1)

            feat_m=torch.tensor(x)

            ys = self.classifiers(x)

            #y = self.sigmoid(y)

        return ys
def cv_imread(file_path):

	cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)

	return cv_img 



def change_size(image):

	#image=cv2.imread(read_file,1) #读取图片 image_name应该是变量



	b=cv2.threshold(image,15,255,cv2.THRESH_BINARY)          #调整裁剪效果

	binary_image=b[1]               #二值图--具有三通道

	binary_image=cv2.cvtColor(binary_image,cv2.COLOR_BGR2GRAY)

	print(binary_image.shape)       #改为单通道



	x=binary_image.shape[0]

	print("高度x=",x)

	y=binary_image.shape[1]

	print("宽度y=",y)

	edges_x=[]

	edges_y=[]



	for i in range(x):



		for j in range(y):



			if binary_image[i][j]==255:

			 # print("横坐标",i)

			 # print("纵坐标",j)

			 edges_x.append(i)

			 edges_y.append(j)



	left=min(edges_x)               #左边界

	right=max(edges_x)              #右边界

	width=right-left                #宽度



	bottom=min(edges_y)             #底部

	top=max(edges_y)                #顶部

	height=top-bottom               #高度



	pre1_picture=image[left:left+width,bottom:bottom+height]        #图片截取



	return pre1_picture                                             #返回图片数据





def crop_image1(img,tol=7):

	# img is image data

	# tol  is tolerance

		

	mask = img>tol

	return img[np.ix_(mask.any(1),mask.any(0))]



def crop_image_from_gray(img,tol=7):

	if img.ndim ==2:

		mask = img>tol

		return img[np.ix_(mask.any(1),mask.any(0))]

	elif img.ndim==3:

		gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		mask = gray_img>tol

		

		check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

		if (check_shape == 0): # image is too dark so that we crop out everything,

			return img # return original image

		else:

			img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

			img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

			img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

	#         print(img1.shape,img2.shape,img3.shape)

			img = np.stack([img1,img2,img3],axis=-1)

	#         print(img.shape)

		return img

def load_ben_color(image, sigmaX=10):

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = crop_image_from_gray(image)

	image = cv2.resize(image, (492, 492))

	image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

		

	return image





def findCircle(image):

	#print(image.shape)

	hsv_img=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

	h_img=image[:,:,0]

	s_img=image[:,:,1]

	v_img=image[:,:,2]

	height,width=v_img.shape

	#print(int(max(height,width)/16+1)*2)

	mask_v_a=cv2.adaptiveThreshold(v_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,int(max(height,width)/16)*2+1,1)

	#ret,mask_v_a = cv2.threshold(h_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)



	#cv2.imshow('msk_v_a',mask_v_a)

	#cv2.waitKey(0)

	ratio=128/min(height,width)

	msk=cv2.resize(mask_v_a,(int(width*ratio),int(height*ratio)),interpolation=cv2.INTER_CUBIC )

	h,w=msk.shape

	msk_expand=np.zeros((3*h,3*w),np.uint8)

	msk_expand[h:2*h,w:2*w]=msk

	#cv2.imshow('msk_expand',msk_expand)

	#cv2.waitKey(0)

	long_edge=max(h,w)

	r0=round(0.3*long_edge)

	r1=round(0.7*long_edge)

	#print(msk_expand,r0,r1)

	circles=cv2.HoughCircles(msk_expand,cv2.HOUGH_GRADIENT,1,90,param1 = 50,param2 = 5,minRadius = r0,maxRadius = r1) 

	#print(circles is None)



	if circles is  None:

		c_x=width/2

		c_y=height/2

		radius=0.55*max(height,width)

	else:	

		#print(circles.shape)

		circles = np.uint16(np.around(circles))

		c_x=(circles[0,0,0]-w)/ratio

		c_y=(circles[0,0,1]-h)/ratio

		radius=circles[0,0,2]/ratio

	'''

	c_x=int(c_x)

	c_y=int(c_y)

	radius=int(radius)

	#print(circles.shape)

	#for i in circles[0,:]:

	cv2.circle(image,(c_x,c_y),radius,(255,0,0),8) 

	cv2.circle(image,(c_x,c_y),2,(0,0,255),10)

	hsv_img=cv2.resize(image,(int(0.3*image.shape[1]),int(0.3*image.shape[0])),interpolation=cv2.INTER_CUBIC )

	cv2.imshow('circle',hsv_img)

	cv2.waitKey(0)

	'''

	return c_x,c_y,radius



def circleCrop(c_x,c_y,radius,height,width):

	if math.floor(radius+c_y)>height:

		y0=max(math.ceil(c_y-radius),0);

		y1=height;

		if math.floor(radius+c_x)>width:

			x1=width

		else:

			x1=math.floor(radius+c_x)

		if math.floor(c_x-radius<0):

			x0=0

		else:

			x0=math.floor(c_x-radius)

	elif math.ceil(c_y-radius)<0:

		y0=0

		y1=min(math.floor(c_y+radius),height)

		if math.floor(radius+c_x)>width:

			x1=width

		else:

			x1=math.floor(radius+c_x)

		if math.floor(c_x-radius<0):

			x0=0

		else:

			x0=math.floor(c_x-radius)

	else:

		y0=math.ceil(c_y-radius)

		y1=math.floor(c_y+radius)

		x0=math.ceil(c_x-radius)

		x1=math.floor(c_x+radius)



	return x0,x1,y0,y1

			



def trimFundus(image):

	c_x,c_y,radius=findCircle(image)

	height=image.shape[0]

	width=image.shape[1]

	x0,x1,y0,y1=circleCrop(c_x,c_y,radius,height,width)

	#print(x0,x1,y0,y1)

	trimed=image[y0:y1,x0:x1,:]

	return trimed



def load_ben_yuan(image,sigmaX=10):

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = crop_image_from_gray(image)

	image = cv2.resize(image, (512, 512))

	return image





PARAM = 92



def Radius_Reduction(img,PARAM):

    h,w,c=img.shape

    Frame=np.zeros((h,w,c),dtype=np.uint8)

    cv2.circle(Frame,(int(math.floor(w/2)),int(math.floor(h/2))),int(math.floor((h*PARAM)/float(2*100))), (255,255,255), -1)

    Frame1=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)

    img1 =cv2.bitwise_and(img,img,mask=Frame1)

    return img1



def info_image(im):

    # Compute the center (cx, cy) and radius of the eye

    cy = im.shape[0]//2

    midline = im[cy,:]

    midline = np.where(midline>midline.mean()/3)[0]

    if len(midline)>im.shape[1]//2:

        x_start, x_end = np.min(midline), np.max(midline)

    else: # This actually rarely happens p~1/10000

        x_start, x_end = im.shape[1]//10, 9*im.shape[1]//10

    cx = (x_start + x_end)/2

    r = (x_end - x_start)/2

    return cx, cy, r





def resize_image(im, img_size, augmentation=False):

    # Crops, resizes and potentially augments the image to IMG_SIZE

    cx, cy, r = info_image(im)

    scaling = img_size/(2*r)

    rotation = 0

    if augmentation:

        scaling *= 1 + 0.3 * (np.random.rand()-0.5)

        rotation = 360 * np.random.rand()

    M = cv2.getRotationMatrix2D((cx,cy), rotation, scaling)

    M[0,2] -= cx - img_size/2

    M[1,2] -= cy - img_size/2

    return cv2.warpAffine(im, M, (img_size, img_size)) # This is the most important line





def subtract_median_bg_image(im):

    k = np.max(im.shape)//20*2+1

    bg = cv2.medianBlur(im, k)

    return cv2.addWeighted (im, 4, bg, -4, 128)





def subtract_gaussian_bg_image(im):

    # k = np.max(im.shape)/10

    bg = cv2.GaussianBlur(im ,(0,0) , 10)

    return cv2.addWeighted (im, 4, bg, -4, 128)



def open_img(fn, size):

    "Open image in `fn`, subclass and overwrite for custom behavior."

    image = cv2.imread(fn)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = resize_image(image, size)



    # changing line here.

    image = subtract_gaussian_bg_image(image)

    image = Radius_Reduction(image, PARAM)

    image = crop_image_from_gray(image)

    image = cv2.resize(image,(512,512))

    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

    #return Image(pil2tensor(image, np.float32).div_(255))



def get_preds(arr):

    mask = arr == 0

    return np.clip(np.where(mask.any(1), mask.argmax(1), 5) - 1, 0, 4)





cnt_t=0

class eye_dataset(Dataset):

	"""docstring for data"""

	def __init__(self, txt_path,transform=None,transform2=None):

		imgs = []

		for img in txt_path:

			imgs.append(img)

		self.imgs = imgs

		self.transform = transform

		self.transform2 = transform2        

	def __getitem__(self, index):

		fn= self.imgs[index]

		#img = Image.open(fn).convert('RGB') 

		#img= Image.open(fn).convert('L')

		

		img=cv2.imread('/kaggle/input/aptos2019-blindness-detection/test_images/'+fn)

		#img=load_ben_color(img)

		img_copy=img.copy()

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		img_copy=img.copy()

		try:

			img=trimFundus(img)

			img = cv2.resize(img, (512, 512))

		except Exception as e:

			#raise e

			print(e)

			img = crop_image_from_gray(img_copy)

			img = cv2.resize(img, (512, 512))

		#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		#img=load_ben_yuan(img)

		img = Image.fromarray(img)

		#cv2.imwrite('./save_test/test_'+str(index)+'.jpg',img)

		#cnt_t+=1

		if self.transform is not None:

			img1 = self.transform(img)

			img2 = self.transform2(img)

		#img = img.unsqueeze(0) 

		return img1,img2, fn[:-4]

	def __len__(self):

		return len(self.imgs)

class eye_dataset_circle(Dataset):

	"""docstring for data"""

	def __init__(self, txt_path,transform=None):

		imgs = []

		for img in txt_path:

			imgs.append(img)

		self.imgs = imgs

		self.transform = transform

	def __getitem__(self, index):

		fn= self.imgs[index]

		#img = Image.open(fn).convert('RGB') 

		#img= Image.open(fn).convert('L')

		

		#img=cv2.imread('/kaggle/input/aptos2019-blindness-detection/test_images/'+fn)

		img=open_img('/kaggle/input/aptos2019-blindness-detection/test_images/'+fn,530)

		img = Image.fromarray(img)

		#cv2.imwrite('./save_test/test_'+str(index)+'.jpg',img)

		#cnt_t+=1

		if self.transform is not None:

			img = self.transform(img)

		#img = img.unsqueeze(0) 

		

		return img, fn[:-4]

	def __len__(self):

		return len(self.imgs)



class eye_dataset_orl(Dataset):

	"""docstring for data"""

	def __init__(self, txt_path,transform=None,transform2=None,transform3=None,transform4=None):

		imgs = []

		for img in txt_path:

			imgs.append(img)

		self.imgs = imgs

		self.transform = transform

		self.transform2 = transform2

		self.transform3 = transform3

		self.transform4 = transform4

	def __getitem__(self, index):

		fn= self.imgs[index]

		#img = Image.open(fn).convert('RGB') 

		#img= Image.open(fn).convert('L')

		

		img=cv2.imread('/kaggle/input/aptos2019-blindness-detection/test_images/'+fn)

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		img=crop_image_from_gray(img)

		img3_=open_img('/kaggle/input/aptos2019-blindness-detection/test_images/'+fn,530)

		img4_=open_img('/kaggle/input/aptos2019-blindness-detection/test_images/'+fn,530)

		img3_ = Image.fromarray(img3_)

		img4_ = Image.fromarray(img4_)

		img=cv2.resize(img, (512, 512))     

		img = Image.fromarray(img)

		#cv2.imwrite('./save_test/test_'+str(index)+'.jpg',img)

		#cnt_t+=1

		if self.transform is not None:

			img1 = self.transform(img)

			img2 = self.transform2(img)

			img3 = self.transform3(img3_)

			img4 = self.transform4(img4_)

			#img5 = self.transform4(img4_)

		#img = img.unsqueeze(0) 

		

		return img1,img2,img3,img4, fn[:-4]

	def __len__(self):

		return len(self.imgs)
def load_para_dict(model1):

    state_dict_1=torch.load(model1)

    new_state_dict = OrderedDict()

    for k, v in state_dict_1.items():

        if 'module' in k:

            name = k[7:] # add `module.`

        else:

            name=k

        new_state_dict[name] = v

    return new_state_dict
if __name__ == '__main__':



	use_gpu = torch.cuda.is_available()

	if use_gpu:

		cudnn.benchmark = True

		torch.cuda.manual_seed_all(0)

	else:

		print("Currently using CPU (GPU is highly recommended)")





	transform2 = transforms.Compose([

			transforms.Resize((512, 512)),

			#transforms.ColorJitter(brightness=20,contrast=0.2,saturation=20,hue=0.1),

			transforms.ToTensor(), # 转为Tensor

			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

			#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化

							 ])	

	transform_eff = transforms.Compose([

			transforms.Resize((512, 512)),

			#transforms.ColorJitter(brightness=20,contrast=0.2,saturation=20,hue=0.1),

			transforms.ToTensor(), # 转为Tensor

			#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

			#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化

							 ])	

	transform_eff_456 = transforms.Compose([

			transforms.Resize((456, 456)),

			#transforms.ColorJitter(brightness=20,contrast=0.2,saturation=20,hue=0.1),

			transforms.ToTensor(), # 转为Tensor

			#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

			#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化

							 ])	

	transform_eff_380 = transforms.Compose([

			transforms.Resize((380, 380)),

			#transforms.ColorJitter(brightness=20,contrast=0.2,saturation=20,hue=0.1),

			transforms.ToTensor(), # 转为Tensor

			#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

			#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化

							 ])	

	transform_eff_528 = transforms.Compose([

			transforms.Resize((528, 528)),

			#transforms.ColorJitter(brightness=20,contrast=0.2,saturation=20,hue=0.1),

			transforms.ToTensor(), # 转为Tensor

			#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

			#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化

							 ])	

	transform_eff_512 = transforms.Compose([

			transforms.Resize((512, 512)),

			#transforms.ColorJitter(brightness=20,contrast=0.2,saturation=20,hue=0.1),

			transforms.ToTensor(), # 转为Tensor

			#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

			#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化

							 ])	

	name_file='../input/aptos2019-blindness-detection/test.csv'

	csv_file=csv.reader(open(name_file,'r'))

	content=[]

	for line in csv_file:

		#print(line) 

		content.append(line[0]+'.png')

	content=content[1:]



	test_data=eye_dataset_orl(content,transform2,transform_eff_512,transform_eff_512,transform_eff_512)

	#net=inception_resnet_v1()

	#net.load_state_dict(torch.load('model_dict.pkl'))

	#net=inception_v3()

	net1=Baseline_single_gem(num_classes=5)

	net1_2=Baseline_single(num_classes=5)

	#net = EfficientNet.from_pretrained(args.model)

	net2 = EfficientNet.from_name('efficientnet-b6')

	feature = net2._fc.in_features

	net2._fc = nn.Linear(in_features=feature,out_features=5,bias=True)

	net2._avg_pooling=GeM()

	net3 = EfficientNet.from_name('efficientnet-b6')

	feature = net3._fc.in_features

	net3._fc = nn.Linear(in_features=feature,out_features=5,bias=True)

	#net3._avg_pooling=GeM()

    

	net3_2 = EfficientNet.from_name('efficientnet-b6')

	feature = net3_2._fc.in_features

	net3_2._fc = nn.Linear(in_features=feature,out_features=5,bias=True)

	net3_2._avg_pooling=GeM()



	net4 = EfficientNet.from_name('efficientnet-b5')

	feature = net4._fc.in_features

	net4._fc = nn.Linear(in_features=feature,out_features=5,bias=True)

	net2_2 = EfficientNet.from_name('efficientnet-b6')

	feature = net2_2._fc.in_features

	net2_2._fc = nn.Linear(in_features=feature,out_features=5,bias=True)

	if use_gpu:

		net1=net1.cuda()

		net1_2=net1_2.cuda()

		net2=net2.cuda()

		net3=net3.cuda()

		net3_2=net3_2.cuda()

		net4=net4.cuda()

		net2_2=net2_2.cuda()

	net1.load_state_dict(load_para_dict('/kaggle/input/temp-file/model_yuan512_dense_00001_adam_combine_trim_bce_newest_gem_8001.pkl'))

	net1_2.load_state_dict(load_para_dict('/kaggle/input/temp-file/model_yuan_dense201_00001_adam_combine_trim_bce_newest_5.pkl'))

	net2.load_state_dict(load_para_dict('/kaggle/input/temp-file/model_yuan512_efficientnet-b6_00001_adam_combine_trim_bce_maxest_gem_8068.pkl'))

	net2_2.load_state_dict(load_para_dict('/kaggle/input/temp-file/model_yuan456_efficientnet-b6_00001_adam_combine_trim_bce_maxest_07785.pkl'))

	net3.load_state_dict(load_para_dict('/kaggle/input/temp-file/model_yuan512_efficientnet-b6_00001_adam_combine_circle_bce_maxest_slow_8139.pkl'))

	#net3_2.load_state_dict(load_para_dict('/kaggle/input/temp-file/model_yuan512_efficientnet-b6_00001_adam_combine_circle_bce_gem_maxest_slow_793.pkl'))

	net4.load_state_dict(load_para_dict('/kaggle/input/temp-file/model_yuan456_efficientnet-b5_00001_adam_combine_circle_bce_newest_slow_7857.pkl'))



	criterion = nn.CrossEntropyLoss()

	#criterion = nn.MSELoss()

	#optimizer=optim.RMSprop(net.parameters(),lr=args.lr,alpha=0.9)

	#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

	#torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=1)

	#scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[3], gamma=0.1, last_epoch=-1)

	dataloader_test=DataLoader(

		test_data,batch_size=1, shuffle = False, num_workers= 4)

	idx=0

	max_correct=0

	with open('/kaggle/working/submission.csv',"a+", newline='')as f:

		f_csv = csv.writer(f)

		f_csv.writerow(['id_code','diagnosis'])

	with torch.no_grad():

		net1.eval()

		net1_2.eval()

		net2.eval()

		net2_2.eval()

		net3.eval()

		net3_2.eval()

		net4.eval()

		total=0

		total_loss=0

		correct=0

		for id,item in enumerate(dataloader_test):

			print('id ',id)            

			data1,data2,data3,data4,name=item

			if use_gpu:

				data1=data1.cuda()

				data2=data2.cuda()

				data3=data3.cuda()

				data4=data4.cuda() 

				#data5=data5.cuda()  

			#print(data.size())

			#out,aux=net(data)

			#out,feat_map,feat=net(data)

			out1=net1(data1)

			out1_2=net1_2(data1)

			out2_2=net2_2(data2)

			out2=net2(data2)

			out3=net3(data3)

			#out3_2=net3_2(data3)

			out4=net4(data4)

			#out5=net5(data5)

			out=(0.1375*out1+0.1375*out1_2+0.1375*out2_2+0.1375*out2+0.2*out3+0.25*out4)

			print(out)          

			predicted=get_preds((torch.sigmoid(out) > 0.5).cpu().numpy())

			#_, predicted = torch.max(out, 1)

			hh=[str(name[0]),str(predicted[0])]

			print(hh)

			with open('/kaggle/working/submission.csv',"a+", newline='')as f:

				f_csv = csv.writer(f)

				f_csv.writerow(hh)

	print(pd.read_csv('/kaggle/working/submission.csv').diagnosis.value_counts())
