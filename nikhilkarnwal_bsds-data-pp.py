!wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz

!wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-human.tgz

!tar zxvf *-images.tgz

!tar zxvf *-human.tgz

!ls
import numpy as np

import os

import torchvision.transforms as tf

import matplotlib.pyplot as plt

import torch as th

from PIL import Image

import re

import torch.nn as nn

import torch.nn.functional as F

import torchvision.models as models
def get_seg(file):

  with open(file, "r") as seg_file:

    d_seen = False

    header = {}

    image = np.full((500,500),33,dtype=np.int16)

    for line in seg_file:

      if not d_seen:

        if('data' in line):

          d_seen = True

        else:

          header[line.split(' ')[0]]=re.split(' |\n',line)[1]

      else:

        col = np.asarray(line.split(' '),dtype=np.int64)

        image[col[1],col[2]:(col[3]+1)]=col[0] if col[0]< 21 else 33

  return header['image'], image[:int(header['height']),:int(header['width'])]
class DataSet:

  size = (224,224)

  def __init__(self, image_train, image_test, label, base_path):

    self.im_train = image_train

    self.im_test = image_test

    self.label = label

    self.base_path = base_path

  

  def get_img_feat(self, img_dir):

    print("def get_img_feat---")

    cnt=0

    img_feats = {}

    trf = tf.Compose([tf.Resize(self.size),

                      tf.ToTensor(), 

                      tf.Normalize(mean = [0.485, 0.456, 0.406], 

                                   std = [0.229, 0.224, 0.225])])

    for img in os.listdir(img_dir):

      #print(img)

      img_path = os.path.join(img_dir,img);

      img_tensr = trf(Image.open(img_path))

      img_feats[img.split('.')[0]]=img_tensr

      cnt=cnt+1

      if cnt%100 == 0:

        print("Cnt - ", cnt)

    return img_feats



  def get_seg_labels(self):

    print("def get_seg_labels---")

    cnt=0

    seg_labels = {}

    trf = tf.Compose([tf.ToPILImage(mode=None),

                      tf.Resize(self.size)])

    for user in os.listdir(self.label):

      user_dir = os.path.join(self.label,user)

      for seg in os.listdir(user_dir):

        seg_path = os.path.join(user_dir,seg)

        img_id, seg_label = get_seg(seg_path)

        #print(img_id)

        seg_tensor = tf.ToTensor()(np.array(trf(th.from_numpy(seg_label).to(th.int)), dtype=np.int16))

        if img_id not in seg_labels:

          seg_labels[img_id]=[]

        seg_labels[img_id].append(seg_tensor)

        cnt=cnt+1

        if cnt%100 == 0:

          print("Cnt - ", cnt)

    return seg_labels



  def gen_data(self):

    train_feat = self.get_img_feat(self.im_train)

    test_feat = self.get_img_feat(self.im_test)

    seg_labels = self.get_seg_labels()

    train_data = []

    train_label = []

    test_data = []

    test_label = []

    print("Augmenting Data")

    for id in seg_labels:

      for seg_lab in seg_labels[id]:

        if id in train_feat and len(train_data)<1500:

          train_data.append(np.array(train_feat[id]))

          train_label.append(np.array(seg_lab))

        if id in test_feat and len(test_data) < 500:

          test_data.append(np.array(test_feat[id]))

          test_label.append(np.array(seg_lab))

    

    self.train_data = np.array(train_data)

    self.train_label = np.array(train_label)

    self.test_data = np.array(test_data)

    self.test_label = np.array(test_label)



  def save_data(self):

    print("Saving Data")

    np.save(os.path.join(self.base_path,"train_data"), self.train_data);

    np.save(os.path.join(self.base_path,"train_label"), self.train_label);

    np.save(os.path.join(self.base_path,"test_data"), self.test_data);

    np.save(os.path.join(self.base_path,"test_label"), self.test_label);



  def load_data(self):

    print("Loading Data")

    self.train_data = np.load(os.path.join(self.base_path,"train_data.npy"));

    self.train_label = np.load(os.path.join(self.base_path,"train_label.npy"));

    self.test_data = np.load(os.path.join(self.base_path,"test_data.npy"));

    self.test_label = np.load(os.path.join(self.base_path,"test_label.npy"));



  def get_data(self,old=False):

    if old:

      self.load_data()

    else:

      self.gen_data()

      self.save_data()



base_dir = "BSDS300"

base_gendata = os.path.join(base_dir,'GenData') 

if not os.path.exists(base_gendata):

    os.mkdir(base_gendata)

images = os.path.join(base_dir,'images')

train = os.path.join(images,'train')

test = os.path.join(images,'test')

label = os.path.join(base_dir,'human/color')

dataset = DataSet(train,test,label,base_gendata)

dataset.get_data()

print(dataset.train_data.shape)

print(dataset.train_label.shape)

print(dataset.test_data.shape)

print(dataset.test_label.shape)
