# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

        print(dirname)



# Any results you write to the current directory are saved as output.
base_path = "../input/kuzushiji-recognition/"

train = pd.read_csv("../input/kuzushiji-recognition/train.csv")
train.head(5)
print(train["labels"][0])
import cv2

import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

s = train["image_id"][0]

img_0_path = base_path + "train_images/" + s + ".jpg"

img_0 = cv2.imread(img_0_path)

img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)

plt.imshow(img_0)
#preprocessing on a single image



plt.figure(figsize=(20,20))

img_0_orig = cv2.imread(img_0_path)

img_0_orig = cv2.cvtColor(img_0_orig, cv2.COLOR_BGR2RGB)

img_0 = cv2.imread(img_0_path)

img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(img_0,(3,3),0)

sharp_mask = np.subtract(img_0, blur)

img_0 = cv2.addWeighted(img_0,1, sharp_mask,10, 0)

ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel_1 = np.ones((5,5),np.uint8)

kernel_2 = np.ones((1,1),np.uint8)

opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_1)

closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_2)

mask = cv2.bitwise_not(closing)

mask = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)

img = cv2.add(img_0_orig,mask)

blur_1 = cv2.GaussianBlur(img, (13,13), 0)

sharp_mask_1 = np.subtract(img,blur_1)

sharp_mask_1 = cv2.GaussianBlur(sharp_mask_1, (7,7), 0)

img = cv2.addWeighted(img,1,sharp_mask_1,-10, 0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)

print(img_0_orig.shape)
pwd = os.getcwd()

print(pwd)
train_imgs_path = base_path + "train_images/"

filelist = os.listdir(train_imgs_path)

#print(filelist[:5])

trainlist = train["image_id"].tolist()

trainlist = [x + ".jpg" for x in trainlist]
#print(trainlist[:5])
train_out = train["labels"].tolist()

#train_out[:5]

yolo_labels = []

img_label = []

for l in train_out:

    voc_out = str(l).split()

    for i in range(len(voc_out)//5):

        start_idx = 5*i

        img_label.append(voc_out[start_idx:start_idx+5])

    yolo_labels.append(img_label)

    img_label = []

#print(yolo_labels[:5])
trainlist[0]
def pre(Image):

    img_0_orig = cv2.imread(Image)

    img_0_orig = cv2.cvtColor(img_0_orig, cv2.COLOR_BGR2RGB)

    img_0 = cv2.imread(Image)

    img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img_0,(3,3),0)

    sharp_mask = np.subtract(img_0, blur)

    img_0 = cv2.addWeighted(img_0,1, sharp_mask,10, 0)

    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel_1 = np.ones((5,5),np.uint8)

    kernel_2 = np.ones((1,1),np.uint8)

    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_1)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_2)

    mask = cv2.bitwise_not(closing)

    mask = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)

    img = cv2.add(img_0_orig,mask)

    blur_1 = cv2.GaussianBlur(img, (13,13), 0)

    sharp_mask_1 = np.subtract(img,blur_1)

    sharp_mask_1 = cv2.GaussianBlur(sharp_mask_1, (7,7), 0)

    img = cv2.addWeighted(img,1,sharp_mask_1,-10, 0)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

    
plt.imshow(pre(train_imgs_path+trainlist[0]), cmap = 'gray')

print(pre(train_imgs_path+trainlist[0]).shape)
#yolo_model = make_yolov3_model()

#yolo_model.load_weights("yolo.h5")
#yolo_model.summary()
uni_lab = pd.read_csv("../input/kuzushiji-recognition/unicode_translation.csv")
#uni_lab.head()
#uni_lab.tail()
uni_list = uni_lab["Unicode"].to_list()

#print(uni_list[:5])
with open('class.names', 'w') as f:

    for item in uni_list:

        f.writelines(item+"\n")

#!ls
#open('class.names', 'w').close()

with open("class.names", "r") as f_r:

    test_list = f_r.readlines()

#print(test_list[0][:-1])
uni_dict = ((uni_list[i],i) for i in range(len(uni_list)))

uni_dict = dict(uni_dict)        
#uni_dict["U+0031"]
for labels in yolo_labels:

    for label in labels:

        label[0] = uni_dict[label[0]]

        label[1:5] = list(map(int, label[1:5]))

yolo_labels[:5]
for i in range(len(trainlist)):

    labels = yolo_labels[i]

    img = trainlist[i]

    img_path = base_path + "train_images/" + img

    image = Image.open(img_path)

    w, h = image.size

    for label in labels:

        id_1 = label[0]

        label[0],label[2] = list(map(int,[label[1]-label[3]/2,label[1]+label[3]/2]))

        label[1],label[3] = list(map(int,[label[2]-label[4]/2,label[2]+label[4]/2]))

        label[4] = id_1
#yolo_labels[:3]
#str(yolo_labels[0][0][0])
lines = [train_imgs_path+img_name for img_name in trainlist]

#print(lines[:3])

annot = []

for labels in yolo_labels:

    ann_fin = ""

    for label in labels:

        ann = " " + str(label[0])+','+str(label[1])+','+str(label[2])+','+str(label[3])+','+str(label[4])

        ann_fin += ann

    annot.append(ann_fin)

#print(annot[0])

for i in range(len(lines)):

    lines[i] = lines[i]+annot[i]

#print(lines[0])