# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
os.listdir("../input")
df_classes_trainable = pd.read_csv("../input/classes-trainable.csv")
df_classes_trainable.describe()
df_classes_trainable.head()
df_classes_trainable.shape
df_train_human_labels = pd.read_csv("../input/train_human_labels.csv")
df_train_human_labels.describe()
df_train_human_labels.head()
df_train_human_labels.shape
df_stage_1_sample_submission = pd.read_csv("../input/stage_1_sample_submission.csv")
df_stage_1_sample_submission.describe()
df_stage_1_sample_submission.head()
df_stage_1_sample_submission.shape
df_tuning_labels = pd.read_csv("../input/tuning_labels.csv", header=None, names=['id', 'labels'])
df_tuning_labels.describe()
df_tuning_labels.head()
df_tuning_labels.shape
df_stage_1_attributions = pd.read_csv("../input/stage_1_attributions.csv")
df_stage_1_attributions.describe()
df_stage_1_attributions.head()
df_stage_1_attributions.shape
df_train_bounding_boxes = pd.read_csv("../input/train_bounding_boxes.csv")
df_train_bounding_boxes.describe()
df_train_bounding_boxes.head()
df_train_bounding_boxes.shape
df_class_descriptions = pd.read_csv("../input/class-descriptions.csv")
df_class_descriptions.describe()
df_class_descriptions.description
df_class_descriptions.head()
df_class_descriptions.shape
df_train_machine_labels = pd.read_csv("../input/train_machine_labels.csv")
df_train_machine_labels.describe()
df_train_machine_labels.head()
df_train_machine_labels.shape
import cv2
from matplotlib import pyplot as plt
%matplotlib inline

m_labels = df_tuning_labels.labels.str.split().tolist()
#print(m_labels)
# get the descriptions and translate
map_label_to_des = dict(zip(df_class_descriptions.label_code.values, df_class_descriptions.description.values))
num_of_imgs = 16
des_labels = []
for i in np.arange(num_of_imgs):
    j = [map_label_to_des.get(item, item) for item in m_labels[i]]
    des_labels.append(j)
    
# pull images and plot
img_list = ['../input/stage_1_test_images/{}.jpg'.format(id_) for id_ in df_tuning_labels.id.values]
fig, ax = plt.subplots()
fig.set_size_inches(25, 25)
ax.set_axis_off()
for n, (image, label) in enumerate(zip(img_list, des_labels)):
    a = fig.add_subplot(num_of_imgs//4, num_of_imgs//4, n+1)
    img = cv2.imread(image, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.title(label, fontsize=15)
    plt.imshow(img)

count = pd.DataFrame(df_tuning_labels['labels'].str.split().apply(lambda x: len(x)))
print(count)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sns.countplot(data=count, x='labels')
plt.title("number of labels")

# tmp = df_tuning_labels[count['labels'] > 7]
# tmp['labels'].apply(lambda x: x.split()).values
## make a dictionary file
d={}
for i,j in zip(df_class_descriptions.label_code.values, df_class_descriptions.description.values):
    d[i]=j
d
tmp = df_tuning_labels[count['labels'] > 6]
my_list = ['../input/stage_1_test_images/{}.jpg'.format(img_id) for img_id in tmp.id.values]

ax = plt.figure(figsize=(12, 12))
for num, i in enumerate(tmp['labels'].apply(lambda x: x.split()).values):
    plt.subplot(3,2, 2*num + 1)
    plt.axis('off')
    #print(num)
    file_name = my_list[num]
    img = cv2.imread(file_name)
    plt.imshow(img)
    
    names = [d[j] for j in i]
    print(names)
    
    for n, i in enumerate(names):
        plt.text(1500,10+n*100, i, fontsize = 14, horizontalalignment='right')
        
