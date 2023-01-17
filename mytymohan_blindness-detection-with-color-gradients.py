import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



sns.set()



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
train_labels = pd.read_csv("../input/train.csv")

print(train_labels.head())



print("There are {0} samples in the Training dataset".format(train_labels.shape[0]))
test_labels = pd.read_csv("../input/test.csv")

print("We need to predict {0} patients as what severity of diabetic retinopathy they have".format(test_labels.shape[0]))
from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())



diag_q = """

select diagnosis, count(distinct id_code) as cnt

From train_labels

GROUP BY diagnosis;

"""



diag_df = pysqldf(diag_q)



import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



fig = {

  "data": [

    {

      "values": diag_df.cnt,

      "labels": diag_df.diagnosis,

      "domain": {"x": [0, .5]},

      "hoverinfo":"label+percent",

      "hole": .2,

      "type": "pie"

    },],

 "layout": {

        "title":"Severity Proportion of Diabetic Retinopathy"

    }

}



iplot(fig)
from IPython.display import Image

from IPython.display import display

im_0 = Image(filename ='../input/train_images/002c21358ce6.png') 

im_1 = Image(filename ='../input/train_images/0024cdab0c1e.png')

im_2 = Image(filename ='../input/train_images/000c1434d8d7.png')

im_3 = Image(filename ='../input/train_images/0104b032c141.png')

im_4 = Image(filename ='../input/train_images/02685f13cefd.png')

display(im_0, im_1, im_2, im_3, im_4)
DATA_PATH = '../input/'

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train_images')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test_images')

TRAIN_LABEL_PATH = os.path.join(DATA_PATH, 'train.csv')

TEST_LABEL_PATH = os.path.join(DATA_PATH, 'test.csv')



train_df = pd.read_csv(TRAIN_LABEL_PATH)

test_df = pd.read_csv(TEST_LABEL_PATH)



train_labels_0 = train_df[train_df.diagnosis == 0].reset_index()

train_labels_1 = train_df[train_df.diagnosis == 1].reset_index()

train_labels_2 = train_df[train_df.diagnosis == 2].reset_index()

train_labels_3 = train_df[train_df.diagnosis == 3].reset_index()

train_labels_4 = train_df[train_df.diagnosis == 4].reset_index()
%matplotlib inline

from PIL import Image

import matplotlib.pyplot as plt

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#0 - No DR

for i in range(5):

    img_path = TRAIN_IMG_PATH+'/'+train_labels_0['id_code'][i]+'.png'

    img = Image.open(img_path)

    img.thumbnail((200,200))

    ax[i].imshow(img)

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 0 - No DR")

plt.show()
from PIL import Image

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False

#1 - Mild DR

for i in range(5):

    img_path = TRAIN_IMG_PATH+'/'+train_labels_1['id_code'][i]+'.png'

    img = Image.open(img_path)

    img.thumbnail((200,200))

    ax[i].imshow(img)

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 1 - Mild DR")

plt.show()
from PIL import Image

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#2 - Moderate DR

for i in range(5):

    img_path = TRAIN_IMG_PATH+'/'+train_labels_2['id_code'][i]+'.png'

    img = Image.open(img_path)

    img.thumbnail((200,200))

    ax[i].imshow(img)

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 2 - Moderate DR")

plt.show()
from PIL import Image

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#3 - Severe DR

for i in range(5):

    img_path = TRAIN_IMG_PATH+'/'+train_labels_3['id_code'][i]+'.png'

    img = Image.open(img_path)

    img.thumbnail((200,200))

    ax[i].imshow(img)

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 3 - Severe DR")

plt.show()
from PIL import Image

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#4 - Proliferative DR

for i in range(5):

    img_path = TRAIN_IMG_PATH+'/'+train_labels_4['id_code'][i]+'.png'

    img = Image.open(img_path)

    img.thumbnail((200,200))

    plt.title(train_labels_4['id_code'][i])

    ax[i].title.set_text(train_labels_4['id_code'][i])

    ax[i].imshow(img)

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 4 - Proliferative DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#0 - No DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_0['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="jet")

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 0 - No DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#1 - Mild DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_1['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="jet")

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 1 - Mild DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#2 - Moderate DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_2['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="jet")

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 2 - Moderate DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#3 - Severe DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_3['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="jet")

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 3 - Severe DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#4 - Proliferative DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_4['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="jet")

    ax[i].set_axis_off()  

print("Diabetic Retinopathy of Severity 4 - Proliferative DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#0 - No DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_0['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="PiYG")

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 0 - No DR")

plt.show()

import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#1 - Mild DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_1['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="PiYG")

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 1 - Mild DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#2 - Moderate DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_2['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="PiYG")

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 2 - Moderate DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#3 - Severe DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_3['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="PiYG")

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 3 - Severe DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#4 - Proliferative DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_4['id_code'][i]+'.png',0)

    edges = cv2.Canny(img,100,200)

    plt.imshow(img)

    ax[i].imshow(img, cmap="PiYG")

    ax[i].set_axis_off()  

print("Diabetic Retinopathy of Severity 4 - Proliferative DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#0 - No DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_0['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="gray")

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 0 - No DR")

plt.show()

import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#1 - Mild DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_1['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="gray")

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 1 - Mild DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#2 - Moderate DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_2['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="gray")

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 2 - Moderate DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#3 - Severe DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_3['id_code'][i]+'.png',0)

    plt.imshow(img)

    ax[i].imshow(img, cmap="gray")

    ax[i].set_axis_off() 

print("Diabetic Retinopathy of Severity 3 - Severe DR")

plt.show()
import cv2

f,ax = plt.subplots(1,5, figsize=(15,15))

plt.rcParams["axes.grid"] = False



#4 - Proliferative DR

for i in range(5):

    img = cv2.imread(TRAIN_IMG_PATH+'/'+train_labels_4['id_code'][i]+'.png',0)

    edges = cv2.Canny(img,100,200)

    plt.imshow(img)

    ax[i].imshow(img, cmap="gray")

    ax[i].set_axis_off()  

print("Diabetic Retinopathy of Severity 4 - Proliferative DR")

plt.show()