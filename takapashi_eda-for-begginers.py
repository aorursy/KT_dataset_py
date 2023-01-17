# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import cv2



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



train_filename_list = []



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        train_filename_list.append(os.path.join(dirname, filename))

#         print(os.path.join(dirname, filename))



print(train_filename_list[0])

train_filename_list = train_filename_list[2:]





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import train data

train_df = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")



# Data path

data_path = '/kaggle/input/global-wheat-detection/train/'



train_df.head()
# get unique list of image_id

train_img_ids = train_df["image_id"].unique()

train_img_ids
# display wheat image and boundingbox

def show_img(img_id):

    img = cv2.cvtColor(cv2.imread(data_path + train_img_ids[img_id] + '.jpg'), cv2.COLOR_BGR2RGB)

    # plt.imshow(img)



    bboxes = train_df[train_df.image_id==train_img_ids[img_id]].bbox.tolist()

    J = img.copy()

    for i in range(len(bboxes)):

        x = int(str(bboxes[i][1:-1]).split(',')[0][:-2])

        y = int(str(bboxes[i][1:-1]).split(',')[1][1:-2])

        xw = x + int(str(bboxes[i][1:-1]).split(',')[2][1:-2])

        yh = y + int(str(bboxes[i][1:-1]).split(',')[3][1:-2])

        cv2.rectangle(J,(x,y),(xw,yh),(180,190,0),4)

    plt.imshow(J);

    title = "image id : " + str(img_id)

    plt.title(title)

        



num_figs = 51 # number of images

num_cols = 5

num_rows = num_figs // num_cols + 1 if num_figs % num_cols else num_figs // num_cols



plt.figure(figsize=(30,60))

for i in range(1,num_figs+1):

    plt.subplot(num_rows,num_cols,i)

    plt.grid(False)

    show_img(i)