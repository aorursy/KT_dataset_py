# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import os
# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         os.path.join(dirname, filename)

# # Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import os
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import scipy.ndimage as ndi
import cv2
dirname = '/kaggle/input'
train_path = os.path.join(dirname, 'chest-xray-pneumonia/chest_xray/chest_xray/train')
train_nrml_pth = os.path.join(train_path, 'NORMAL')
train_pnm_pth = os.path.join(train_path, 'PNEUMONIA')
test_path = os.path.join(dirname, 'chest-xray-pneumonia/chest_xray/chest_xray/test')
test_nrml_pth = os.path.join(test_path, 'NORMAL')
test_pnm_pth = os.path.join(test_path, 'PNEUMONIA')
val_path = os.path.join(dirname, 'chest-xray-pneumonia/chest_xray/chest_xray/test')
val_nrml_pth = os.path.join(val_path, 'NORMAL')
val_pnm_pth = os.path.join(val_path, 'PNEUMONIA')
def plot_imgs(item_dir, num_imgs=25):
    all_item_dirs = os.listdir(item_dir)
    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_imgs]

    plt.figure(figsize=(10, 10))
    for idx, img_path in enumerate(item_files):
        plt.subplot(5, 5, idx+1)

        img = plt.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)

    plt.tight_layout()
plot_imgs(train_nrml_pth)
plot_imgs(train_pnm_pth)
def plot_img_hist(item_dir, num_img=6):
    all_item_dirs = os.listdir(item_dir)
    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_img]
  
  #plt.figure(figsize=(10, 10))
    for idx, img_path in enumerate(item_files):
        fig1 = plt.figure(idx,figsize=(10, 10))
        fig1.add_subplot(2, 2, 1)
        img = mpimg.imread(img_path, )
        img = cv2. cvtColor(img, cv2. COLOR_BGR2RGB)
        plt.imshow(img)
        fig1.add_subplot(2, 2, 2)
        plt.hist(img.ravel(),bins=256, fc='k', ec='k')
    
    plt.tight_layout()
plot_img_hist(train_pnm_pth,1)
dirname_work = '/kaggle'
dir_chest_xray = os.path.join('/kaggle', 'chest_xray')
os.mkdir('/kaggle/chest_xray/')
os.mkdir('/kaggle/chest_xray/train')
os.mkdir('/kaggle/chest_xray/train/NORMAL')
os.mkdir('/kaggle/chest_xray/train/PNEUMONIA')
train_path_work = os.path.join(dir_chest_xray, 'train')
train_nrml_pth_work = os.path.join(train_path_work, 'NORMAL')
train_pnm_pth_work = os.path.join(train_path_work, 'PNEUMONIA')


os.mkdir('/kaggle/chest_xray/test')
os.mkdir('/kaggle/chest_xray/test/NORMAL')
os.mkdir('/kaggle/chest_xray/test/PNEUMONIA')
test_path_work = os.path.join(dir_chest_xray, 'test')
test_nrml_pth_work = os.path.join(test_path_work, 'NORMAL')
test_pnm_pth_work = os.path.join(test_path_work, 'PNEUMONIA')
def image_resizing(path_from, path_to, height=500, width=500):
    size = height, width
    i=1
    files = os.listdir(path_from)
    for file in files: 
        try:
            file_dir = os.path.join(path_from, file)
            file_dir_save = os.path.join(path_to, file)
            img = Image.open(file_dir)
            img = img.resize(size, Image.ANTIALIAS)
            img = img.convert("RGB")
            img.save(file_dir_save) 
            i=i+1
        except:
            continue
image_resizing(train_nrml_pth, train_nrml_pth_work, 300, 300)
image_resizing(train_pnm_pth, train_pnm_pth_work, 300, 300)
image_resizing(test_nrml_pth, test_nrml_pth_work, 300, 300)
image_resizing(test_pnm_pth, test_pnm_pth_work, 300, 300)
plot_imgs(train_nrml_pth_work)
def  hist_equal(path_from, path_to):
    i=1
    files = os.listdir(path_from)
    for file in files: 
        try:
            file_dir = os.path.join(path_from, file)
            file_dir_save = os.path.join(path_to, file)
            img = Image.open(file_dir)
            img = ImageOps.equalize(img)
            #img = img.convert("RGB") #konwersja z RGBA do RGB, usuniecie kanału alfa zeby zapisać do jpg
            img.save(file_dir_save) 
            i=i+1
        except:
            continue
hist_equal(train_pnm_pth_work, train_pnm_pth_work)
hist_equal(train_nrml_pth_work, train_nrml_pth_work)

hist_equal(test_pnm_pth_work, test_pnm_pth_work)
hist_equal(test_nrml_pth_work, test_nrml_pth_work)
plot_img_hist(train_pnm_pth_work, 2)
