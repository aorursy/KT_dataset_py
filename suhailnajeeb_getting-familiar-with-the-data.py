# Importing necessary libraries
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cv2
# Declare constants which will be used while plotting the data
FS_AXIS_LABEL=14
FS_TITLE=17
FS_TICKS=12
FIG_WIDTH=20
ROW_HEIGHT=2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
data_dir=os.path.join('..','input')
print(os.listdir(data_dir))
paths_train_a=glob.glob(os.path.join(data_dir,'training-a','*.png'))
paths_train_b=glob.glob(os.path.join(data_dir,'training-b','*.png'))
paths_train_e=glob.glob(os.path.join(data_dir,'training-e','*.png'))
paths_train_c=glob.glob(os.path.join(data_dir,'training-c','*.png'))
paths_train_d=glob.glob(os.path.join(data_dir,'training-d','*.png'))

paths_test_a=glob.glob(os.path.join(data_dir,'testing-a','*.png'))
paths_test_b=glob.glob(os.path.join(data_dir,'testing-b','*.png'))
paths_test_e=glob.glob(os.path.join(data_dir,'testing-e','*.png'))
paths_test_c=glob.glob(os.path.join(data_dir,'testing-c','*.png'))
paths_test_d=glob.glob(os.path.join(data_dir,'testing-d','*.png'))
paths_test_f=glob.glob(os.path.join(data_dir,'testing-f','*.png'))+glob.glob(os.path.join(data_dir,'testing-f','*.jpg'))
paths_test_auga=glob.glob(os.path.join(data_dir,'testing-auga','*.png'))
paths_test_augc=glob.glob(os.path.join(data_dir,'testing-augc','*.png'))
path_label_train_a=os.path.join(data_dir,'training-a.csv')
path_label_train_b=os.path.join(data_dir,'training-b.csv')
path_label_train_e=os.path.join(data_dir,'training-e.csv')
path_label_train_c=os.path.join(data_dir,'training-c.csv')
path_label_train_d=os.path.join(data_dir,'training-d.csv')
def get_img(path,mode=cv2.IMREAD_GRAYSCALE):
    # get image data
    img=cv2.imread(path)
    # opencv stores color images in BGR format by default, so transforming to RGB colorspace
    if len(img.shape)>2:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if img is not None:
        return img 
    else:
        raise FileNotFoundError('Image does not exist at {}'.format(path))
def imshow_group(paths,n_per_row=10):
    # plot multiple digits in one figure, by default 10 images are plotted per row
    n_sample=len(paths)
    j=np.ceil(n_sample/n_per_row)
    fig=plt.figure(figsize=(FIG_WIDTH,ROW_HEIGHT*j))
    for i, path in enumerate(paths):
        img=get_img(path)
        plt.subplot(j,n_per_row,i+1)
        if len(img.shape)<3: # if grayscale
            plt.imshow(img,cmap='gray')  
        else:
            plt.imshow(img)  
        plt.axis('off')
    return fig
def get_key(path):
    # separate the key from the filepath of an image
    return path.split(sep=os.sep)[-1]
paths=np.random.choice(paths_train_a,size=40)
fig=imshow_group(paths)
fig.suptitle('Samples from {} training images in dataset a'.format(len(paths_train_a)), fontsize=FS_TITLE)
plt.show()
paths=np.random.choice(paths_train_b,size=40)
fig=imshow_group(paths)
fig.suptitle('Samples from {} training images in dataset b'.format(len(paths_train_b)), fontsize=FS_TITLE)
plt.show()
paths=np.random.choice(paths_train_c,size=40)
fig=imshow_group(paths)
fig.suptitle('Samples from {} training images in dataset c'.format(len(paths_train_c)), fontsize=FS_TITLE)
plt.show()
paths=np.random.choice(paths_train_d,size=40)
fig=imshow_group(paths)
fig.suptitle('Samples from {} training images in dataset d'.format(len(paths_train_d)), fontsize=FS_TITLE)
plt.show()
paths=np.random.choice(paths_train_e,40)
fig=imshow_group(paths)
fig.suptitle('Samples from {} training images in dataset e'.format(len(paths_train_e)), fontsize=FS_TITLE)
plt.show()
paths=np.random.choice(paths_test_f,size=40)
fig=imshow_group(paths)
fig.suptitle('Samples from {} training images in dataset f'.format(len(paths_test_f)), fontsize=FS_TITLE)
plt.show()
paths=np.random.choice(paths_test_auga,size=40)
fig=imshow_group(paths)
fig.suptitle('Samples from {} training images in dataset auga'.format(len(paths_test_auga)), fontsize=FS_TITLE)
plt.show()
paths=np.random.choice(paths_test_augc,size=40)
fig=imshow_group(paths)
fig.suptitle('Samples from {} training images in dataset augc'.format(len(paths_test_augc)), fontsize=FS_TITLE)
plt.show()