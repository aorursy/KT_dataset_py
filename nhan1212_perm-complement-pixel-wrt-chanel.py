import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt

data_dir = '/kaggle/input/image-grid'
data_file = os.listdir(data_dir)
data_file.sort()
data_file
n = len(data_file)
fig, ax = plt.subplots(1, 4, figsize=(25, 5.5))
for k in range(n):
    img = cv2.imread(os.path.join(data_dir, data_file[k]))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ax[k].imshow(img, aspect="auto")
    ax[k].set_title('original image : "%s"'%(data_file[k]))
plt.show()
def replace_pixel_chanel(ID, chanel_1, chanel_2):
    """
        This function return the plug_in_chanel by another chanel
        Input args:
            ID (str): path to image with the original_order_chanel = [0, 1, 2]
            chanel_1, chanel_2 (int of {0, 1, 2}): is the order_of_plug_in_chanels. 
                         For example: (chanel_1, chanel_2) = (0, 2); meant replace all pixel in chanel[0](red) to chanel[2](blue)
            Noting that (chanel_1, chanel_2) = (0, 2) is not the same with (chanel_1, chanel_2) = (2, 0)
    """
    img = cv2.imread(ID)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img[:, :, chanel_1] = img[:, :, chanel_2]
    
    plt.imshow(img)
    plt.title('plug all pixel chanels_%s in chanel_%s'%(chanel_1, chanel_2))
    plt.axis('off')
ID1 = os.path.join(data_dir, data_file[0])
plt.figure(figsize = (20, 15))
for i in range(3):
    for j in range(3): 
        plt.subplot(3, 3, 3*i + 1 + j)
        replace_pixel_chanel(ID1, i, j)
ID2 = os.path.join(data_dir, data_file[2])
plt.figure(figsize = (20, 15))
for i in range(3):
    for j in range(3): 
        plt.subplot(3, 3, 3*i + 1 + j)
        replace_pixel_chanel(ID2, i, j)
def perm_img_chanel(ID, perm_ls):
    """
        This function return the plug_in_chanel by another chanel
        Input args:
            ID (str): path to image with the original_order_chanel = [0, 1, 2]
            perm_ls (list of 3 intergers)
    """
    img = cv2.imread(ID)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for k in range(3):
        img[:, :, k] = img[:, :, perm_ls[k]]
    plt.imshow(img)
    plt.title('perm_list = %s'%(perm_ls))
    plt.axis('off')
idx = 0
plt.figure(figsize = (25, 39))
for i in range(3):
    for j in range(3):
        for k in range(3):
            plt.subplot(9, 3, idx + 1)
            perm_img_chanel(ID1, [i, j, k])
            idx += 1
idx = 0
plt.figure(figsize = (25, 15))
for i in range(3):
    for j in range(3):
        for k in range(3):
            plt.subplot(3, 9, idx + 1)
            perm_img_chanel(ID2, [i, j, k])
            idx += 1
def compl_img_chanel(ID, chanels):
    """
        This function returns the complement_of_pixels in the chanels
        Input_args:
            ID : path to image
            chanels (list of intergers of {0, 1, 2}): chanels of the image
    """
    img = cv2.imread(ID)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    n = len(chanels)
    for k in range(n):
        img[:, :, chanels[k]] = ~img[:, :, chanels[k]]
    plt.imshow(img)
    plt.title('chanel_complement = %s'%(chanels))
    plt.axis('off')
idx = 0
plt.figure(figsize = (25, 45))
for i in range(3):
    for j in range(3):
        for k in range(3):
            plt.subplot(9, 3, idx + 1)
            compl_img_chanel(ID1, [i, j, k])
            idx += 1
idx = 0
ID3 = os.path.join(data_dir, data_file[-1])
plt.figure(figsize = (25, 45))
for i in range(3):
    for j in range(3):
        for k in range(3):
            plt.subplot(9, 3, idx + 1)
            compl_img_chanel(ID3, [i, j, k])
            idx += 1
def perm_and_compl(ID, perm_ls):
    """
        This function return the plug_in_chanel by another chanel
        Input args:
            ID (str): path to image with the original_order_chanel = [0, 1, 2]
            perm_ls (list of 3 intergers)
    """
    img = cv2.imread(ID)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for k in range(3):
        img[:, :, k] = ~img[:, :, perm_ls[k]]
    plt.imshow(img)
    plt.title('Complement(perm_list = %s)'%(perm_ls))
    plt.axis('off')
idx = 0
plt.figure(figsize = (25, 45))
for i in range(3):
    for j in range(3):
        for k in range(3):
            plt.subplot(9, 3, idx + 1)
            perm_and_compl(ID1, [i, j, k])
            idx += 1