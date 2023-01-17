from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
! ls -l ../input/lego
! ls -l ../input/lego/01-Inventory/
image = mpimg.imread("../input/lego/01-Inventory/inventory.jpg")
plt.figure(figsize=(15, 15))
plt.imshow(image)
plt.show()
! ls -l ../input/lego/02-MessyInventory/
image = mpimg.imread("../input/lego/02-MessyInventory/messy_inventory.jpg")
plt.figure(figsize=(15, 15))
plt.imshow(image)
plt.show()
! ls -l ../input/lego/05-Pieces_Dataset/
pictures_subset = "../input/lego/05-Pieces_Dataset/01-Subdataset_Lego"
images = [pictures_name for pictures_name in os.listdir(pictures_subset) if 'jpg' in pictures_name]
print("There are {} available images for this single piece.".format(len(images)))
fig, axes = plt.subplots(9, 9, figsize=(15, 15))
for i in range(81):
    image = mpimg.imread(pictures_subset + '/' + images[i])
    axes[i//9, i%9].imshow(image)
plt.show()
! ls -l ../input/lego/06-Partial_Assembly_Dataset
# example for the 4th assembly
pictures_subset = "../input/lego/06-Partial_Assembly_Dataset/04-Partial_assembly/"
images = [pictures_name for pictures_name in os.listdir(pictures_subset) if 'jpg' in pictures_name]
print("Number of available images: {}".format(len(images)))
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for i in range(9):
    image = mpimg.imread(pictures_subset + '/' + images[i])
    axes[i//3, i%3].imshow(image)
plt.show()
def format_string(i):
    p = str(i)
    return '0'*(2-len(p))+p
path_assembly = '../input/lego/06-Partial_Assembly_Dataset/'
folder_names = sorted([name+'/' for name in os.listdir(path_assembly) if 'Partial' in name])
file_names = []
for folder_name in folder_names:
    pictures_subset = path_assembly + folder_name
    images = [pictures_name for pictures_name in os.listdir(pictures_subset) if 'jpg' in pictures_name]
    image_name = pictures_subset + images[0]
    file_names.append(image_name)
count = 0
for image_name in file_names:
    count += 1
    plt.figure(figsize=(5, 5))
    image = plt.imread(image_name)
    plt.title('Assembly step n°{}'.format(count))
    plt.imshow(image)
    plt.show()
for folder in sorted(os.listdir('../input/lego')):
    print(folder)
# Exploring the single pieces
path = '../input/lego/05-Pieces_Dataset/'
for folder in sorted([i for i in os.listdir(path) if 'Lego' in i]):
    print('Number of picture for {}: {}'.format(folder, len(os.listdir(path + folder))))
# Exploring the single pieces
path = '../input/lego/06-Partial_Assembly_Dataset/'
for folder in sorted([i for i in os.listdir(path) if 'Partial' in i]):
    print('Number of picture for {}: {}'.format(folder, len(os.listdir(path + folder))))