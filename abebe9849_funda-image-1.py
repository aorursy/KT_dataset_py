import glob

import matplotlib.pyplot as plt

%matplotlib inline



path_list=glob.glob("/kaggle/input/glaucomadataset/Glaucoma/*")

print(len(path_list))

print(path_list[0])
import cv2

import skimage.io
for i in range(500):

    img = skimage.io.MultiImage(path_list[i])[0]

    plt.imshow(img)

    plt.show()
n_path_list=glob.glob("/kaggle/input/glaucomadataset/Non Glaucoma/*")

print(len(n_path_list))

print(n_path_list[0])



for i in range(len(n_path_list)):

    img = skimage.io.MultiImage(n_path_list[i])[0]

    plt.imshow(img)

    plt.show()