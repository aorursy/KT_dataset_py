!mkdir ./source-frames
import matplotlib.pyplot as plt

from PIL import Image

import cv2

import numpy as np



source = '/kaggle/input/ldv1960/LaDolceVita.mp4'

target_path ="./source-frames/"



vidcap = cv2.VideoCapture(source)

fps = vidcap.get(cv2.CAP_PROP_FPS)

print (f"Frames per second: {fps}")

codec = vidcap.get(cv2.CAP_PROP_FOURCC);

print (f"codec: {codec}")



success,image = vidcap.read()

image = image[150:550,:,:]

count = 0

j=0

while success:

    frame = f"frame{count:05d}.jpg"

    cv2.imwrite(target_path + frame, image)     # save frame as JPEG file      

    success,image = vidcap.read()

    if success:

        image = image[150:550,:,:]



        if count % 250 == 0:

            if j==0:

                fig = plt.figure(figsize= (20, 5))

                axes = fig.subplots(nrows=1, ncols=4)



            axes[j].imshow(image)

            axes[j].title.set_text(frame)

            j =(j +1) % 4

            if j==0:

                plt.show()

    count += 1

print(f"# frames:{count}")
!git clone --branch inference-tf-2.x https://github.com/steubk/White-box-Cartoonization.git

!pip install --upgrade tf_slim
import sys

sys.path.append('./White-box-Cartoonization/test_code')



import cartoonize

import os
model_path = './White-box-Cartoonization/test_code/saved_models'

load_folder = "./source-frames"

save_folder = './cartoonized_images'

if not os.path.exists(save_folder):

    os.mkdir(save_folder)

    

cartoonize.cartoonize(load_folder, save_folder, model_path)
name_list = sorted(os.listdir(save_folder))



for i,name in enumerate(name_list):

  

  if i % 250 == 0 and i > 0: 

    fig = plt.figure(figsize= (30, 15))

    axes = fig.subplots(nrows=1, ncols=2)



    image = load_folder + "/" + name

    cartoon = save_folder + "/" + name

    im = Image.open(image) 

    im_array = np.asarray(im)

    axes[0].imshow(im_array)



    im = Image.open(cartoon) 

    im_array = np.asarray(im)

    axes[1].imshow(im_array)

    plt.show()
!ffmpeg -y -framerate 30  -i ./cartoonized_images/frame%5d.jpg  output.mp4
from IPython.display import Video



Video("output.mp4", embed=True)
!rm ./source-frames/*.jpg

!rm ./cartoonized_images/*.jpg

!rm ./White-box-Cartoonization/test_code/test_images/*.jpg