# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from PIL import Image



import matplotlib.animation as animation

import numpy as np

import matplotlib.pyplot as plt

import cv2

import pydicom as dicom

import imghdr

import pylab as pl

import os

        

        

        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

# Any results you write to the current directory are saved as output.
path = '/kaggle/input/basic-images/'

images = ["cta_scan_index.bmp", "mri.jpg", "retinaRGB.jpg", "abdomen.png", "cameraman.tif", 

          "house.tif", "lake.tif", "lena_color_512.tif", "peppers_color.tif", "rxpie-rodilla.tif"]



# Lectura de imágenes dcm

ds = dicom.dcmread(path + 'Anonymized20200210.dcm')

data = ds.pixel_array

print("Tipo de imagen: {}, tamaño: {}x{}, tipo de dato: {}"

      .format(imghdr.what(path + 'Anonymized20200210.dcm'), 

             data.shape[0], data.shape[1],

             type(ds)))

plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

plt.show()



# Lectura de imágenes Tiff

for image in images:

    img = cv2.imread(path + image,1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("Tipo de imagen: {}, tamaño: {}, tipo de dato: {}"

      .format(imghdr.what(path + image), 

             img.shape,

             type(img)))

    plt.imshow(img)

    plt.show()

    

    # con opencv

    #cv2.imshow('image',img)

    #cv2.waitKey(0)

    #cv2.destroyAllWindows()

image = cv2.imread(path + 'peppers_color.tif') 

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

 

#Rango de colores detectados:

#Verdes:

green_low = np.array([49,50,50])

green_high = np.array([107, 255, 255])



mask_green = cv2.inRange(hsv, green_low, green_high)



#Azules:

blue_low = np.array([100,65,75], dtype=np.uint8)

blue_high = np.array([130, 255, 255], dtype=np.uint8)



mask_blue = cv2.inRange(hsv, blue_low, blue_high)



#Rojos:

red_low1 = np.array([0,65,75])

red_high1 = np.array([12, 255, 255])

red_low2 = np.array([240,65,75])

red_high2 = np.array([256, 255, 255])

mask_red1 = cv2.inRange(hsv, red_low1, red_high1)

mask_red2 = cv2.inRange(hsv, red_low2, red_high2)



mask = cv2.add(mask_red1, mask_red2)

mask = cv2.add(mask, mask_green)

mask = cv2.add(mask, mask_blue)

image_rgb = cv2.bitwise_and(image, image, mask=mask)



plt.imshow(image_rgb)

plt.show()
escalas = [cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2YUV, cv2.COLOR_RGB2HSV]

for escala in escalas:

    img = cv2.imread(path + "retinaRGB.jpg",1)

    img = cv2.cvtColor(img, escala)

    plt.imshow(img)

    plt.show()
img = cv2.imread(path + "cta_scan_index.bmp",1)

firsth = int(img.shape[0]/3)

firstw = int(img.shape[1]/3)



h = int(img.shape[0] - firsth)

w = int(img.shape[1] - firstw)



print("Esquina inferior derecha:")

img = img[h:, w:]

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)

plt.show()



print("Parte central:")

img = cv2.imread(path + "cta_scan_index.bmp",1)

img = img[firsth:h, firstw:w]

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)

plt.show()
img = cv2.imread(path + 'lena_color_512.tif',1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows,cols,_ = img.shape



for degrees in [45, 90, 180]:

    print("Lena in {} degrees:".format(degrees))

    M = cv2.getRotationMatrix2D((cols/2,rows/2),degrees,1)

    dst = cv2.warpAffine(img,M,(cols,rows))



    plt.imshow(dst)

    plt.show()
fig = plt.figure()



list_images = []

for dirname, _, filenames in os.walk("/kaggle/input/basic-images/DICOM/"):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        ds = dicom.dcmread(path)

        im = plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

        list_images.append([im])





ani = animation.ArtistAnimation(fig, list_images, 

                                interval=10, blit=True, repeat_delay=1000)



# ani.save('basic_animation.gif', fps=30)

plt.show()