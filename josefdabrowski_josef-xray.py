# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



import cv2

from matplotlib import pyplot as plt

import matplotlib

from PIL import Image





path_train_pneu = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'

path_train_norm = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL'

path_val_pneu = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA'

path_val_norm = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL'

path_test_pneu = '/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA'

path_test_norm = '/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL'



path_working = '/kaggle/working'



def image_hist(pathname, histname):

    im=Image.open(pathname).convert('L')

    a = np.array(im.getdata())



    fig, ax = plt.subplots(figsize=(10,4))

    n,bins,patches = ax.hist(a, bins=range(256), edgecolor='none')

    ax.set_title(histname)

    ax.set_xlim(0,255)

    #cm = plt.cm.get_cmap('cool')

    #norm = matplotlib.colors.Normalize(vmin=bins.min(), vmax=bins.max())

    #for b,p in zip(bins,patches):

    #    p.set_facecolor(cm(norm(b)))

        

    #print(type(bins))

    #print(bins)

    #print(bins[50])

    

    plt.show()

    

def blend_imgs(path1, path2):

    im1 = Image.open(path1).convert('L')

    im2 = Image.open(path2).convert('L')

    

    im1, im2 = square_crop(im1, im2)



    new_img = Image.blend(im1, im2, 0.5)

    return new_img



def square_crop(im1, im2):

    width1, height1 = im1.size

    width2, height2 = im2.size

    

    hmin, wmin, sidelen = 0, 0, 0

    

    if height1 <= height2:

        hmin = height1

    else:

        hmin = height2

        

    if width1 <= width2:

        wmin = width1

    else:

        wmin = width2

        

    if hmin < wmin:

        sidelen = hmin

    else:

        sidelen = wmin



    first = [int((width1 - sidelen)/2), int((height1 - sidelen)/2), int((width1 + sidelen)/2), int((height1 + sidelen)/2)]

    second = [int((width2 - sidelen)/2), int((height2 - sidelen)/2), int((width2 + sidelen)/2), int((height2 + sidelen)/2)]

    

    im1 = im1.crop((first[0], first[1], first[2], first[3]))

    im2 = im2.crop((second[0], second[1], second[2], second[3]))

    

    return im1, im2



def gen_avrg_folder(path, keyword):

    foundfirst = False

    blended = False

    prev_img_path = None

    new_img = None

    count = 5

    for img in os.listdir(path):

        if count > 0 and img.find(keyword):

            print("Blending img")

            img_path = os.path.join(path_train_pneu, img)

            if not foundfirst:

                foundfirst = True

                prev_img_path = img_path

            else:

                new_img = blend_imgs(prev_img_path, img_path)

                new_img.save('blended_img.jpeg')

                prev_img_path = os.path.join(path_working, 'blended_img.jpeg')

            count -= 1

    return new_img

    

image_hist(os.path.join(path_train_pneu, 'person100_virus_184.jpeg'), 'Pneumonia, Virus')

image_hist(os.path.join(path_train_pneu, 'person1_bacteria_1.jpeg'), 'Pneumonia, Bacteria')

image_hist(os.path.join(path_train_norm, 'IM-0115-0001.jpeg'), 'Normal')



#blend_imgs(os.path.join(path_train_pneu, 'person100_virus_184.jpeg'),

          #os.path.join(path_train_pneu, 'person1_bacteria_1.jpeg')).save('test_blend.jpg')



gen_avrg_folder(path_train_pneu, 'virus').save('virus_blend.jpg')

gen_avrg_folder(path_train_pneu, 'bacteria').save('bacteria_blend.jpg')

gen_avrg_folder(path_train_norm, 'jpeg').save('normal_blend.jpg')



    

#os.path.join(dirname, filename)