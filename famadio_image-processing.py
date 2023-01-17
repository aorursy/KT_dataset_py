import numpy as np

import cv2

from PIL import Image

import os
print('Dataset Folders ',(os.listdir("../input/kaggle/")))

print('Train ',len(os.listdir("../input/kaggle/Train")))

print('Test ',len(os.listdir("../input/kaggle/test")))

print('Images not identified ',len(os.listdir("../input/kaggle/non_identified")))

print('treated 1 ',len(os.listdir("../input/kaggle/treated_1")))

print('treated 2 ',len(os.listdir("../input/kaggle/treated_2")))

print('treated 3 ',len(os.listdir("../input/kaggle/treated_3")))

print('treated 4 ',len(os.listdir("../input/kaggle/treated_4")))
im = Image.open('../input/kaggle/Train/3as1.png')

im
# -------------------

#    Treatment 1

#

#  Remove Background

# -------------------





for i, pic in enumerate(os.listdir('../input/kaggle/Train')):

        # Read image as RGBA

        im = Image.open(os.path.join('../input/kaggle/Train/', pic))

        #print(pic)

        im = im.convert('RGBA')

        data = np.array(im)

        

        # just use the rgb values for comparison

        rgb = data[:,:,:3]

        color = [192, 192, 192]   # Original value - color to be changed

        white = [255,255,255,255] # Color to change - new color

        mask = np.all(rgb == color, axis = -1)



        # change all pixels that match color to white

        data[mask] = white



        #set of colors to change

        rgb = data[:,:,:3]

        color = [240, 248, 255]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [190, 190, 190]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)

        

        rgb = data[:,:,:3]

        color = [191, 191, 191]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [253, 245, 230]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [1, 255, 255]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [1, 254, 254]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)





        rgb = data[:,:,:3]

        color = [1, 253, 253]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)





        rgb = data[:,:,:3]

        color = [224, 255, 255]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [240, 248, 255]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [255, 228, 225]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [240, 248, 255]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [252, 244, 229]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [218, 112, 146]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [156, 188, 156]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [254, 247, 219]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [0, 255, 255]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [0, 254, 254]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [251, 243, 228]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [230, 230, 230]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)

        

        rgb = data[:,:,:3]

        color = [173, 173, 173]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)

        

        rgb = data[:,:,:3]

        color = [254, 227, 224]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [223, 254, 254]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [239, 247, 254]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [255, 248, 220]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [217, 111, 146]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [253, 246, 218]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [238, 246, 253]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)



        rgb = data[:,:,:3]

        color = [219, 112, 147]   # Original value

        mask = np.all(rgb == color, axis = -1)

        data[mask] = white

        new_im = Image.fromarray(data)

        

        # save treated image as the same file name to mantain the text solved

        # new_im.save(os.path.join('../input/kaggle/treated_1/', pic))

        # don't know how to save files here in kaggle



im = Image.open('../input/kaggle/treated_1/3as1.png')

im
# -------------------

#    Treatment 2

# -------------------



for i, pic in enumerate(os.listdir('../input/kaggle/treated_1/')):

        # Read image as RGBA

        image = cv2.imread(os.path.join('../input/kaggle/treated_1/', pic))

        #print(pic)



        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU )[1]

        gray = cv2.medianBlur(gray, 3)



        # don't know how to save files here in kaggle

        # cv2.imwrite(os.path.join('../input/kaggle/treated_2/', pic), gray)



im = Image.open('../input/kaggle/treated_2/3as1.png')

im
# -------------------

#    Treatment 3

# -------------------





for i, pic in enumerate(os.listdir('../input/kaggle/treated_1/')):

        # Read image as RGBA

        image = cv2.imread(os.path.join('../input/kaggle/treated_1/', pic))

        #print(pic)



        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)



        # don't know how to save files here in kaggle

        # cv2.imwrite(os.path.join('../input/kaggle/treated_2/', pic), gray)



im = Image.open('../input/kaggle/treated_3/3as1.png')

im
# -------------------

#    Treatment 4

#

#  Isolate letters and numbers

# -------------------





for i, pic in enumerate(os.listdir('../input/kaggle/treated_1/')):

        # Read image as RGBA

        frame = cv2.imread(os.path.join('../input/kaggle/treated_1/', pic))

        # print(pic)



        lower = np.array([230,230,230])

        upper = np.array([255,255,255])

        my_mask = cv2.inRange(frame, lower, upper)



        gray = cv2.threshold(my_mask, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        gray = cv2.medianBlur(gray, 3)

        

        # don't know how to save files here in kaggle

        # cv2.imwrite(os.path.join('../input/kaggle/treated_4/', pic), gray)



im = Image.open('../input/kaggle/treated_4/3as1.png')

im