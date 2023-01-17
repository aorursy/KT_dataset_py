import pandas as pd

import numpy as np

import os

import time

import cv2

import matplotlib.pyplot as plt



print(os.listdir("../input"))
FOLDER = '../input/'

IMAGES = FOLDER + 'train_images/'

len(os.listdir(IMAGES))
df_train = pd.read_csv(FOLDER + 'train.csv')

# df_sub = pd.read_csv(FOLDER + 'sample_submission.csv')

unicode_map = {codepoint: char for codepoint, char in pd.read_csv(FOLDER + 'unicode_translation.csv').values}
df_train_idx = df_train.set_index("image_id")

idx_train = df_train['image_id']
df_char_train = pd.DataFrame()

start = time.time()



# for idx in idx_train: 

for idx in idx_train[:100]: # for displaying only

    label = df_train_idx.loc[idx]

    try:

        label_arr = np.array(label['labels'].split(' ')).reshape(-1, 5) # labels are configured as [unicode, x, y, w, h]

    except:

        continue

    df_char = pd.DataFrame(label_arr, columns=['unicode', 'x', 'y', 'w', 'h'])

    df_char['image_id'] = idx

    df_char_train = pd.concat([df_char_train, df_char], axis=0)

elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
unicode_arr = df_char_train['unicode'].unique()

unicode = unicode_arr[0]



print('-'*10, unicode, ' : ',  unicode_map[unicode], '-'*10)



df_code_char0 = df_char_train.query('unicode == "{}"'.format(unicode))

images_char0 = df_code_char0['image_id'].unique()



cnt = 0



num = len(images_char0)

# for n in range(num):

for n in range(3):

    

    fname = images_char0[n]

    print('-'*10, fname, '-'*10)



    image_path = IMAGES + fname + '.jpg'

    im_original = cv2.imread(image_path)

    im_original = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)

    positions = df_code_char0.query('image_id == "{}"'.format(fname))[['x', 'y', 'w', 'h']].values.astype('int')



    for pos in positions:

        x, y, w, h = pos

        im = im_original[y:y+h, x:x+w]

        plt.imshow(im) # to be canceled for saving images

        

#         cv2.imwrite("{}_{}.jpg".format(unicode, cnt), im)

#         cnt += 1

        

        plt.show() # to be canceled for saving images
# unicode_arr = df_char_train['unicode'].unique()

# for unicode in unicode_arr:

#     print('-'*10, unicode, ' : ',  unicode_map[unicode], '-'*10)



#     df_code_char0 = df_char_train.query('unicode == "{}"'.format(unicode))

#     images_char0 = df_code_char0['image_id'].unique()



#     cnt = 0

#     num = len(images_char0)

#     for n in range(num):

#         fname = images_char0[n]

#         print('-'*10, fname, '-'*10)



#         image_path = IMAGES + fname + '.jpg'

#         im_original = cv2.imread(image_path)

#         im_original = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)

#         positions = df_code_char0.query('image_id == "{}"'.format(fname))[['x', 'y', 'w', 'h']].values.astype('int')



#         for pos in positions:

#             x, y, w, h = pos

#             im = im_original[y:y+h, x:x+w]  

#             cv2.imwrite(PROCESSED_DATA + "{}_{}.jpg".format(unicode, cnt), im)

#             cnt += 1
