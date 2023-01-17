# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

from skimage import io
im = io.imread('../input/sentinel18/sentinel2018/Sentinel2018/S2B_MSIL1C_20180207T030859_N0206_R075_T48PYU_20180207T090608.tif')

COORD = [(4489, 1,    6625, 637),

         (6857, 201,  7147, 465),

         (4729, 577,  5049, 921),

         (5441, 569,  5905, 1097),

         (6961, 593,  7553, 1193),

         (5009, 961,  5649, 1601),

         (4369, 1233, 4889, 1783),

         (4833, 1561, 5761, 2569),

         (5801, 1753, 6305, 2385),

         (4169, 2265, 5033, 3697),

         (6225, 1217, 6721, 1783),

         (5265, 2945, 6289, 3553),

         (4729, 3529, 5521, 4353),

         (5593, 3457, 6689, 5201),

         (6537, 3121, 7561, 5841),

         (2169, 3713, 2745, 4329),

         (921,  3641, 1745, 4337),

         (2585, 4025, 4361, 5081),

         (4617, 4425, 5689, 5521),

         (593,  3897, 1753, 6241),

         (1921, 4585, 2441, 5217),

         (1617, 5361, 2497, 5729),

         (1985, 5965, 3065, 6713),

         (3373, 5325, 4129, 6037),

         (4601, 5461, 5473, 5853),

         (6129, 5085, 6657, 5813),

         (3505, 6181, 4377, 7317),

         (5449, 6005, 6985, 6781),

         (2137, 7005, 3273, 7621),

         (4617, 6709, 6177, 7717),

         (6497, 6669, 7321, 7565),

         (2673, 7861, 3265, 8973),

         (3113, 7437, 4953, 9125),

         (4793, 8221, 5657, 9061),

         (2665, 9069, 3201, 9829),

         (3241, 8973, 4897, 9525),

         (6321, 7565, 7505, 8733)]
palette = {0: (255, 255, 255),

           1: (0, 0, 255),

           2: (0, 255, 255),

           3: (0, 255, 0),

           4: (255, 255, 0),

           5: (255, 0, 0),

           6: (0, 0, 0),

           7: (255, 0, 255)

}



invert_palette = {v: k for k, v in palette.items()}
import numpy as np

from skimage import io



GT = "../input/sample2018/sample2018_PYU.tif"



def convert_to_color(arr_2d, palette=palette):

    """ Numeric labels to RGB-color encoding """

    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)



    for c, i in palette.items():

        m = arr_2d == c

        arr_3d[m] = i



    return arr_3d





def convert_from_color(arr_3d, palette=invert_palette):

    """ RGB-color encoding to gray scale labels """

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)



    for c, i in palette.items():

        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)

        arr_2d[m] = i



    return arr_2d





gt = io.imread(GT)

gt = np.where(gt > 7, 0, gt)

color_gt = convert_to_color(gt)

io.imsave('color_gt.png', color_gt)



import matplotlib.pyplot as plt



def plot_pie(n_pixel):

    labels_ = ['Con lai {}', 'Mat nuoc {}', 'Rung rung la {}', 'Rung trong {}', 'Rung thuong xanh {}',

               'Rung non {}', 'Rung hon giao {}', 'Dan cu {}']

    colors = ['gray', 'blue', 'cyan', 'green', 'yellow', 'red', 'black', 'violet']

    explode = (0, 0, 0, 0, 0, 0, 0, 0)  # explode 1st slice



    total_pixel = np.sum(np.array(n_pixel))

    i = 0

    labels = []

    for l in labels_:

        labels.append(l.format(n_pixel[i] / total_pixel))

        i += 1



    # Plot

    plt.pie(n_pixel, explode=explode, labels=labels, colors=colors)

    plt.axis('equal')

    plt.show()





fig = plt.figure()

plt.subplot(1, 1, 1)

plt.imshow(color_gt)

plt.title('ground truth')

plt.show()



array_gt = gt.reshape(83228400)

n_pixel = [0, 0, 0, 0, 0, 0, 0, 0]

for i in array_gt:

    n_pixel[i] += 1

plot_pie(n_pixel)

print(n_pixel) # [27479094, 157293, 0, 13868, 366927, 721287, 204145, 7600]
patch = [io.imread(GT)[x[1]:x[3], x[0]:x[2]] for x in COORD]



len(patch)
i = 1

for p in patch:

    fig = plt.figure()

    plt.subplot(1, 1, 1)

    plt.imshow(p)

    plt.title('ground truth')

    plt.show()

    i = i + 1
i = 1

for p in patch:

    print(p.shape)
n_pixel = [0, 0, 0, 0, 0, 0, 0, 0]

for p in patch:

    array_p = p.reshape(p.shape[0]*p.shape[1])

    for i in array_p:

        n_pixel[i] += 1





plot_pie(n_pixel)

print(n_pixel)
import csv

from IPython.display import clear_output



im_patch = [im[:,c[1]:c[3],c[0]:c[2]] for c in COORD]



with open('imgtocsv.csv', mode='w+') as file:

    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    limit = 0

    for k in range(0,len(patch)):

        p_k = patch[k]

        p_k = p_k.reshape(p_k.shape[0]*p_k.shape[1]) 

        im_p_k = im_patch[k]

        im_p_k = im_p_k.transpose((1,2,0)).reshape(im_p_k.shape[1]*im_p_k.shape[2],13)

        for i in range(0, len(p_k)):

            if p_k[i] <= 7:

                if p_k[i] > 0:

                    tmp = np.append(im_p_k[i],p_k[i])

                    #print('Writing line {0}: {1}'.format(i, tmp))

                    writer.writerow(tmp)

                elif limit < 1000000:

                    tmp = np.append(im_p_k[i],p_k[i])

                    #print('Writing line {0}: {1}'.format(i, tmp))

                    writer.writerow(tmp)

                    limit += 1                 

file.close()
import pandas as pd

file_name = "./imgtocsv.csv"

file_name_output = "imgtocsv_without_dupes.csv"



df = pd.read_csv(file_name,header=None)

df
df.describe()
df1 = df.drop_duplicates()

df1
df1.to_csv(file_name_output)