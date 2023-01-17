# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from scipy.io import loadmat

import numpy as np

from matplotlib import pyplot as plt

from glob import glob

from os import path
mat_fps = glob(path.join('/kaggle/input/chiu-2015/2015_BOE_Chiu/', '*.mat'))
mat = loadmat(mat_fps[0])
fluid_class = 9
def get_valid_idx(manualLayer):

    idx = []

    for i in range(0,61):

        temp = manualLayer[:,:,i]

        if np.sum(temp) != 0:

            idx.append(i)

    return idx
# 0~9



def get_valid_img_seg(mat):

    manualLayer = np.array(mat['manualLayers1'], dtype=np.uint16)

    manualFluid = np.array(mat['manualFluid1'], dtype=np.uint16)

    img = np.array(mat['images'], dtype=np.uint8)

    valid_idx = get_valid_idx(manualLayer)





    manualFluid = manualFluid[:, :, valid_idx]

    manualLayer = manualLayer[:, :, valid_idx]



    print(manualLayer.shape)



    seg = np.zeros((496, 768, 11))

    seg[manualFluid > 0] = fluid_class

    max_col = -100

    min_col = 900

    for b_scan_idx in range(0, 11):

        for col in range(768):

            cur_col = manualLayer[:, col, b_scan_idx]

            if np.sum(cur_col) == 0:

                continue



            max_col = max(max_col, col)

            min_col = min(min_col, col)



            labels_idx = cur_col.tolist()

    #         print(f'{b_scan_idx} {labels_idx}')

    #         labels_idx.append(-1)

    #         labels_idx.insert(0, 0)

            last_st = None

            last_ed = None

            for label, (st, ed) in enumerate(zip([0]+labels_idx, labels_idx+[-1])):

    #             print(st, ed)

                if st == 0 and ed == 0:

                    st = last_ed

                    # 穿越第一层

                    while(seg[st, col, b_scan_idx] == fluid_class):

                        st += 1



                    while(seg[st, col, b_scan_idx] != fluid_class):

                        seg[st, col, b_scan_idx] = label

                        st += 1

                        if st >= 496:

                            break

                    continue

                if ed == 0:

                    ed = st + 1

                    while(seg[ed, col, b_scan_idx] != fluid_class):

                        ed += 1



                if st == 0 and label != 0:

                    st = ed-1

                    while(seg[st, col, b_scan_idx] != fluid_class):

                        st -= 1

                    st += 1



                seg[st:ed, col, b_scan_idx] = label

                last_st = st

                last_ed = ed



    seg[manualFluid > 0] = fluid_class

    

    seg = seg[:, min_col:max_col+1]

    img = img[:, min_col:max_col+1]

    return img, seg
img, seg = get_valid_img_seg(mat)
plt.imshow(seg[:, :, 0], cmap=plt.cm.jet, vmax=9)
plt.imshow(img[:, :, 0])
plt.imshow(seg[:, :, 3], cmap=plt.cm.jet, vmax=9)