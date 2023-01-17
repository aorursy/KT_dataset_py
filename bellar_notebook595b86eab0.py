# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import PIL.Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
im = PIL.Image.open("../input/train/00087a6bd4dc_05.jpg")

im.thumbnail((200, 150), PIL.Image.ANTIALIAS)

im
img = plt.imread("../input/train/00087a6bd4dc_05.jpg")

dims = img.shape[:2]



for k in range(2, 1000):

    if dims[0] % k == 0 and dims[1] % k == 0:

        print(k, dims)
def get_num_twos(n, k=0):

    return k if n % 2 == 1 else get_num_twos(int(n/2.0), k=k+1)

    

def get_nums_with_min_num_of_even_divisions(n_max, num_splits):

    ar = [(k, get_num_twos(k)) for k in range(2**num_splits - 1, n_max+1)]

    return [(k,num) for k,num in ar if num >= num_splits]



# (320, 480)

def get_offsets_with_min_num_of_even_divisions(y_dim, x_dim, num_splits):

    x_dim_cands = get_nums_with_min_num_of_even_divisions(x_dim, num_splits=num_splits)[-6:]

    y_dim_cands = get_nums_with_min_num_of_even_divisions(y_dim, num_splits=num_splits)[-6:]

    

    print("candidate sizes for img dim {} with {} splits required:\n".format((y_dim, x_dim), num_splits))

    

    print("\nY={} candidates:\n".format(y_dim))

    for y_dim_cand,n in reversed(y_dim_cands):

        print("new_y:{:4} offset:{:4} num_splits:{}".format(y_dim_cand, y_dim-y_dim_cand, n))

        

    print("\nX={} candidates:\n".format(x_dim))

    for x_dim_cand,n in reversed(x_dim_cands):

        print("new_x:{:4} offset:{:4} num_splits:{}".format(x_dim_cand, x_dim-x_dim_cand, n))

    

img_dims = (320, 480)

get_offsets_with_min_num_of_even_divisions(img_dims[0], img_dims[1], 5)