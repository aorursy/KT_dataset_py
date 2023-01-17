# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os



train_dir = "../input/train/"

sub_dirs = os.listdir(train_dir)

sub_dirs.remove(".DS_Store")



fish_type=dict(zip(sorted(sub_dirs), np.arange(len(sub_dirs))))



images = [(train_dir+"/"+sub_dir+"/"+image, sub_dir) for sub_dir in sub_dirs for image in os.listdir(train_dir+sub_dir)]

print(images[0][1])

print(len(images))



inputs = np.zeros(shape=(3777,500,500), dtype= float)