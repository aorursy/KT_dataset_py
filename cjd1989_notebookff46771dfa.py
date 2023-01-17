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

from PIL import Image
fish_species = os.listdir("/kaggle/input/train")

print(fish_species)

for item in fish_species:

    if item!='.DS_Store':

        print(item, len(os.listdir("/kaggle/input/train/"+item)))
with open('/kaggle/input/train/.DS_Store',"rb") as f:

    for line in f:

        line=line.strip()

        print(line)
with open("/kaggle/input/train/ALB/img_02227.jpg","rb") as f:

    for line in f:

        line = line.strip()

        print(line)
Image.open("/kaggle/input/train/ALB/img_01300.jpg")

Image.open("/kaggle/input/train/ALB/img_02227.jpg")
#print(os.listdir("/kaggle/input/test_stg1/"))



Image.open("/kaggle/input/test_stg1/img_07659.jpg")
data = pd.read_csv('/kaggle/input/sample_submission_stg1.csv')

data
data2 = data.ix[:,data.columns!='image']

print(data2.sum(axis=0))

print(data2.sum(axis=1))