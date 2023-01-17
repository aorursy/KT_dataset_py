# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from PIL import Image

import csv

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def createFileList(myDir, format='.jpg'):

    fileList = []

    print(myDir)

    for root, dirs, files in os.walk(myDir, topdown=False):

        print(myDir)

        print(root)

        for name in files:

            if name.endswith(format.upper()) or name.endswith(format):

                fullName = os.path.join(root, name)

                fileList.append(fullName)

                print(fullName)

        

    return fileList



# load the original image

myFileList = createFileList('/kaggle/input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset/yes')



for file in myFileList:

    print(file)

    img_file = Image.open(file)

    # img_file.show()



    # get original image parameters...

    width, height = 240,240

    format = img_file.format

    mode = img_file.mode



    # Make image Greyscale

    img_grey = img_file.convert('L')

    #img_grey.save('result.png')

    #img_grey.show()



    # Save Greyscale values

    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))

    value = value.flatten()

    print(value)

    with open("img_pixels_2.csv", 'a') as f:

        writer = csv.writer(f)

        writer.writerow(value)