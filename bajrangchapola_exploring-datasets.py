# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
sample_path = '../input/sample images'

df = pd.read_csv(sample_path + '/train-rle-sample.csv',header=None)

df.head(len(df))
#These are the functions provided by kaggle to convert a mask to rle and vice-versa.

import numpy as np



def mask2rle(img, width, height):

    rle = []

    lastColor = 0;

    currentPixel = 0;

    runStart = -1;

    runLength = 0;



    for x in range(width):

        for y in range(height):

            currentColor = np.where(img[x][y] > 0.2,1,0)

            if currentColor != lastColor: 

                if currentColor == 1:

                    runStart = currentPixel;

                    runLength = 1;

                else:

                    rle.append(str(runStart));

                    rle.append(str(runLength));

                    runStart = -1;

                    runLength = 0;

                    currentPixel = 0;

            elif runStart > -1:

                runLength += 1

            lastColor = currentColor;

            currentPixel+=1;



    return " ".join(rle)



def rle2mask(rle, width, height):

    

    mask= np.zeros(width* height)

    if rle == ' -1' or rle == '-1':

        return mask.reshape(width,height)

    array = np.asarray([int(x) for x in rle.split()])

    

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        current_position += start

        mask[current_position:current_position+lengths[index]] = 255

        current_position += lengths[index]



    return mask.reshape(width, height)
imageId = df[0]

encodedPixels = df[1]
plt.imshow(rle2mask(df[1][7],1024,1024))

plt.show()