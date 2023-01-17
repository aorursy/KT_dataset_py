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
os.listdir("/kaggle/input/")


from PIL import Image 

im = Image.open("/kaggle/input/equationOog.png")  

im
import numpy

imarray = numpy.asarray(im)



oogabooga = imarray.shape





imarray.shape
arr = []

for i in range(imarray.shape[1]):

    tempSet= set()

    for j in range(imarray.shape[0]):



        temp = 0

        for k in range(3):

            #if(imarray[i][j][k] != 255):

            #    print(imarray[i][j][k], "tuple: " , (i,j,k))

            temp = temp + imarray[j][i][k]

            

        tempSet.add(temp)

   # print(tempSet, " len: ", len(tempSet))

    

    #CHANGE BACK TO 1 FOR MICHAEL

    if len(tempSet) > 2:

        arr.append("no")

    else:

        arr.append("yes")

    

    



fool = arr[0]

barr = []

for i in range(len(arr)):

    if (arr[i] is not fool):

        barr.append(i)

        fool = arr[i]
arr
barr




bar = []

for i in range(len(barr)-1):

    bar.append((barr[i] + barr[i+1])/2)

bar.append(imarray.shape[1]-1)
bar
im = Image.open("/kaggle/input/equationOog.png")
im
from PIL import ImageDraw

draw = ImageDraw.Draw(im)

draw.line((0, 0, 0, im.size[0]), fill=(0,0,0,255))



for i in range(len(bar)):

    if i%2 == 1:

        draw.line((bar[i], 0, bar[i], im.size[0]), fill=(0,0,0,255))

    



im
bar
!pip install imageslicer