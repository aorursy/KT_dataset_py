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
os.listdir("/kaggle/input/greenbean")


from PIL import Image 

im = Image.open("/kaggle/input/greenbean/Screen Shot 2020-02-01 at 1.36.54 PM.png")  

im
import numpy

imarray = numpy.asarray(im)



oogabooga = imarray.shape





imarray.shape
arr = []

arrhorizontal = []



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

barrhorizontal = []

for i in range(len(arr)):

    if (arr[i] is not fool):

        barr.append(i)

        fool = arr[i]

bar = []

for i in range(len(barr)-1):

    bar.append((barr[i] + barr[i+1])/2)

bar.insert(0, 0)

bar.append(imarray.shape[1]-1)

arrayColNum = []

for i in range(len(bar)):

    if i % 2 == 0:

        arrayColNum.append(bar[i])
bar
arrayColNum
imarray.shape[0]
im = Image.open("/kaggle/input/greenbean/Screen Shot 2020-02-01 at 1.36.54 PM.png")
im
from PIL import ImageDraw

draw = ImageDraw.Draw(im)



for i in range(len(arrayColNum)):

        draw.line((arrayColNum[i], 0, arrayColNum[i], im.size[0]), fill=(0,0,0,255))

for i in range(len(barrhorizontal)):

    draw.line((0, barrhorizontal[i], im.size[0], barrhorizontal[i] ), fill=(0,0,0,255))

    



im
def split_horizontal(array):

    arr = []

    arrhorizontal = []



    for i in range(array.shape[0]):

        tempSet= set()

        for j in range(array.shape[1]):



            temp = 0

            for k in range(3):

                #if(imarray[i][j][k] != 255):

                #    print(imarray[i][j][k], "tuple: " , (i,j,k))

                temp = temp + array[i][j][k]



            tempSet.add(temp)

       # print(tempSet, " len: ", len(tempSet))



        #CHANGE BACK TO 1 FOR MICHAEL

        if len(tempSet) > 2:

            arr.append("no")

        else:

            arr.append("yes")

            

    fool = arr[0]

    barr = []

    barrhorizontal = []

    for i in range(len(arr)):

        if (arr[i] is not fool):

            barr.append(i)

            fool = arr[i]

    return barr
import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



arrayOfImgs = []



arrs = np.split(imarray, [int(i) for i in arrayColNum], axis = 1)[1:]

for i in arrs:

    print(i.shape)

    arr_temp= split_horizontal(i)

    int1, int2 = (0,0)

    if(len(arr_temp) == 0):

        int1 = 0

        int2 = i.shape[0] - 1

    elif(len(arr_temp) == 1):

        int1 = arr_temp[0]

        int2 = i.shape[0] - 1

    else:

        int1 = arr_temp[0]

        int2 = arr_temp[-1]

    

    print(int1, int2)

    arrayOfImgs.append(np.split(i, [int1, int2], axis = 0)[1])

    

    





plt.imshow(arrayOfImgs[0])

plt.imshow(arrayOfImgs[1])

plt.imshow(arrayOfImgs[2])

plt.imshow(arrayOfImgs[3])

plt.imshow(arrayOfImgs[4])

plt.imshow(arrayOfImgs[5])

plt.imshow(arrayOfImgs[6])

plt.imshow(arrayOfImgs[7])
