# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

 # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import cv2

import numpy as np

import matplotlib.pyplot as plt

#from subprocess import check_output

#print(check_output(["ls","../input/data"]).decode("utf8"))

#print(os.listdir("../input"))

gray_bricks=cv2.imread('../input/bricks.jpg')

gray_bricks=cv2.cvtColor(gray_bricks,cv2.COLOR_BGR2RGB)

gray_bricks=cv2.cvtColor(gray_bricks,cv2.COLOR_RGB2GRAY)

hist,bins=np.histogram(gray_bricks.flatten(),256,[0,256])

cdf=hist.cumsum()

cdfn=cdf*hist.max()/cdf.max()

plt.plot(cdfn)

plt.show()



plt.hist(gray_bricks.flatten(),256,[0,256])

plt.title('cumulative distribution function graph')

plt.xlim([0,256])

#plt.ylim([0,4000])

plt.show()



#cdf_m = np.ma.masked_equal(cdf,0)\

cdf_m = np.array(cdf)

cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())

cdf=cdf_m[gray_bricks]

print(cdf_m)



plt.imshow(cdf)

plt.show()

plt.imshow(gray_bricks)

plt.show()





# Any results you write to the current directory are saved as output.