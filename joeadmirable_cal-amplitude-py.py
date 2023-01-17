# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy
from json import JSONEncoder
import math
import cmath
import json
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


L_shape=numpy.array([1,0,1,0,1,1])
# L_shape=numpy.array([[1,0],[1,0],[1,1]])
# Serialization
numpyData = {
    "size":{
        "width": 2,
        "height": 3
        },
        "data": L_shape
        }
print("serialize NumPy array into JSON and write into a file")
with open("L_shape.json", "w") as write_file:
    json.dump(numpyData, write_file, cls=NumpyArrayEncoder)
print("Done writing serialized NumPy array into file")

# Deserialization
print("Started Reading JSON file")
with open("L_shape.json", "r") as read_file:
    print("Converting JSON encoded data into Numpy array")
    decodedArray = json.load(read_file)

    finalNumpyArray = numpy.asarray(decodedArray["data"])
    print("NumPy Array")
    print(finalNumpyArray)
class assocG:
    def __init__(self, name, w,h,data):
        self.name = name
        self.data = data
        self.w = w
        self.h = h

    def load(self):
        #changed here
#         self.assocW = np.round((np.array(self.data)).reshape(self.w,self.h),1)
        self.assocW = np.round((np.array(self.data)).reshape(self.h,self.w),1)
#         print(self.assocW.shape)
#         self.render = np.round((1-np.array(self.data)).reshape(self.w,self.h),1)
        self.render = np.round((1-np.array(self.data)).reshape(self.h,self.w),1)
        self.xticklabels=np.arange(0,self.w)
        self.yticklabels=np.arange(0,self.h)
        
    def calLinks(self):
        self.wLink = np.ones_like(np.arange((self.h)*(self.w-1)).reshape(self.h,self.w-1)).astype(float)
        self.hLink = np.ones_like(np.arange((self.h-1)*(self.w)).reshape(self.h-1,self.w)).astype(float)

        for i in range(self.h):
            for j in range(self.w-1):
                if i <self.h-1 or j<self.w-1:
#                     print(i,j)
                    self.wLink[i,j] = (min(self.assocW[i,j], self.assocW[i,j+1])*10)

        for i in range(self.h-1):
            for j in range(self.w):
                if i <self.h-1 or j<self.w-1:
                    self.hLink[i,j] = (min(self.assocW[i,j], self.assocW[i+1,j])*10)

        self.wLink=self.wLink.astype(int)
        self.hLink=self.hLink.astype(int)
    
    def plot(self,fig,ax):
        self.ax = ax
        self.fig= fig
        
        self.ax.imshow(self.render, cmap='gray', vmin=0, vmax=1)

        # draw gridlines
        self.ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        self.ax.set_xticks(np.arange(-.5, self.w, 1));
        self.ax.set_yticks(np.arange(-.5, self.h, 1));

        self.ax.set_xticklabels(self.xticklabels)
        self.ax.set_yticklabels(self.yticklabels)
        #style="Simple,tail_width=0.1,head_width=10,head_length=8"
        style="-"
        kw = dict(arrowstyle=style, color="k")


        for i in range(len(self.yticklabels)):
            for j in range(len(self.xticklabels)):
                text = ax.text(j, i, self.assocW[i,j],ha="center", va="center", color="w")

                if (i<self.h-1 and j < self.w-1):
                    arrowg = linkGen(self.wLink[i,j],(j,i),(j+1,i)).generate()
                    for arx in arrowg:
                        plt.gca().add_patch(arx)
                    arrowg = linkGen(self.hLink[i,j],(j,i),(j,i+1)).generate()
                    for ary in arrowg:
                        plt.gca().add_patch(ary)
                elif(j == self.w-1 and i < self.h-1):                    
                    arrowg = linkGen(self.hLink[i,j],(j,i),(j,i+1)).generate()
                    for ary in arrowg:
                        plt.gca().add_patch(ary)
                elif(i == self.h-1 and j < self.w-1):
                    arrowg = linkGen(self.wLink[i,j],(j,i),(j+1,i)).generate()
                    for arx in arrowg:
                        plt.gca().add_patch(arx)
data = decodedArray
print(data)
perfect_L = assocG("L_shape", data['size']['width'],data['size']['height'],data['data'])
perfect_L.load()
perfect_L.calLinks()
# print('perfect_L.wLink',perfect_L.wLink)
# print('perfect_L.wLink/10',perfect_L.wLink/10)
# print('numpy.squeeze(perfect_L.wLink/10)',numpy.squeeze(perfect_L.wLink/10))
# print('list(numpy.squeeze(perfect_L.wLink/10))',list(numpy.squeeze(perfect_L.wLink/10)))
# print('perfect_L.hLink/10',perfect_L.hLink/10)
# print('numpy.squeeze(perfect_L.hLink/10)',numpy.squeeze(perfect_L.hLink/10))
# print('list(numpy.squeeze(perfect_L.hLink/10))',list(numpy.squeeze(perfect_L.hLink/10)))
# allJab=list(numpy.squeeze(perfect_L.wLink/10)).extend(list(numpy.squeeze(perfect_L.hLink/10)))
# print('perfect_L.wLink/10.flatten()',(perfect_L.wLink/10).flatten())
# print('perfect_L.hLink/10.flatten()',(perfect_L.hLink/10).flatten())
allwJab=list((perfect_L.wLink/10).flatten())
allhJab=list((perfect_L.hLink/10).flatten())
allwJab.extend(allhJab)
allJab=allwJab
print('allJab',allJab)

mean=numpy.mean(allJab)
print('mean', mean)

std=numpy.std(allJab)
print('std', std)
xi= 1 # don't know
A=1
for jab in allJab:
    power=-((jab-mean)**2)/(2*std)
    eachItem=(2*jab+1) * (math.e**(power))*(math.e**(-1j*xi*jab))
    A*=eachItem

print('amplitude',A)
# w = cmath.polar(A) 
# print(w)
print('prediction',abs(A)**2)

print('using 𝛿Φ, prediction',abs((0.20932699490074924)*A)**2)
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def near(a, b, rtol=1e-5, atol=1e-8):
    return abs(a - b) < (atol + rtol * abs(b))
def crosses(line1, line2):
    """
    Return True if line segment line1 intersects line segment line2 and 
    line1 and line2 are not parallel.
    """
    (x1,y1), (x2,y2) = line1
    (u1,v1), (u2,v2) = line2
    (a,b), (c,d) = (x2-x1, u1-u2), (y2-y1, v1-v2)
    e, f = u1-x1, v1-y1
    denom = float(a*d - b*c)
    if near(denom, 0):
        # parallel
        return False
    else:
        t = (e*d - b*f)/denom
        s = (a*f - e*c)/denom
        # When 0<=t<=1 and 0<=s<=1 the point of intersection occurs within the
        # line segments
        return 0<=t<=1 and 0<=s<=1
import math
import matplotlib.pyplot as plt

x_array = []
y_array = []
init = -5
for num in range(0, 1000):
    x = init+ (num/100)
    y = abs(A*x)**2
    x_array.append(x)
    y_array.append(y)

x_maximums = []
y_maximums = []

for i in range(2, len(y_array) - 2):
    if y_array[i - 2] < y_array[i - 1] and y_array[i - 1] < y_array[i] and y_array[i + 2] < y_array[i + 1] and y_array[i + 1] < y_array[i]:
        y_maximums.append(y_array[i])
        x_maximums.append(x_array[i])


plt.plot(x_array, y_array)
plt.scatter(x_maximums, y_maximums, color='k')
plt.show()
plt.plot(x_array, y_array)
plt.axhline(y=1, linestyle = '--', color = 'grey')

yys = [1]
xx, yy = [],[]
xo,yo = x_array,y_array
d = 1
optimal = 1
for i in range(1,len(y_array)):
    for k in yys:
        p1 = np.array([xo[i-1],yo[i-1]],dtype='float')
        p2 = np.array([xo[i],yo[i]],dtype='float')
        k1 = np.array([xo[i-1],k],dtype='float')
        k2 = np.array([xo[i],k],dtype='float')
        if crosses((p2,p1),(k1,k2)):
            seg = line_intersection((p2,p1),(k1,k2))
            if seg is not None:
                xx.append(seg[0])
                yy.append(seg[1]-d)
                plt.scatter(seg[0],seg[1],c='red')
                plt.annotate(seg[0], (seg[0],seg[1]))
                optimal = seg[0]
plt.ylim(2,0)
plt.xlim(0,7)
plt.tight_layout()
print('optimal',optimal)
plt.show()