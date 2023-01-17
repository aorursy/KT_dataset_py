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
import os,cv2,numpy

TRAIN_DIR = "../input/train"

TEST_DIR = "../input/test"

IMG_LN = 32*32*3





def norm(arr):

    return (arr - 0) / 255



def readImg(dirNam):

    rtXarr = list()

    rtYarr = list()

    namDict = dict()

    curCn = 0

    for img in os.listdir(dirNam):

        cim = cv2.imread(dirNam+'/'+img)

        if img.__contains__('cat'):

            rtYarr.append(0)

        else:

            rtYarr.append(1)

        namDict[curCn] = img

        curCn = curCn + 1

        cim = cv2.resize(cim, (32,32))

        np = numpy.array(cim).flatten()

        np = norm(np)

        rtXarr.append(np)

    

    return rtXarr,rtYarr,namDict





XArr,YArr,nmDict = readImg(TRAIN_DIR)

    

print(len(YArr))

print(len(XArr))



import math

Y = numpy.array(YArr).T.reshape(1, len(YArr))

#W = numpy.zeros((IMG_LN , 1))

cachW = numpy.zeros((IMG_LN , 1)) # For adagrad



#B = 0

cachB = 0

W = numpy.random.randn(IMG_LN , 1) * .001

B = numpy.asscalar(numpy.random.randn(1,1) * .001)

trm = len(XArr)

X = numpy.array(XArr).T

#W = numpy.random.normal(0, .1, IMG_LN).reshape(IMG_LN , 1)

#B = 0

print(X.shape)

print(Y.shape)

print(W.shape)
def forwardMov(CX):

    YPred = (W.T).dot(CX) + B

    YPred = -1 * YPred

    YPred = 1 / (1 + numpy.exp(YPred))

    return YPred
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.animation as animation





fig = plt.figure(figsize=(16, 12))

ax = fig.add_subplot(111)

# You can initialize this with whatever

im = ax.imshow(np.random.rand(6, 10), cmap='bone_r', interpolation='nearest')





def animate(i):

    aux = np.zeros(60)

    aux[i] = 1

    image_clock = np.reshape(aux, (6, 10))

    im.set_array(image_clock)



ani = animation.FuncAnimation(fig, animate, frames=60, interval=1000)



plt.show()
runArr = []

loArr = []



for run in range(0 , 1000):

    #print (run)

    curPrd = 0

    #dW = numpy.zeros((1, IMG_LN))

    #dB = 0

    YPred = forwardMov(X)

    lo = -1 * (Y * numpy.log(YPred) + (1 - Y) * numpy.log(1 - YPred))

    #curPrd = numpy.sum(map(lbl , YPred))

    dZ = YPred - Y

    #print (dZ)

    dW = X.dot(dZ.T)

    dW = dW / trm;

    tmpCacheDw = dW * dW

    #print("89")

    dB = numpy.sum(dZ)

    dB = dB / trm;

    tmpCacheDb = dB * dB

    

    cachW = cachW + tmpCacheDw 

    #W = W - 0.001*(dW)

    W = W - 0.01*(dW / numpy.sqrt(cachW) + .00001);

    

    cachB = cachB + tmpCacheDb 

    #B = B - 0.001*dB

    B = B - 0.01*(dB / numpy.sqrt(cachB)+ .00001);

    

    curL = (numpy.sum(lo)) / trm

    runArr.append(run)

    loArr.append(curL)

    

    #line.set_xdata(runArr)

    #line.set_ydata(loArr)

    #ax.relim()

    #ax.autoscale_view()

    #plt.draw() 

    if run % 100 == 0:

        

        curL = (numpy.sum(lo)) / trm

        curPrd = 0

        for ct in range(0 , trm):

            if YPred[0][ct] < .5:

                curvl = 0

            else:

                curvl = 1

            if curvl == Y[0][ct]:

                curPrd = curPrd + 1

        print ("run : " + str(run) + " lo : " + str(curL) + " acc : " + str(curPrd/trm))

        

"""

import math

Y = numpy.array(YArr).reshape(len(YArr) , 1)

W = numpy.zeros((1, IMG_LN))

B = 0

X = numpy.array(XArr)



    

for run in range(0 , 10000):

    curPrd = 0

    dW = numpy.zeros((1, IMG_LN))

    dB = 0

    for cnt in range(0, len(XArr)):

        YPred = XArr[cnt].dot(W.T) + B

        YPrdVl = numpy.asscalar(YPred.flatten())

        YPrdVl = 1 / (1 + numpy.exp(-YPrdVl))

        curvl = -1

        if YPrdVl < .5:

            curvl = 0

        else:

            curvl = 1

        if curvl == Y[cnt]:

            curPrd = curPrd + 1

        # dW = (YPred - Y) * XArr[cnt]

        #W = W - dW

        dW = dW + ((YPrdVl - Y[cnt]) * XArr[cnt])

        dB = dB + (YPrdVl - Y[cnt])

    

    dW = dW / len(XArr)

    W = W - 0.0000001*dW

    

    dB = dB / len(XArr)

    B = B - 0.0000001*dB

    if run % 10 == 0:

        print (" " + str(run) + ":" + str(curPrd/len(XArr)))

"""
def readTestImg(dirNam):

    rtXarr = list()

    rtYarr = dict()

    curCn = 0

    for img in os.listdir(dirNam):

        cim = cv2.imread(dirNam+'/'+img)

        imgName = img.strip().split('.')[0]

        rtYarr[curCn] = imgName 

        curCn = curCn+1

        

        cim = cv2.resize(cim, (32,32))

        np = numpy.array(cim).flatten()

        np = norm(np)

        rtXarr.append(np)

    

    return rtXarr,rtYarr 
TArr,mgMp = readTestImg(TEST_DIR)

#print(len(TArr))

TData = numpy.array(TArr).T

YTat = forwardMov(TData)

#print(len(YTat))

fVal = list()

for cnt in range(0 , len(TArr)):

    fVal.append((mgMp[cnt] , YTat[0][cnt]))

    

fVal = sorted(fVal , key=lambda x:int(x[0]))

fl = open("sample_submission.csv" , "w")

fl.write("id,label\n")

for cnt in range(0,len(fVal)):

    fl.write(str(fVal[cnt][0])+","+str(fVal[cnt][1])+"\n")



fl.close()



print("id,label")

for cnt in range(0,len(fVal)):

    print(str(fVal[cnt][0])+","+str(fVal[cnt][1]))



#print (fVal)
