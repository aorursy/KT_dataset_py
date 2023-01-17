# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/real-and-fake-face-detection/real_and_fake_face_detection/real_and_fake_face/'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        pass

# Any results you write to the current directory are saved as output.
import subprocess

import sys



try:

    import dlib

except:

    subprocess.call([sys.executable, "-m", "pip", "install", 'dlib'])

    import dlib

import os

import cv2
import cv2

import matplotlib.pyplot as plt



fakeDir='/kaggle/input/real-and-fake-face-detection/real_and_fake_face_detection/real_and_fake_face/training_fake/'

realDir='/kaggle/input/real-and-fake-face-detection/real_and_fake_face_detection/real_and_fake_face/training_real/'
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('/kaggle/input/dlibpackage/shape_predictor_68_face_landmarks.dat')

import matplotlib.pyplot as plt

from PIL import Image



fakeImgList=os.listdir('/kaggle/input/real-and-fake-face-detection/real_and_fake_face_detection/real_and_fake_face/training_fake/')

realImgList=os.listdir('/kaggle/input/real-and-fake-face-detection/real_and_fake_face_detection/real_and_fake_face/training_real/')



def genResult(fileList):

    nameDict={}

    for nameString in fileList:

        temp=nameString.split('.')[0]

        infoList=temp.split('_')

        lvl=infoList[0]

        filNum=infoList[1]

        rgs=[int(x) for x in str(infoList[2])]

        nameDict[nameString]=[lvl,filNum,rgs]

    return nameDict



def genOrigin(fileList):

    nameDict={}

    for nmStr in fileList:

        nameDict[int(nmStr.split('.')[0].split('_')[1])]=nmStr

    return nameDict



fakeDictionary=genResult(fakeImgList)

realDictionary=genOrigin(realImgList)



def oneImgPipeline(imgIdx):

    fakeImgName=fakeImgList[imgIdx]



    metaInfo=fakeDictionary[fakeImgName]

    realImgName=realDictionary[int(metaInfo[1])]



    img = cv2.imread(fakeDir+fakeImgName, 1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_in_image = detector(img_gray, 0)

    

    if len(faces_in_image)!=1:

        return 

    

    for face in faces_in_image:

        landmarks = predictor(img_gray, face)

        landmarks_list = []

        for i in range(0, landmarks.num_parts):

            landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))



    xaxis=[x[0] for x in landmarks_list]

    yaxis=[x[1] for x in landmarks_list]

    img = Image.open(fakeDir+fakeImgName)

    img.load()

    

    data = np.asarray(img, dtype="int32" )

    

    fakeImg = Image.open(fakeDir+fakeImgName)

    fakeImg.load()

    

    dataFake = np.asarray(fakeImg, dtype="int32" )

        

    fig = plt.figure(figsize=(100, 40))

    ax1 = fig.add_subplot(1,3,1)

    ax1.imshow(img)

    

    def maskRightEye(xlst,ylst):

        hotXs=xlst[36:42]

        hotYs=ylst[36:42]

        minX=min(hotXs)

        maxX=max(hotXs)



        minY=min(hotYs)

        maxY=max(hotYs)



        leftX=max(0,minX-(maxX-minX)/2)

        rightX=min(data.shape[1],maxX+(maxX-minX)/2+1)



        botY=min(data.shape[0],maxY+maxY-minY+1)

        topY=max(0,minY-maxY+minY)



        boxXLst=[leftX,rightX,rightX,leftX,leftX]

        boxYLst=[topY,topY,botY,botY,topY]

        ax1.plot(boxXLst,boxYLst,linewidth=10.0)

        return int(leftX),int(rightX),int(topY),int(botY)

        

    def maskLeftEye(xlst,ylst):

        hotXs=xlst[42:48]

        hotYs=ylst[42:48]

        minX=min(hotXs)

        maxX=max(hotXs)



        minY=min(hotYs)

        maxY=max(hotYs)



        leftX=max(0,minX-(maxX-minX)/2)

        rightX=min(data.shape[1],maxX+(maxX-minX)/2+1)



        botY=min(data.shape[0],maxY+maxY-minY+1)

        topY=max(0,minY-maxY+minY)



        boxXLst=[leftX,rightX,rightX,leftX,leftX]

        boxYLst=[topY,topY,botY,botY,topY]

        ax1.plot(boxXLst,boxYLst,linewidth=10.0)

        return int(leftX),int(rightX),int(topY),int(botY)

    

    def maskNose(xlst,ylst):

        hotXs=xlst[27:36]

        hotYs=ylst[27:36]

        minX=min(hotXs)

        maxX=max(hotXs)



        minY=min(hotYs)

        maxY=max(hotYs)



        leftX=max(0,minX-(maxX-minX)/5)

        rightX=min(data.shape[1],maxX+(maxX-minX)/5+1)



        botY=min(data.shape[0],maxY+(maxY-minY)/10+1)

        topY=max(0,minY-(maxY-minY)/10)



        boxXLst=[leftX,rightX,rightX,leftX,leftX]

        boxYLst=[topY,topY,botY,botY,topY]

        ax1.plot(boxXLst,boxYLst,linewidth=10.0)

        return int(leftX),int(rightX),int(topY),int(botY)

    

    def maskMouth(xlst,ylst):

        hotXs=xlst[48:68]

        hotYs=ylst[48:68]

        minX=min(hotXs)

        maxX=max(hotXs)



        minY=min(hotYs)

        maxY=max(hotYs)



        leftX=max(0,minX-(maxX-minX)/10)

        rightX=min(data.shape[1],maxX+(maxX-minX)/10+1)



        botY=min(data.shape[0],maxY+(maxY-minY)/10+1)

        topY=max(0,minY-(maxY-minY)/10)



        boxXLst=[leftX,rightX,rightX,leftX,leftX]

        boxYLst=[topY,topY,botY,botY,topY]

        ax1.plot(boxXLst,boxYLst,linewidth=10.0)

        return int(leftX),int(rightX),int(topY),int(botY)

    

    rightEyeInfo=maskRightEye(xaxis,yaxis)

    leftEyeInfo=maskLeftEye(xaxis,yaxis)

    noseInfo=maskNose(xaxis,yaxis)

    mouthInfo=maskMouth(xaxis,yaxis)

    

    reconShape=data.shape

    level=metaInfo[0]

    base=np.zeros(reconShape)

    fillVal=0

    

    if metaInfo[0]=='easy':

        fillVal=85.0

    elif metaInfo[0]=='mid':

        fillVal=170.0

    elif metaInfo[0]=='hard':

        fillVal=255.0

    

    if int(metaInfo[2][0])==1:

        for x in range(rightEyeInfo[0],rightEyeInfo[1]):

            for y in range(rightEyeInfo[2],rightEyeInfo[3]):

                base[y][x]=fillVal

    if int(metaInfo[2][1])==1:

        for x in range(leftEyeInfo[0],leftEyeInfo[1]):

            for y in range(leftEyeInfo[2],leftEyeInfo[3]):

                base[y][x]=fillVal

    if int(metaInfo[2][2])==1:

        for x in range(noseInfo[0],noseInfo[1]):

            for y in range(noseInfo[2],noseInfo[3]):

                base[y][x]=fillVal

    if int(metaInfo[2][3])==1:

        for x in range(mouthInfo[0],mouthInfo[1]):

            for y in range(mouthInfo[2],mouthInfo[3]):

                base[y][x]=fillVal

    ax2 = fig.add_subplot(1,3,2)

    ax2.imshow(base/255.0)

    ax3 = fig.add_subplot(1,3,3)

    ax3.imshow(data)



    return metaInfo

    
oneImgPipeline(201)