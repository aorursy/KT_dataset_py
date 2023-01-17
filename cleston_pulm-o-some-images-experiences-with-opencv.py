import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
from matplotlib import pyplot
import cv2 
import random
import os
os.listdir('../input')
treino = pd.read_csv('../input/stage_1_train_labels.csv')
pac = pd.read_csv('../input/stage_1_detailed_class_info.csv')
img =[]
                  
for pid in treino['patientId']:
    DICOM = pydicom.read_file('../input/stage_1_train_images/{}.dcm'.format(pid))
    img.append(DICOM)                 
    
pyplot.imshow(img[random.randrange(len(img))].pixel_array)
pyplot.imshow(img[random.randrange(len(img))].pixel_array, cmap = 'gray')
numIm = [4,4]
lista=[]
listaId=[]
for i in range(numIm[0]*numIm[1]):
    lista.append(img[random.randrange(len(img))].pixel_array)
    listaId.append(img[random.randrange(len(img))].PatientID)
    
graf, loc= pyplot.subplots(numIm[0],numIm[1], figsize=(20,20))
i=0
for lo in loc:
    for l in lo:
        l.imshow(lista[i])
        i =i+1
positives = treino[treino['Target'] == 1]
rand = random.randrange(len(positives))
tempInfo = positives.iloc[rand]
for pid in img:
    if(pid.PatientID == tempInfo.patientId):
        temp = pid.pixel_array
        t1 = pid.PatientID
graf, loc= pyplot.subplots(2, figsize=(20,20))
temp2 = temp.copy()
temp2=cv2.rectangle(temp2, (int(tempInfo['x']),int(tempInfo['y'])), (int(tempInfo['x']+tempInfo['width']), int(tempInfo['y']+tempInfo['height'])) , 255, 3)
loc[0].imshow(temp2)

temp4 = temp2[  int(tempInfo['y']) : int(tempInfo['y']+tempInfo['height']+1), int(tempInfo['x']) : int(tempInfo['x']+tempInfo['width']+1) ]
loc[1].imshow(temp4)

rand = random.randrange(len(positives))
tempInfo = positives.iloc[rand]
for pid in img:
    if(pid.PatientID == tempInfo.patientId):
        temp = pid.pixel_array
        t1 = pid.PatientID
#temp with the marked place
tempM = temp.copy()
tempM=cv2.rectangle(tempM, (int(tempInfo['x']),int(tempInfo['y'])), (int(tempInfo['x']+tempInfo['width']), int(tempInfo['y']+tempInfo['height'])) , 255, 3)

kernel=(5,5)
temp2 = cv2.blur(temp, (kernel))
temp2 = cv2.blur(temp2, (kernel))
temp2 = cv2.blur(temp2, (kernel))
temp2 = cv2.blur(temp2, (kernel))

graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM)
loc[1].imshow(temp2)
kernel=(3,3)
temp2 =  cv2.GaussianBlur(temp,(kernel),0)
temp2 =  cv2.GaussianBlur(temp2,(kernel),0)
temp2 =  cv2.GaussianBlur(temp2,(kernel),0)
temp2 =  cv2.GaussianBlur(temp2,(kernel),0)

graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM)
loc[1].imshow(temp2)
temp2 = cv2.medianBlur(temp, 5)
temp2 = cv2.medianBlur(temp2, 5)
temp2 = cv2.medianBlur(temp2, 5)
temp2 = cv2.medianBlur(temp2, 5)

graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM)
loc[1].imshow(temp2)
kernel =(5,5)
temp2 = cv2.erode(temp, (kernel),1)
temp2 = cv2.dilate(temp2, (kernel),1)
for i in range(50):
    temp2 = cv2.erode(temp2, (kernel),1)
for i in range(50):
    temp2 = cv2.dilate(temp2, (kernel),1)
graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM)
loc[1].imshow(temp2)
kernel =(3,3)
temp2 = cv2.erode(temp, (kernel),1)
temp2 = cv2.erode(temp2, (kernel),1)
temp3 = cv2.dilate(temp, (kernel),1)
temp3 = cv2.dilate(temp3, (kernel),1)
temp2 = temp3-temp2
pyplot.figure(figsize=(15,15))

temp2=cv2.rectangle(temp2, (int(tempInfo['x']),int(tempInfo['y'])), (int(tempInfo['x']+tempInfo['width']), int(tempInfo['y']+tempInfo['height'])) , 255, 3)
pyplot.imshow(temp2)
temp2 =  cv2.GaussianBlur(temp,(5,5),0)
ret, temp2 = cv2.threshold(temp2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM)
loc[1].imshow(temp2)
ret, temp2 = cv2.threshold(temp2,175,255,cv2.THRESH_BINARY)
graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM)
loc[1].imshow(temp2)
ret, temp2 = cv2.threshold(temp,70,1,1)
graf, loc= pyplot.subplots(1,2, figsize=(15,15))
loc[0].imshow(tempM, cmap="gray")
temp2=cv2.addWeighted(temp, 1.2, temp2, -50, 1.0)
loc[1].imshow(temp2, cmap="gray")
graf, loc= pyplot.subplots(1,3, figsize=(15,15))
ret, temp2 = cv2.threshold(temp,50,255,1)
loc[0].imshow(tempM)
temp2 = temp-temp2
loc[1].imshow(temp2)
ret, temp2 = cv2.threshold(temp2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
loc[2].imshow(temp2)
temp2= cv2.convertScaleAbs(temp, alpha=3, beta=-350)
temp3= cv2.convertScaleAbs(temp, alpha=5, beta=-700) 
temp4= cv2.convertScaleAbs(temp, alpha=3.5, beta=-500)

graf, loc= pyplot.subplots(2,2, figsize=(15,15))

loc[0,0].imshow(tempM, cmap='gray')
loc[0,1].imshow(temp2, cmap='gray')
loc[1,0].imshow(temp3, cmap='gray')
loc[1,1].imshow(temp4, cmap='gray')
loc[1,1].set_title('the selected values after some observation')
rand = random.randrange(len(positives))
tempInfo = positives.iloc[rand]
for pid in img:
    if(pid.PatientID == tempInfo.patientId):
        temp = pid.pixel_array
        t1 = pid.PatientID
def density(imag, x,y, alt, larg):    
    dLin=[0]*alt
    dCol=[0]*larg
    for col in range(larg):       
        for row in range(alt):  
            dLin[row] = imag[y+row, x+col] + dLin[row]
            dCol[col] = imag[y+row, x+col] + dCol[col]  
    return({'lin':dLin , 'col':dCol})          
temp2 = cv2.convertScaleAbs(temp, alpha=3.5, beta=-500)
ret = density(temp2, int(tempInfo['x']), int(tempInfo['y']),int(tempInfo['height']), int(tempInfo['width']) )
ret2= density(temp2, 0, 0, len(temp), len(temp) )

graf, loc= pyplot.subplots(3,2, figsize=(20,20))
loc[0,0].barh(range(len(ret['lin'])),list(reversed(ret['lin'])))
loc[0,0].set_title('Vertical density (roi)')
loc[0,1].bar(range(len(ret['col'])),ret['col'])
loc[0,1].set_title('Horizontal density (roi)')
loc[2,0].barh(range(len(ret2['lin'])),list(reversed(ret2['lin'])))
loc[2,0].set_title('Vertical density (whole image)')
loc[2,1].bar(range(len(ret2['col'])),ret2['col'])
loc[2,1].set_title('Horizontal density (whole image)')
loc[1,0].imshow(temp2[  int(tempInfo['y']) : int(tempInfo['y']+tempInfo['height']+1), int(tempInfo['x']) : int(tempInfo['x']+tempInfo['width']+1) ],cmap='gray')
loc[1,0].set_title('ROI')
loc[1,1].imshow(temp2, cmap='gray')
loc[1,1].set_title('whole image')
graf, loc= pyplot.subplots(3,2, figsize=(20,20))
temp2 = temp.copy()
temp2=cv2.rectangle(temp2, (int(tempInfo['x']),int(tempInfo['y'])), (int(tempInfo['x']+tempInfo['width']), int(tempInfo['y']+tempInfo['height'])) , 255, 3)
loc[0,0].imshow(temp2)
loc[0,0].set_title('Original')

temp2= cv2.convertScaleAbs(temp, alpha=3.5, beta=-500)
temp2=cv2.rectangle(temp2, (int(tempInfo['x']),int(tempInfo['y'])), (int(tempInfo['x']+tempInfo['width']), int(tempInfo['y']+tempInfo['height'])) , 255, 3)
loc[0,1].imshow(temp2)
loc[0,1].set_title('Alfa/Beta changes')

temp2 = cv2.medianBlur(temp, 5)
temp2 = cv2.medianBlur(temp2, 5)
temp2 = cv2.medianBlur(temp2, 5)
temp2= cv2.convertScaleAbs(temp2, alpha=3.5, beta=-500)
temp2=cv2.rectangle(temp2, (int(tempInfo['x']),int(tempInfo['y'])), (int(tempInfo['x']+tempInfo['width']), int(tempInfo['y']+tempInfo['height'])) , 255, 3)
loc[1,0].imshow(temp2)
loc[1,0].set_title('Alfa/Beta + blur')

temp3 = cv2.resize(temp, (64,64))
temp3= cv2.convertScaleAbs(temp3, alpha=3.5, beta=-500)
loc[1,1].imshow(temp3)
loc[1,1].set_title('Alfa/Beta + resize')

temp4 = temp2[  int(tempInfo['y']) : int(tempInfo['y']+tempInfo['height']+1), int(tempInfo['x']) : int(tempInfo['x']+tempInfo['width']+1) ]
loc[2,0].imshow(temp4)
loc[2,0].set_title('Alfa/Beta + blur on ROI')

temp5= cv2.resize(temp4, (32,32))
loc[2,1].imshow(temp5)
loc[2,1].set_title('Alfa/Beta + resize on ROI')
negatives = treino[treino['Target'] == 0]
rand = random.randrange(len(negatives))
tempInfoN = negatives.iloc[rand]
for pid in img:
    if(pid.PatientID == tempInfoN.patientId):
        tempN = pid.pixel_array
        t1 = pid.PatientID
        
tempN = cv2.convertScaleAbs(tempN, alpha=3.5, beta=-500)
rand = random.randrange(len(positives))
tempInfo = positives.iloc[rand]
for pid in img:
    if(pid.PatientID == tempInfo.patientId):
        temp = pid.pixel_array
        t1 = pid.PatientID
temp = cv2.convertScaleAbs(temp, alpha=3.5, beta=-500)
graf, loc= pyplot.subplots( 2 , figsize=(15,15))

temp = temp[int(tempInfo['y']) : int(tempInfo['y']+tempInfo['height']+1), int(tempInfo['x']) : int(tempInfo['x']+tempInfo['width']+1)]
temp = cv2.resize(temp, (32,32))

loc[0].imshow(temp)
loc[0].set_title('random positive sample')
tam = 1024 - 32

x= random.randrange(tam)
y= random.randrange(tam)

loc[1].imshow(tempN[y:y+32 , x:x+32] )       
loc[1].set_title('random negative sample')
