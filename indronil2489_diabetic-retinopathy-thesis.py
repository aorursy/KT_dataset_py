# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import math 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
#for dirname, _, filenames in os.walk('/kaggle/input/'):
    #for filename in filenames:
       #os.path.join(dirname, filename)

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/diabetic-retinopathy-resized/trainLabels_cropped.csv", header=None)
df = df.iloc[1:]
num = len(df)
num
data_size= num
s="/kaggle/input/diabetic-retinopathy-resized/resized_train_cropped/resized_train_cropped/"
class_list=[]
img=[]
two_cnt=0
one_cnt=0
zero_cnt=0
for i in range(0,data_size):
    imgloc = s+df.iloc[i,2]+'.jpeg'
    if(df.iloc[i,3]=='0'):
        zero_cnt=zero_cnt+1
        if(zero_cnt%5==0):
            class_list.append(df.iloc[i,3])
            img1 = cv2.imread(imgloc,1)
            img1 = cv2.resize(img1,(350,350))
            img.append(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
    elif(df.iloc[i,3]=='2'):
        two_cnt=two_cnt+1
        if(two_cnt%4==0):    
            class_list.append(df.iloc[i,3])
            img1 = cv2.imread(imgloc,1)
            img1 = cv2.resize(img1,(350,350))
            img.append(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
            class_list.append(df.iloc[i,3])
            #img1 = cv2.flip(img1,1)
            img.append(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
    elif(df.iloc[i,3]=='1'):
        one_cnt=one_cnt+1
        if(one_cnt%2==0):    
            class_list.append(df.iloc[i,3])
            img1 = cv2.imread(imgloc,1)
            img1 = cv2.resize(img1,(350,350))
            img.append(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
            class_list.append(df.iloc[i,3])
            #img1 = cv2.flip(img1,1)
            img.append(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
    else:
            class_list.append(df.iloc[i,3])
            img1 = cv2.imread(imgloc,1)
            img1 = cv2.resize(img1,(350,350))
            img.append(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
            class_list.append(df.iloc[i,3])
            #img1 = cv2.flip(img1,1)
            img.append(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
#df[3]=pd.to_numeric(df[3])
zero = 0
one = 0
two = 0
three = 0
four = 0
for i in range(0,len(class_list)):
    if(class_list[i]=='0'): zero= zero+1
    elif(class_list[i]=='1'): one= one+1
    elif(class_list[i]=='2'): two= two+1
    elif(class_list[i]=='3'): three= three+1
    elif(class_list[i]=='4'): four= four+1
print(zero, one, two, three, four)
new_data_size=len(class_list)
new_data_size

area_of_exudate=[]
gre = []
for i in range(0,new_data_size):
    img2 = np.array(img[i])
    r,greencha,b=cv2.split(img2)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8)) 
    curImg = clahe.apply(greencha)
    gre.append(curImg)
    strEl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    curImg = cv2.dilate(curImg, strEl)
    curImg = cv2.medianBlur(curImg,5)
    retValue, curImg = cv2.threshold(curImg, 235, 255, cv2.THRESH_BINARY)
    #curImg= cv2.cvtColor(curImg,cv2.COLOR_BGR2RGB)
    
    rv2, optical_disk = cv2.threshold(b, 240, 255, cv2.THRESH_BINARY)
    curImg = cv2.subtract(curImg,optical_disk)
    
    
    moment = cv2.moments(curImg)
    huMoments = cv2.HuMoments(moment)

    #for i in range(0,7):
    #    huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
    #Humoments2 = -np.sign(Humoments)*np.log10(np.abs(Humoments))
    area_of_exudate.append(huMoments[0])
flattened_list = [y for x in area_of_exudate for y in x]
#flattened_list = np.shape(flattened_list)
area_of_exudate = flattened_list
print(area_of_exudate)
kernel_for_bv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

def extract_bv(image):

    contrast_enhanced_green_fundus = image
   
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
   
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
   
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
   # cv2.imshow('contrast_enhanced_green_fundus',contrast_enhanced_green_fundus)
    f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)
   # cv2.imshow('f5',f5)
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    #print(mask)
   # _, contours, _ = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)

   
    fundus_eroded = cv2.bitwise_not(newfin) 
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)
    blood_vessels = cv2.bitwise_not(finimage)
    return blood_vessels
area_of_bloodvessel=[]

for i in range(0,new_data_size):
    bloodvessel = extract_bv(gre[i])
    bloodvessel = cv2.resize(bloodvessel,(350,350))
    count = 0
    bloodvessel =255- bloodvessel
    retValue, bloodvessel = cv2.threshold(bloodvessel, 235, 255, cv2.THRESH_BINARY)
    bloodvessel = cv2.dilate(bloodvessel,kernel_for_bv,iterations = 1)
   # bloodvessel= cv2.cvtColor(bloodvessel,cv2.COLOR_BGR2RGB)
    
    moment = cv2.moments(bloodvessel)
    huMoments = cv2.HuMoments(moment)

  #  for i in range(0,7):
  #      huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
 
    area_of_bloodvessel.append(huMoments[0])
flattened_list = [y for x in area_of_bloodvessel for y in x]
#flattened_list = np.shape(flattened_list)
area_of_bloodvessel = flattened_list

print(area_of_bloodvessel)   
kernelmicro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
def extract_ma(image):
     
    median = cv2.medianBlur(image,3)

    erosion_ma =255- cv2.erode(median, kernelmicro,iterations = 1)
    ret3,thresh2 = cv2.threshold(erosion_ma,215,255,cv2.THRESH_BINARY)
    closing_ma = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernelmicro)
    mask = np.ones(closing_ma.shape[:2], dtype="uint8") * 255
    contours_mn, hierarchy_mn = cv2.findContours(closing_ma, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
   
    for cnt_mn in contours_mn:
        if cv2.contourArea(cnt_mn) <= 70:
            cv2.drawContours(mask, [cnt_mn], -1, 0, -1)
    final_ma = cv2.bitwise_and(closing_ma, closing_ma, mask=mask)
    sub_ma = cv2.subtract(closing_ma,final_ma)
    sub_ma = cv2.morphologyEx(sub_ma, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
    sub_ma =cv2.erode(sub_ma,kernelmicro,iterations = 1)
    return sub_ma

area_of_micro = []
for i in range(0,new_data_size):
    count = 0
    mcran = extract_ma(gre[i])
    
    moment = cv2.moments(mcran)
    huMoments = cv2.HuMoments(moment)

  #  for i in range(0,7):
  #      huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
 
    area_of_micro.append(huMoments[0])
    
flattened_list = [y for x in area_of_micro for y in x]
#flattened_list = np.shape(flattened_list)
area_of_micro = flattened_list
print(area_of_micro)
  
X = list(zip(area_of_exudate,area_of_bloodvessel,area_of_micro))
print(len(X))
y = class_list
#df.iloc[0:new_data_size,3:4].values
#print((y))


from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(X,y,test_size = .25 ,random_state =0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
Classifier = RandomForestClassifier(n_estimators = 3000, criterion='gini', max_features = 'sqrt',  random_state=1, oob_score=True)
Classifier.fit(X_train, y_train)
y_pred = Classifier.predict(X_test)
#print(y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
from mpl_toolkits import mplot3d
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (16,16))
ax = plt.axes(projection="3d")
class_list=pd.to_numeric(class_list)

z_points = area_of_micro
x_points = area_of_bloodvessel
y_points = area_of_exudate
ax.scatter3D(x_points, y_points, z_points, zdir=x_points, c=class_list, cmap=cm.Set1);
plt.show()
import plotly.express as px
fig = px.scatter_3d(x=area_of_bloodvessel, y=area_of_micro, z=area_of_exudate,
              color=class_list)
fig.update_traces(marker=dict(size=2,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()