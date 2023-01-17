import cv2

import numpy as np

import matplotlib.pyplot as plt

import os

%matplotlib inline

print(os.listdir("../input"))
image=("../input/coin.jpg")

img=cv2.imread(image,1)

img_orig=img.copy()

img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.rcParams["figure.figsize"]=(16,9)

plt.imshow(img,cmap='gray')
img=cv2.GaussianBlur(img,(29,29),cv2.BORDER_DEFAULT)

plt.rcParams["figure.figsize"]=(16,9)

plt.imshow(img)
all_coins=cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,0.9,120,param1=50,param2=30,minRadius=100,maxRadius=185)

all_coins_r=np.uint16(np.around(all_coins))

#Detect the coins
a=all_coins_r

print(all_coins_r.shape)

print('found'+str(all_coins_r.shape[1]))

print(all_coins_r)

#the below list contains x,y,r meaning x,y axes points of the coins and their respective radius
c=1

for i in all_coins_r[0,:]:

    cv2.circle(img_orig,(i[0],i[1]),i[2],(50,200,200),5)

    cv2.circle(img_orig,(i[0],i[1]),2,(255,0,0),3)

    cv2.putText(img_orig,"coin"+str(c),(i[0]-70,i[1]+30),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,0,0),10)

    c+=1
plt.rcParams["figure.figsize"]=(16,9)

plt.imshow(img_orig)

#Detected coins circled


b=[]

for i in a :

    for j in i:

       b.append(j)  

print(b)     
b=sorted(b,key=lambda x:x[0])
for i in b:

    print(i)

#sort the array wrt to x axis     
from collections import defaultdict

#for coins before a particular coin find if the value(x+r) of all coins before it is greater than or equal to the x of the coin

#if yes append it to the dict

#similarly for coins after a particular coin check as done below

#this is done for each and every coin(13 coins)

def algo(b):

    c=defaultdict(list)

    for i in range(len(b)):

        for j in range(0,i):

            r=b[j][0]+b[j][2]

            if(r>=b[i][0]):

                c[i].append(j)

        for k in range(i+1,len(b)):

            r=b[k][0]-b[k][2]

            if(r<=b[i][0]):

                c[i].append(k)

    return(c)

#the algorithm for finding coins which fall under the same line
algo(b)
img_orig2=img_orig.copy()

img_orig2=cv2.line(img_orig2,(706,0),(706,4000),(255,0,0),10)

img_orig2=cv2.line(img_orig2,(1888,0),(1888,4000),(255,0,0),10)

img_orig2=cv2.line(img_orig2,(2630,0),(2630,4000),(255,0,0),10)



plt.imshow(img_orig2)