# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math as m

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
prr40_1 = pd.read_csv("../input/input1/prr40_1.csv")

prr40_1 = np.asarray(prr40_1)
z=np.zeros((len(prr40_1),1))

prr40_1=np.append(prr40_1,z,axis=1)
for i in range(len(prr40_1)):

    if(prr40_1[i][2]>=90):

        prr40_1[i][4]=1
np.savetxt('data.csv', prr40_1, delimiter=',', fmt='%d')
data = pd.read_csv("../input/input2/data.csv")
# input 

x = data.iloc[:, [0,1,2,3]].values 

  

# output 

y = data.iloc[:, 4].values 
from sklearn.model_selection import train_test_split 

xtrain, xtest, ytrain, ytest = train_test_split( 

        x, y, test_size = 0.25, random_state = 0) 
from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) 

classifier.fit(xtrain, ytrain) 
y_pred = classifier.predict(xtest) 
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(ytest, y_pred) 

  

print ("Confusion Matrix : \n", cm) 
from sklearn.metrics import accuracy_score 

print ("Accuracy : ", accuracy_score(ytest, y_pred)) 
n=40

best=[[[-1,-1]for i in range(n)]for j in range(n)]
for i in range(len(prr40_1)):

    x,y=int(prr40_1[i][0]),int(prr40_1[i][1])

    x=x-1

    y=y-1

    if best[x][y][0]<prr40_1[i][2]:

        best[x][y][0]=prr40_1[i][2]

        best[x][y][1]=int(prr40_1[i][3])
best[12][16][1]
count_0=[0 for i in range (16)]

count_90=[0 for i in range(16)]
for i in range(len(best)):

    for j in range(len(best)):

        if best[i][j][0]>=90:

            x=int(best[i][j][1])

            x=x-11

            count_0[x]+=1

            count_90[x]+=1

        elif best[i][j][0]>0:

            x=int(best[i][j][1])

            x=x-11

            count_0[x]+=1
count_0
count_90
x=[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]

#y=[count_90,count_0]

#plt.bar(x,y=[count_0,count_90],rot=40)

x
z =np.arange(16)

ax1 = plt.subplot(1,1,1)

w = 0.3

ninety =ax1.bar(z+11,count_90, width=w, color='b', align='center')

ax2 = ax1.twinx()

zero =ax2.bar(z +11+ w, count_0, width=w,color='g',align='center')

plt.xlabel('Channel')

plt.ylabel('Number of links')

plt.legend([ninety,zero ],['PRR>=90', 'PRR>0'])

plt.show()

merged_10csv = pd.read_csv("../input/180files-csv/180_files_csv.csv")

merged_10csv = np.asarray(merged_10csv)

n=40

reward=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for i in range(0,16):

    reward[i]=[[[0,0]for i in range(n)]for j in range(n)]    
temp=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

count=0

for i in range(len(merged_10csv)):

    x,y=merged_10csv[i][0],merged_10csv[i][1]

    x=x-1

    y=y-1

    k=merged_10csv[i][3]

    k=k-11

    reward[k][x][y][0]+=pow((merged_10csv[i][2]-100),2)

    temp[k]+=1
for i in range(0,16):

    for x in range(0,40):

        for y in range(0,40):

            reward[i][x][y][0]/=180
for i in range(0,16):

    for x in range(0,40):

        for y in range(0,40):

            reward[i][x][y][0]=m.sqrt(reward[i][x][y][0])
for i in range(0,16):

    for x in range(40):

        for y in range(40):

            if(reward[i][x][y][0]>=90):

                reward[i][x][y][1]=1

            elif(reward[i][x][y][0]>=80):

                reward[i][x][y][1]=2

            elif(reward[i][x][y][0]>=70):

                reward[i][x][y][1]=3

            elif(reward[i][x][y][0]>=60):

                reward[i][x][y][1]=4

            elif(reward[i][x][y][0]>=50):

                 reward[i][x][y][1]=5

            elif(reward[i][x][y][0]>=40):

                 reward[i][x][y][1]=6

            elif(reward[i][x][y][0]>=30):

                 reward[i][x][y][1]=7

            elif(reward[i][x][y][0]>=20):

                 reward[i][x][y][1]=8

            elif(reward[i][x][y][0]>=10):

                 reward[i][x][y][1]=9

            elif(reward[i][x][y][0]>=0):

                 reward[i][x][y][1]=10
x=input("Enter the source node: ")

y=input("Enter the destination node: ")

x=int(x)

y=int(y)

x=x-1

y=y-1

blacklist=[]

whitelist=[]

max_reward=6

best_channel=0

for i in range(0,16):

    if(reward[i][x][y][1]<7):

        blacklist.append(i+11)

    else:

        whitelist.append(i+11)

        if(reward[i][x][y][1]>max_reward):

            max_reward=reward[i][x][y][1]

            best_channel=i+11

print("The blacklisted channels are:")

print(blacklist)

print("The whitelisted channels are:")

print(whitelist)

print("The best channel for this link according to module_3:")

print(best_channel)

print("The best channel according to module_2:")

print(best[x][y][1])