import pandas as pd

import matplotlib.pyplot as plt

import cv2

import seaborn as sns

df = pd.read_csv('../input/train.csv')





Cases = {

            'No_DR' : df['id_code'][df['diagnosis']==0][:9],

            'Mild' : df['id_code'][df['diagnosis']==1][:9],

            'Moderate' : df['id_code'][df['diagnosis']==2][:9],

            'Severe' : df['id_code'][df['diagnosis']==3][:9],

            'Proliferative_DR' : df['id_code'][df['diagnosis']==4][:9]

}

def readImg(lis):

    ret = []

    for i in lis:

        img = cv2.resize(cv2.cvtColor(cv2.imread('../input/train_images/'+i+'.png'), cv2.COLOR_BGR2RGB), (300,300))

        ret.append(img)

    return ret
fig, ax = plt.subplots(3,3, figsize=(20,20))

t=list(Cases)[0]

lis = readImg(Cases[t])

for i in range(3):

    for j in range(3):

        ax[i][j].imshow(lis[i*3+j])
fig, ax = plt.subplots(3,3, figsize=(20,20))

t=list(Cases)[1]

lis = readImg(Cases[t])

for i in range(3):

    for j in range(3):

        ax[i][j].imshow(lis[i*3+j])



fig, ax = plt.subplots(3,3, figsize=(20,20))

t=list(Cases)[2]

lis = readImg(Cases[t])

for i in range(3):

    for j in range(3):

        ax[i][j].imshow(lis[i*3+j])



fig, ax = plt.subplots(3,3, figsize=(20,20))

t=list(Cases)[3]

lis = readImg(Cases[t])

for i in range(3):

    for j in range(3):

        ax[i][j].imshow(lis[i+j])



fig, ax = plt.subplots(3,3, figsize=(20,20))

t=list(Cases)[4]

lis = readImg(Cases[t])

for i in range(3):

    for j in range(3):

        ax[i][j].imshow(lis[i*3+j])



sns.countplot(df.diagnosis)

plt.title("Distrubation of Classes")

plt.show()
df.head()
df.describe()
df.info()
df.diagnosis.value_counts()