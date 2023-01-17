# Importing the usual libraries and filter warnings

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pyplot import xticks

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#train = pd.read_csv('train.csv')

#test = pd.read_csv('test.csv')



train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



print(train.shape,test.shape)

#In the beginning it's important to check the size of your train and test data which later helps in 

#deciding the sample size while testing your model on train data
train.head(5)
test.head(5)
# Lets see if we have a null value in the whole dataset

#Usuall we will check isnull().sum() but here in our dataset we have 784 columns and a groupby won't fit the buffer

print(np.unique([train.isnull().sum()]))

print(np.unique([test.isnull().sum()]))
y = train['label']

df_train = train.drop(columns=["label"],axis=1)

print(y.shape,df_train.shape)
sns.countplot(y)
#Lets see the first 50 images of the dataset

df_train_img = df_train.values.reshape(-1,28,28,1)

plt.figure(figsize=(15,8))

for i in range(50):

    plt.subplot(5,10,i+1)

    plt.imshow(df_train_img[i].reshape((28,28)),cmap='gray')

    plt.axis("off")

plt.show()
def augment(df_aug,y):

    col_list = df_aug.columns.tolist()

    col_list = ['label']+col_list

    list1=[]

    list2=[]

    list3=[]

    global df1

    global df2

    global df3

    df_train_img = df_aug.values.reshape(-1,28,28,1)

    for i in range(len(df_aug)):

        list1.append([y[i]]+np.rot90(df_train_img[i],1).flatten().tolist())

        list2.append([y[i]]+np.rot90(df_train_img[i],2).flatten().tolist())

        list3.append([y[i]]+np.rot90(df_train_img[i],3).flatten().tolist())

    df1= pd.DataFrame(list1,columns=col_list)

    df2 = pd.DataFrame(list2,columns=col_list)

    df3 = pd.DataFrame(list3,columns=col_list)
#Function is called

augment(df_train,y)
#3 new dataframes are created with same size as tarining set as expected

print(df1.shape,df2.shape,df3.shape)
df_train1 = df1.drop(columns=["label"],axis=1)

df_train_img1 = df_train1.values.reshape(-1,28,28,1)

#Lets see the first 50 images of the dataset

plt.figure(figsize=(15,8))

for i in range(50):

    plt.subplot(5,10,i+1)

    plt.imshow(df_train_img1[i].reshape((28,28)),cmap='gray')

    plt.axis("off")

plt.show()

df_train2 = df2.drop(columns=["label"],axis=1)

df_train_img2 = df_train2.values.reshape(-1,28,28,1)

#Lets see the first 50 images of the dataset

plt.figure(figsize=(15,8))

for i in range(50):

    plt.subplot(5,10,i+1)

    plt.imshow(df_train_img2[i].reshape((28,28)),cmap='gray')

    plt.axis("off")

plt.show()

df_train3 = df3.drop(columns=["label"],axis=1)

df_train_img3 = df_train3.values.reshape(-1,28,28,1)

#Lets see the first 50 images of the dataset

plt.figure(figsize=(15,8))

for i in range(50):

    plt.subplot(5,10,i+1)

    plt.imshow(df_train_img3[i].reshape((28,28)),cmap='gray')

    plt.axis("off")

plt.show()

#Lets merge all the dataframes

frames = [train,df1,df2,df3]

final_df = pd.concat(frames)

final_df.shape
y = final_df['label']

df_train = final_df.drop(columns=["label"],axis=1)

print(y.shape,df_train.shape)
# Normalize the dataset

df_train = df_train / 255

test = test / 255
#Looks like the values are equally distributed in the dataset

y.value_counts()
sns.countplot(y)
fig,axs = plt.subplots(2,2)

axs[0,0].imshow(df_train_img[26].reshape((28,28)),cmap='gray')

axs[0,1].imshow(df_train_img[27].reshape((28,28)),cmap='gray')

axs[1,0].imshow(df_train_img2[26].reshape((28,28)),cmap='gray')

axs[1,1].imshow(df_train_img2[27].reshape((28,28)),cmap='gray')

axs[0, 0].set_title("label="+str(y.iloc[26]))

axs[0, 1].set_title("label="+str(y.iloc[27]))

axs[1, 0].set_title("label="+str(y.iloc[26]))

axs[1, 1].set_title("label="+str(y.iloc[27]))

fig.tight_layout()