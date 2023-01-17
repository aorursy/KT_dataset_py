# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input"))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os, sys

import matplotlib.pyplot as plt

import cv2

from sklearn.model_selection import train_test_split

from PIL import Image

import warnings

warnings.filterwarnings("ignore")

IMG_SIZE = 512
df_train=pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

df_test=pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
df_train.head()
x=df_train["id_code"]

y=df_train["diagnosis"]
labels=["Train","Test"]

sizes=[len(df_train),len(df_test)]



plt.pie(sizes,labels=labels,autopct='%1.1f%%')

plt.show()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42)

print("x_train shape: ", x_train.shape)

print("x_test shape: ", x_test.shape)

print("y_train shape: ", y_train.shape)

print("y_test shape: ", y_test.shape)
plt.hist(y_train,label="y_train")

plt.hist(y_test,label="y_test")

plt.title("Retinopathy Type and number")

plt.xlabel("0:No DR, 1:Mild, 2:Moderate, 3:Severe, 4:Proliferative DR")

plt.ylabel("Number")

plt.legend()

plt.show()
 %%time

fig = plt.figure(figsize=(25, 16))

# display 10 images from each class

for i in sorted(y_train.unique()):

    for j, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == i].sample(5, random_state=42).iterrows()):

        ax = fig.add_subplot(5, 5, i * 5 + j + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        plt.imshow(image)

        ax.set_title('Label: %d-%d-%s' % (i, idx, row['id_code']) )

        

%%time

fig=plt.figure(figsize=(25,16))

for i in sorted(y_train.unique()):

    for j ,(idx,row) in enumerate(df_train.loc[df_train["diagnosis"]==i].sample(5,random_state=42).iterrows()):

        ax = fig.add_subplot(5, 5, i*5+j+1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))



        plt.imshow(image)

        ax.set_title('Label: %d-%d-%s' % (i, idx, row['id_code']) )
%%time

fig=plt.figure(figsize=(25,16))

for i in sorted(y_train.unique()):

    for j ,(idx,row) in enumerate(df_train.loc[df_train["diagnosis"]==i].sample(5,random_state=42).iterrows()):

        ax = fig.add_subplot(5, 5, i*5+j+1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))



        plt.imshow(image,cmap="gray")

        ax.set_title('Label: %d-%d-%s' % (i, idx, row['id_code']) )
dpi=80



path=f"../input/aptos2019-blindness-detection/train_images/838c87c63422.png"

image = cv2.imread(path)

image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

height,width=image.shape

print(height,width)

SCALE=2

figsize=(width/float(dpi))/SCALE,(height/float(dpi))/SCALE

image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

fig=plt.figure(figsize=figsize)

plt.imshow(image,cmap="gray")
fig=plt.figure(figsize=(25,16))

for i in sorted(y_train.unique()):

    for j,(idx,row) in enumerate(df_train.loc[df_train["diagnosis"]==i].sample(5,random_state=42).iterrows()):

        

        ax = fig.add_subplot(5, 5, i*5+j+1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , IMG_SIZE/10) ,-4 ,128) 



        plt.imshow(image, cmap='gray')

        ax.set_title('Label: %d-%d-%s' % (i, idx, row['id_code']) )
def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img
def load_ben_color(path, sigmaX=10):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

        

    return image
NUM_SAMP=7

fig = plt.figure(figsize=(25, 16))

for i in sorted(y_train.unique()):

    for j, (idx, row) in enumerate(df_train.loc[df_train['diagnosis'] == i].sample(NUM_SAMP, random_state=42).iterrows()):

        ax = fig.add_subplot(5, NUM_SAMP, i* NUM_SAMP + j + 1, xticks=[], yticks=[])

        path=f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png"

        image = load_ben_color(path,sigmaX=30)



        plt.imshow(image)

        ax.set_title('%d-%d-%s' % (i, idx, row['id_code']) )
def load_ben_color2(path, sigmaX=10):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

  

        

    return image
dpi = 80 #inch



path=f"../input/aptos2019-blindness-detection/train_images/838c87c63422.png" 

image = load_ben_color(path,sigmaX=10)



height, width = IMG_SIZE, IMG_SIZE

print(height, width)



SCALE=1

figsize = (width / float(dpi))/SCALE, (height / float(dpi))/SCALE



fig = plt.figure(figsize=figsize)

plt.imshow(image, cmap='gray')



dpi = 80 #inch



path=f"../input/aptos2019-blindness-detection/train_images/838c87c63422.png" 

image = load_ben_color2(path,sigmaX=10)



height, width = IMG_SIZE, IMG_SIZE

print(height, width)



SCALE=1

figsize = (width / float(dpi))/SCALE, (height / float(dpi))/SCALE



fig = plt.figure(figsize=figsize)

plt.imshow(image, cmap='gray')





dpi = 80 #inch



path=f"../input/aptos2019-blindness-detection/train_images/78937523f7a8.png" 

image = load_ben_color(path,sigmaX=10)



height, width = IMG_SIZE, IMG_SIZE

print(height, width)



SCALE=1

figsize = (width / float(dpi))/SCALE, (height / float(dpi))/SCALE



fig = plt.figure(figsize=figsize)

plt.imshow(image, cmap='gray')

dpi = 80 #inch





path=f"../input/aptos2019-blindness-detection/train_images/838c87c63422.png" 

image = load_ben_color(path,sigmaX=10)



height, width = IMG_SIZE, IMG_SIZE

print(height, width)



SCALE=1

figsize = (width / float(dpi))/SCALE, (height / float(dpi))/SCALE



fig = plt.figure(figsize=figsize)

plt.imshow(image, cmap='gray')





train=pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train.head()
patient=train[train["diagnosis"]==4]

healt=train[train["diagnosis"]==0]

train_df=pd.concat([patient,healt])
train_df["diagnosis"]=[1 if i==4 else 0 for i in train_df.diagnosis]
train_df.shape
x=train_df.drop(columns=["diagnosis"])

y=train_df.diagnosis
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
dn_x = [ Image.open('../input/aptos2019-blindness-detection/train_images/'+i+'.png') for i in x_train.id_code[:5]]
dn_x
for i in dn_x:

    

    plt.figure(figsize=(5,3))

    i = cv2.resize(np.asarray(i),(64,64))

    i= cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)

    plt.imshow(i,cmap="gray")

    plt.axis("off")

    plt.show
x_train = [cv2.resize(np.asarray(Image.open('../input/aptos2019-blindness-detection/train_images/'+i+'.png').convert("L")),(64,64)) for i in x_train.id_code]
x_test = [cv2.resize(np.asarray(Image.open('../input/aptos2019-blindness-detection/train_images/'+i+'.png').convert("L")),(64,64)) for i in x_test.id_code]
x_test=np.array(x_test)

x_test.shape
x_train=np.array(x_train)

x_train.shape
x_train_flatten=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])

x_test_flatten=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
x_train=x_train_flatten.T

y_train=y_train.T

x_test=x_test_flatten.T

y_test=y_test.T

print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
y_train=np.array(y_train)

y_test=np.array(y_test)
y_train=y_train.reshape(-1,1)

y_test=y_test.reshape(-1,1)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
def initialize_weights_and_bias(dimension):

    w=np.full((dimension,1),0.01)

    b=0.0

    return w,b
def sigmoid(z):

    # sigmoid function is 1/(1+e^-z)

    y_head=1/(1+np.exp(-z))

    return y_head
def forward_propagation(w,b,x_train,y_train):

    z=np.dot(w.T,x_train)+b

    y_head=sigmoid(z)

    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost=(np.sum(loss))/x_train.shape[1]

    return cost
def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z=np.dot(w.T,x_train)+b

    y_head=sigmoid(z)

    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost=(np.sum(loss))/x_train.shape[1]

    #backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]

    gradients={"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}

    return cost,gradients

    
# Updating(learning) parameters



def update(w,b,x_train,y_train,learning_rate,number_of_iteration):

    cost_list=[]

    cost_list2=[]

    index=[]

    for i in range(number_of_iteration):

        cost,gradients=forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        w=w-learning_rate*gradients["derivative_weight"]

        b=b-learning_rate*gradients["derivative_bias"]

    parameters ={"weight":w,"bias":b}

    

    return parameters,gradients,cost_list
# prediction

def predict(w,b,x_test):

    z=sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction=np.zeros((1,x_test.shape[1]))

    for i in range(z.shape[1]):

        if z[0,i]<=0.5:

            Y_prediction[0,i]=0

        else:

            Y_prediction[0,i]=1

    return Y_prediction   

# Logistic regression initialize

def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations):

    dimension=x_train.shape[0]

    w,b=initialize_weights_and_bias(dimension)

    parameters,gradients,cost_list=update(w,b,x_train,y_train,learning_rate,num_iterations)

    y_prediction_test=predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train=predict(parameters["weight"],parameters["bias"],x_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 50)
from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)

print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train).score(x_test.T, y_test)))