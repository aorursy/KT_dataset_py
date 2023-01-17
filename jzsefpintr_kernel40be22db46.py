import numpy as np 

import pandas as pd 

import os

from PIL import Image

from matplotlib.image import imread
picnames=[] #Betöltöm a képek neveit.

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        picnames.append(os.path.join(dirname, filename))

picnames=picnames[2:] #Az első kettő az más fájlok.

picnames.sort()
df=pd.read_csv('/kaggle/input/football-player-number-13/train_solutions.csv')

trainnevek=df["Id"]

trainnevek=list(trainnevek)

for i in range(len(trainnevek)):

    trainnevek[i]=trainnevek[i].replace("-","_")

    trainnevek[i]+=".jpg"

    trainnevek[i]='/kaggle/input/football-player-number-13/images/'+trainnevek[i]

trainnevek.sort()
X_train=[]

n=0

for i in trainnevek:

    n+=1

    if n%500==0:

        print(n)

    img = Image.open(i).resize((480,270)) # Ekkora dimenziós a legkisebb kiskép

    img = np.asarray(img)

    X_train.append(img)
X_train=np.array(X_train)

X_train.shape
y_train=list(df.sort_values(by='Id')['Predicted'])

for i in range(len(y_train)):

    if y_train[i]==True:

        y_train[i]=1

    else:

        y_train[i]=0

y_train=np.array(y_train)

y_train
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg19 import preprocess_input

from keras.layers import Input, Flatten, Dense

from keras.models import Model, Sequential



base_model = VGG16(

    weights="imagenet",

    include_top=False,

    input_shape=(270,480,3)

)





model = Sequential()

model.add(base_model)

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



model.layers[0].trainable = False



model.compile(

    loss='binary_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)



model.summary()
model.fit(X_train,y_train, epochs=15)
from collections import Counter
df2=df=pd.read_csv('/kaggle/input/football-player-number-13/sampleSubmissionAllZeros.csv')
testnevek=list(df2['Id'])

for i in range(len(testnevek)):

    testnevek[i]=testnevek[i].replace("-","_")

    testnevek[i]='/kaggle/input/football-player-number-13/images/'+testnevek[i]+".jpg"
X_test=[]

n=0

for i in testnevek:

    n+=1

    if n%500==0:

        print(n)

    img = Image.open(i).resize((480,270)) # Ekkora dimenziós a legkisebb kiskép

    img = np.asarray(img)

    X_test.append(img)

X_test=np.array(X_test)
y_test=model.predict(X_test)

y_test=y_test.tolist()

for i in range(len(y_test)):

    y_test[i]=y_test[i][0]
for i in range(len(y_test)):

    if y_test[i]>0.5:

        y_test[i]=True

    else:

        y_test[i]=False

Counter(y_test)
df2["Predicted"]=y_test
df2.to_csv('predictions77.csv',index=False)
import re 
picnames=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        picnames.append(os.path.join(dirname, filename))

picnames=picnames[2:]

picnames2=[]

for i in picnames:

    if bool(re.search("orig",i)):

        picnames2.append(i)

picnames2.sort()
X_asarray=[]
X_asarray=np.array(X_asarray)
df=pd.read_csv('/kaggle/input/football-player-number-13/train_solutions.csv')

y_train=list(df.sort_values(by='Id')['Predicted'])

trainnevek=df["Id"]

trainnevek2=[]

for i in range(len(trainnevek)):

    if i%16==0:

        trainnevek2.append(trainnevek[i][0:len(trainnevek[i])-2]) #Mindegyik 'képcsoportból' egy elem kell, de azt leveszem, hogy a 0.

print(trainnevek2[0:5])

picnames3=[]

for i in trainnevek2: #Azok kellenek, amik originalok és trainek.

    for j in picnames2:

        if bool(re.search(i,j)):

            picnames3.append(j)
X_train=[]

n=0

for i in picnames3:

    n+=1

    if n%40==0:

        print(n)

    img = Image.open(i).resize((960,540)) # Nagyobb méretnél sajnos OOM

    img = np.asarray(img)

    X_train.append(img)
X_train=np.array(X_train)

X_train.shape
y_train=list(df.sort_values(by='Id')['Predicted'])

for i in range(len(y_train)):

    if y_train[i]==True:

        y_train[i]=1

    else:

        y_train[i]=0

y_train=np.array(y_train)
y_train2=[]

for i in range(int(len(y_train)/16)):

    l=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for j in range(16):

        if j==0:

            l[0]=y_train[i*16+0]

        if j==1:

            l[9]=y_train[i*16+1]

        if j==2:

            l[10]=y_train[i*16+2]

        if j==3:

            l[11]=y_train[i*16+3]

        if j==4:

            l[12]=y_train[i*16+4]

        if j==5:

            l[13]=y_train[i*16+5]

        if j==6:

            l[14]=y_train[i*16+6]

        if j==7:

            l[15]=y_train[i*16+7]

        if j==8:

            l[1]=y_train[i*16+8]

        if j==9:

            l[2]=y_train[i*16+9]

        if j==10:

            l[3]=y_train[i*16+10]

        if j==11:

            l[4]=y_train[i*16+11]

        if j==12:

            l[5]=y_train[i*16+12]

        if j==13:

            l[6]=y_train[i*16+13]

        if j==14:

            l[7]=y_train[i*16+14]

        if j==15:

            l[8]=y_train[i*16+15]          

    y_train2.append(l)
y_train2=np.array(y_train2)

y_train2[0:5]
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

from keras.layers import Input, Flatten, Dense

from keras.models import Model

import numpy as np



#Get back the convolutional part of a VGG network trained on ImageNet



base_model = VGG16(

    weights="imagenet",

    include_top=False,

    input_shape=(540,960,3)

)



vgg16 = Sequential()

vgg16.add(base_model)

vgg16.add(Flatten())

vgg16.add(Dropout(0.5))

vgg16.add(Dense(16, activation='sigmoid'))





vgg16.layers[0].trainable = False

vgg16.compile(loss='binary_crossentropy',

              optimizer="sgd",

              metrics=["accuracy"])



vgg16.summary()
vgg16.fit(X_train, y_train2, epochs=15, validation_split=0.3)
df2=pd.read_csv('/kaggle/input/football-player-number-13/sampleSubmissionAllZeros.csv')

testnevek=df2["Id"]

testnevek2=[]

for i in range(len(testnevek)):

    if i%16==0:

        testnevek2.append(testnevek[i][0:len(testnevek[i])-2])

picnames4=[]

for i in testnevek2:

    for j in picnames2:

        if bool(re.search(i,j)):

            picnames4.append(j)
X_test=[]

n=0

for i in picnames4:

    n+=1

    if n%40==0:

        print(n)

    img = Image.open(i).resize((960,540))

    img = np.asarray(img)

    X_test.append(img)

X_test=np.array(X_test)
y_test=vgg16.predict(X_test)

y_predict=[]

for i in y_test: #Mivel most nem 16 hosszú listára van szükségünk, hanem a subpicture-ökhöz 0-1 labelekre.

    for j in i:

        y_predict.append(j)
predictions=[]

for i in range(len(y_predict)):

    if y_predict[i]>0.5:

        predictions.append(True)

    else:

        predictions.append(False)

Counter(predictions)
df2["Predicted"]=predictions
df2.to_csv('predictions81.csv',index=False)