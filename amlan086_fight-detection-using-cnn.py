# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
img_rows, img_cols = 128, 64

input_shape = (img_rows, img_cols, 1)

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix
import math

import numpy as np

from PIL import Image

import os

import random

mm=0

X=[]

y=[]

for z in range(1,501):

    cnt=0

    if z%4!=0:

        com_list=[]

        

        ss='/kaggle/input/h_f (2)/h_f/h_f_selected/data_'+str(z)

        if not os.path.exists(ss):

            continue

        os.chdir(ss)

        

        file_list = os.listdir(ss) 

        number_of_files = len(file_list)

        #print(number_of_files)

        







        for x in range(number_of_files-1):

            res=[]

            #print(x)

            s=file_list[x]

            #print(s)

            im = Image.open(s, 'r')

            gray_list= list(im.getdata())

            X.append(gray_list)

            y.append(1)

            

            

        
import math

import numpy as np

from PIL import Image

import os

import random

mm=0



for z in range(1,501):

    cnt=0

    if z%6!=0:

        com_list=[]

        

        ss='/kaggle/input/h_f (2)/h_f/h_nf_selected/ndata_'+str(z)

        if not os.path.exists(ss):

            continue

        os.chdir(ss)

        

        file_list = os.listdir(ss) 

        number_of_files = len(file_list)

        #print(number_of_files)

        







        for x in range(number_of_files-1):

            res=[]

            #print(x)

            s=file_list[x]

            #print(s)

            im = Image.open(s, 'r')

            gray_list= list(im.getdata())

            X.append(gray_list)

            y.append(0)

            

            

        
import math

import numpy as np

from PIL import Image

import os

import random

mm=0

X_t=[]

y_t=[]

for z in range(1,501):

    cnt=0

    if z%5==0:

        com_list=[]

        

        ss='/kaggle/input/h_f (2)/h_f/h_f_selected/data_'+str(z)

        if not os.path.exists(ss):

            continue

        os.chdir(ss)

        

        file_list = os.listdir(ss) 

        number_of_files = len(file_list)

        #print(number_of_files)

        







        for x in range(number_of_files-1):

            res=[]

            #print(x)

            s=file_list[x]

            #print(s)

            im = Image.open(s, 'r')

            gray_list= list(im.getdata())

            X_t.append(gray_list)

            y_t.append(1)

            

            

        
import math

import numpy as np

from PIL import Image

import os

import random

mm=0



for z in range(1,501):

    cnt=0

    if z%5==0:

        com_list=[]

        

        ss='/kaggle/input/h_f (2)/h_f/h_nf_selected/ndata_'+str(z)

        if not os.path.exists(ss):

            continue

        os.chdir(ss)

        

        file_list = os.listdir(ss) 

        number_of_files = len(file_list)

        #print(number_of_files)

        







        for x in range(number_of_files-1):

            res=[]

            #print(x)

            s=file_list[x]

            #print(s)

            im = Image.open(s, 'r')

            gray_list= list(im.getdata())

            X_t.append(gray_list)

            y_t.append(0)

            

            

        
X=np.array(X)

y=np.array(y)

X_train=X

y_train=y


X_t=np.array(X_t)

y_t=np.array(y_t)



X_test=X_t

y_test=y_t
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 62)
#normalizing the data

X_train, X_test = X_train / 255.0, X_test / 255.0
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(X_train)

X_train = sc.transform(X_train)

X_test= sc.transform(X_test)

import keras

from keras.utils import np_utils 

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

y_train[900]
X_train.shape 
X_test.shape
X_train = X_train.reshape(X_train.shape[0],128,64,1)

X_test = X_test.reshape(X_test.shape[0],128,64,1)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten

#create model

model = Sequential()

#add model layers

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(128,64,1)))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20)
ss='/kaggle/input/h_f (2)/h_f/h_f_selected/data_10'

if os.path.exists(ss):

    os.chdir(ss)



file_list = os.listdir(ss) 

number_of_files = len(file_list)

res=[]

s=file_list[4]

            #print(s)

im = Image.open(s, 'r')

gray_list= list(im.getdata())



x=np.array(gray_list)

x=x.reshape(1,128,64,1)

y_prob = model.predict(x) 

y_classes = y_prob.argmax(axis=-1)

print(y_classes)





import math

import numpy as np

from PIL import Image

import os

import random

mm=0

acc=0

tot=0

TN=0

TP=0

FN=0

FP=0



for z in range(1,501):

    cnt=0

    if z%5==0:

        tot=tot+1

        X_t=[]

        y_t=[]

        com_list=[]

        

        ss='/kaggle/input/h_f (2)/h_f/h_f_selected/data_'+str(z)

        if not os.path.exists(ss):

            continue

        os.chdir(ss)

        

        file_list = os.listdir(ss) 

        number_of_files = len(file_list)

        #print(number_of_files)

        







        for x in range(number_of_files-1):

            res=[]

            #print(x)

            s=file_list[x]

            #print(s)

            im = Image.open(s, 'r')

            gray_list= list(im.getdata())

            x=np.array(gray_list)

            x=x.reshape(1,128,64,1)

            y_prob = model.predict(x) 

            y_classes = y_prob.argmax(axis=-1)

            cnt=cnt+y_classes

       

                   

        #print(cnt)

        if cnt>(number_of_files/2):

            acc=acc+1

            print(str(z)+"fight")

            TP=TP+1

        else:

            print(str(z)+"non fight")

            FN=FN+1    

        

        

        





for z in range(1,501):

    cnt=0

    if z%5==0:

        tot=tot+1

        X_t=[]

        y_t=[]

        com_list=[]

        

        ss='/kaggle/input/h_f (2)/h_f/h_nf_selected/ndata_'+str(z)

        if not os.path.exists(ss):

            continue

        os.chdir(ss)

        

        file_list = os.listdir(ss) 

        number_of_files = len(file_list)

        #print(number_of_files)

        







        for x in range(number_of_files-1):

            res=[]

            #print(x)

            s=file_list[x]

            #print(s)

            im = Image.open(s, 'r')

            gray_list= list(im.getdata())

            x=np.array(gray_list)

            x=x.reshape(1,128,64,1)

            y_prob = model.predict(x) 

            y_classes = y_prob.argmax(axis=-1)

            cnt=cnt+y_classes

       

                   

        #print(cnt)

        

        if cnt>(number_of_files/2):

            print(str(z)+"fight")

            FP=FP+1

        else:

            acc=acc+1

            print(str(z)+"non fight")

            TN=TN+1  

        

        

        

acuracy=(acc/tot)*100

print('accuracy : '+str(acuracy))

precision=TP/(TP+FP)

recall=TP/(TP+FN)

f1=2*(precision*recall)/(precision+recall)

mathews=float(TP*TN-FP*FN)/math.sqrt(float((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))





print(str(TN)+' '+str(FP))

print(str(FN)+' '+str(TP))



print('precision : '+str(precision))

print('recall : '+str(recall))

print('f1-score : '+str(f1))

print('mathews coefficient : '+str(mathews))

            

        

            

        
import math

import numpy as np

from PIL import Image

import os

import random

mm=0

acc=0

TN=0

TP=0

FN=0

FP=0



for z in range(1,501):

    cnt=0

    if z%5==0:

        X_t=[]

        y_t=[]

        com_list=[]

        

        ss='/kaggle/input/h_f (2)/h_f/h_f_selected/data_'+str(z)

        if not os.path.exists(ss):

            continue

        os.chdir(ss)

        

        file_list = os.listdir(ss) 

        number_of_files = len(file_list)

        #print(number_of_files)

        







        for x in range(number_of_files-1):

            res=[]

            #print(x)

            s=file_list[x]

            #print(s)

            im = Image.open(s, 'r')

            gray_list= list(im.getdata())

            x=np.array(gray_list)

            x=x.reshape(1,128,64,1)

            y_prob = model.predict(x) 

            y_classes = y_prob.argmax(axis=-1)

            cnt=cnt+y_classes

       

                   

        #print(cnt)

        

        if cnt>(number_of_files/2):

            print("fight")

            FP=FP+1

        else:

            acc=acc+1

            print("non fight")

            TN=TN+1  

        

        

        

acuracy=(acc/tot)*100

print('accuracy : '+str(acuracy))

precision=TP/(TP+FP)

recall=TP/(TP+FN)

f1=2*(precision*recall)/(precision+recall)

#mathews=float(TP*TN-FP*FN)/math.sqrt(float((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))





print(str(TN)+' '+str(FP))

print(str(FN)+' '+str(TP))



print('precision : '+str(precision))

print('recall : '+str(recall))

print('f1-score : '+str(f1))

#print('mathews coefficient : '+str(mathews))

            

        


labels = np.argmax(pred, axis=-1)    

print(labels)
import math

import numpy as np

from PIL import Image

import os

import random

mm=0



for z in range(1,501):

    cnt=0

    if z%5==0:

        X_t=[]

        y_t=[]

        com_list=[]

        

        ss='/kaggle/input/h_f (2)/h_f/h_nf_selected/ndata_'+str(z)

        if not os.path.exists(ss):

            continue

        os.chdir(ss)

        

        file_list = os.listdir(ss) 

        number_of_files = len(file_list)

        #print(number_of_files)

        







        for x in range(number_of_files-1):

            res=[]

            #print(x)

            s=file_list[x]

            #print(s)

            im = Image.open(s, 'r')

            gray_list= list(im.getdata())

            X_t.append(gray_list)

            y_t.append(1)

        X_t=np.array(X_t)

        y_t=np.array(y_t)

        X_t=X_t/255.0

       

        X_t = X_t.reshape(X_t.shape[0],128,64,1)

        print(model.predict(X_t))

        

        

        



            

        
labels = np.argmax(pred1, axis=-1)    

print(labels)
X_t=np.array(X_t)

y_t=np.array(y_t)



X_t=X_t/255.0



from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(X_t)

X_t = sc.transform(X_t)

X_test= sc.transform(X_test)

X_t.shape[0]
X_t = X_t.reshape(X_t.shape[0],128,64,1)
model.predict(X_t)