# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=[16,10]

import seaborn as sns



%matplotlib inline

plt.rcParams['axes.unicode_minus']=False

import cv2

import os

import matplotlib.image as img

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_csv=pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

train_csv_idcode_as_index=train_csv.set_index('id_code')

#check if all images entry is once in csv

#class frequency

labels_df=train_csv['diagnosis']

labels_df.value_counts()



#ploting of each class data

plt.title='Test Data'

plt.xlabel='Class'

plt.ylabel='Count'

plt.hist(train_csv['diagnosis'], facecolor='peru',edgecolor='blue', bins=10)

x=np.arange(5);

plt.xticks(x, ['0','1','2','4','3'])

plt.show()
#Image

temp=Image.open("/kaggle/input/aptos2019-blindness-detection/train_images/000c1434d8d7.png")

temp_data=np.asarray(temp)

print(temp.size)

print(temp.mode)

print(temp_data.shape)

#Directory for storing resized images

output_dir = "/kaggle/input/resized"

os.mkdir(output_dir)

#Image rezie to 200*200

def resizeImage(infile, output_dir, flag):

    if flag:

        t_infile="/kaggle/input/aptos2019-blindness-detection/train_images/"+infile

    else:

        t_infile="/kaggle/input/aptos2019-blindness-detection/test_images/"+infile

    temp=Image.open(t_infile)

    temp=temp.resize((200,200))

    temp.save(output_dir+"/"+infile)

    

 

train_dir="/kaggle/input/aptos2019-blindness-detection/train_images/"

for file in os.listdir(train_dir):

    resizeImage(file,output_dir, True)

            
#Resized image status

all_train_image_dir=os.listdir(output_dir)

len(all_train_image_dir)
#For 2 fold validation and 1 train set

from sklearn.model_selection import train_test_split

train_img,kfold_img=train_test_split(all_train_image_dir,test_size=0.15, random_state=10)

#kfold_img2,kfold_img1=train_test_split(kfold_img,test_size=0.50, random_state=10)

print("training size",len(train_img))

print("kfold_img size", len(kfold_img))
from os import listdir

from keras.preprocessing import image 

def getTensors(img1, output_dir,csvfile):

    targets=[]

    labels=[]

    images_t=[]

    list_of_tensors=[]

    loaded_images = list()

    for i in img1:

        img_temp=img.imread(output_dir+"/"+i)

        images_t.append(img)

        temp=i.strip(".png")

        #targets.append(train_csv_idcode_as_index.loc[temp]['diagnosis'])

        targets.append(csvfile.loc[temp]['diagnosis'])

        labels.append(temp)

        temp=Image.open(output_dir+"/"+i)

        temp=image.load_img(output_dir+"/"+i)

        x = image.img_to_array(temp)

        list_of_tensors.append(np.expand_dims(x, axis=0))

        temp=np.vstack(list_of_tensors)

        tensors=temp.astype('float32')/255

        

    return targets,labels,tensors

        

    


#for Training

train_array=(np.asarray(train_csv['id_code'])).tolist()

train_targets, train_labels, train_tensors =getTensors(train_img, output_dir, train_csv_idcode_as_index)

#for validation

valid_targets, valid_labels, valid_tensors =getTensors(kfold_img, output_dir, train_csv_idcode_as_index)
#Current status of a image

temp=Image.open(output_dir+"/000c1434d8d7.png")

temp_data=np.asarray(temp)

print(temp.size)

print(temp.mode)

print(temp_data.shape)  
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential



model = Sequential()



### TODO: Define your architecture.

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu',input_shape=(200,200,3)))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(512,activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(5,activation='softmax'))

model.summary()



model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train_targets=np.asarray(train_targets)

valid_targets=np.asarray(valid_targets)

train_tensors=train_tensors.reshape(train_tensors.shape[0], 200,200, 3)

valid_tensors=valid_tensors.reshape(valid_tensors.shape[0], 200,200,3)

print(train_tensors.shape)

print(train_targets.shape)

print(valid_targets.shape)
history=model.fit(x=train_tensors, y=train_targets, validation_data=(valid_tensors,valid_targets), epochs=90, verbose=2, callbacks=None,  

          validation_split=0.0, shuffle=True,initial_epoch=0,steps_per_epoch=None)
#test data

test_dir = "/kaggle/input/testresized/"

os.mkdir(test_dir)


temp_dir="/kaggle/input/aptos2019-blindness-detection/test_images/"

for file in os.listdir(temp_dir):

    resizeImage(file,test_dir, False)

    

all_test_image_dir=os.listdir(test_dir)

test_csv=pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

test_csv_idcode_as_index=train_csv.set_index('id_code')

out_dict={}

out_dict['id_code']=[]

out_dict['diagnosis_predicted']=[]

def test_model(test_dir,model):

    for file in os.listdir(test_dir):

        list_of_tensors=[]

        id_code=file.strip(".png")

        out_dict['id_code'].append(id_code)

        temp=image.load_img(test_dir+"/"+file)

        x = image.img_to_array(temp)

        list_of_tensors.append(np.expand_dims(x, axis=0))

        temp=np.vstack(list_of_tensors)

        tensors=temp.astype('float32')/255

        temp=model.predict_classes(tensors)

        out_dict['diagnosis_predicted'].append(temp)

        #print(id_code,temp)



#temp=model.predict_classes(test_model(test_dir))

#print(temp)

test_model(test_dir,model)
# store the prediction in prediction.csv file

output_df=pd.DataFrame(out_dict,columns=['id_code', 'diagnosis_predicted'])

output_df.to_csv('prediction.csv',index=False)

output_df.head
from tqdm import tqdm

train_messay=train_dir
for image in tqdm(os.listdir(train_messay)): 

    path = os.path.join(train_messay, image)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

    img = cv2.resize(img, (200, 200)).flatten()   

    np_img=np.asarray(img)

    

def train_data(train_messay):

    train_data_messy = [] 

    for image1 in tqdm(os.listdir(train_messay)): 

        path = os.path.join(train_messay, image1)

        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

        img1 = cv2.resize(img1, (200, 200))

        train_data_messy.append(img1)

        train_data=np.asarray(train_data_messy)

    return train_data 

train_data=train_data(train_messay)
x_train, x_test, y_train, y_test = train_test_split(train_data, train_csv['diagnosis'], test_size=0.15, random_state=42)

number_of_train = x_train.shape[0]

number_of_test = x_test.shape[0]

print(number_of_train)
x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])

x_test_flatten = x_test .reshape(number_of_test,x_test.shape[1]*x_test.shape[2])

print("X train flatten",x_train_flatten.shape)

print("X test flatten",x_test_flatten.shape)
x_train = x_train_flatten.T

x_test = x_test_flatten.T

y_test = y_test.T

y_train = y_train.T

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w, b



def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head



def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    

    for i in range(number_of_iterarion):

        

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 100 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list



def predict(w,b,x_test):

    

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))



    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction



def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):



    dimension =  x_train.shape[0]

    w,b = initialize_weights_and_bias(dimension)



    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    

    print("Test Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100,2)))

    print("Train Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100,2)))
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 1500)
#apply Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

grid={"C":np.logspace(-3,3,7),"penalty":["l1","l2"]},

logistic_regression=LogisticRegression(random_state=42)

log_reg_cv=GridSearchCV(logistic_regression,grid,cv=10)

log_reg_cv.fit(x_train,y_train)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve

model = KNeighborsClassifier(n_neighbors=3)

model.fit(x_train, y_train)
