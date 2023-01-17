# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt 

import math

from glob import glob 

import itertools

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.convolutional import (Conv2D,MaxPooling2D)

from keras.layers import BatchNormalization

#from keras.callbacks import (ModelCheckpoint,ReduceLROnPlateau,CSVLogger)

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import cv2

import seaborn as sns

import numpy as np 

import pandas as pd

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from keras import layers

from keras import models

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers. normalization import BatchNormalization

from keras.optimizers import RMSprop

from keras.optimizers import Adam


#-------------------------------------Reading Data-------------------------------------------------------#



#this is the path of training data to read the data

#star is used here becuase folder path are dynamic and data is to be read from each folder after going back

train_data_path = '../input/plant-seedlings-classification/train/*/*.png'

data = glob(train_data_path)



training_data=[]

training_label=[]



data_count= len(data)



print("Reading Training Data")



for d in data:

    training_data.append(cv2.resize(cv2.imread(d),(70,70)))

    training_label.append(d.split('/')[-2])

    

training_data = np.asarray(training_data)

training_label = pd.DataFrame(training_label)


#-------------------------------------Opertions on training data-------------------------------------------------------#



#new_train = []





def ImageOperation(data):

    #create a new list to store the modified images

    new_data = []

    # this step is done so as the proper operations can be performed on the dataset

    image_Display=True

    for t in data:

        

        #applying hsv

        lower_hsv = np.array([25, 100, 50])

        upper_hsv = np.array([95, 255, 255])

        hsv = cv2.cvtColor(t,cv2.COLOR_BGR2HSV)

        

        #masking image

        masking = cv2.inRange(hsv,lower_hsv,upper_hsv)

        structuring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

        masking = cv2.morphologyEx(masking,cv2.MORPH_CLOSE,structuring)

        

        #boolean masking image

        boolean = masking>0

        

        #removing backgraound from the image

        new = np.zeros_like(t,np.uint8)

        new[boolean] = t[boolean]

        new_data.append(new)

    

        #the image will be dsiplayed through this code for each operation

        if image_Display == True:

            #showing original image

            plt.subplot(2, 3, 1); plt.imshow(t)  

            #showing hsv image

            plt.subplot(2, 3, 2); plt.imshow(hsv) 

            # showing mask image

            plt.subplot(2, 3, 3); plt.imshow(masking)  

            # showing boolean mask image

            plt.subplot(2, 3, 4); plt.imshow(boolean) 

            #showing image without the background

            plt.subplot(2, 3, 5); plt.imshow(new)

        

        #once it is turned false, now when the next time loop runs, it will not show the image again

        image_Display = False

        

    return new_data



new_training_data= ImageOperation(training_data)

new_training_data = np.asarray(new_training_data) 

new_training_data=new_training_data/255

#print(new_training_data.shape)

    
#-----------------------Data Visulaisation ----------------------------------------------------#



category = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',

              'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse',

              'Small-flowered Cranesbill', 'Sugar beet']

print_data = {}



print('------ Data Contains----------')



for s in category:

    count= len(os.listdir(os.path.join('../input/plant-seedlings-classification/train',s )))

    print('{} data for - {} category'.format(count,s))

    print_data[s] = count

    

plt.figure(figsize=(23, 8))  

sns.barplot(list(print_data.keys()), list(print_data.values()))

LabelEncode = preprocessing.LabelEncoder()

LabelEncode.fit(training_label[0])

new_label = LabelEncode.transform(training_label[0])

clearalllabels = np_utils.to_categorical(new_label)



print(new_training_data.shape, clearalllabels.shape)
#-----------------------------------Splitting the test and training data----------------------------------





x_train,x_test,y_train,y_test = train_test_split(new_training_data,clearalllabels,test_size=0.1,random_state=1,stratify=clearalllabels)



print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


#--------------------------------------BUILDING MODEL-------------------------------------------------------#



#initialising sequential model

model = Sequential()





#----------------- 1st convolution layer --------------------------



#adding convolution layer with 64 filter and imput shape 70 x70 with relu function



model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(70, 70, 3), activation='relu'))

#normalising batch

model.add(BatchNormalization(axis=3))





#----------------- 1st convolution layer --------------------------



#adding convolution layer with 64 filters



model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

#maxpooling

model.add(MaxPooling2D((2, 2)))

#normalising batch

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))





#----------------- 1st convolution layer --------------------------





#adding convolution layer with 128 filters



model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

#normalising batch

model.add(BatchNormalization(axis=3))





#----------------- 1st convolution layer --------------------------





#adding convolution layer with 128 filters



model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

#maxpooling

model.add(MaxPooling2D((2, 2)))

#normalising batch

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))





#----------------- 1st convolution layer --------------------------





#adding convolution layer with 128 filters with 256 filters

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

#normalising batch

model.add(BatchNormalization(axis=3))



'''



#----------------- 1st convolution layer --------------------------



#adding convolution layer with 256 filters

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

#maxpooling

model.add(MaxPooling2D((2, 2)))

#normalising batch

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



'''



#------------------- flattening layer-------------------------

model.add(Flatten())



#------------------- Dense layer-------------------------



#adding dense layer

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



#------------------- Dense layer-------------------------



#adding dense layer

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



#------------------- Dense layer-------------------------





#adding dense layer with same output as no of cateogries, in our case 12 category with softmax function

model.add(Dense(12, activation='softmax'))



    
from keras.optimizers import Adam



optimizer = Adam()

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    

model.summary()



#Designing imagedategenrator so as to obtain as much accuracy, and to make sure same image is not fed into the newtork

# for this we will rotate image, randomdy zoom the imahe, shoft the image vertically, horizontally and flipping it

generate_data = ImageDataGenerator(

        rotation_range=180,

        horizontal_flip=True,

        vertical_flip=True,

        zoom_range = 0.1, 

        width_shift_range=0.1,

        height_shift_range=0.1,

        

    )  



generate_data.fit(x_train)

# Running the model with different batch size of 4 ,8, 16 ,32, 64

batch_size = [32]

for i in batch_size:

    history = model.fit_generator(generate_data.flow(x_train, y_train,batch_size=i),

                            steps_per_epoch= (x_train.shape[0] // i),

                            epochs = 200,

                            workers=4,

                            validation_data=(x_test,y_test),

                            verbose =2

                            )

    score, acc = model.evaluate(x_test,y_test)

    score2, acc2 = model.evaluate(x_train,y_train)

    print('---------------')

    print('Test score:', score,'   Test accuracy:', acc)

    print('Train score:', score2,'   Train accuracy:',acc2)

    print('---------------')
#Final score and accuracy of the model



score, acc = model.evaluate(x_test,y_test)

score2, acc2 = model.evaluate(x_train,y_train)

print('--------------------------------------------------------------')

print('Test score:', score,'   Test accuracy:', acc)

print('Train score:', score2,'   Train accuracy:',acc2)

print('--------------------------------------------------------------')
# evaluating model on test data

path_to_test = '../input/plant-seedlings-classification/test/*.png'

pics = glob(path_to_test)



#creating two list for isnseritng testing and testing labels

testimages = []

tests = []

num = len(pics)



# performing same operations on the testing data as on training data

for i in pics:

    tests.append(i.split('/')[-1])

    testimages.append(cv2.resize(cv2.imread(i),(70,70)))



newtestimages = []

newtestimages = ImageOperation(testimages)    

testimages = np.asarray(testimages)

newtestimages = np.asarray(newtestimages)
#using the model to predict the values on the testing data

prediction = model.predict(newtestimages)



pred = np.argmax(prediction,axis=1)

predStr = LabelEncode.classes_[pred]

submission = {'file':tests,'species':predStr}

submission = pd.DataFrame(submission)

submission.to_csv("Prediction.csv",index=False)
#------------------------------- Creating confusion matrix ---------------------------------



import seaborn as sns

ypred = model.predict(x_test)



y_correct = np.argmax(y_test, axis=1)

y_pred = np.argmax(ypred, axis=1)



cm = confusion_matrix(y_correct, y_pred)



plt.figure(figsize=(12, 12))

ax = sns.heatmap(cm, cmap=plt.cm.Greens_r, annot=True, square=True)



ax.set_ylabel('Correct', fontsize=40)

ax.set_xlabel('Predicted', fontsize=40)
#------------------------ Creating the accuracy and losss curve -----------------------------



from matplotlib import pyplot



#plotting the accuracy and the loss curve using seaborn



pyplot.plot(history.history['accuracy'], label='train')

pyplot.plot(history.history['val_accuracy'], label='test')

pyplot.legend()

pyplot.show()



pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

pyplot.show()
from sklearn.model_selection import GridSearchCV,cross_val_score

from sklearn.ensemble import (GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier,VotingClassifier)

from xgboost import XGBClassifier

from glob import glob

import os



category = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',

              'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse',

              'Small-flowered Cranesbill', 'Sugar beet']



x = []

y = []



def ImageOperation2(image):

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([25, 100, 50])

    upper_hsv = np.array([95, 255, 255])

    masking = cv2.inRange(image_hsv, lower_hsv, upper_hsv)

    structuring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))

    mask = cv2.morphologyEx(masking, cv2.MORPH_CLOSE, structuring)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output



train=[] 



print('--------- Reading training data -----------')



for c in category:

    num_samples= os.path.join('../input/plant-seedlings-classification/train', c)

    for i in glob(os.path.join(num_samples, "*.png")):

        

        image = cv2.imread(i, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (150, 150))

        

        image = ImageOperation2(image)

        

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (45,45))

        

        image = image.flatten()

        

        x.append(image)

        y.append(c)

        

x = np.array(x)

y = np.array(y)



print(x.shape)

print()

print(y.shape)

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



label_to_id_dict = {a:b for b,a in enumerate(np.unique(y))}

id_to_label_dict = {a: b for b, a in label_to_id_dict.items()}

label_ids = np.array([label_to_id_dict[a] for a in y])





print('------------------ Creating Visualisation --------------------')



images_scaled = StandardScaler().fit_transform(x)



pca = PCA(n_components=2)

pca_200= PCA(n_components=200)



# -------------------------- for 2 components ---------------------



pca_result = pca.fit_transform(images_scaled)

pca_result_scaled = StandardScaler().fit_transform(pca_result)



# -------------------------- for 200 components ---------------------



pca_result_200 = pca_200.fit_transform(images_scaled)

pca_result_scaled_200= StandardScaler().fit_transform(pca_result_200)





def Plotting(data, label,figsize=(20,20)):

    plt.figure(figsize=figsize)

    plt.grid()

    noOfClass = len(np.unique(label))

    for l in np.unique(label):

        plt.scatter(data[np.where(label == l), 0],

                    data[np.where(label == l), 1],

                    marker='*',

                    color= plt.cm.Set1(l / float(noOfClass)),

                    linewidth='1',

                    alpha=0.8,

                    label=id_to_label_dict[l])

    plt.legend(loc='best')

    



tsne = TSNE(n_components=2, perplexity=40.0)



tsne_result = tsne.fit_transform(pca_result)

tsne_result_200 = tsne.fit_transform(pca_result_200)



#print(tsne_result)

#print()

#print(tsne_result.shape)



tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

tsne_result_scaled_200 = StandardScaler().fit_transform(tsne_result_200)



Plotting(pca_result_scaled_200, label_ids)

Plotting(tsne_result_scaled, label_ids)

Plotting(tsne_result_scaled_200, label_ids)
xgb= XGBClassifier()

ada= AdaBoostClassifier()

gb= GradientBoostingClassifier()

rf= RandomForestClassifier()

from sklearn import svm

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.decomposition import PCA



pca = PCA(n_components=100)

pca_result = pca.fit_transform(images_scaled)



print(pca_result.shape)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)



svm= svm.SVC()



eclf = VotingClassifier(estimators=[('rf',rf),('xgb',xgb),('svm',svm),('adaboost', ada)], voting='hard')



for clf, label in zip([ rf,xgb,svm,ada, eclf],['Random Forest', 'XGB', 'SVM', 'adaboost','Combine']):

    score=cross_val_score(clf,pca_result,y,scoring='accuracy',cv=kfold)

    #print(score)

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (score.mean(), score.std(), label))



#eclf.fit(x_train,y_train)

#prediction = eclf.predict(x_test)