from PIL import Image, ImageFilter ## To read and filter the image

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
im = Image.open("/kaggle/input/test-image/shutterstock_aero.jpg")
plt.figure(figsize=(10,8))

plt.axis('off')

imshow(im);
emboss = [-2,-1,0,-1,1,1,0,1,2] # This perticular matrix is called emboss

kernel = ImageFilter.Kernel((3,3), emboss)

 

im2 = im.filter(kernel)

plt.figure(figsize=(10,8))

plt.xticks=[]

imshow(im2);
from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

import numpy as np

import pandas as pd

from keras.utils import to_categorical # to convert our lables to catagories

import collections #to get count the number of each label

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
import pylab

pylab.rc('figure', figsize=(10,7))



SMALL_SIZE = 8

MEDIUM_SIZE = 10

BIGGER_SIZE = 12



plt.rc('font', size=SMALL_SIZE)          # controls default text sizes

plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title

plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
Train_data=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

Test_data=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
y=Train_data['label'].values

Train=Train_data.drop('label',axis=1).values

Test=Test_data.values
plt.figure(figsize=(10,10))

for i in range(16):

    plt.subplot(4,4,i+1)

    plt.title(f"Label: {y[i]}",fontdict={'size'   : 12})

    plt.gca().set_xticks([])

    #plt.xticks([])

    #plt.yticks([])

    plt.grid(False)

    plt.axis('off')

    plt.imshow(Train.reshape(Train.shape[0],28,28)[i], cmap='gray')

plt.show()
def sort_dic(dic):  # Sort the counter dictionaries 

    index=sorted(dic)

    sort_list=[]

    label=[]

    for i in index:

        sort_list.append(dic[i])

        label.append(i)

    return sort_list,label
train_count=collections.Counter(y)

train_count,labels=sort_dic(train_count)
x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, train_count, width, label='Train_label_Count')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Frequency')

ax.set_xlabel('Labels')

ax.set_title('Digit Counts in Training data')

ax.set_xticks(x)

ax.set_xticklabels(labels)





def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)



fig.tight_layout()



plt.show()
X=Train.reshape(Train.shape[0],28,28,1)

X_test=Test.reshape(Test.shape[0],28,28,1)
X_t, X_v, y_t, y_v = train_test_split( X, y, test_size=0.1, random_state=42) #Taking a small validation size so that we have more examples to train from.
y_train=to_categorical(y_t,10)

y_val=to_categorical(y_v,10)



X_train=X_t/255

X_val=X_v/255
classifier=Sequential()



classifier.add(Conv2D(64,(3,3),input_shape=(28,28,1),activation='relu')) #by default the stride is 1

classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

classifier.add(Dropout(0.2))

classifier.add(Conv2D(64,(3,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

classifier.add(Dropout(0.2))

classifier.add(Flatten())



classifier.add(Dense(units=256,activation='relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(units=10,activation='softmax'))
classifier.summary()
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#classifier.fit(X_train,y_train,epochs=5,batch_size=128)

#score=classifier.evaluate(X_val,y_val)

#print("The accuracy of the model on test data is:",score[1])
T=25

datagen = ImageDataGenerator(

        rotation_range=10, 

        zoom_range = 0.1,  

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        )  





datagen.fit(X_train)



history = classifier.fit_generator(datagen.flow(X_train,y_train, batch_size=128),

                              epochs = T, validation_data = (X_val,y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // 128

                              )

## Plotting the accuracies



plt.plot(range(1,T+1),history.history['accuracy'],label='Training accuracy',c='r')

plt.plot(range(1,T+1),history.history['val_accuracy'],label='Validation accuracy',c='b')

plt.legend()

plt.xlabel("# epochs")

plt.ylabel("Accuracy")

plt.title("Training and Validation accuracy wint number of epochs")

plt.show()
## Plotting the losses



plt.plot(range(1,T+1),history.history['loss'],label='Training loss',c='r')

plt.plot(range(1,T+1),history.history['val_loss'],label='Validation loss',c='b')

plt.legend()

plt.xlabel("# epochs")

plt.ylabel("Loss")

plt.title("Training and Validation loss with number of epochs")

plt.show()
y_pred=classifier.predict(X_val)

pred_label=np.argmax(y_pred,axis=1)  ##Convert the results to class labels

actual_label=np.argmax(y_val,axis=1)
error=pred_label!=y_v

missclassified_images=X_v[error]

missclassified_label=pred_label[error]

correct_label=y_v[error]
pred_counts=collections.Counter(pred_label)  ##Getting the frequencies

labels_count=collections.Counter(y_v)



prediction_count,labels=sort_dic(pred_counts) ##Sorting the dictionary

actual_count,labels=sort_dic(labels_count)
x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, prediction_count, width, label='Prediction')

rects2 = ax.bar(x + width/2, actual_count, width, label='Actual')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Frequency')

ax.set_xlabel('Labels')

ax.set_title('Label counts (Actual vs Prediction )')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()





def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)



fig.tight_layout()



plt.show()
plt.figure(figsize=(10,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.title(f"Correct Label {correct_label[i]}\n Prediction {missclassified_label[i]}",fontdict={'size'   : 12})

    #plt.xticks([])

    #plt.yticks([])

    plt.axis('off')

    plt.grid(False)

    plt.imshow(missclassified_images.reshape(missclassified_images.shape[0],missclassified_images.shape[1],missclassified_images.shape[2])[i], cmap='gray')

plt.show()
#test_pred=classifier.predict(X_test)

#test_pred=np.argmax(test_pred,axis=1)

#submission =  pd.DataFrame({

#        "ImageId": Test_data.index+1,

#        "Label": test_pred

#    })

#submission.to_csv('submission.csv', index=False)
submission=pd.read_csv("/kaggle/input/submission-mnist/submission.csv")  ##uploading locally predicted file

submission.to_csv('submission.csv', index=False)