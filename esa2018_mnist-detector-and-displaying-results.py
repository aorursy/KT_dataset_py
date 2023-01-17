# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd



path = "/kaggle/input/digit-recognizer/"

dataset = pd.read_csv(path+"train.csv")



# Show the first 10 rows of the dataset

dataset.head(3)



# To check profile of the dataset

#dataset.describe(include='all')  
# The dataset consists of 785 columns. The 1st column is the character class. There are 10 characters

# that will be recognised. The value could be from "0" to "9". The following 784 columns, the 2nd to 784th, 

# are obtained by flattening a sample image with size 28 x 28 pixels. 



X = dataset.iloc[:,1:785]

Y = dataset.iloc[:,0]



# X is an input data and Y is the labelled target.

print("Input")

print(X)

print("Target")

print(Y)

# Preparing training and test dataset



from keras import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)



print("X_train shape")

print(X_train.shape)



print("X_test shape")

print(X_test.shape)

X_test_ori = X_test.copy()



print("y_train shape")

print(y_train.shape)



print("y_test shape")

print(y_test.shape)





#The input values need to be standardised (normalised)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)



print(X_train)

print(X_train.shape)





# Just want to check the size of training and test dataset

# N_total = len(X)

# N_train = len(X_train)

# N_test = len(X_test)

# print("N_total "+str(N_total))

# print("Check the total value: "+str(N_train)+" + "+str(N_test)+" = "+str(N_train + N_test))



Nc = len(dataset.columns)

print("Number of columns "+str(Nc))



# This step is required since we will have a multiclass output. Since, we have 10 possible output values, 

# the ConvNet output should be stored as matrix with 10 columns.  



from keras.utils import to_categorical

train_labels = to_categorical(y_train)



print(y_train)

print(train_labels)



y_train = to_categorical(y_train)



showASingleTestedImage = False
# Create ANN architecture. The input is 784 nodes and the output is 10 nodes. Therefore, the nodes between 

# input and output is around (784+10)/2 = 397 ~ 400 nodes.



from keras import optimizers

inputSz = Nc-1



model = Sequential()



model.add(Dense(400, activation='relu', kernel_initializer='random_normal', input_dim=inputSz)) 

model.add(Dense(200, activation='relu', kernel_initializer='random_normal'))

model.add(Dense(100, activation='relu', kernel_initializer='random_normal'))

model.add(Dense(10, activation='softmax', kernel_initializer='random_normal'))

model.summary()



model.compile(optimizer ='adam',loss='categorical_crossentropy', metrics =['accuracy'])

# Starting to train the dataset



import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint



checkpoint = ModelCheckpoint("best_model1.hdf5", monitor='loss', verbose=1,

    save_best_only=True, mode='auto', period=1)



batch_size=20 

epochs= 30





#Fitting the data to the training dataset

history = model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_split=0.2)





print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# Save the trained model and its weights



model_json = model.to_json()

with open("model1.json", "w") as json_file:

    json_file.write(model_json)



# serialize weights to HDF5

model.save_weights("weight_of_model1.h5")

print("Saved model to disk")

N_total = len(X)

print(N_total)

N_train = len(X_train)

print(N_train)

N_test = len(X_test)

print(N_test)



print("N_total "+str(N_total))

print("Check the total value: "+str(N_train)+" + "+str(N_test)+" = "+str(N_train + N_test))





print(X_test.shape)

print(y_test.shape)



print(y_test.iloc[0])
X_test_lab = X_test.copy()

X_test_lab = sc.fit_transform(X_test_lab)

N = len(X_test_lab)



y_lab = model.predict(X_test_lab)





def formatList(a):

    N = len(a)

    s = ""

    for i in range(N):

        vS = "%2.5f"%a[i]

        s = s +" "+vS

    return s



classIdx = []

#CM = np.zeros((10,10),dtype ="int")

for i in range(N):

    yItem = y_lab[i,:]

    classIdx.append(np.argmax(yItem))

    #print("["+str(i)+"] --------- char: "+str(y_test.iloc[i])+" >>> "+str(classIdx[i])+" -----------")   

    #print(formatList(y[i,:]))



y_pred = classIdx.copy()    

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



print(cm)



className = []

for i in range(10):

    className.append(str(i))



cmap=plt.cm.Reds

fig, ax = plt.subplots()

im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

ax.figure.colorbar(im, ax=ax)



ax.figure.set_size_inches(8,6,True)



ax.set(xticks=np.arange(cm.shape[1]),

  yticks=np.arange(cm.shape[0]),

  xticklabels=className, yticklabels=className,

  title='',

  ylabel='True character',

  xlabel='Predicted character')



# Rotate the tick labels and set their alignment.

#plt.setp(ax.get_xticklabels(), rotation=90, ha="right",

#  rotation_mode="anchor")



fmt = 'd'

thresh = cm.max() / 2.

for i in range(cm.shape[0]):

  for j in range(cm.shape[1]):

    ax.text(j, i, format(cm[i, j], fmt),

    ha="center", va="center",

    color="white" if cm[i, j] > thresh else "black")



fig.tight_layout()





print(y_test.shape)

print(y_test.iloc[2])

#plt.imshow(cm, cmap='binary')



#Another statistical parameters:



from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import roc_auc_score



acc   = accuracy_score(y_test,y_pred)

prec  = precision_score(y_test,y_pred, average =None)

recl  = recall_score(y_test,y_pred, average =None)

f1_sc = f1_score(y_test,y_pred, average =None)

cohkp = cohen_kappa_score(y_test,y_pred)

#roc   = roc_auc_score(y_test,y_pred)



print("Accuracy          : %3.4f "%(acc))

print("Average precision : %3.4f "%(np.average(prec)))

print("Average recall    : %3.4f "%(np.average(recl)))

print("Average F1-score  : %3.4f "%(np.average(f1_sc)))

print("Cohen-Kappa score : %3.4f "%(cohkp))



# Sample Images

import random as rd





def convListToImg(list1):

    N = len(list1)

    N2 = int(np.sqrt(N))

    img = np.zeros((N2,N2), dtype ="uint8")

    idx = 0

    for i in range(N2):

        for j in range(N2):

            img[i,j] = list1[idx]

            idx = idx + 1

    return img



idxList = []

for i in range(10):

    idxList.append(rd.randrange(N_test))

#print(idxList)



ix = 0

fig, axs = plt.subplots(2,5,figsize=(13,6))  # Width: 12 Height: 6

for i in range(2):

    for j in range(5):

        ixx = idxList[ix]

        list1 = X_test_ori.iloc[ixx,0:785]

        img = convListToImg(list1)

        axs[i,j].set_title("["+str(ixx)+"] Label:"+str(y_test.iloc[ixx])+" >> "+str(y_pred[ixx]))

        axs[i,j].imshow(img, cmap="gray")

        ix = ix + 1



# Load test dataset from unused dataset in training stages

testDataset = pd.read_csv(path+"test.csv")

testDataset.head(10)
X_test= testDataset.iloc[:,0:785]

X_test_ori = X_test.copy()

X_test = sc.fit_transform(X_test)

N = len(X_test)







y_pred_noLab = model.predict(X_test)



def formatList(a):

    N = len(a)

    s = ""

    for i in range(N):

        vS = "%2.5f"%a[i]

        s = s +" "+vS

    return s



final_detected_class = []

for i in range(N):

    yItem = y_pred_noLab[i,:]

    final_detected_class.append(np.argmax(yItem))

    print("["+str(i)+"] --------- char: "+str(final_detected_class[i])+"------------")    

    print(formatList(y_pred_noLab[i,:]))



# Show a single image from test dataset

import matplotlib.pyplot as plt





if (showASingleTestedImage==True):



    print("Give the index! Max: "+str(N))

    testIdx = int(input())





    list1 = X_test_ori.iloc[testIdx,0:785]

    #print(list1)

    img = convListToImg(list1)



    plt.imshow(img,cmap="gray")

    plt.title("Detected as "+str(final_detected_class[testIdx]))

    plt.show()

import random as rd

idxList = []

for i in range(12):

    idxList.append(rd.randrange(N))

print(idxList)



ix = 0

fig, axs = plt.subplots(2,5,figsize=(13,6))

for i in range(2):

    for j in range(5):

        list1 = X_test_ori.iloc[idxList[ix],0:785]

        img = convListToImg(list1)

        axs[i,j].set_title("["+str(idxList[ix])+"] >> "+str(final_detected_class[idxList[ix]]))

        axs[i,j].imshow(img, cmap="gray")

        ix = ix + 1



# This block is used to store the prediction result as CSV file



csvFileName = "sample_submission.csv"

csvFile = open(csvFileName,"w+")

N_detClass = len(final_detected_class)

headerStr = "ImageId,Label"

csvFile.write(headerStr+"\n")

for i in range(N_detClass):

    strVal = ""

    strVal = str(i)

    strVal = strVal +","+ str(final_detected_class[i]) 

    csvFile.write(strVal+"\n")



csvFile.close()



print("File "+csvFileName+" has been saved")


