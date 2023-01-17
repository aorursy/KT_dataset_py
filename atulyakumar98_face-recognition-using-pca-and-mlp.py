import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Visualization

import matplotlib.pyplot as plt



#Machine Learning

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import metrics



#System

import os

print(os.listdir("../input"))
import warnings

warnings.filterwarnings('ignore')

print("Warnings ignored!!")
data=np.load("../input/olivetti_faces.npy")

target=np.load("../input/olivetti_faces_target.npy")
print("There are {} images in the dataset".format(len(data)))

print("There are {} unique targets in the dataset".format(len(np.unique(target))))

print("Size of each image is {}x{}".format(data.shape[1],data.shape[2]))

print("Pixel values were scaled to [0,1] interval. e.g:{}".format(data[0][0,:4]))
print("unique target number:",np.unique(target))
def show_40_distinct_people(images, unique_ids):

    #Creating 4X10 subplots in  18x9 figure size

    fig, axarr=plt.subplots(nrows=4, ncols=10, figsize=(18, 9))

    #For easy iteration flattened 4X10 subplots matrix to 40 array

    axarr=axarr.flatten()

    

    #iterating over user ids

    for unique_id in unique_ids:

        image_index=unique_id*10

        axarr[unique_id].imshow(images[image_index], cmap='gray')

        axarr[unique_id].set_xticks([])

        axarr[unique_id].set_yticks([])

        axarr[unique_id].set_title("face id:{}".format(unique_id))

    plt.suptitle("There are 40 distinct people in the dataset")
show_40_distinct_people(data, np.unique(target))
def show_10_faces_of_n_subject(images, subject_ids):

    cols=10# each subject has 10 distinct face images

    rows=(len(subject_ids)*10)/cols #

    rows=int(rows)

    

    fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(18,9))

    #axarr=axarr.flatten()

    

    for i, subject_id in enumerate(subject_ids):

        for j in range(cols):

            image_index=subject_id*10 + j

            axarr[i,j].imshow(images[image_index], cmap="gray")

            axarr[i,j].set_xticks([])

            axarr[i,j].set_yticks([])

            axarr[i,j].set_title("face id:{}".format(subject_id))

    
#You can playaround subject_ids to see other people faces

show_10_faces_of_n_subject(images=data, subject_ids=[0,5, 21, 24, 36])
#We reshape images for machine learnig  model

X=data.reshape((data.shape[0],data.shape[1]*data.shape[2]))

print("X shape:",X.shape)
X_train, X_test, y_train, y_test=train_test_split(X, target, test_size=0.3, stratify=target, random_state=0)

print("X_train shape:",X_train.shape)

print("y_train shape:{}".format(y_train.shape))
y_frame=pd.DataFrame()

y_frame['subject ids']=y_train

y_frame.groupby(['subject ids']).size().plot.bar(figsize=(15,8),title="Number of Samples for Each Classes")
import mglearn
mglearn.plots.plot_pca_illustration()
from sklearn.decomposition import PCA

pca=PCA(n_components=2)

pca.fit(X)

X_pca=pca.transform(X)
number_of_people=10

index_range=number_of_people*10

fig=plt.figure(figsize=(10,8))

ax=fig.add_subplot(1,1,1)

scatter=ax.scatter(X_pca[:index_range,0],

            X_pca[:index_range,1], 

            c=target[:index_range],

            s=10,

           cmap=plt.get_cmap('jet', number_of_people)

          )



ax.set_xlabel("First Principle Component")

ax.set_ylabel("Second Principle Component")

ax.set_title("PCA projection of {} people".format(number_of_people))



fig.colorbar(scatter)
pca=PCA()

pca.fit(X)



plt.figure(1, figsize=(12,8))



plt.plot(pca.explained_variance_, linewidth=2)

 

plt.xlabel('Components')

plt.ylabel('Explained Variaces')

plt.show()
n_components=90
pca=PCA(n_components=n_components, whiten=True)

pca.fit(X_train)
fig,ax=plt.subplots(1,1,figsize=(8,8))

ax.imshow(pca.mean_.reshape((64,64)), cmap="gray")

ax.set_xticks([])

ax.set_yticks([])

ax.set_title('Average Face')
number_of_eigenfaces=len(pca.components_)

eigen_faces=pca.components_.reshape((number_of_eigenfaces, data.shape[1], data.shape[2]))



cols=10

rows=int(number_of_eigenfaces/cols)

fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))

axarr=axarr.flatten()

for i in range(number_of_eigenfaces):

    axarr[i].imshow(eigen_faces[i],cmap="gray")

    axarr[i].set_xticks([])

    axarr[i].set_yticks([])

    axarr[i].set_title("eigen id:{}".format(i))

plt.suptitle("All Eigen Faces".format(10*"=", 10*"="))
X_train_pca=pca.transform(X_train)

X_test_pca=pca.transform(X_test)
print(X_train_pca.shape)

print(y_train.shape)

print(X_test_pca.shape)

print(y_test.shape)

print(y_train)



from keras.utils import to_categorical



y_train=to_categorical(y_train,40)

y_test=to_categorical(y_test,40)

print(y_train.shape)
from keras.models import Sequential

from keras.layers import Dense, Activation



from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

from keras.layers import Dropout



from keras import regularizers
model=Sequential()

model.add(Dense(256, activation='relu', input_dim=90))

model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.05))

model.add(Dense(40, activation='softmax'))



epochs=100

batch_size=128

red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=0.0001, patience=2, verbose=1)

model.compile(optimizer=Adam(lr=1e-3),loss='categorical_crossentropy',metrics=['accuracy'])



model.summary()
History = model.fit(X_train_pca,y_train, epochs = epochs, validation_data = (X_test_pca,y_test),batch_size=128, verbose = 1)
plt.plot(History.history['acc'])

plt.plot(History.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(History.history['loss'])

plt.plot(History.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()