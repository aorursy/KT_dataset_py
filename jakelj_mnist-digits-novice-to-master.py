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
#Import Packages



import random

import keras



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import matplotlib.patches as mpatches



from skimage.filters import threshold_otsu

from skimage.segmentation import clear_border

from skimage.measure import label, regionprops

from skimage.morphology import closing, square

from skimage.color import label2rgb

from math import sqrt





from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import LearningRateScheduler

from keras.callbacks import ModelCheckpoint
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')



train.head()
y_train = train['label']

X = train.drop(['label'],axis=1)

X.shape
test.shape




plt.imshow(X.values[3].reshape(28,28))



# code for this plot taken from https://www.kaggle.com/josephvm/kannada-with-pytorch



fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(15,15))



# I know these for loops look weird, but this way num_i is only computed once for each class

for i in range(10): # Column by column

    num_i = X[y_train == i]

    ax[0][i].set_title(i)

    for j in range(10): # Row by row

        ax[j][i].axis('off')

        ax[j][i].imshow(num_i.iloc[j, :].to_numpy().astype(np.uint8).reshape(28, 28))

import seaborn as sns

lab, val = np.unique(y_train,return_counts=True)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,7))

sns.barplot(lab,val)

plt.title('distribution of samples in the training data')
labels =[]

frequencies = []





for i in range(len(y_train)):

    lab, freq = str(y_train[i]), len([n for n in X.values[i] if n > 0])

    labels.append(lab)

    frequencies.append(freq)

    

    

data = {'Labels':labels, 'Frequencies':frequencies}



df = pd.DataFrame(data)
# mean number of pixels per label

df.groupby('Labels').mean()


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))

sns.boxplot(x = 'Labels', y = 'Frequencies', data = df)

# apply threshold

image = X.values[3].reshape(28,28)

thresh = threshold_otsu(image)

bw = closing(image > thresh, square(3))



# label image regions

label_image = label(bw)

image_label_overlay = label2rgb(label_image, image=image)



fig, ax = plt.subplots(figsize=(10, 6))

ax.imshow(image_label_overlay)



for region in regionprops(label_image):

    # take regions with large enough areas

    if region.area >= 30:

        # draw rectangle around segmented coins

        minr, minc, maxr, maxc = region.bbox

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,

                                  fill=False, edgecolor='green', linewidth=2)

        ax.add_patch(rect)

plt.text(1,1, f'Width: {maxc-minc} Height: {maxr -minr} Diagonal: {round(sqrt((((maxc-minc)**2) + (maxr-minr)**2)),2)}', color = 'w')

ax.set_axis_off()

plt.tight_layout()





def measurements(images):

    

    widths = []

    heights = []

    diags = []

    

    for i in range(len(images)):

        # apply threshold

        image = images[i].reshape(28,28)

        thresh = threshold_otsu(image)

        bw = closing(image > thresh, square(3))



        # label image regions

        label_image = label(bw)

        image_label_overlay = label2rgb(label_image, image=image)



        for region in regionprops(label_image):

            # take regions with large enough areas

            if region.area >= 30:

                # draw rectangle around segmented coins

                minr, minc, maxr, maxc = region.bbox

                

        widths.append(maxc-minc)

        heights.append(maxr -minr)

        diags.append(sqrt((((maxc-minc)**2) + (maxr-minr)**2)))

                     

    return widths, heights, diags

                     





widths, heights, diags = measurements(X.values)

                     

data = {'Labels':labels, 'diagonals':diags, 'widths':widths, 'heights':heights, 'Area':frequencies}



df = pd.DataFrame(data)

df.head()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))

sns.boxplot(x = 'Labels', y = 'diagonals', data = df)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))

sns.boxplot(x = 'Labels', y = 'heights', data = df)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))

sns.boxplot(x = 'Labels', y = 'widths', data = df)
from sklearn.ensemble import RandomForestClassifier

test_y = df['Labels']



test_x = df

test_x.drop('Labels', axis =1, inplace = True)

X_train1, X_val1, y_train1, y_val1 = train_test_split(test_x, test_y, test_size=0.1, random_state=1337)
clf_rfc = RandomForestClassifier(n_estimators = 100)

clf_rfc.fit(X_train1,y_train1)
from sklearn.metrics import accuracy_score



accuracy_score(clf_rfc.predict(X_val1), y_val1)
from sklearn.neighbors import KNeighborsClassifier





clf_knn = KNeighborsClassifier()

clf_knn.fit(X_train1,y_train1)



accuracy_score(clf_knn.predict(X_val1), y_val1)
imaginery_data = [[1,2,3,4,5], [3,3,3,3,3]]



plt.scatter(imaginery_data[0],imaginery_data[1])





pca_test = PCA()

pca_test.fit(imaginery_data).explained_variance_ratio_
imaginary_data = np.random.rand(28,2)







plt.scatter(imaginary_data[:,0], imaginary_data[:,1])



print(imaginary_data[0:10])

pca = PCA(n_components = 1)

pca.fit(imaginary_data).explained_variance_ratio_
imaginary_data_pca = pca.transform(imaginary_data)



print("original shape:   ", imaginary_data.shape)

print("transformed shape:", imaginary_data_pca.shape)
imaginary_data_pca_new = pca.inverse_transform(imaginary_data_pca)

plt.scatter(imaginary_data[:, 0], imaginary_data[:, 1], alpha=0.2)

plt.scatter(imaginary_data_pca_new[:, 0], imaginary_data_pca_new[:, 1], alpha=0.8)

plt.axis('equal')
from mpl_toolkits.mplot3d import Axes3D

imaginary_data = np.random.rand(728,3)



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



i = imaginary_data[:,0]

j = imaginary_data[:,1]

k = imaginary_data[:,2]







ax.scatter(i, j, k, c='r', marker='o')



ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')

ax.set_zlabel('Z Label')



plt.show()



pca = PCA(n_components=2) # project from 784 to 2 dimensions so we can view them 

pca.fit(imaginary_data)

imaginary_data_pca = pca.transform(imaginary_data)

imaginary_data_pca_new = pca.inverse_transform(imaginary_data_pca)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



i = imaginary_data[:,0]

j = imaginary_data[:,1]

k = imaginary_data[:,2]







ax.scatter(i, j, k, c='r', marker='o')



ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')

ax.set_zlabel('Z Label')





l = imaginary_data_pca_new[:,0]

m = imaginary_data_pca_new[:,1]





ax.scatter(l, m, c='b', marker='o')





plt.axis('equal')





pca = PCA(n_components=2) # project from 784 to 2 dimensions so we can view them 

principalComponents = pca.fit(X)



print('Explained variance for the 1st two components',principalComponents.explained_variance_ratio_)



principalComponents = pca.transform(X)

principal_df = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

plt.scatter(principalComponents[:, 0], principalComponents[:, 1], s= 1, c=y_train, cmap='Spectral')

plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10));

plt.xlabel('PC1')

plt.ylabel('PC2')



data = {'Labels':labels, 'diagonals':diags, 'widths':widths, 'heights':heights, 'Area':frequencies, 'PC1':principalComponents[:, 0],'PC2':principalComponents[:, 1]}



df = pd.DataFrame(data)

df

test_y = df['Labels']



test_x = df

test_x.drop('Labels', axis =1, inplace = True)



X_train1, X_val1, y_train1, y_val1 = train_test_split(test_x, test_y, test_size=0.1, random_state=1337)



clf_knn = KNeighborsClassifier()

clf_knn.fit(X_train1,y_train1)



accuracy_score(clf_knn.predict(X_val1), y_val1)
from sklearn.decomposition import MiniBatchDictionaryLearning



mbdl = MiniBatchDictionaryLearning(n_components = 2)



mbdl.fit(X)
comps = mbdl.transform(X)
principal_df = pd.DataFrame(data = principalComponents, columns = ['C1', 'C2'])

plt.scatter(comps[:, 0], comps[:, 1], s= 1, c=y_train, cmap='Spectral')

plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10));

plt.xlabel('C1')

plt.ylabel('C2')
def test_model(d1,d1_lab, d2, d2_lab,data):

    



    data = {'Labels':labels, 'diagonals':diags, 'widths':widths, 'heights':heights, 'Area':frequencies, 'PC1':principalComponents[:, 0],'PC2':principalComponents[:, 1]}

    data[d1_lab] = d1

    data[d2_lab] = d2

    

    df = pd.DataFrame(data)

    

    print(df.columns)

    test_y = df['Labels']



    test_x = df

    test_x.drop('Labels', axis =1, inplace = True)



    X_train1, X_val1, y_train1, y_val1 = train_test_split(test_x, test_y, test_size=0.1, random_state=1337)



    clf_knn = KNeighborsClassifier()

    clf_knn.fit(X_train1,y_train1)



    return accuracy_score(clf_knn.predict(X_val1), y_val1)





test_model(comps[:, 0],'d1', comps[:, 1],'d2', data)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



lda = LinearDiscriminantAnalysis(n_components = 2, )

comps = lda.fit_transform(X,y_train.values)
principal_df = pd.DataFrame(data = principalComponents, columns = ['C1', 'C2'])

plt.scatter(comps[:, 0], comps[:, 1], s= 1, c=y_train, cmap='Spectral')

plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10));

plt.xlabel('C1')

plt.ylabel('C2')
test_model(comps[:, 0],'lda1', comps[:, 1],'lda2', data)
'''from sklearn.manifold import TSNE



tsne = TSNE(n_components =2)

comps = tsne.fit_transform(X) 



principal_df = pd.DataFrame(data = principalComponents, columns = ['C1', 'C2'])

plt.scatter(comps[:, 0], comps[:, 1], s= 1, c=y_train, cmap='Spectral')

plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10));

plt.xlabel('C1')

plt.ylabel('C2')



'''



#test_model(comps[:, 0],'tsne1', comps[:, 1],'tsne2', data)#




# Normalize the data

X = X / 255.0

test = test / 255.0



# re-shaping the data so that keras can use it, this is something that trips me up every time



X = X.values.reshape(X.shape[0], 28, 28,1)

test = test.values.reshape(test.shape[0], 28, 28,1)



# This modifies some images slightly, I have seen this in a few tutorials and it usually makes the model more accurate. As a beginner, it goes without saying I don't fully understand all the parameters



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

valid_datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images
y_train = to_categorical(y_train,num_classes=10) # the labels need to be one-hot encoded, this is something else I usually forget



# function to modify the learning rate, if the loss does not change. Also saving the best weights of the model for prediction.



learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', 

                                            patience=3, 

                                            verbose=10, 

                                            factor=0.5, 

                                            min_lr=0.00001)



# Using this notebook as a guide https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist if you haven't read it, and you want to learn about cnn's do yourself a favour and read it



def build_model(input_shape=(28, 28, 1), classes = 10):

    

    activation = 'relu'

    padding = 'same'

    gamma_initializer = 'uniform'

    

    input_layer = Input(shape=input_shape)

    

    hidden=Conv2D(32, (3,3), padding=padding,activation = activation, name="conv1")(input_layer)

    hidden=BatchNormalization(name="batch1")(hidden)

    hidden=Conv2D(32, (3,3), padding=padding,activation = activation, name="conv2")(hidden)

    hidden=BatchNormalization(name="batch2")(hidden)

    hidden=Conv2D(32, (5,5), padding=padding,activation = activation, name="conv3")(hidden)

    hidden=BatchNormalization(name="batch3")(hidden)

    hidden=MaxPool2D(pool_size=2, padding=padding, name="max1")(hidden)

    hidden=Dropout(0.4)(hidden)



    

    

    hidden=Conv2D(64, (3,3), padding =padding, activation = activation,  name="conv4")(hidden)

    hidden=BatchNormalization(name = 'batch4')(hidden)

    hidden=Conv2D(64, (3,3), padding =padding, activation = activation,  name="conv45")(hidden)

    hidden=BatchNormalization(name = 'batch5')(hidden)

    hidden=Conv2D(64, (5,5), padding =padding, activation = activation,  name="conv6")(hidden)

    hidden=BatchNormalization(name = 'batch6')(hidden)

    hidden=MaxPool2D(pool_size=2, padding="same", name="max2")(hidden)

    hidden=Dropout(0.4)(hidden)

    



    hidden=Flatten()(hidden)

    hidden=Dense(264,activation = activation, name="Dense1")(hidden)

    hidden=Dropout(0.3)(hidden)

    output = Dense(classes, activation = "softmax")(hidden)

    

    model = Model(inputs=input_layer, outputs=output)

    

    return model
#keras.backend.clear_session()

epochs = 50

initial_learningrate=2e-3

batch_size = 264
# Define the optimizer

#optimizer = Adam(learning_rate=initial_learningrate, beta_1=0.9, beta_2=0.999, amsgrad=False)

optimizer = Adam(learning_rate=initial_learningrate)



# Compile the model





model = build_model(input_shape=(28, 28, 1), classes = 10)





model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size=0.1, random_state=1337)

datagen.fit(X_train)

valid_datagen.fit(X_val)

callbacks = [learning_rate_reduction]

history = model.fit_generator(datagen.flow(X_train,y_train),

                              epochs = epochs,

                              validation_data=valid_datagen.flow(X_val,y_val),

                              verbose = 1,

                            callbacks = callbacks)



# On the first attempt I forgot to add the 'learning_rate_reduction'



#history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size ),

#                              epochs = epochs,

 #                             validation_data=valid_datagen.flow(X_val,y_val),

  #                            validation_steps = 50,

   #                           verbose = 1,

    #                          steps_per_epoch = X_train.shape[0] // batch_size,

     #                       callbacks = callbacks)





model.summary()
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best')



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best')
import seaborn as sns



# used the code from https://www.kaggle.com/shahules/indian-way-to-learn-cnn to create this



y_pre_test=model.predict(X_val)

y_pre_test=np.argmax(y_pre_test,axis=1)

y_test=np.argmax(y_val,axis=1)



conf=confusion_matrix(y_test,y_pre_test)

conf=pd.DataFrame(conf,index=range(0,10),columns=range(0,10))

plt.figure(figsize=(8,6))

sns.heatmap(conf, annot=True)
print('out of {} samples, we got {} incorrect'.format(len(X_train), round(len(X_train) - history.history['accuracy'][-1] * len(X_train))))





# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_val,axis = 1) 



# Errors are difference between predicted labels and true labels

errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = X_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 3

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize = (10,10))

    fig.tight_layout()

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 9 errors 

most_important_errors = sorted_dela_errors[-9:]



# Show the top 6 errors

display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
predictions = model.predict(test)
plt.imshow(test[0].reshape(28,28))
predictions = predictions.argmax(axis = -1)

predictions


submission['Label'] = predictions
submission.head()
submission.to_csv('submission.csv',index=False)