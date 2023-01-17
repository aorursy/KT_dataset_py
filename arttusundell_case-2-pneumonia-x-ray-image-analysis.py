%pylab inline

import numpy as np #linear algebra

import os #operatin system commands

import shutil #high-level operating for files (for copying contents)

import random #random generator

from tensorflow import keras #keras neural network library

from tensorflow.keras import layers, models, optimizers #keras tools needed

from keras.layers import BatchNormalization, Dropout, Flatten, Activation,Conv2D, MaxPooling2D #CNN layers used 

from tensorflow.keras.preprocessing.image import ImageDataGenerator #image preprocessor for CNN

from tensorflow.keras.metrics import FalseNegatives,FalsePositives, SpecificityAtSensitivity #metrics used in model performance evaluation

from sklearn.metrics import classification_report, confusion_matrix, roc_curve #metrics used in final network performance evaluation
# importing dataset



norm_dir1 = "/kaggle/input/chestxray2017/chest_xray/train/NORMAL"

norm_dir2 = "/kaggle/input/chestxray2017/chest_xray/test/NORMAL"



sick_dir1 = "/kaggle/input/chestxray2017/chest_xray/train/PNEUMONIA"

sick_dir2 = "/kaggle/input/chestxray2017/chest_xray/test/PNEUMONIA"

# creating a pool directory and all the data subset directories

pool_dir = './pool'

testset = './test'

trainset = './train'

valset = './val'





allnewdirs=[pool_dir,trainset,valset, testset] 

# creating directories described above and creating NORMAL and PNEUMONIA subfolders for each new directory

try:

    for d in allnewdirs:

        os.mkdir(d)

        os.mkdir(os.path.join(d, 'NORMAL'))

        os.mkdir(os.path.join(d, 'PNEUMONIA'))

except:

    pass
# list the filenames of normal images used for copying

normal_images1=[n for n in os.listdir(norm_dir1) if n.endswith(".jpeg")]

normal_images2=[n for n in os.listdir(norm_dir2) if n.endswith(".jpeg")]

no_normal_images1=len(normal_images1)

no_normal_images2=len(normal_images2)



# list the filenames of pneumonia images used for copying

pneumonia_images1 = [n for n in os.listdir(sick_dir1) if n.endswith(".jpeg")]

pneumonia_images2 = [n for n in os.listdir(sick_dir2) if n.endswith(".jpeg")]

no_pneumonia_images1=len(pneumonia_images1)

no_pneumonia_images2=len(pneumonia_images2)



# printing out the indices of normal and pneumonia images

print('total normal images:', no_normal_images1 + no_normal_images2)

print('total pneumonia images:', no_pneumonia_images1 + no_pneumonia_images2)
# making the pool dataset consisting of original train and test images



# copying normal imagesets 1& 2 into pool "NORMAL"

for fname in normal_images1:

    src = os.path.join(norm_dir1, fname)

    dst = os.path.join(pool_dir,'NORMAL', fname)

    shutil.copyfile(src,dst)

    

for fname in normal_images2:

    src = os.path.join(norm_dir2, fname)

    dst = os.path.join(pool_dir,'NORMAL', fname)

    shutil.copyfile(src,dst)

    

 # copying pneumonia imagesets 1& 2 into pool "PNEUMONIA"

for fname in pneumonia_images1:

    src = os.path.join(sick_dir1, fname)

    dst = os.path.join(pool_dir,'PNEUMONIA', fname)

    shutil.copyfile(src,dst)

    

for fname in pneumonia_images2:

    src = os.path.join(sick_dir2, fname)

    dst = os.path.join(pool_dir,'PNEUMONIA', fname)

    shutil.copyfile(src,dst)
# get filenames of normal and pneumonia images and shuffle for a more randomised train val and test set split



# list of normal image filenames in original dataset test and train folders combined

normal_images=normal_images1+normal_images2

# list of pneumonia image filenames in original dataset test and train folders combined

pneumonia_images=pneumonia_images1+pneumonia_images2



# shuffled the list in random order

random.shuffle(normal_images)

random.shuffle(pneumonia_images)
# splitting the pool set for training, validation and test datasets using 60:20:20 ratio

## splitted to similar fractions of normal and pneumonia pictures in all datasets



###Copying images from pool to test set

#Test set (in total of 1000 for reliable metrics)

##Normal images

for fname in normal_images[:300]:

    src = os.path.join(pool_dir,'NORMAL', fname)

    dst = os.path.join(testset,'NORMAL', fname)

    shutil.copyfile(src,dst)

    

##Pneumonia images    

for fname in pneumonia_images[:700]:

    src = os.path.join(pool_dir,'PNEUMONIA', fname)

    dst = os.path.join(testset,'PNEUMONIA', fname)

    shutil.copyfile(src,dst)



#-------------------------------------------------

###Copying images from pool to validation set



##Valset (in total of 1000)

##Normal images

for fname in normal_images[301:601]:

    src = os.path.join(pool_dir,'NORMAL', fname)

    dst = os.path.join(valset,'NORMAL', fname)

    shutil.copyfile(src,dst)



##Pneumonia images    

for fname in pneumonia_images[701:1401]:

    src = os.path.join(pool_dir,'PNEUMONIA', fname)

    dst = os.path.join(valset,'PNEUMONIA', fname)

    shutil.copyfile(src,dst)

#---------------------------------------------------

###Copying images from pool to training set



##Trainset = the rest

##Normal images

for fname in normal_images[602:]:

    src = os.path.join(pool_dir,'NORMAL', fname)

    dst = os.path.join(trainset,'NORMAL', fname)

    shutil.copyfile(src,dst)



##Pneumonia images    

for fname in pneumonia_images[1402:]:

    src = os.path.join(pool_dir,'PNEUMONIA', fname)

    dst = os.path.join(trainset,'PNEUMONIA', fname)

    shutil.copyfile(src,dst)
## Image and data generation



# generate two Data Generators for data processing



# image generator for training set utilizing image augmentation

traingen = ImageDataGenerator(rescale=1./255, # All images will be rescaled by 1./255

                rotation_range=20, #set random rotational shifts

                width_shift_range=0.2, #set random horisontal shifts

                height_shift_range=0.2, #set random vertical shifts

                shear_range=0.2, #random shearing

                brightness_range= [0.1, 0.4], #randoms brightness shifts

                horizontal_flip=True, #random 

                ) 



#-----------------------------------------------------------------------------------

# image generator for validation and test set without augmentation

devgen = ImageDataGenerator(rescale=1./255) # All images will be rescaled by 1./255





# constant variables used for image generators

TS = (227,227) # Variable for Image Target Size

BS = 16 # Variable for Image Batch Size



# generating batches of tensor image data for training utilizing train generator created above



print('Training set:')

train_generator = traingen.flow_from_directory(

    # this is the target directory

    trainset,

    

    # all images will be resized according to TS

    target_size = TS,

    

    # reading images in batches of batch size according to BS

    batch_size = BS,



    # create binary labels

    class_mode = 'binary' ##### used binary classification since predicting values 0 or 1

)



#-----------------------------------------------------------------------------------

# generating batches of tensor image data for validation utilizing dev generator created above



print('Validation set:')

dev_generator = devgen.flow_from_directory(

    valset,

    target_size = TS,

    batch_size = BS,

    shuffle = False, #no shuffle for validation

    class_mode = 'binary'

)



#-----------------------------------------------------------------------------------

# generating batches of tensor image data for testing utilizing dev generator created above



print('Test set:')

test_generator = devgen.flow_from_directory(

    testset,

    target_size = TS,

    batch_size = BS,

    shuffle = False, #no shuffle for test

    class_mode = 'binary'

)
# checking labels for the data (train)

train_generator.class_indices
# check for the generator output

i = 0

for data_batch, labels_batch in train_generator:

    print('data batch shape: ', data_batch.shape)

    print('labels batch shape: ', labels_batch.shape)

    i = i+1

    if i > 5:

        break
# Check the first 16 images from the batch



figure(figsize = (10,10))

for i in range(16):

    subplot(4,4, i+1)

    imshow(data_batch[i])

show()



# Check the last labels batch

labels_batch
### Building the models



#variable to use to make changin the models easier

inputshape=(227,227,3)



# Build a simple convolutional neural network (CNN) - Model #1



# convolutional layers

model = models.Sequential()

model.add(layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = inputshape))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

# dense layers

model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dense(256, activation = 'relu'))

# output layer

model.add(layers.Dense(1, activation = 'sigmoid'))





# Build a simple convolutional neural network (CNN) - Model #2



# convolutional layers

model2 = models.Sequential()

model2.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = inputshape))

model2.add(layers.MaxPooling2D((2, 2)))

model2.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model2.add(layers.MaxPooling2D((2, 2)))

model2.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model2.add(layers.MaxPooling2D((2, 2)))

model2.add(layers.Flatten())

# dense layers

model2.add(layers.Dense(128, activation = 'relu'))

model2.add(layers.Dense(64, activation = 'relu'))

# output layer

model2.add(layers.Dense(1, activation = 'sigmoid'))



# Build a single layer version of AlexNet (CNN)



model3 = models.Sequential()

# convolutional layers

model3.add(layers.Conv2D(filters=96, input_shape=inputshape, kernel_size=(11,11), strides=(4,4), padding="valid", activation = "relu"))

model3.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"))

model3.add(layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding="same", activation = "relu"))

model3.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"))

model3.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu"))

model3.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu"))

model3.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu"))

model3.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"))

model3.add(layers.Flatten())

# dense layers

model3.add(layers.Dense(units = 9216, activation = "relu"))

model3.add(layers.Dense(units = 4096, activation = "relu"))

model3.add(layers.Dense(4096, activation = "relu"))

# output layer

model3.add(layers.Dense(1, activation = "sigmoid")) #As we have two classes



# a variable to define metrics for easier changes

myMetrics=['acc',FalseNegatives(),FalsePositives(),SpecificityAtSensitivity(0.9)]



## compiling the models

# !! use slower learning rate than the default rate to make the optimizer update the parameters in a more subtle manner

model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop (lr = 1e-4), metrics=myMetrics)

model2.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop (lr = 1e-4), metrics=myMetrics)

model3.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop (lr = 1e-4), metrics=myMetrics)
## Train the networks with the training and validation data



# variable to change the number of epoch easily

nEpochs=20





#Model 1 training

history = model.fit_generator(

    train_generator,

    steps_per_epoch = None, # modifying steps_per_epoch, (steps_per_epoch = None) using all 3852 images

    verbose = 0,

    epochs = nEpochs,

    validation_data = dev_generator,

    validation_steps = None)



#Model 2 training

history2 = model2.fit_generator(

    train_generator,

    steps_per_epoch = None,

    verbose = 0,

    epochs = nEpochs,

    validation_data = dev_generator,

    validation_steps = None)



#Model 3 training

history3 = model3.fit_generator(

    train_generator,

    steps_per_epoch = None,

    verbose = 0, 

    epochs = nEpochs,

    validation_data = dev_generator,

    validation_steps = None)
# Create variables for history metrics for the fitted models (accuracy, loss)



#Metrics for Model 1

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

sas=history.history['specificity_at_sensitivity']

val_sas=history.history['val_specificity_at_sensitivity']

fn=history.history['false_negatives']

vfn=history.history['val_false_negatives']

fp=history.history['false_positives']

vfp=history.history['val_false_positives']

epochs = range(len(acc))



#Metrics for Model 2

acc2 = history2.history['acc']

val_acc2 = history2.history['val_acc']

loss2 = history2.history['loss']

val_loss2 = history2.history['val_loss']

sas2=history2.history['specificity_at_sensitivity']

val_sas2=history2.history['val_specificity_at_sensitivity']

fn2=history2.history['false_negatives']

vfn2=history2.history['val_false_negatives']

fp2=history2.history['false_positives']

vfp2=history2.history['val_false_positives']

epochs2 = range(len(acc2))



#Metrics for Model 3

acc3 = history3.history['acc']

val_acc3 = history3.history['val_acc']

loss3 = history3.history['loss']

val_loss3 = history3.history['val_loss']

sas3=history3.history['specificity_at_sensitivity']

val_sas3=history3.history['val_specificity_at_sensitivity']

fn3=history3.history['false_negatives']

vfn3=history3.history['val_false_negatives']

fp3=history3.history['false_positives']

vfp3=history3.history['val_false_positives']

epochs3 = range(len(acc3))
# Check the sensitivity at specificity level 0.9 for the training done for each model



plt.figure(figsize=(20,10))

plt.plot(epochs, sas, 'bo-', label='Model 1 training')

plt.plot(epochs2, sas2, 'go-', label='Model 2 training')

plt.plot(epochs3, sas3, 'ro-', label='Model 3 training')

plt.plot(epochs, val_sas, 'bo--', alpha=0.7, label='Model 1 validation')

plt.plot(epochs2, val_sas2, 'go--', alpha=0.5, label='Model 2 validation')

plt.plot(epochs3, val_sas3, 'ro--', alpha=0.5, label='Model 3 validation')



plt.title('Sensitivity at specificity level 0.9')

plt.xlabel('Epochs')

plt.ylabel('sensitivity at specificity level 0.9')

plt.grid()

plt.legend()



plt.show()
# Check the accuracy and loss graphs for the training done for each model



plt.figure(figsize=(20,10))

plt.plot(epochs, acc, 'bo-', label='Model 1 training')

plt.plot(epochs2, acc2, 'go-', label='Model 2 training')

plt.plot(epochs3, acc3, 'ro-', label='Model 3 training')

plt.plot(epochs, val_acc, 'bo--', alpha=0.7, label='Model 1 validation')

plt.plot(epochs2, val_acc2, 'go--', alpha=0.5, label='Model 2 validation')

plt.plot(epochs3, val_acc3, 'ro--', alpha=0.5, label='Model 3 validation')



plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.grid()

plt.legend()



plt.figure(figsize=(20,10))

plt.plot(epochs, loss, 'bo-', label='Model 1 training')

plt.plot(epochs2, loss2, 'go-', label='Model 2 training')

plt.plot(epochs3, loss3, 'ro-', label='Model 3 training')

plt.plot(epochs, val_loss, 'bo--', alpha=0.7, label='Model 1 validation')

plt.plot(epochs2, val_loss2, 'go--', alpha=0.7, label='Model 2 validation')

plt.plot(epochs3, val_loss3, 'ro--', alpha=0.7, label='Model 3 validation')



plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.grid()



plt.show()




#Plot false negatives

plt.figure(figsize=(20,10))

plot(epochs, fn, 'bo-', label = "Model 1 training")

plot(epochs, vfn, 'b*--', alpha=0.5,  label = "Model 1 validation")

plot(epochs, fn2, 'go-', label = "Model 2 training")

plot(epochs, vfn2, 'g*--',alpha=0.5, label = "Model 2 validation")

plot(epochs, fn3, 'ro-', label = "Model 3 training")

plot(epochs, vfn3, 'r*--', alpha=0.5, label = "Model 2 validation")

plt.xlabel('epochs')

plt.ylabel('n false negatives')

title('False negatives')

grid()

legend()



#Plot false positives

plt.figure(figsize=(20,10))

plot(epochs, fp, 'bo-', label = "Model 1 training")

plot(epochs, vfp, 'b*--', alpha=0.5,  label = "Model 1 validation")

plot(epochs, fp2, 'go-', label = "Model 2 training")

plot(epochs, vfp2, 'g*--',alpha=0.5, label = "Model 2 validation")

plot(epochs, fp3, 'ro-', label = "Model 3 traiining")

plot(epochs, vfp3, 'r*--', alpha=0.5, label = "Model 3 validation")

plt.xlabel('epochs')

plt.ylabel('n false positives')

title('False positives')

grid()

legend()



show()
# Model predictions using validation data



# assign the labels



labels = dev_generator.classes



# predict the results for each model



predicted = model.predict_generator(dev_generator).flatten()

predicted2 = model2.predict_generator(dev_generator).flatten()

predicted3 = model3.predict_generator(dev_generator).flatten()
# Plot the predicted and true labels for model 1



plt.plot (predicted, 'b', label = 'Predicted')

plt.plot (labels, 'k', label = 'True value')

plt.legend()

plt.xlabel ('Case index')

plt.title('predicted and true labels for model 1')

plt.grid()
# Plot the predicted and true labels for model 2



plt.plot (predicted2,'g', label = 'Predicted')

plt.plot (labels,'k', label = 'True value')

plt.legend()

plt.xlabel ('Case index')

plt.title('predicted and true labels for model 2')

plt.grid()
# Plot the predicted and true labels for model 3



plt.plot (predicted3,'r', label = 'Predicted')

plt.plot (labels,'k', label = 'True value')

plt.legend()

plt.xlabel ('Case index')

plt.title('predicted and true labels for model 3')

plt.grid()
# Create a confusion matrix and calculate classification report for analysis



# creation for model 1



print ('Confusion matrix (medical presentation) and classification report for model 1:\n')

tn, fp, fn, tp =  confusion_matrix(labels, predicted > 0.5).ravel()

cm= array([[tp,fn],[fp,tn]])

print(cm)



# maximum value of sensitivity at specificity of 0,9 from the validation set

maximumsas=max(val_sas)

print("\nModel 1 Highest validation sensitivity at specificity 0,9: ",maximumsas,"\n")

print("\nModel 1 validation sensitivity at specificity 0,9 at the end of training: ", val_sas,"\n")



cr = classification_report(labels, predicted > 0.5, target_names = ['Normal (0)', 'Pneumonia (1)'])

print(cr)





# creation for model 2

print ('\nConfusion matrix (medical presentation) and classification report for model 2:\n')

tn, fp, fn, tp =  confusion_matrix(labels, predicted2 > 0.5).ravel()

cm2= array([[tp,fn],[fp,tn]])

print(cm2)



maximumsas=max(val_sas2)

print("\nModel 2 Highest validation sensitivity at specificity 0,9: ", maximumsas,"\n")

print("\nModel 2 validation sensitivity at specificity 0,9 at the end of training: ", val_sas2,"\n")



cr2 = classification_report(labels, predicted2 > 0.5, target_names = ['Normal (0)', 'Pneumonia (1)'])

print(cr2)





# creation for model 3

print ('\nConfusion matrix (medical presentation) and classification report for model 3:\n')

tn, fp, fn, tp =  confusion_matrix(labels, predicted3 > 0.5).ravel()

cm3= array([[tp,fn],[fp,tn]])

print(cm3)



maximumsas=max(val_sas3)

print("\nModel 3 Highest validation sensitivity at specificity 0,9: ", maximumsas,"\n")

print("\nModel 3 validation sensitivity at specificity 0,9 at the end of training: ", val_sas3,"\n")



cr3 = classification_report(labels, predicted3 > 0.5, target_names = ['Normal (0)', 'Pneumonia (1)'])

print(cr3)







# Calculate the ROC curves for further analysis



fpr, tpr, thresholds = roc_curve(labels, predicted, pos_label = 1)

fpr2, tpr2, thresholds = roc_curve(labels, predicted2, pos_label = 1)

fpr3, tpr3, thresholds = roc_curve(labels, predicted3, pos_label = 1)



# Show the ROC curve plot



plt.plot (fpr, tpr,'b', label = 'Model 1')

plt.plot (fpr2, tpr2,'g', label = 'Model 2')

plt.plot (fpr3, tpr3,'r', label = 'Model 3')

plt.plot ([0, 1], [0, 1], 'r:')

plt.xlabel('False positive rate')

plt.ylabel('True positivite rate')

plt.title('ROC Curve')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.legend()

plt.grid()
# Find the labels from test generator



labels_test = test_generator.classes



# Predict the results using the unseen test data



predicted_test = model.predict_generator(test_generator).flatten()

predicted_test2 = model2.predict_generator(test_generator).flatten()

predicted_test3 = model3.predict_generator(test_generator).flatten()
# Create a confusion matrix and calculate classification report for observations



# creation for model 1

print ('Confusion matrix (medical presentation) and classification report for model 1:\n')

tn, fp, fn, tp =  confusion_matrix(labels_test, predicted_test > 0.5).ravel()

cm= array([[tp,fn],[fp,tn]])

print(cm)

cr = classification_report(labels_test, predicted_test > 0.5, target_names = ['Normal (0)', 'Pneumonia (1)'])

print(cr)



# creation for model 2

print ('Confusion matrix (medical presentation) and classification report for model 2:\n')

tn, fp, fn, tp =  confusion_matrix(labels_test, predicted_test2 > 0.5).ravel()

cm2= array([[tp,fn],[fp,tn]])

print(cm2)

cr2 = classification_report(labels_test, predicted_test2 > 0.5, target_names = ['Normal (0)', 'Pneumonia (1)'])

print(cr2)



# creation for model 3

print ('Confusion matrix (medical presentation) and classification report for model 3:\n')

tn, fp, fn, tp =  confusion_matrix(labels_test, predicted_test3 > 0.5).ravel()

cm3= array([[tp,fn],[fp,tn]])

print(cm3)

cr3 = classification_report(labels_test, predicted_test3 > 0.5, target_names = ['Normal (0)', 'Pneumonia (1)'])

print(cr3)