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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from pathlib import Path

import datetime

from keras.models import Sequential

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from keras import regularizers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.layers.convolutional import MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline
INPUT = Path("../input/digit-recognizer")

os.listdir(INPUT)
# Load the data

X_train = pd.read_csv(INPUT/"train.csv")

X_test = pd.read_csv(INPUT/"test.csv")
print(" Training Data Shape: " + str(X_train.shape))
print(" Test Data Shape: " + str(X_test.shape))
X_train.head()
id_train = X_train.index.values

y_train = X_train['label']

y_valid_pred = 0*y_train



X_train.drop(labels=['label'], axis=1, inplace=True)



# prepare the test data set by removing the id

#X_test.drop(labels=['id'], axis=1, inplace=True)



#prepare test data

X_test = X_test.astype('float32') / 255.

X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1).astype('float32')



my_init = 'glorot_uniform'

my_activ = 'relu'

my_optimiser = 'adam'

my_epsilon = 1e-8

nb_classes = 10
fig,ax=plt.subplots(5,10)

for i in range(5):

    for j in range(10):

        ax[i][j].imshow(X_test[np.random.randint(0,X_test.shape[0]),:,:,0],cmap=plt.cm.binary)

        ax[i][j].axis('off')

plt.subplots_adjust(wspace=0, hspace=0)        

fig.set_figwidth(15)

fig.set_figheight(7)

fig.show()
def build_network(input_shape):    

    model = Sequential()

    # For an explanation on conv layers see http://cs231n.github.io/convolutional-networks/#conv

    # For an explanation on pooling layers see http://cs231n.github.io/convolutional-networks/#pool

    # By default the stride/subsample is 1 and there is no zero-padding.

    # use padding="same" if you want to preserve dimensions

    

    #model.add(ZeroPadding2D(padding=(2, 2), data_format="channels_last", input_shape=input_shape))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), input_shape=input_shape, padding="same", activation='relu', kernel_initializer=my_init))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))    

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding="same", activation='relu', kernel_initializer=my_init))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))    

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=my_init))

    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))    

    model.add(Dropout(0.25))

    

    # Flatten the 3D output to 1D tensor for a fully connected layer to accept the input

    model.add(Flatten())

    

    

    #Fully Connected Layer

    model.add(Dense(1024, kernel_initializer=my_init))

    model.add(Activation(my_activ))

    model.add(BatchNormalization())

    model.add(Dropout(0.4)) #dropout is a type of regularisation. Regularisation helps to control overfitting

    #Fully Connected Layer

    model.add(Dense(512, kernel_initializer=my_init))

    model.add(Activation(my_activ))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    

    #Output layer

    model.add(Dense(nb_classes, activation='softmax')) 



    model.compile(optimizer=my_optimiser,

        loss='categorical_crossentropy',

        metrics=['accuracy'])

    



    return model
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        shear_range=20, #move top of image along without moving the bottom or vice versa

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False,  # randomly flip images

        data_format="channels_last"         )
def PlotLoss(his, epoch):

    plt.style.use("ggplot")

    plt.figure()

    plt.plot(np.arange(1, epoch + 1), his.history["loss"], label="train_loss")

    plt.plot(np.arange(1, epoch + 1), his.history["val_loss"], label="val_loss")

    plt.title("Training Loss")

    plt.xlabel("Epoch #")

    plt.ylabel("Loss")

    plt.legend(loc="upper right")

    plt.show()



def PlotAcc(his, epoch):

    plt.style.use("ggplot")

    plt.figure()

    plt.plot(np.arange(1, epoch + 1), his.history["accuracy"], label="train_accuracy")

    plt.plot(np.arange(1, epoch + 1), his.history["val_accuracy"], label="val_accuracy")

    plt.title("Training and Validation Accuracy")

    plt.xlabel("Epoch #")

    plt.ylabel("Accuracy")

    plt.legend(loc="lower right")

    plt.show()
def epoch_cv(df_cv_per_epoch_val_acc, fold_num):

    #Find the best epoch by the best single and the best moving average

    #Update the index to start at 1 with epoch 1 for the validation accuracy data

    if fold_num==0:

        df_cv_per_epoch_val_acc.index += 1 #needed so length of values matches length of index

    df_cv_per_epoch_val_acc['mean_val_acc'] = df_cv_per_epoch_val_acc.mean(axis=1)

    #Calculate an epoch moving average

    num_epochs = df_cv_per_epoch_val_acc.shape[0]

    

    for i in range(1, num_epochs+1):

        #print(i)

        if i<moving_average_period+1:

            df_cv_per_epoch_val_acc.at[i, 'moving_average'] = df_cv_per_epoch_val_acc.iloc[:i]['mean_val_acc'].mean()

        else:

            df_cv_per_epoch_val_acc.at[i, 'moving_average'] = df_cv_per_epoch_val_acc.iloc[i-moving_average_period:i]['mean_val_acc'].mean()

    

    #Locate the Best Epoch Number (not the value but the epoch number) by the Mean per epoch

    best_epoch_cv = df_cv_per_epoch_val_acc['mean_val_acc'].idxmax()

    #Locate the Best Epoch Number (not the value but the epoch number) by the Moving Average

    best_epoch_cv_by_moving_average = df_cv_per_epoch_val_acc['moving_average'].idxmax()

    

    return df_cv_per_epoch_val_acc, best_epoch_cv, best_epoch_cv_by_moving_average
Run_CV = "Y"

Run_Kaggle_Pred = "Y"

n_epochs = 80 #333 #number of epochs

my_batch_size = 96

my_verbose = 0 #how much information keras shows per epoch 0 shows least, 1 shows moving arrows as each epoch progresses, 2 displays accuracy at the end of each epoch

K = 6 #number of folds

len_test = len(X_test)

moving_average_period = 10

mean_chart_lower_epoch_bound = 20

initial_learningrate = 2.2e-3
# Set a learning rate annealer

def lr_decay(epoch):#lrv

    return initial_learningrate * 0.99 ** epoch
print(" Epochs: " + str(n_epochs))

print(" Cross Validation Requested: " + Run_CV)

print(" Kaggle Prediction Requested: " + Run_Kaggle_Pred)
if Run_CV=="Y":

    print(" Batch Size: " + str(my_batch_size))

    print(" Number of K-Folds: " + str(K))





    print(('Fold Preparation: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))

    kfold = StratifiedKFold(n_splits = K, 

                            random_state = 2002, 

                            shuffle = True) 



    oof_pred = None

    df_cv_per_epoch_train_acc = pd.DataFrame()

    df_cv_per_epoch_val_acc = pd.DataFrame()

    

    print(('KFold Model Starting: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))

    

    for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

        my_optimiser = Adam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=my_epsilon, decay=0.0, amsgrad=False)

        

        # Create data for this fold

        X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()

        #X_train_f = X_train_f.values

        #X_val_f = X_val_f.values

    

        # Normalize and reshape

        X_train_f = X_train_f.astype('float32') / 255.

        X_train_f = X_train_f.values.reshape(X_train_f.shape[0], 28, 28, 1).astype('float32') #Fabien Tence suggests this shape suits Tensorflow but Theano requires 1, 28, 28

        X_val_f = X_val_f.astype('float32') / 255.

        X_val_f = X_val_f.values.reshape(X_val_f.shape[0], 28, 28, 1).astype('float32') #Fabien Tence suggests this shape suits Tensorflow but Theano requires 1, 28, 28



        #Identify the input_shape - the cnn needs this

        input_shape = X_train_f.shape[1:]

        #nnet_model = build_model(input_shape=input_shape, classes = nb_classes)

        #nnet_model.compile(loss='categorical_crossentropy', optimizer=my_optimiser, metrics=['accuracy'])

        nnet_model = build_network(input_shape)

     

        y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]

        y_train_f = y_train_f.values

        y_val_f = y_val_f.values

        y_val_f_series = y_val_f



        y_train_f = to_categorical(y_train_f, num_classes = nb_classes)

        y_val_f = to_categorical(y_val_f, num_classes = nb_classes)

    

        print(('Augmenting Data: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))    

        datagen.fit(X_train_f) #This step must be after reshaping

        # Run model for this fold

        print(('Model Fitting: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))

        print('Fold: ' + str(i))

        history = nnet_model.fit_generator(datagen.flow(X_train_f,y_train_f, batch_size=my_batch_size), epochs=n_epochs, verbose=my_verbose, 

                                           steps_per_epoch=X_train.shape[0] // my_batch_size, validation_data=(X_val_f,y_val_f), callbacks=[LearningRateScheduler(lr_decay)])

                

        df_cv_per_epoch_train_acc['fold_'+str(i)] = history.history['accuracy']

        df_cv_per_epoch_val_acc['fold_'+str(i)] = history.history['val_accuracy']

        

        # Generate validation predictions for this fold

        y_preds = nnet_model.predict(X_val_f)

        

        if Run_Kaggle_Pred=="Y":

            if i==0:

                test_preds = nnet_model.predict(X_test)

            else:

                test_preds = test_preds + nnet_model.predict(X_test)

        

        y_preds_series = np.argmax(y_preds,axis = 1)

        y_preds_series = pd.Series(y_preds_series,name="label")



        fold_accuracy = accuracy_score(y_val_f_series, y_preds_series)



        print( " Fold Accuracy = %3.6f"% (fold_accuracy)) # Report the accuracy of the prediction



        if oof_pred is None:

            oof_pred = y_preds_series

            oof_pred_ids = outf_ind

        else:

            oof_pred = np.hstack((oof_pred, y_preds_series))

            oof_pred_ids = np.hstack((oof_pred_ids, outf_ind))

            

        df_cv_per_epoch_train_acc, best_train_epoch_cv, best_train_epoch_cv_by_moving_average = epoch_cv(df_cv_per_epoch_train_acc, i)

        df_cv_per_epoch_val_acc, best_epoch_cv, best_epoch_cv_by_moving_average = epoch_cv(df_cv_per_epoch_val_acc, i)

        print("Best Single Epoch: " + str(best_epoch_cv))

        #print("Best Epoch Moving Average Period: " + str(moving_average_period) + " || Best Epoch: "  + str(best_epoch_cv_by_moving_average))



        df_cv_per_epoch_val_acc['mean_train_acc'] = df_cv_per_epoch_train_acc['mean_val_acc']

        

        PlotAcc(history, n_epochs) # plot the accuracy for this fold

        

    

    #Output CV Epoch Data

    df_cv_per_epoch_val_acc.to_csv('df_cv_epoch_{:%Y%m%d%H%M%S}.csv'.format(datetime.datetime.now()), index=True)

    

    #Deal with oof preds

    oof_pred = np.column_stack((oof_pred_ids, oof_pred))

    df_oof_pred = pd.DataFrame(oof_pred,index=oof_pred[:,0])

    df_oof_pred.columns = ['ImageId', 'Label']

    df_oof_pred = df_oof_pred.sort_values(by=('ImageId'), ascending=True)

    

    y_valid_pred = df_oof_pred['Label'].values

    

    oof_accuracy = accuracy_score(y_train, y_valid_pred)

    print( " Overall Out-of-Fold Accuracy = %3.4f"% (oof_accuracy))   

else:

    print("Cross-Validation skipped")



if Run_Kaggle_Pred=="Y":

    test_preds = test_preds / K

    #format prediction

    results = np.argmax(test_preds,axis = 1)

    results = pd.Series(results,name="Label")



    print(('Writing Prediction: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))

    submission = pd.concat([pd.Series(range(1,len_test+1),name = "ImageId"),results],axis = 1)

    submission.to_csv("submission.csv",index=False)
lbound = min(n_epochs-1, mean_chart_lower_epoch_bound)

ubound = max(n_epochs, lbound) + 1
plt.figure(figsize=(12,8))

plt.plot(np.arange(lbound, ubound), df_cv_per_epoch_val_acc["mean_train_acc"][lbound-1:], label="mean_train_acc")

plt.plot(np.arange(lbound, ubound), df_cv_per_epoch_val_acc["mean_val_acc"][lbound-1:], label="mean_val_acc")

plt.title("Mean Accuracy across folds")

plt.xlabel("Epoch #")

plt.ylabel("Accuracy")

plt.legend(loc="lower right")

plt.show()
if Run_Kaggle_Pred=="Y" or Run_CV=="Y":

    print(nnet_model.summary())

else:

    print("Select Cross-Validation Run, Kaggle Prediction, or both.")
print(('Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))