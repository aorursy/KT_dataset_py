import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential, Model
#from keras.layers import, Dropout, Flatten, Conv2D, MaxPool2D, Input, ZeroPadding2D
from keras.layers import Dense, Dropout, Flatten, Input, Activation, Add, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, MaxPool2D
from keras import backend as K
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.initializers import glorot_uniform

sns.set(style='white', context='notebook', palette='deep')

# Load the dataset for training and test
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# get labels from training set to train models
Y_train = train["label"] 
# drop labels from original training set, and generate a new training set
X_train = train.drop(labels = ["label"],axis = 1)  
# free some RAM space 
del train

# We check that neither training set nor test set includes null value :
#     X_train.isnull().any().describe() 
#     test.isnull().any().describe()

# Normalization, transform from [0,255] to [0,1], to speed up CNN running
X_train = X_train / 255.0
test = test / 255.0

# Tranfor DataFrame to ndarray and Reshape them in 28*28*1 
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1,28,28,1)

# Label encoding
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
#                                        0 -> [1,0,0,0,0,0,0,0,0,0]
#                                        5 -> [0,0,0,0,0,1,0,0,0,0]
#                                        9 -> [0,0,0,0,0,0,0,0,0,1]
Y_train = to_categorical(Y_train, num_classes = 10)

# Split the train and the validation set for the fitting
random_seed = 0
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, 
                                                  random_state=random_seed)
def Simpel_CNN(InputShape=(28,28,1), filters=[32,32,64,64]):
    '''
    Set the CNN model 
    Input -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
    
    Arguments:
        InputShape -- input tensor of shape (n_H_prev, n_W_prev, n_C_prev) (in this case, it is (28,28,1))
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
                   [32,32,64,64] in this case
    '''
    F1, F2, F3, F4 = filters
    model = Sequential()

    model.add(Conv2D(filters = F1, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu', input_shape = InputShape))
    model.add(Conv2D(filters = F2, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(filters = F3, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(Conv2D(filters = F4, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))
    
    return model
              



def Compile_Fit_Simple_CNN():
    # Define the optimizer
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # Compile the Simple_CNN model
    model = Simpel_CNN()
    model.compile(optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) , 
              loss = "categorical_crossentropy", metrics=["accuracy"])
    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)
    # Data augmentation:
    #     Randomly rotate some training images by 10 degrees
    #     Randomly Zoom by 10% some training images
    #     Randomly shift images horizontally by 10% of the width
    #     Randomly shift images vertically by 10% of the height
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

    datagen.fit(X_train)

    epochs = 20 # Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 86
    history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                                  epochs = epochs, 
                                  validation_data = (X_val,Y_val),
                                  verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size, 
                                  callbacks=[learning_rate_reduction])
    return history
# Construct a ResNet50 (learned from Andrew Ng's online-course)

# step 1. Identity block function

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', 
              kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c',
              kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X


# step 2. Convolutional block function
def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', padding='valid',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b', padding='same',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides = (1, 1), name = conv_name_base + '2c', padding='valid',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides = (s, s), name = conv_name_base + 'l', padding='valid',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name = bn_name_base + 'l')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

# step 3. Build ResNet50
def ResNet50(input_shape, classes):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(1, 1))(X)

    X = Dropout(rate=0.3)(X)  # Set into ResNet50

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, f=3, filters = [128,128,512], stage=3, block='b')
    X = identity_block(X, f=3, filters = [128,128,512], stage=3, block='c')
    X = identity_block(X, f=3, filters = [128,128,512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, f=3, filters = [256,256,1024], stage=4, block='b')
    X = identity_block(X, f=3, filters = [256,256,1024], stage=4, block='c')
    X = identity_block(X, f=3, filters = [256,256,1024], stage=4, block='d')
    X = identity_block(X, f=3, filters = [256,256,1024], stage=4, block='e')
    X = identity_block(X, f=3, filters = [256,256,1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, f=3, filters = [512,512,2048], stage=5, block='b')
    X = identity_block(X, f=3, filters = [512,512,2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2,2))(X)
    
    X = Dropout(rate=0.3)(X) # add to ResNet50

    # output layer
    X = Flatten()(X)
    
    X = Dense(36, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)  ##
   # X = Dropout(rate=0.2)(X)                                                            ###  Set into ResNet50
    #X = Dense(36, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)  ## 
    
    X = Dense(36, activation='sigmoid', kernel_initializer = glorot_uniform(seed=0))(X)  ##
    X = Dropout(rate=0.3)(X)                                                            ###  Set into ResNet50
    X = Dense(36, activation='sigmoid', kernel_initializer = glorot_uniform(seed=0))(X)  ## 
    
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
def Compile_Fit_ResNet50(input_shape, number_of_classes) :
    '''
    Compile and fit a ResNet50 model with the Digit Recognizer training images' set
    
    Arguments :
    input_shape -- shape of the images of the dataset
    number_of_classes -- integer, number of classes
    
    Returns:
    A fitted ResNet50 model using data augmentation
    '''
    Digit_ResNet50_Model = ResNet50(input_shape=input_shape, classes=number_of_classes)
    Digit_ResNet50_Model.compile(optimizer="Adam", loss="categorical_crossentropy",
                                 metrics = ["accuracy"])
    
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

    datagen.fit(X_train)
    
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)
    
    epochs = 20 # Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 86
    
    history = Digit_ResNet50_Model.fit_generator(
                                  datagen.flow(X_train,Y_train, batch_size=batch_size),
                                  epochs = epochs, 
                                  validation_data = (X_val,Y_val),
                                  verbose = 1, 
                                  steps_per_epoch=X_train.shape[0] // batch_size, 
                                  callbacks=[learning_rate_reduction])
    return history
    
def Visualize_Learning_Curve(fittedModel, model_name):
    '''
    Visualize the Simple_CNN's learning curve
    
    Argument:
        fittedModel --> a fitting result of a model, model.fit()
        model_name  --> a string, will be same as the plot's title
    '''
    history = fittedModel
    fig, ax = plt.subplots(1,2)
    plt.suptitle(model_name)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    # end def
def Get_Confusion_Matrix(fittedModel, X_val, Y_val):
    '''
    Compute a confusion matrix w.r.t a fitted model
    
    Arguments:
        fittedModel --> a fitting of model, model.fit()
        X_val       --> image set for validation
        Y_val       --> labels, corresponding to X_val
    
    Return:
        ConfutionMatrix, utilization of the function confusion_matrx()
    '''
    model = fittedModel.model
    # Do prediction from the validation dataset, get probabilities for each label
    Y_predicted_prob = model.predict(X_val)     # shape = (2100,10)
    # The highest probability's index gives the predicted digit
    Y_predicted_digits = np.argmax(Y_predicted_prob,axis = 1)  # shape = (2100,)
    # Get the true digits from the encoded labels in the validation set 
    Y_true_digits = np.argmax(Y_val,axis = 1)          # shape = (2100,)

    ConfusionMatrix = confusion_matrix(Y_true_digits, Y_predicted_digits) 
    
    return ConfusionMatrix



def plot_confusion_matrix(cm, classes, model_name,
                          normalize=False,
                          title='Confusion matrix ',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + model_name)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # end def
def Preparation_for_display_errors(fittedModel, X_val, Y_val, topNumber):
    '''
    Generate parameters will be used to display errors
    
    Arguments :
        fittedModel --> a fitting result of a model, model.fit()
        X_val       --> image set for validation
        Y_val       --> labels, corresponding to X_val
        topNumber   --> integer, number of top errors will be displayed
    
    Return :
        most_important_errors (errors_index) --> indices of errors will be displayed. 
        True_pixel (img_errors)              --> true pixel-values corresponding to the wrong prediction
        Wrong_predicted_digits (pred_errors) --> digits (labels) are wrong prediction
        True_digits (obs_errors)             --> true digits (labels) corresponding to the wrong prediction
    '''
    
    model = fittedModel.model
    # Do prediction from the validation dataset, get probabilities for each label
    Y_predicted_prob = model.predict(X_val)     # shape = (2100,10)
    # The highest probability's index gives the predicted digit
    Y_predicted_digits = np.argmax(Y_predicted_prob,axis = 1)  # shape = (2100,)
    # Get the true digits from the encoded labels in the validation set 
    Y_true_digits = np.argmax(Y_val,axis = 1)          # shape = (2100,)

    # Visualize error prediction v.s. real digits in validation set
    errors = (Y_true_digits - Y_predicted_digits != 0) # [a boolean list]
    Wrong_predicted_digits = Y_predicted_digits[errors] # these digits are wrong prediction
    Wrong_predicted_prob   = Y_predicted_prob[errors]  # these probabilities are wrong prediction

    # Probabilities of the wrong predicted digits
    Wrong_predicted_prob_digits = np.max(Wrong_predicted_prob, axis=1) 

    True_digits = Y_true_digits[errors] # true digits corresponding to the wrong prediction
    True_pixel = X_val[errors] # true pixel-values corresponding to the wrong prediction

    # Predicted probabilities of the true values in the error set
    true_prob_errors = np.diagonal(np.take(Wrong_predicted_prob, True_digits, axis=1))

    # Difference between the probability of the predicted label and the true label
    delta_pred_true_errors = Wrong_predicted_prob_digits - true_prob_errors

    # Sorted list of the delta prob errors
    sorted_dela_errors = np.argsort(delta_pred_true_errors)
    
    # Top errors 
    most_important_errors = sorted_dela_errors[-topNumber:]
    
    return most_important_errors, True_pixel, Wrong_predicted_digits, True_digits


def display_errors(errors_index,img_errors,pred_errors, obs_errors, model_name):
    """ 
    This function shows 6 images with their predicted and real labels
    
    Arguments:
        errors_index --> indices of errors will be displayed
        img_errors   --> true pixel-values corresponding to the wrong prediction
        pred_errors  --> digits (labels) are wrong prediction
        obs_errors   --> true digits (labels) corresponding to the wrong prediction
    """
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    plt.suptitle("Top 6 errors for " + model_name, y=1.05)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],
                                                                               obs_errors[error]))
            n += 1
            
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
    plt.tight_layout()   # adjust the distance between subplots  
            
    # end 
def Predict_and_Submit(fittedModel, TestSet, csv_name):
    '''
    Use a fitted model to generate prediction on test set and a submission csv
    
    Argument:
    fittedModel --> a fitted model, expected form: history.model
    TestSet     --> imported data set for test and scoring
    csv_name    --> string, name of the returned csv file
    
    Return:
    csv file for submission
    '''
    prediction =  fittedModel.predict(TestSet) # an array of probabilities 
    predicted_Digits = np.argmax(prediction, axis=1) # a highest probability in each row gives a predicted digit
    predicted_Digits = pd.Series(predicted_Digits, name="Label")
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),
                            predicted_Digits],axis = 1)
    submission_CSV = submission.to_csv(csv_name+".csv", index=False)
    
    return submission_CSV
    
Modelling_SimpleCNN = Compile_Fit_Simple_CNN()
Modelling_ResNet50 = Compile_Fit_ResNet50(input_shape=X_train.shape[1:], number_of_classes=10)
Visualize_Learning_Curve(Modelling_ResNet50, "ResNet50")
Visualize_Learning_Curve(Modelling_SimpleCNN, "SimpleCNN")
ConfusionMatrix_ResNet50 = Get_Confusion_Matrix(Modelling_ResNet50, X_val=X_val, Y_val=Y_val)
plot_confusion_matrix(ConfusionMatrix_ResNet50, classes = range(10), model_name="ResNet50") 


ConfusionMatrix_SimpleCNN = Get_Confusion_Matrix(Modelling_SimpleCNN, X_val=X_val, Y_val=Y_val)
plot_confusion_matrix(ConfusionMatrix_SimpleCNN, classes = range(10), model_name="SimpleCNN") 
most_important_errors, True_pixel, Wrong_predicted_digits, True_digits = Preparation_for_display_errors(Modelling_ResNet50, 
                                                                                                       X_val, Y_val, 6)
display_errors(most_important_errors, True_pixel, Wrong_predicted_digits, True_digits, "ResNet50")
most_important_errors, True_pixel, Wrong_predicted_digits, True_digits = Preparation_for_display_errors(Modelling_SimpleCNN, 
                                                                                                        X_val, Y_val, 6)
display_errors(most_important_errors, True_pixel, Wrong_predicted_digits, True_digits, "SimpleCNN")
submission_SimpleCNN = Predict_and_Submit(Modelling_SimpleCNN.model, test, "submission_SimpleCNN")
submission_ResNet50 = Predict_and_Submit(Modelling_ResNet50.model, test, "submission_ResNet50")