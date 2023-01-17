# Display video link below

from IPython.lib.display import YouTubeVideo

YouTubeVideo('JeF-OTbDcGg')
# First, we'll import pandas and numpy, two data processing libraries

import pandas as pd

import numpy as np



# We'll also import seaborn and matplot, twp Python graphing libraries

import seaborn as sns

import matplotlib.pyplot as plt

# Import the needed sklearn libraries

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder



# The Keras library provides support for neural networks and deep learning

print ("====== This should generate a FutureWaring on Conversion ===== ignore this warning")

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Lambda, Flatten, LSTM

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import Adam, RMSprop

from keras.utils import np_utils, to_categorical



# We will turn off some warns in this notebook to make it easier to read for new students

import warnings

warnings.filterwarnings('ignore')
# Read data from the actual Kaggle download files stored in a raw file in GitHub

github_folder = 'https://raw.githubusercontent.com/CIS3115-Machine-Learning-Scholastica/CIS3115ML-Units7and8/master/petfinder-adoption/'

kaggle_folder = '../input/'



data_folder = github_folder

# Uncomment the next line to switch from using the github files to the kaggle files for a submission

#data_folder = kaggle_folder



train = pd.read_csv(data_folder + 'train/train.csv')

submit = pd.read_csv(data_folder + 'test/test.csv')



sample_submission = pd.read_csv(data_folder + 'test/sample_submission.csv')

labels_breed = pd.read_csv(data_folder + 'breed_labels.csv')

labels_color = pd.read_csv(data_folder + 'color_labels.csv')

labels_state = pd.read_csv(data_folder + 'state_labels.csv')



print ("training data shape: " ,train.shape)

print ("submission data shape: : " ,submit.shape)
train.head(5)
# Select which features to use

pet_train = train[['Age','Dewormed','Health', 'Sterilized', 'Fee']]

# Everything we do to the training data we also should do the the submission data

pet_submit = submit[['Age','Dewormed','Health','Sterilized', 'Fee']]



# Convert output to one-hot encoding

pet_adopt_speed = to_categorical( train['AdoptionSpeed'] )



print ("pet_train data shape: " ,pet_train.shape)

print ("pet_submit data shape: " ,pet_submit.shape)

print ("pet_adopt_speed data shape: " ,pet_adopt_speed.shape)

# Add any columns to the list below that you want dummy variables created

cat_columns = ['Health','Vaccinated','Dewormed','Sterilized']



# You should not need to change any code below this line

# =======================================================



# Create the dummy variables for the columns listed above

dfTemp = pd.get_dummies( train[cat_columns], columns=cat_columns )

pet_train = pd.concat([pet_train, dfTemp], axis='columns')



# Do the same to the submission data

dfSummit = pd.get_dummies( submit[cat_columns], columns=cat_columns )

pet_submit = pd.concat([pet_submit, dfSummit], axis='columns')

# Get missing columns in the submission data

missing_cols = set( pet_train.columns ) - set( pet_submit.columns )

# Add a missing column to the submission set with default value equal to 0

for c in missing_cols:

    pet_submit[c] = 0

# Ensure the order of column in the test set is in the same order than in train set

pet_submit = pet_submit[pet_train.columns]



# We should check the that the number of features is not too large and that the training and submission data still have the same number of features







# print out the current data

print ("Size of pet_train = ", pet_train.shape)

print ("Size of pet_submit = ", pet_submit.shape)

pet_train.head(5)
# Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)

#encodedVaccinated = train[['Vaccinated']] 

def fixVac( value ):

    if value == 1: return +1

    elif value == 2: return -1

    else: return 0



#train['encodedVaccinated'] = list(map(lambda a: 0 if (a>1) else a,train['Vaccinated']))

pet_train['encodedVaccinated'] = list(map(fixVac,train['Vaccinated']))

# Do the same thing to the submission data

pet_submit['encodedVaccinated'] = list(map(fixVac,submit['Vaccinated']))



pet_train.head(10)
print ("pet_train data shape: " ,pet_train.shape)

print ("pet_adopt_speed data shape: " ,pet_adopt_speed.shape)

print ("pet_submit data shape: " ,pet_submit.shape)

# Scale the data to put large features like area_mean on the same footing as small features like smoothness_mean

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()

pet_train_scaled = scaler.fit_transform(pet_train)

pet_submit_scaled = scaler.fit_transform(pet_submit)



pet_train_scaled
# Split the data into 80% for training and 10% for testing out the models

X_train, X_test, y_train, y_test = train_test_split(pet_train_scaled, pet_adopt_speed, test_size=0.1)



print ("X_train training data shape of 28x28 pixels greyscale: " ,X_train.shape)

print ("X_test submission data shape of 28x28 pixels greyscale: : " ,X_test.shape)



print ("y_train training data shape of 28x28 pixels greyscale: " ,y_train.shape)

print ("y_test submission data shape of 28x28 pixels greyscale: : " ,y_test.shape)
# Set up the Neural Network

input_Size = X_test.shape[1]     # This is the number of features you selected for each pet

output_Size = y_train.shape[1]   # This is the number of categories for adoption speed, should be 5



NN = Sequential()

NN.add(Dense(50, activation='relu', input_dim=(input_Size)))

NN.add(Dropout(0.3))

NN.add(Dense(35, activation='relu'))

NN.add(Dense(20, activation='relu'))

NN.add(Dense(10, activation='relu'))

NN.add(Dropout(0.3))

NN.add(Dense(output_Size, activation='softmax'))



# Compile neural network model

NN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint



learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 

                                            patience=5, 

                                            verbose=2, 

                                            factor=0.5,                                            

                                            min_lr=0.000001)



early_stops = EarlyStopping(monitor='val_loss', 

                            min_delta=0, 

                            patience=20, 

                            verbose=2, 

                            mode='auto')



checkpointer = ModelCheckpoint(filepath = 'cis3115_MNIST.{epoch:02d}-{accuracy:.6f}.hdf5',

                               verbose=2,

                               save_best_only=True, 

                               save_weights_only = True)

# Fit model on training data for network with dense input layer



history = NN.fit(X_train, y_train,

          epochs=200,

          verbose=1,

          callbacks=[learning_rate_reduction, early_stops],

          validation_data=(X_test, y_test))

# 10. Evaluate model on test data

print ("Running final scoring on test data")

score = NN.evaluate(X_test, y_test, verbose=1)

print ("The accuracy for this model is ", format(score[1], ",.2f"))
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)



ax[0].plot(history.history['acc'], color='b', label="Training accuracy")

ax[0].plot(history.history['val_acc'], color='r',label="Testing accuracy")

ax[0].set_title("Accruacy")

legend = ax[0].legend(loc='best', shadow=True)

ax[0].set_ylim([0, 1])

              

ax[1].plot(history.history['loss'], color='b', label="Training loss")

ax[1].plot(history.history['val_loss'], color='r', label="Testing loss",axes =ax[1])

ax[1].set_title("Loss")

legend = ax[1].legend(loc='best', shadow=True)

print ("pet_train data shape: " ,pet_train.shape)

print ("submit data shape: " ,submit.shape)

print ("pet_submit data shape: " ,pet_submit_scaled.shape)

predictions = NN.predict_classes(pet_submit_scaled, verbose=1)



submissions=pd.DataFrame({'PetID': submit.PetID})

submissions['AdoptionSpeed'] = predictions



submissions.to_csv("submission.csv", index=False, header=True)



submissions.head(10)