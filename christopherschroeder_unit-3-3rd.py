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

# Use the updated Keras library from Tensorflow -- provides support for neural networks and deep learning

import tensorflow.keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Lambda, Flatten, LSTM

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import Adam, RMSprop

#from tensorflow.keras.utils import np_utils

from tensorflow.keras.utils import to_categorical





# We will turn off some warns in this notebook to make it easier to read for new students

import warnings

warnings.filterwarnings('ignore')



print ("All libraries imported")
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
#from tensorflow.keras.utils import to_categorical



# Select which features to use

pet_train = train[['Type','Gender','Age','Health','Sterilized','Vaccinated','Dewormed']]

# Everything we do to the training data we also should do the the submission data

pet_submit = submit[['Type','Gender','Age','Health','Sterilized','Vaccinated','Dewormed']]



# Convert output to one-hot encoding

pet_adopt_speed = to_categorical( train['AdoptionSpeed'] )



print ("pet_train data shape: " ,pet_train.shape)

print ("pet_submit data shape: " ,pet_submit.shape)

print ("pet_adopt_speed data shape: " ,pet_adopt_speed.shape)

# Add any columns to the list below that you want dummy variables created

cat_columns = ['Type','Gender','Health','Sterilized','Vaccinated','Dewormed']



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
# Type - Pet type (1 = dog, 2 = cat)

#encodedType = train[['Type']] 

def fixVac( value ):

    if value == 1: return +1

    elif value == 2: return -1

    else: return 0



#train['encodedType'] = list(map(lambda a: 0 if (a>1) else a,train['Type']))

pet_train['encodedType'] = list(map(fixVac,train['Type']))

# Do the same thing to the submission data

pet_submit['encodedType'] = list(map(fixVac,submit['Type']))



# Gender - Pet gender (1 = M, 2 = F, 3=group)

#encodedType = train[['Type']] 

def fixVac( value ):

    if value == 1: return +1

    elif value == 2: return -1

    else: return 0



#train['encodedGender'] = list(map(lambda a: 0 if (a>1) else a,train['Gender']))

pet_train['encodedGender'] = list(map(fixVac,train['Gender']))

# Do the same thing to the submission data

pet_submit['encodedGender'] = list(map(fixVac,submit['Gender']))



# Health - Pet Health (1 = health, 2 = minor injury, 3 = serious injury, 0 = N/A)

#encodedHealth = train[['Health']] 

def fixVac( value ):

    if value == 1: return +1

    elif value == 2: return -1

    elif value == 3: return +2

    else: return 0



#train['encodedHealth'] = list(map(lambda a: 0 if (a>1) else a,train['Health']))

pet_train['encodedHealth'] = list(map(fixVac,train['Health']))

# Do the same thing to the submission data

pet_submit['encodedHealth'] = list(map(fixVac,submit['Health']))



# Sterilized - Pet has been fixed (1 = Yes, 2 = No, 3 = Not Sure)

#encodedSterilized = train[['Sterilized']] 

def fixVac( value ):

    if value == 1: return +1

    elif value == 2: return -1

    else: return 0



#train['encodedSterilized'] = list(map(lambda a: 0 if (a>1) else a,train['Sterilized']))

pet_train['encodedSterilized'] = list(map(fixVac,train['Sterilized']))

# Do the same thing to the submission data

pet_submit['encodedSterilized'] = list(map(fixVac,submit['Sterilized']))



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



# Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)

#encodedDewormed = train[['Dewormed']] 

def fixVac( value ):

    if value == 1: return +1

    elif value == 2: return -1

    else: return 0



#train['encodedDewormed'] = list(map(lambda a: 0 if (a>1) else a,train['Dewormed']))

pet_train['encodedDewormed'] = list(map(fixVac,train['Dewormed']))

# Do the same thing to the submission data

pet_submit['encodedDewormed'] = list(map(fixVac,submit['Dewormed']))



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

X_train, X_test, y_train, y_test = train_test_split(pet_train_scaled, pet_adopt_speed, test_size=0.05)



print ("X_train training data shape of 28x28 pixels greyscale: " ,X_train.shape)

print ("X_test submission data shape of 28x28 pixels greyscale: : " ,X_test.shape)



print ("y_train training data shape of 28x28 pixels greyscale: " ,y_train.shape)

print ("y_test submission data shape of 28x28 pixels greyscale: : " ,y_test.shape)
# Set up the Neural Network

input_Size = X_test.shape[1]     # This is the number of features you selected for each pet

output_Size = y_train.shape[1]   # This is the number of categories for adoption speed, should be 5



model = Sequential()

model.add(Dense(500, activation='relu', input_dim=(input_Size)))

#model.add(Dropout(0.3))

model.add(Dense(300, activation='relu'))

model.add(Dense(200, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(output_Size, activation='softmax'))



# Compile neural network model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



print ("Neural Network created")

model.summary()
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint



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



checkpointer = ModelCheckpoint(filepath = 'cis6115_PetFinder.{epoch:02d}-{accuracy:.6f}.hdf5',

                               verbose=2,

                               save_best_only=True, 

                               save_weights_only = True)

# Fit model on training data for network with dense input layer



history = model.fit(X_train, y_train,

          epochs=500,

          verbose=1,

          callbacks=[learning_rate_reduction, early_stops],

          validation_data=(X_test, y_test))

# 10. Evaluate model on test data

print ("Running final scoring on test data")

score = model.evaluate(X_test, y_test, verbose=1)

print ("The accuracy for this model is ", format(score[1], ",.2f"))
# We will display the loss and the accuracy of the model for each epoch

# NOTE: this is a little fancy display than is shown in the textbook

def display_training_curves(training, validation, title, subplot):

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    #ax.set_ylim(0.28,1.05)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)

display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
print ("pet_train data shape: " ,pet_train.shape)

print ("submit data shape: " ,submit.shape)

print ("pet_submit data shape: " ,pet_submit_scaled.shape)

predictions = model.predict_classes(pet_submit_scaled, verbose=1)



submissions=pd.DataFrame({'PetID': submit.PetID})

submissions['AdoptionSpeed'] = predictions



submissions.to_csv("submission.csv", index=False, header=True)



submissions.head(10)