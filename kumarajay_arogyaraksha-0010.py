# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

from PIL import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



%matplotlib inline

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
skin_df=pd.read_csv("../input/HAM10000_metadata.csv")
skin_df.head()
base_skin_dir = os.path.join('..', 'input')

print(base_skin_dir)
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x

                    for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}
skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df
lesion_type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}

skin_df['target'] = skin_df['dx'].map(lesion_type_dict.get) 

skin_df['target_cat'] = pd.Categorical(skin_df['target']).codes
skin_df.head()
label_cat_ids=dict(zip(skin_df['target_cat'],skin_df['dx']))
label_cat_ids
skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)
skin_df['target'].value_counts()
skin_df['sex'].value_counts()
skin_df['localization'].value_counts()
skin_df['dx_type'].value_counts()
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,100))))




classes = list(skin_df['dx'].unique())

train_data = pd.DataFrame(columns = list(skin_df.columns))

test_data = pd.DataFrame(columns = list(skin_df.columns))







def split(temp_df,split_ratio):

    train, test = train_test_split(temp_df, test_size=split_ratio)

    return train,test



from sklearn.model_selection import train_test_split

for cl in classes:

    mask= skin_df['dx']==cl

    temp_df = skin_df[mask]

    if cl=='nv':

        split_ratio=0.65

    else:

        split_ratio=0.2

    trn,tst = split(temp_df,split_ratio)

    train_data=pd.concat([train_data,trn],ignore_index = True)

    test_data=pd.concat([test_data,tst],ignore_index = True)

    
train_data.head()

print(len(train_data))

print(len(test_data))



train_data['image'].map(lambda x: x.shape).value_counts()
features=train_data.drop(columns=['target_cat'],axis=1)

target=train_data['target_cat']
x_test=test_data.drop(columns=['target_cat'],axis=1)

y_test=test_data['target_cat']
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=1234)
x_train = np.asarray(x_train_o['image'].tolist())

x_validate = np.asarray(x_test_o['image'].tolist())

x_test = np.asarray(x_test['image'].tolist())



x_train_mean = np.mean(x_train)

x_train_std = np.std(x_train)



x_valid_mean = np.mean(x_validate)

x_valid_std = np.std(x_validate)



x_test_mean = np.mean(x_test)

x_test_std = np.std(x_test)



x_train = (x_train - x_train_mean)/x_train_std

x_validate = (x_validate - x_valid_mean)/x_valid_std

x_test = (x_test - x_test_mean)/x_test_std

from keras.utils.np_utils import to_categorical 

y_train = to_categorical(y_train_o, num_classes = 7)

y_validate = to_categorical(y_test_o, num_classes = 7)

y_test = to_categorical(y_test, num_classes = 7)
x_train = x_train.reshape(x_train.shape[0], *(100, 100, 3))

x_test = x_test.reshape(x_test.shape[0], *(100, 100, 3))

x_validate = x_validate.reshape(x_validate.shape[0], *(100, 100, 3))


from PIL import Image

from sklearn.preprocessing import label_binarize

from sklearn.metrics import confusion_matrix

import itertools



import keras

from keras.utils.np_utils import to_categorical 

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras import backend as K

import itertools

from keras.layers.normalization import BatchNormalization

from keras.utils.np_utils import to_categorical 



from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping

from sklearn.model_selection import train_test_split
input_shape = (100, 100, 3)

num_classes = 7



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))

model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))

model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Dropout(0.20))



model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.20))



model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))

model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.20))



model.add(Conv2D(256, (3, 3), activation='relu',padding = 'Same'))

model.add(Conv2D(256, (3, 3), activation='relu',padding = 'Same'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.20))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.summary()
# Define the optimizer

optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.9, decay=0.0)
# Compile the model

model.compile(optimizer = "Adam" , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=5, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

model_check_point = ModelCheckpoint(filepath="best_model.h5", verbose=1, save_best_only=True)



early_stopping=EarlyStopping(monitor='val_acc', patience=10, verbose=1)
datagen = ImageDataGenerator(

        rotation_range=25,

        zoom_range = 0.1, 

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        horizontal_flip=True,  

        vertical_flip=True)  



datagen.fit(x_train)
# Fit the model

epochs = 50 

batch_size = 10

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_validate,y_validate),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction,model_check_point,early_stopping])
from keras.models import load_model

model = load_model('best_model.h5')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

model.save("model.h5")
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

# Function to plot confusion matrix    

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

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



Y_pred = model.predict(x_validate)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_validate,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)



 



plot_confusion_matrix(confusion_mtx, classes = range(7)) 
# Function to plot confusion matrix    

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

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



Y_pred = model.predict(x_test)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_test,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plot_confusion_matrix(confusion_mtx, classes = range(7)) 
def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy

    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])

    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)

    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()
plot_model_history(history)