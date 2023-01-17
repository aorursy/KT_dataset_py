import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.image as mimg

%matplotlib inline

plt.rcParams["figure.figsize"] = (10,7)

from PIL import Image

from scipy import misc



import os



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix





# DEEP LEARNING IMPORTS

from keras.models import Sequential, load_model

from keras.layers import Dense, Conv2D, Activation, Dropout, Flatten, MaxPooling2D

from keras.callbacks import ModelCheckpoint, EarlyStopping
print(os.listdir("../input"))
#one hot encoding function

def one_hot_encoder(df_name, df_column_name, suffix=''):

    temp = pd.get_dummies(df_name[df_column_name]) #get dummies is used to create dummy columns

    df_name = df_name.join(temp, lsuffix=suffix) #join the newly created dummy columns to original dataframe

    df_name = df_name.drop(df_column_name, axis=1) #drop the old column used to create dummy columnss

    return df_name





#function to draw confusion matrix

def draw_confusion_matrix(true,preds):

    conf_matx = confusion_matrix(true, preds)

    sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="viridis")

    plt.show()

    #return conf_matx
train_images = pd.read_csv("../input/fashion-mnist_train.csv")
train_images_x = train_images.iloc[:,1:]
train_images_array = train_images_x.values

train_x = train_images_array.reshape(train_images_array.shape[0], 28, 28, 1)

train_x_scaled = train_x/255
IMAGE_SIZE = (28, 28, 1)
### read the image labels and one hot encode the labels

train_images_y = train_images[['label']]



#do one hot encoding with the earlier created function

train_images_y_encoded = one_hot_encoder(train_images_y, 'label', 'lab')

print(train_images_y_encoded.head())



#get the labels as an array

train_images_y_encoded = train_images_y_encoded.values
#check to see if distribution of target labels are equal (if not equal we need to assign weights to classes)

plt.bar(train_images_y['label'].value_counts().index, train_images_y['label'].value_counts().values)
test_images = pd.read_csv("../input/fashion-mnist_test.csv")
test_images_x = test_images.iloc[:,1:]



test_images_array = test_images_x.values

test_x = test_images_array.reshape(test_images_array.shape[0], 28, 28, 1)

test_x_scaled = test_x/255
test_images_y = test_images[['label']]

test_images_y_encoded = one_hot_encoder(test_images_y, 'label', 'lab')

#get the labels as an array

test_images_y_encoded = test_images_y_encoded.values
train_x, test_x, train_y, test_y = train_test_split(train_x_scaled, train_images_y_encoded, random_state = 101, 

                                                   test_size=0.25)
def cnn_model(size, num_cnn_layers):

    NUM_FILTERS = 32

    KERNEL = (3, 3)

    #MIN_NEURONS = 20

    MAX_NEURONS = 120

    

    model = Sequential()

    

    for i in range(1, num_cnn_layers+1):

        if i == 1:

            model.add(Conv2D(NUM_FILTERS*i, KERNEL, input_shape=size, activation='relu', padding='same'))

        else:

            model.add(Conv2D(NUM_FILTERS*i, KERNEL, activation='relu', padding='same'))



    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(int(MAX_NEURONS), activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(int(MAX_NEURONS/2), activation='relu'))

    model.add(Dense(10, activation='softmax'))

    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    #print(model.summary())

    

    return model
model = cnn_model(IMAGE_SIZE, 2)
model.summary()
#set early stopping criteria

pat = 5 #this is the number of epochs with no improvment after which the training will stop

early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)



#define the model checkpoint callback -> this will keep on saving the model as a physical file

model_checkpoint = ModelCheckpoint('fas_mnist_1.h5', verbose=1, save_best_only=True)



#define a function to fit the model

def fit_and_evaluate(t_x, val_x, t_y, val_y, EPOCHS=20, BATCH_SIZE=128):

    model = None

    model = cnn_model(IMAGE_SIZE, 2)

    results = model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, model_checkpoint], 

              verbose=1, validation_split=0.1)  

    print("Val Score: ", model.evaluate(val_x, val_y))

    return results
n_folds=3

epochs=20

batch_size=128



#save the model history in a list after fitting so that we can plot later

model_history = [] 



for i in range(n_folds):

    print("Training on Fold: ",i+1)

    t_x, val_x, t_y, val_y = train_test_split(train_x, train_y, test_size=0.1, 

                                               random_state = np.random.randint(1,1000, 1)[0])

    model_history.append(fit_and_evaluate(t_x, val_x, t_y, val_y, epochs, batch_size))

    print("======="*12, end="\n\n\n")
plt.title('Accuracies vs Epochs')

plt.plot(model_history[0].history['acc'], label='Training Fold 1')

plt.plot(model_history[1].history['acc'], label='Training Fold 2')

plt.plot(model_history[2].history['acc'], label='Training Fold 3')

plt.legend()

plt.show()
plt.title('Train Accuracy vs Val Accuracy')

plt.plot(model_history[0].history['acc'], label='Train Accuracy Fold 1', color='black')

plt.plot(model_history[0].history['val_acc'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")

plt.plot(model_history[1].history['acc'], label='Train Accuracy Fold 2', color='red', )

plt.plot(model_history[1].history['val_acc'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")

plt.plot(model_history[2].history['acc'], label='Train Accuracy Fold 3', color='green', )

plt.plot(model_history[2].history['val_acc'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")

plt.legend()

plt.show()
#Load the model that was saved by ModelCheckpoint

model = load_model('fas_mnist_1.h5')
model.evaluate(test_x, test_y)
model.evaluate(test_x_scaled, test_images_y_encoded)
#function for converting predictions to labels

def prep_submissions(preds_array, file_name='abc.csv'):

    preds_df = pd.DataFrame(preds_array)

    predicted_labels = preds_df.idxmax(axis=1) #convert back one hot encoding to categorical variabless

    return predicted_labels

    '''

    ### prepare submissions in case you need to submit

    submission = pd.read_csv("test.csv")

    submission['label'] = predicted_labels

    submission.to_csv(file_name, index=False)

    print(pd.read_csv(file_name).head())

    '''
test_preds = model.predict(test_x_scaled)

test_preds_labels = prep_submissions(test_preds)
print(classification_report(test_images_y, test_preds_labels))
draw_confusion_matrix(test_images_y, test_preds_labels)