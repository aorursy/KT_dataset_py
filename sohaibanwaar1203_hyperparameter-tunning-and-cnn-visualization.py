#!pip install talos # hyperparameter tuning

#!pip install --upgrade pip
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # showing and rendering figures

# io related

import seaborn as sns # for Visualizing my data

from keras.utils import to_categorical # Using keras to_categorical because I need to convert by labels 

# into categorical form

import os # for taking input to my dataframe

import cv2 # for resizing my iamges

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# not needed in Kaggle, but required in Jupyter

%matplotlib inline 



# Here I am building my dataframe (taking patient id  from image coloum and seperating side of image 

# left or right specefying path of image and at last converting my labels into categorical labels)

temp_df=pd.read_csv('../input/diabetic-retinopathy-detection/trainLabels.csv') # uploading csv to my pandas dataframe

print(temp_df.head()) # displaying first 5 objects in dataframe

image=temp_df['image'].str.split('_',n=1,expand=True) #splitting Side and Patient ID 

df = pd.DataFrame()# creating new dataframe object

df['eye_side']=image[1] #taking side of Image

df['patient_id']=image[0]#taking patient id of an Image



df['path']='../input/diabetic-retinopathy-detection/'#Giving paths of the images 

df['path']=df['path'].str.cat(temp_df['image']+'.jpeg')#adding Image path and format 

df['exists'] = df['path'].map(os.path.exists)

df=df[df['exists']]

df['level']=temp_df['level']# taking levels of Image

df['level_cat'] = df['level'].map(lambda x: to_categorical(x, 1+df['level'].max()))#converting my 

# labels to categorical_labels

df.head()
im = plt.imread(df.path.values[2]) # reading Image from its path



plt.imshow(im)# show the image

plt.show()
sizes = df['level'].values #taking values from series because I only want to visualize levels nt index

print(sizes[0:5])#printing first 5 values

sns.distplot(sizes, kde=False); # Visualizing levels in dataset
pd.value_counts(sizes) # viewing the values of levels
import PIL

from PIL import Image

baseheight = 128

img = Image.open('../input/diabetic-retinopathy-detection/1192_right.jpeg')



wsize = 128

img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)

img.save('resized_image.jpg')# i need this for storing my previous image also 
im = plt.imread('resized_image.jpg') # reading Image from its path



plt.imshow(im)# show the image

plt.title("Resized Image")

plt.show()



im = plt.imread('../input/diabetic-retinopathy-detection/1192_right.jpeg') # reading Image from its path



plt.imshow(im)# show the image

plt.title("Orignal Image")

plt.show()
#total Examples of LEVEL [1,2,3,4] So that we are able to make balanced dataset

sum_E=0

for i in range (1,5):

    L1_df=pd.DataFrame()# creating new dataframe object

    L1_df =df [df.level==i]

    x=len(L1_df)

    sum_E=x+sum_E

print(sum_E)
B_df=pd.read_csv('../input/prepossessed-arrays-of-binary-data/1000_Binary Dataframe')

B_df=B_df.drop('Unnamed: 0',axis=1)

B_df.head(10)
sizes =B_df['level'].values

sns.distplot(sizes, kde=False); # Visualizing levels in dataset
Binary_90 = np.load('../input/prepossessed-arrays-of-binary-data/1000_Binary_images_data_90.npz')

X_90=Binary_90['a']

Binary_128 = np.load('../input/prepossessed-arrays-of-binary-data/1000_Binary_images_data_128.npz')

X_128=Binary_128['a']

Binary_264 = np.load('../input/prepossessed-arrays-of-binary-data/1000_Binary_images_data_264.npz')

X_264=Binary_264['a']

y=B_df['level'].values





print(X_90.shape)

print(X_128.shape)

print(X_264.shape)

print(y.shape)
# we need to resize our X because we load array in 2 diminsional and we need it in 4 diminsional

print("Shape before reshaping X_90" +str(X_90.shape))

X_90=X_90.reshape(1000,90,90,3)

print("Shape after reshaping X_90" +str(X_90.shape))

print("\n\n")



print("Shape before reshaping X_128" +str(X_128.shape))

X_128=X_128.reshape(1000,128,128,3)

print("Shape after reshaping X_128" +str(X_128.shape))

print("\n\n")



print("Shape before reshaping X_264" +str(X_264.shape))

X_264=X_264.reshape(1000,264,264,3)

print("Shape after reshaping X_264" +str(X_264.shape))

im = plt.imread(B_df['path'][1]) # reading Image from its path



plt.imshow(im)# show the image

plt.title("Orignal Image")

plt.show()
plt.title("90*90*3 Image")

plt.imshow(X_90[1])

plt.show()



plt.title("128*128*3 Image")

plt.imshow(X_128[1])

plt.show()



plt.title("264*264*3 Image")

plt.imshow(X_264[1])

plt.show()
y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X_128,y, test_size=0.10, random_state=42)

y_train = to_categorical(y_train, num_classes=2)

y_test_Categorical=to_categorical(y_test)

y_categorical =to_categorical(y)
from keras.models import Sequential,Model

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,Activation

from keras import losses

from keras.optimizers import Adam, Adagrad

from keras.callbacks import EarlyStopping

from keras import regularizers

from sklearn.model_selection import GridSearchCV

import keras

#import talos as ta


def Talos_Model(X_train, y_train, X_test, y_test, params):

    #parameters defined

    lr = params['lr']

    epochs=params['epochs']

    dropout_rate=params['dropout']

    optimizer=params['optimizer']

    loss=params['loss']

    last_activation=params['last_activation']

    activation=params['activation']

    clipnorm=params['clipnorm']

    decay=params['decay']

    momentum=params['momentum']

    l1=params['l1']

    l2=params['l2']

    No_of_CONV_and_Maxpool_layers=params['No_of_CONV_and_Maxpool_layers']

    No_of_Dense_Layers =params['No_of_Dense_Layers']

    No_of_Units_in_dense_layers=params['No_of_Units_in_dense_layers']

    Kernal_Size=params['Kernal_Size']

    Conv2d_filters=params['Conv2d_filters']

    pool_size_p=params['pool_size']

    padding_p=params['padding']

    

    #model sequential

    model=Sequential()

    

    for i in range(0,No_of_CONV_and_Maxpool_layers):

        model.add(Conv2D(Conv2d_filters, Kernal_Size ,padding=padding_p))

        model.add(Activation(activation))

        model.add(MaxPooling2D(pool_size=pool_size_p,strides=(2,2)))

    

    

    model.add(Flatten())

    

    for i in range (0,No_of_Dense_Layers):

        model.add(Dense(units=No_of_Units_in_dense_layers,activation=activation, kernel_regularizer=regularizers.l2(l2),

                  activity_regularizer=regularizers.l1(l1)))

    

    

    model.add(Dense(units=20,activation=activation))

    

    model.add(Dense(units=2,activation=activation))

    if optimizer=="Adam":

        opt=keras.optimizers.Adam(lr=lr, decay=decay, beta_1=0.9, beta_2=0.999)

    if optimizer=="Adagrad":

        opt=keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=decay)

    if optimizer=="sgd":

        opt=keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)

    

    model.compile(loss=loss,optimizer=opt,

                 metrics=['accuracy'])

    

    out = model.fit(X_train, y_train, epochs=params['epochs'])



    return out,model




params = {'lr': (0.1, 0.01,1 ),

     'epochs': [10,5,15],

     'dropout': (0, 0.40, 0.8),

     'optimizer': ["Adam","Adagrad","sgd"],

     'loss': ["binary_crossentropy","mean_squared_error","mean_absolute_error"],

     'last_activation': ["softmax","sigmoid"],

     'activation' :["relu","selu","linear"],

     'clipnorm':(0.0,0.5,1),

     'decay':(1e-6,1e-4,1e-2),

     'momentum':(0.9,0.5,0.2),

     'l1': (0.01,0.001,0.0001),

     'l2': (0.01,0.001,0.0001),

     'No_of_CONV_and_Maxpool_layers':[2,3],

     'No_of_Dense_Layers': [2,3,4],

     'No_of_Units_in_dense_layers':[128,64,32,256],

     'Kernal_Size':[(2,2),(4,4),(6,6)],

     'Conv2d_filters':[60,40,80,120],

     'pool_size':[(2,2),(4,4)],

     'padding':["valid","same"]

    }



def Randomized_Model(lr=0.01,dropout=0.5,optimizer="adam",loss='mean_squared_error',

                    last_activation="softmax",activation="relu",clipnorm=0.1,

                    decay=1e-2,momentum=0.5,l1=0.01,l2=0.001,No_of_CONV_and_Maxpool_layers=3,

                    No_of_Dense_Layers=3,No_of_Units_in_dense_layers=24,Conv2d_filters=60):

       

    

    

    #model sequential

    model=Sequential()

    

    for i in range(0,No_of_CONV_and_Maxpool_layers):

        model.add(Conv2D(Conv2d_filters, (2,2) ,padding="same"))

        model.add(Activation(activation))

        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    

    

    model.add(Flatten())

    

    for i in range (0,No_of_Dense_Layers):

        model.add(Dense(units=No_of_Units_in_dense_layers,activation=activation, kernel_regularizer=regularizers.l2(l2),

                  activity_regularizer=regularizers.l1(l1)))

    

    model.add(Dropout(dropout))

    model.add(Dense(units=20,activation=activation))

    

    model.add(Dense(units=2,activation=activation))

    if optimizer=="Adam":

        opt=keras.optimizers.Adam(lr=lr, decay=decay, beta_1=0.9, beta_2=0.999)

    if optimizer=="Adagrad":

        opt=keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=decay)

    if optimizer=="sgd":

        opt=keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)

    

    model.compile(loss=loss,optimizer=opt,

                 metrics=['accuracy'])

    

    



    return model




params = {'lr': (0.1, 0.01,1,0.001 ),

     'epochs': [10,5,15,30],

     'dropout': (0, 0.40, 0.8),

     'optimizer': ["Adam","Adagrad","sgd"],

     'loss': ["binary_crossentropy","mean_squared_error","mean_absolute_error"],

     'last_activation': ["softmax","sigmoid"],

     'activation' :["relu","selu","linear"],

     'clipnorm':(0.0,0.5,1),

     'decay':(1e-6,1e-4,1e-2),

     'momentum':(0.9,0.5,0.2),

     'l1': (0.01,0.001,0.0001),

     'l2': (0.01,0.001,0.0001),

     'No_of_CONV_and_Maxpool_layers':[2,3],

     'No_of_Dense_Layers': [2,3,4,5],

     'No_of_Units_in_dense_layers':[128,64,32,256],

     

     'Conv2d_filters':[60,40,80,120,220]

     

     

    }

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import RandomizedSearchCV, KFold

from sklearn.metrics import make_scorer

# model class to use in the scikit random search CV 

model = KerasClassifier(build_fn=Randomized_Model, epochs=10, batch_size=20, verbose=1)

grid = RandomizedSearchCV(estimator=model, cv=KFold(3), param_distributions=params, 

                          verbose=20,  n_iter=10, n_jobs=1)

grid_result = grid.fit(X_train, y_train)
best_params=grid_result.best_params_

best_params


from sklearn.metrics import accuracy_score



y=grid_result.predict(X_test)

random=accuracy_score(y, y_test)

print("Base Accuracy ",random)



best_random = grid_result.best_estimator_

y1=best_random.predict(X_test)

Best=accuracy_score(y1, y_test)

print("Best Accuracy " ,Best)





print('Improvement of {:0.2f}%.'.format( 100 * (Best - random) / random))
def Best_param_Model(best_params):

       

    lr=best_params["lr"]

    dropout=best_params["dropout"]

    optimizer=best_params["optimizer"]

    loss=best_params["loss"]

    last_activation=best_params["last_activation"]

    activation=best_params["activation"]

    clipnorm=best_params["clipnorm"]

    decay=best_params["decay"]

    momentum=best_params["momentum"]

    l1=best_params["l1"]

    l2=best_params["l2"]

    No_of_CONV_and_Maxpool_layers=best_params["No_of_CONV_and_Maxpool_layers"]

    No_of_Dense_Layers=best_params["No_of_Dense_Layers"]

    No_of_Units_in_dense_layers=best_params["No_of_Units_in_dense_layers"]

    Conv2d_filters=best_params["Conv2d_filters"]

    

    #model sequential

    model=Sequential()

    

    for i in range(0,No_of_CONV_and_Maxpool_layers):

        model.add(Conv2D(Conv2d_filters, (2,2) ,padding="same"))

        model.add(Activation(activation))

        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    

    

    model.add(Flatten())

    

    for i in range (0,No_of_Dense_Layers):

        model.add(Dense(units=No_of_Units_in_dense_layers,activation=activation, kernel_regularizer=regularizers.l2(l2),

                  activity_regularizer=regularizers.l1(l1)))

    

    

    model.add(Dense(units=20,activation=activation))

    

    model.add(Dense(units=2,activation=activation))

    if optimizer=="Adam":

        opt=keras.optimizers.Adam(lr=lr, decay=decay, beta_1=0.9, beta_2=0.999)

    if optimizer=="Adagrad":

        opt=keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=decay)

    if optimizer=="sgd":

        opt=keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)

    

    model.compile(loss=loss,optimizer=opt,

                 metrics=['accuracy'])

    

    



    return model


Binary_model=Best_param_Model(best_params)

history =Binary_model.fit(X_train, y_train, epochs=100, validation_data=(X_test,y_test_Categorical))



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])





plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])



plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
Binary_model.evaluate(X_test,y_test_Categorical)
y=B_df['level'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X_128,y, test_size=0.10, random_state=42)

y_train = to_categorical(y_train, num_classes=2)

y_test_Categorical=to_categorical(y_test)


model = Sequential()

model.add(Conv2D(16,kernel_size = (5,5),activation = 'relu', activity_regularizer=regularizers.l2(1e-8)))

model.add(Conv2D(32,kernel_size = (5,5),activation = 'relu', activity_regularizer = regularizers.l2(1e-8)))

model.add(MaxPooling2D(3,3))

model.add(Conv2D(64,kernel_size = (5,5),activation = 'relu', activity_regularizer = regularizers.l2(1e-8)))

model.add(MaxPooling2D(3,3))

model.add(Conv2D(128,activation = 'relu',kernel_size = (3,3),activity_regularizer = regularizers.l2(1e-8)))

model.add(Flatten())

model.add(Dropout(0.4))

model.add(Dense(64,activation = 'tanh',activity_regularizer = regularizers.l2(1e-8)))

model.add(Dropout(0.2))

model.add(Dense(16,activation = 'tanh',activity_regularizer = regularizers.l2(1e-8)))

model.add(Dropout(0.2))

model.add(Dense(2,activation = 'softmax'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer="adam", metrics=["accuracy"])

model.fit(X_train,y_train, epochs = 10 ,batch_size = 16,validation_data=(X_test,y_test_Categorical))

model.summary()

from sklearn.metrics import confusion_matrix

prediction=model.predict(X_test)

y_pred=[]

for i in prediction:

    y_pred.append(i.argmax())

y_pred=np.asarray(y_pred)

true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_test, y_pred).ravel()



print("true_negative: ",true_negative)

print("false_positive: ",false_positive)

print("false_negative: ",false_negative)

print("true_positive: ",true_positive)

print("\n\n Accuracy Measures\n\n")

Sensitivity=true_positive/(true_positive+false_negative)

print("Sensitivity: ",Sensitivity)



False_Positive_Rate=false_positive/(false_positive+true_negative)

print("False_Positive_Rate: ",False_Positive_Rate)



Specificity=true_negative/(false_positive + true_negative)

print("Specificity: ",Specificity)



#FDR à 0 means that very few of our predictions are wrong

False_Discovery_Rate=false_positive/(false_positive+true_positive)

print("False_Discovery_Rate: ",False_Discovery_Rate)



Positive_Predictive_Value =true_positive/(true_positive+false_positive)

print("Positive_Predictive_Value: ",Positive_Predictive_Value)



a=np.expand_dims( X_train[10],axis=0)

a.shape

layer_outputs = [layer.output for layer in model.layers]

activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(a)

def display_activation(activations, col_size, row_size, act_index): 

    activation = activations[act_index]

    activation_index=0

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))

    for row in range(0,row_size):

        for col in range(0,col_size):

            ax[row][col].imshow(activation[0, :, :, activation_index])

            activation_index += 1

display_activation(activations, 4, 4,1)
top_layer = model.layers[0]

plt.imshow(top_layer.get_weights()[0][:, :, :,15 ])