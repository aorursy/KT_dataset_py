##Load Data
import numpy as np
import pandas as pd

input_train = pd.read_csv("../input/digit-recognizer/train.csv")
input_test = pd.read_csv("../input/digit-recognizer/test.csv")
print("Train data Shape : "+str(input_train.shape))
print("Test data Shape : "+str(input_test.shape))
#Check any missing data
print("Any missing data : "+str(np.any(input_train.isnull().sum())))
input_train.head(2)
##Prepare
Y_train = input_train["label"]
X_train = input_train.drop(labels = ["label"],axis = 1) 
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
print("Features Shape : "+str(X_train.shape))
print("Labels Shape : "+str(Y_train.shape))
input_train.groupby("label",axis=0).size()

del input_train # free some space
#View Sample
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(X_train[1][:,:,0])
plt.title("Actual Value : "+str(Y_train.iloc[1]))
plt.axis('off')
plt.show()
#Normalize
X_train = X_train / 255
X_train  = X_train.round()
input_test = input_test / 255
input_test  = input_test.round()
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
Y_train = to_categorical(Y_train, num_classes = 10)
from sklearn.model_selection import train_test_split
random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
from keras.preprocessing.image import ImageDataGenerator
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
###Loading libraries
from keras.models import Sequential, Model
from keras.layers import Input,concatenate,DepthwiseConv2D,Activation, Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
def evaluate_model(model,X_train,Y_train,X_val,Y_val,saveModelName,epochs,batch_size):
    filepath=saveModelName+".hdf5"
    best_accuracy_model = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction,best_accuracy_model]) 
    model.load_weights(filepath)
    return model,history
## Plotting train and test accuracy/loss
def plot_results(history):
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    
# Saving results
result_table_columns = ['CNN Model','Train Accuracy','Test Accuracy']
results = pd.DataFrame(columns=result_table_columns)

def add_result(name,history,results,row):
    results.loc[row,'CNN Model'] = name
    results.loc[row,'Train Accuracy'] = history.history['accuracy'][np.argmax(history.history['val_accuracy'])]
    results.loc[row,'Test Accuracy'] = max(history.history['val_accuracy'])
    return results
##First we try with simple model with no shape reformating
def singelHiddenLayer(pixelcount):
    model = Sequential()
    model.add(Dense(pixelcount, input_dim=pixelcount, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

flat_X_train = X_train.reshape((X_train.shape[0],-1))
model = singelHiddenLayer(flat_X_train.shape[1])
model.fit(flat_X_train, Y_train,epochs=15, batch_size=200, verbose=2)
##To get Idea start with simple CNN model
#[[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
#Method decription
# 2C = 2 Conv2D
# M = MaxPool2D
# Dr = Dropout
# F = Flatten
# D = Dense
def model_2CMDr_2CMDr_FDDr_D():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'same', activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'same', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))
    print(model.summary())
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    return model
epochs = 5 # Change this to 35 for best results
model = model_2CMDr_2CMDr_FDDr_D()
print("Starting evaluation")
model,history = evaluate_model(model,X_train,Y_train,X_val,Y_val,"model_2CMDr_2CMDr_FDDr_D.best",epochs,64)
plot_results(history)

results = add_result("2CMDr_2CMDr_FDDr_D",history,results,0)
results
def vgg_model(n_filters,n_conv):
    visible = Input(shape=(28, 28, 1))
    layers= visible
    for _ in range(n_conv):
        layers = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layers)
    #add max pooling layer
    layers = MaxPooling2D((2,2), strides=(2,2))(layers)
    layers = Flatten()(layers)
    output = Dense(10,activation='softmax')(layers)
    model = Model(inputs=visible, outputs=output)
    print(model.summary())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = vgg_model(64,4)
print("Starting evaluation")
model,history = evaluate_model(model,X_train,Y_train,X_val,Y_val,"model_vgg_64_4.best",epochs,64)
plot_results(history)
results = add_result("vgg_64_4",history,results,1)
results
def model_vgg_batchNormalize():
    model = Sequential()

    model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=3,activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(BatchNormalization())
    model.add(Conv2D(256,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = model_vgg_batchNormalize()
print("Starting evaluation")
model,history = evaluate_model(model,X_train,Y_train,X_val,Y_val,"model_vgg_batchNormalize.best",epochs,64)
plot_results(history)
results = add_result("vgg_batchNormalize",history,results,2)
results
def model_Inception():
    input_img = Input(shape=(28, 28, 1))
    # Layer 1
    layer1_tower_0 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_img)
    layer1_tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_img)
    layer1_tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(layer1_tower_1)
    
    layer1_tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_img)
    layer1_tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(layer1_tower_2)

    layer1_tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    layer1_tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(layer1_tower_3)

    layer1_output = concatenate([layer1_tower_1, layer1_tower_2, layer1_tower_3, layer1_tower_0], axis=1)

    # Layer 2
    layer2_tower_0 = Conv2D(16, (1, 1), padding='same', activation='relu')(layer1_output)
    layer2_tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(layer1_output)
    layer2_tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(layer2_tower_1)

    layer2_tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(layer1_output)
    layer2_tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(layer2_tower_2)

    layer2_tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer1_output)
    layer2_tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(layer2_tower_3)

    layer2_output = concatenate([layer2_tower_1, layer2_tower_2, layer2_tower_3, layer2_tower_0], axis=1)

    # Flatten & Dense
    layer2_output = Flatten()(layer2_output)
    output = Dense(10,activation='softmax')(layer2_output)
    
    inception_Model = Model(inputs=input_img,outputs=output)
    
    print(inception_Model.summary())
    inception_Model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return inception_Model

model = model_Inception()
print("Starting evaluation")
model,history = evaluate_model(model,X_train,Y_train,X_val,Y_val,"model_Inception.best",epochs,64)
plot_results(history)
results = add_result("Inception",history,results,3)
results
def model_Xception():
    inp = Input(shape = (28, 28, 1))
    x = inp
    x = Conv2D(32, (3, 3), strides = 2, padding = "same", activation = "relu")(x)
    x = BatchNormalization(axis = 3)(x)
    x = Dropout(0.4)(x)
    x = Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
    x = BatchNormalization(axis = 3)(x)
    x = Dropout(0.4)(x)

    x1 = DepthwiseConv2D((3, 3), (1, 1), padding = "same", activation = "relu")(x)
    x = BatchNormalization(axis = 3)(x)
    x = Dropout(0.4)(x)
    x1 = DepthwiseConv2D((3, 3), (1, 1), padding = "same", activation = "relu")(x1)
    x = BatchNormalization(axis = 3)(x)
    x = Dropout(0.4)(x)
    x1 = MaxPooling2D((2, 2), strides = 1)(x1)

    x = concatenate([x1, Conv2D(64, (2, 2), strides = 1)(x)])

    x1 = Activation("relu")(x)
    x1 = Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu")(x1)
    x = BatchNormalization(axis = 3)(x)
    x = Dropout(0.4)(x)
    x1 = DepthwiseConv2D((3, 3), strides = 1, padding = "same", activation = "relu")(x1)
    x = BatchNormalization(axis = 3)(x)
    x = Dropout(0.4)(x)
    x1 = DepthwiseConv2D((3, 3), strides = 1, padding = "same")(x1)
    x = BatchNormalization(axis = 3)(x)
    x = Dropout(0.4)(x)
    x1 = MaxPooling2D((2, 2), strides = 1)(x1)

    x = concatenate([x1, Conv2D(256, (2, 2), strides = 1)(x)])

    x = Activation("relu")(x)
    x = Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
    x = BatchNormalization(axis = 3)(x)
    x = Dropout(0.4)(x)
    x = Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
    x = BatchNormalization(axis = 3)(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)

    x = Dense(10, activation = "softmax")(x)

    xception = Model(inp, x)
    print(xception.summary())
    xception.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return xception

model = model_Xception()
print("Starting evaluation")
model,history = evaluate_model(model,X_train,Y_train,X_val,Y_val,"model_Xception.best",epochs,64)
plot_results(history)
results = add_result("Xception",history,results,4)
results
results
## predict results
# model = model_2CMDr_2CMDr_FDDr_D()  #99.214%
# model = vgg_model(64,4) #99.476%
model = model_vgg_batchNormalize()  #99.73%
# model = model_Inception()  #99.21%
# model = model_Xception()  #99.57%

model.load_weights("model_vgg_batchNormalize.best.hdf5")

test_data = input_test.values.reshape(-1,28,28,1)
outputs = model.predict(test_data)

# select the index with the maximum probability
outputs = np.argmax(outputs,axis = 1)

outputs = pd.Series(outputs,name="Label")
outputs
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),outputs],axis = 1)

#submission.to_csv("submision_Inception.csv",index=False)
submission