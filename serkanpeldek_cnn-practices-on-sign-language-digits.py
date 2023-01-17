import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras import layers

from keras import optimizers



from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))
X=np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy")

y=np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy")

print("The dataset loaded...")
def show_model_history(modelHistory, model_name):

    history=pd.DataFrame()

    history["Train Loss"]=modelHistory.history['loss']

    history["Validation Loss"]=modelHistory.history['val_loss']

    history["Train Accuracy"]=modelHistory.history['accuracy']

    history["Validation Accuracy"]=modelHistory.history['val_accuracy']

    

    fig, axarr=plt.subplots(nrows=2, ncols=1 ,figsize=(12,8))

    axarr[0].set_title("History of Loss in Train and Validation Datasets")

    history[["Train Loss", "Validation Loss"]].plot(ax=axarr[0])

    axarr[1].set_title("History of Accuracy in Train and Validation Datasets")

    history[["Train Accuracy", "Validation Accuracy"]].plot(ax=axarr[1]) 

    plt.suptitle(" Convulutional Model {} Loss and Accuracy in Train and Validation Datasets".format(model_name))

    plt.show()
from keras.callbacks import EarlyStopping

def split_dataset(X, y, test_size=0.3, random_state=42):

    X_conv=X.reshape(X.shape[0], X.shape[1], X.shape[2],1)

    

    



    return train_test_split(X_conv,y, stratify=y,test_size=test_size,random_state=random_state)



def evaluate_conv_model(model, model_name, X, y, epochs=100,

                        optimizer=optimizers.RMSprop(lr=0.0001), callbacks=None):

    print("[INFO]:Convolutional Model {} created...".format(model_name))

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    

    

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    print("[INFO]:Convolutional Model {} compiled...".format(model_name))

    

    print("[INFO]:Convolutional Model {} training....".format(model_name))

    earlyStopping = EarlyStopping(monitor = 'val_loss', patience=20, verbose = 1) 

    if callbacks is None:

        callbacks = [earlyStopping]

    modelHistory=model.fit(X_train, y_train, 

             validation_data=(X_test, y_test),

             callbacks=callbacks,

             epochs=epochs,

             verbose=0)

    print("[INFO]:Convolutional Model {} trained....".format(model_name))



    test_scores=model.evaluate(X_test, y_test, verbose=0)

    train_scores=model.evaluate(X_train, y_train, verbose=0)

    print("[INFO]:Train Accuracy:{:.3f}".format(train_scores[1]))

    print("[INFO]:Validation Accuracy:{:.3f}".format(test_scores[1]))

    

    show_model_history(modelHistory=modelHistory, model_name=model_name)

    return model
def decode_OneHotEncoding(label):

    label_new=list()

    for target in label:

        label_new.append(np.argmax(target))

    label=np.array(label_new)

    

    return label

def correct_mismatches(label):

    label_map={0:9,1:0, 2:7, 3:6, 4:1, 5:8, 6:4, 7:3, 8:2, 9:5}

    label_new=list()

    for s in label:

        label_new.append(label_map[s])

    label_new=np.array(label_new)

    

    return label_new

    

def show_image_classes(image, label, n=10):

    label=decode_OneHotEncoding(label)

    label=correct_mismatches(label)

    fig, axarr=plt.subplots(nrows=n, ncols=n, figsize=(18, 18))

    axarr=axarr.flatten()

    plt_id=0

    start_index=0

    for sign in range(10):

        sign_indexes=np.where(label==sign)[0]

        for i in range(n):



            image_index=sign_indexes[i]

            axarr[plt_id].imshow(image[image_index], cmap='gray')

            axarr[plt_id].set_xticks([])

            axarr[plt_id].set_yticks([])

            axarr[plt_id].set_title("Sign :{}".format(sign))

            plt_id=plt_id+1

    plt.suptitle("{} Sample for Each Classes".format(n))

    plt.show()
number_of_pixels=X.shape[1]*X.shape[2]

number_of_classes=y.shape[1]

print(20*"*", "SUMMARY of the DATASET",20*"*")

print("an image size:{}x{}".format(X.shape[1], X.shape[2]))

print("number of pixels:",number_of_pixels)

print("number of classes:",number_of_classes)



y_decoded=decode_OneHotEncoding(y.copy())

sample_per_class=np.unique(y_decoded, return_counts=True)

print("Number of Samples:{}".format(X.shape[0]))

for sign, number_of_sample in zip(sample_per_class[0], sample_per_class[1]):

    print("  {} sign has {} samples.".format(sign, number_of_sample))

print(65*"*")
show_image_classes(image=X, label=y.copy())
def build_conv_model_1():

    model=Sequential()

    

    model.add(layers.Conv2D(64, kernel_size=(3,3),

                           padding="same",

                           activation="relu", 

                           input_shape=(64, 64,1)))

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation="relu"))

    model.add(layers.Dense(number_of_classes, activation="softmax"))

        

    return model
trained_models=dict()

model=build_conv_model_1()

trained_model_1=evaluate_conv_model(model=model, model_name=1, X=X, y=y)



#Will be used for serialization

trained_models["model_1"]=(trained_model_1,optimizers.RMSprop(lr=0.0001) )
def build_conv_model_2():

    model = Sequential()

    model.add(layers.Convolution2D(64, (3, 3), activation='relu', padding="same", input_shape=(64, 64, 1)))

    model.add(layers.MaxPooling2D((2, 2)))

       

    model.add(layers.Convolution2D(64, (3, 3), activation='relu', padding="same"))

    model.add(layers.MaxPooling2D((2, 2)))

        

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

      

    return model
model=build_conv_model_2()

trained_model_2=evaluate_conv_model(model=model, model_name=2, X=X, y=y)



#Will be used for serialization

trained_models["model_2"]=(trained_model_2,optimizers.RMSprop(lr=0.0001) )
def build_conv_model_3():

    model = Sequential()

    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))

    model.add(layers.MaxPooling2D((2, 2)))

    

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

           

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

        

    return model
model=build_conv_model_3()

trained_model_3=evaluate_conv_model(model=model, model_name=3, X=X, y=y)

#Will be used for serialization

trained_models["model_3"]=(trained_model_3,optimizers.RMSprop(lr=0.0001) )
def build_conv_model_4():

    model = Sequential()

    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

    

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

    

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

       

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))



    return model
model=build_conv_model_4()

trained_model_4=evaluate_conv_model(model=model, model_name=4, X=X, y=y)



#Will be used for serialization

trained_models["model_4"]=(trained_model_4,optimizers.RMSprop(lr=0.0001) )
def build_conv_model_5():

    model = Sequential()

    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

    

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

    

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

    

    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

       

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

        

    return model
model=build_conv_model_5()

trained_model_5=evaluate_conv_model(model=model, model_name=5, X=X, y=y)

#Will be used for serialization

trained_models["model_5"]=(trained_model_5,optimizers.RMSprop(lr=0.0001) )
def build_conv_model_6():

    model = Sequential()

    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))

    model.add(layers.MaxPooling2D((2, 2)))



    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))



    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))



    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    

    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

        

    return model
model=build_conv_model_6()

trained_model_6=evaluate_conv_model(model=model, model_name=6, X=X, y=y)

#Will be used for serialization

trained_models["model_6"]=(trained_model_6,optimizers.RMSprop(lr=0.0001) )
def build_conv_model_7():

    model = Sequential()

    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())



    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())



    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())



    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    

    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

        

    return model
model=build_conv_model_7()

trained_model_7=evaluate_conv_model(model=model, model_name=7, X=X, y=y)

#Will be used for serialization

trained_models["model_7"]=(trained_model_7,optimizers.RMSprop(lr=0.0001) )
def build_conv_model_8():

    model = Sequential()

    model.add(layers.Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.25))



    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.25))



    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.25))



    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.25))

    

    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

        

    return model
model=build_conv_model_8()

trained_model_8_1=evaluate_conv_model(model=model, model_name=8, X=X, y=y)

#Will be used for serialization

trained_models["model_8_1"]=(trained_model_8_1,optimizers.RMSprop(lr=0.0001) )
model=build_conv_model_8()

optimizer=optimizers.RMSprop(lr=1e-4)# our default optimizer in evaluate_conv_model function

trained_model_8_2=evaluate_conv_model(model=model, model_name=8, X=X, y=y,optimizer=optimizer, epochs=200)



#Will be used for serialization

trained_models["model_8_2"]=(trained_model_8_2,optimizer )
model=build_conv_model_8()

optimizer=optimizers.Adam(lr=0.001)

trained_model_8_3=evaluate_conv_model(model=model, model_name=8, X=X, y=y, optimizer=optimizer, epochs=250)

#Will be used for serialization

trained_models["model_8_3"]=(trained_model_8_3,optimizer )
model=build_conv_model_8()

optimizer_8_4=optimizers.Adam(lr=0.001)

trained_model_8_4=evaluate_conv_model(model=model, model_name=8, X=X, y=y, optimizer=optimizer_8_4, epochs=300)

#Will be used for serialization

trained_models["model_8_4"]=(trained_model_8_4,optimizer )
from keras.models import model_from_json, model_from_yaml

class Save:

    @classmethod

    def save(self, model, model_file_name, hdf5_file_name):

        if "json" in model_file_name:

            model_format=model.to_json()

        else:

            model_format=model.to_yaml()

        with open(model_file_name, "w") as file:

            file.write(model_format)

        model.save_weights(hdf5_file_name)

class Load:

    @classmethod

    def load(self, model_file_name, hdf5_file_name):

        format_file=open(model_file_name)

        loaded_file=format_file.read()

        format_file.close()

        if "json" in model_file_name:

            model=model_from_json(loaded_file)

        else:

            model=model_from_yaml(loaded_file)

        model.load_weights(hdf5_file_name)

        

        return model

class YAML:

    def __init__(self):

        self.yaml_file_name=None

        self.hdf5_file_name=None

    

    def save(self, model, model_name):

        self.yaml_file_name=model_name+".yaml"

        self.hdf5_file_name=model_name+"_yaml.hdf5"

        Save.save(model,

                  self.yaml_file_name,

                  self.hdf5_file_name)        

        

        print("YAML model and HDF5 weights saved to disk...")

        print("Model file name:{}".format(self.yaml_file_name))

        print("Weights file name:{}".format(self.hdf5_file_name))

    

    def load(self):

              

        print("YAML model and HDF5 loaded from disk...")

        return  Load.load(self.yaml_file_name, self.hdf5_file_name)

        

class JSON:

    def __init__(self):

        self.json_file_name=None

        self.hdf5_file_name=None

        

    def save(self, model, model_name):

        self.json_file_name=model_name+".json"

        self.hdf5_file_name=model_name+"_json.hdf5"

        Save.save(model,

                  self.json_file_name,

                  self.hdf5_file_name)

        

        print("JSON model and HDF5 weights saved to disk...")

        print("Model file name:{}".format(self.json_file_name))

        print("Weights file name:{}".format(self.hdf5_file_name))

        

    

    def load(self):



        print("JSON model and HDF5 weights loaded from disk...")

        return Load.load(self.json_file_name, self.hdf5_file_name)

        

class Serialization():

    def __init__(self, file_format):

        assert file_format in ["json", "yaml"], "There is no such a serialization format"

        self.file_format=file_format

        if self.file_format=="json":

            self.serialization_type=JSON()

        else:

            self.serialization_type=YAML()

    

    def save(self, model, model_name="model"):

        self.serialization_type.save(model, model_name)

    def load(self):

        return self.serialization_type.load()
def test_serialization(trained_models, format_type):

    for model_name, model_pack in trained_models.items():

        model, optimizer=model_pack

        serialization=Serialization(format_type)

        serialization.save(model, model_name=model_name)



        loaded_model=serialization.load()

        X_train, X_test, y_train, y_test=split_dataset(X, y)





        #optimizer=optimizers.RMSprop(lr=0.0001)

        loaded_model.compile(loss="categorical_crossentropy", 

                             optimizer=optimizer,

                             metrics=["accuracy"])



        train_scores = loaded_model.evaluate(X_train, y_train, verbose=0)

        test_scores  = loaded_model.evaluate(X_test, y_test, verbose=0)

        print("Train accuracy:{:.3f}".format(train_scores[1]))

        print("Test accuracy:{:.3f}".format(test_scores[1]))

        print()
test_serialization(trained_models, format_type="json")
import yaml

yaml.warnings({'YAMLLoadWarning': False})

test_serialization(trained_models, format_type="yaml")
print("Created files in Kaggle working directory:")

for file in sorted(os.listdir("../working")):

    if "ipynb" in file:

        continue

    print(file)