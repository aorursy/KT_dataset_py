import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,MaxPooling2D
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.utils import np_utils
from sklearn.utils import shuffle
import glob
import matplotlib.pyplot as plt
from keras.models import model_from_json
# creating dictionary to convert labels of fruits into numeric label which cna be used in our deep learning models
keys =0
fruitname_label_map = {}
# accessing names of fruits from the directory we have 
for sub_folder in glob.glob('/kaggle/input/fruits/fruits-360/Training/*') :
    name = sub_folder.split('/')[-1]
    # creating dictionary with the help of names we obtain
    fruitname_label_map[name] = keys
    keys += 1
    
fruitname_label_map
# reversing above dictionary to convert labels to name of fruit
fruitlabel_name_map = {v: k for k, v in fruitname_label_map.items()}
fruitlabel_name_map
# rotating images to 20degree rescaling images value from range 0-255 to 0-1 and other augmentation process

img_generator = ImageDataGenerator(rotation_range=20, rescale=1./255, horizontal_flip=True, shear_range=0.1)
# shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)

# using flow_from _directory which best suit with the type folders we have in our dataset it labels the class according to the folders we have
# and we don't need to wory about labaling data on our own.
train = img_generator.flow_from_directory(directory="/kaggle/input/fruits/fruits-360/Training/", 
                                                    target_size=(100, 100), # size of each image
                                                    batch_size=32, # creating batches as it will easy computer processing
                                                    class_mode="categorical",
                                                    shuffle=True, # random pick the data
                                                    seed=0   # seed set so that random value fixed for different computers
                                                   )

# performing sam for test data
test_gen = ImageDataGenerator(rescale=1./255)
test = test_gen.flow_from_directory(directory="/kaggle/input/fruits/fruits-360/Test/",
                                     target_size=(100, 100),
                                     color_mode="rgb",
                                     batch_size=32,
                                     class_mode="categorical",
                                     shuffle=True,
                                     seed=0
                                    )
# deep learning model 
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(5,5),activation="relu",input_shape=(100,100,3)))
model.add(MaxPool2D(strides=2, pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(5,5),activation="relu"))
model.add(MaxPool2D(strides=2, pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(5,5),activation="relu"))
model.add(MaxPool2D(strides=2, pool_size=(2,2)))
model.add(Conv2D(filters=128,kernel_size=(5,5),activation="relu"))
model.add(MaxPool2D(strides=2, pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dense(125,activation="softmax"))
model.summary()
checkpoint = ModelCheckpoint("fruits_classifier.h5", monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(), metrics = ['acc'])
# training deep network using fit_generate as we have augmented the data callback is used to monitor the loss and accuracy of the model
# in each step / epoch
trained_model = model.fit_generator(train, epochs=15,shuffle=False,validation_data=test,callbacks=[checkpoint])
# Plots visualize the models we have trained above 
ep = range(1,len(trained_model.history['acc'])+1)
plt.plot(ep, trained_model.history['acc'], label = 'Training')
plt.plot(ep, trained_model.history['val_acc'], label='Validation')
plt.title('Accuracy Plot')
plt.legend()
plt.savefig('Accuracy.jpg')
classifier_json = model.to_json()
with open("fruit_model.json", "w") as json_file:
    json_file.write(classifier_json)
model.save_weights("fruit_model_weights.h5")
print("Model Saved.....")
json_file = open('/kaggle/working/fruit_model.json', 'r')
model_json = json_file.read()
json_file.close()
fruit_clf = model_from_json(model_json)

fruit_clf.load_weights("/kaggle/working/fruit_model_weights.h5")
print("Model loaded.....")

xtest,ytest=test.next()  # get x and y to test on
ypred = fruit_clf.predict(xtest)   # model gives preidction

pred_result = ypred.argmax(axis=1)   # taking class with max value
test_result = ytest.argmax(axis=1)

for i in range(32):
    print("Actual o/p:{} ---- Predicted o/p:{}".format(fruitlabel_name_map[test_result[i]],fruitlabel_name_map[pred_result[i]]))
