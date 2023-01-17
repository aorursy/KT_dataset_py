from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array,load_img
from glob import glob # To understand how many classes we have
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  
import matplotlib.pyplot as plt

train_path = "/kaggle/input/fruits/fruits-360/Training/"
test_path = "/kaggle/input/fruits/fruits-360/Test/"
img = load_img(test_path +"Salak/140_100.jpg") #"+" means add directory
plt.imshow(img)
plt.axis("off")
plt.show()
x = img_to_array(img)
print(x.shape) #100px x 100px x 3(rgb = color code)
className = glob(train_path + '/*') #go through train path after that add all the files names in train path to ClassName
#First method to learn number of classes when that is list.
num_of_classes = len(className)
print("Number of Classes : ",num_of_classes)
examplefruit = glob(train_path + "Nectarine Flat" +  '/*') #go through train path after that add all the files names in train path to ClassName
#First method to learn number of classes when that is list.
num_of_classesex = len(examplefruit)
print("Number of Classes : ",num_of_classesex) 
#Second method to learn num of classes
classname = pd.DataFrame(className) # We convert our list to DataFrame for using nunique function.
classname.nunique() # We got 131 classes
example = glob(train_path + '/*') #go through train path after that add all the files names in train path to ClassName
#First method to learn number of classes when that is list.
num_of_classes = len(className)
print("Number of Classes : ",num_of_classes)
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = x.shape)) # 3x3 32 Filter
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3))) # 3x3 64 Filter
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024)) #1024 Layer
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_of_classes)) #Output layer >> Output layer size must equal to output(this model = classes)
model.add(Activation("softmax"))
model.compile(loss = "categorical_crossentropy",
             optimizer = "rmsprop",
             metrics = ["accuracy"])
batch_size = 32
train_datagen = ImageDataGenerator(rescale = 1./255, #Rgb 0-255 ,we normalized the data
                  shear_range = 0.3, #Randomly rotated
                  horizontal_flip = True, #Rotated horizontally 
                  zoom_range = 0.3)

test_datagen = ImageDataGenerator(rescale = 1./255)

# -------------------------------------------------------------------------------

train_generator = train_datagen.flow_from_directory(train_path,
                                                   target_size = x.shape[:2],
                                                   batch_size = batch_size,
                                                   color_mode = "rgb",
                                                   class_mode = "categorical")
                                                                                             
#To use this method train method have to describe like we did after that there should be classes then the pics.
#The right directory for this method must be like that. 

test_generator = test_datagen.flow_from_directory(test_path,
                                                   target_size = x.shape[:2],
                                                   batch_size = batch_size,
                                                   color_mode = "rgb",
                                                   class_mode = "categorical")
hist = model.fit_generator(
    generator = train_generator, 
    steps_per_epoch = 1600// batch_size,
    epochs = 75,
    validation_data = test_generator,
    validation_steps = 800 // batch_size)
model.save_weights("trial.h5") # Saving our results

print(hist.history.keys())
plt.plot(hist.history["loss"], label ="Train Loss")
plt.plot(hist.history["val_loss"], label ="Test Loss")
plt.legend()
plt.show()

#-----------------------------------------------------------------------

print(hist.history.keys())
plt.plot(hist.history["accuracy"], label ="Train Accuracy")
plt.plot(hist.history["val_accuracy"], label ="Test Accuracy")
plt.legend()
plt.show()
import json 
with open("trial.json","w") as f:
    json.dump(hist.history,f)
import codecs 
with codecs.open("./trial.json","r",encoding = "utf-8") as f:
    h = json.loads(f.read())
# print(h.keys())
# plt.plot(h["loss"], label ="Train Loss")
# plt.plot(h["val_loss"], label ="Test Loss")
# plt.legend()
# plt.show()
# 
# #-----------------------------------------------------------------------
# 
# print(h.keys())
# plt.plot(h["accuracy"], label ="Train Accuracy")
# plt.plot(h["val_accuracy"], label ="Test Accuracy")
# plt.legend()
# plt.show()