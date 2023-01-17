#Import of MobileNet repositories
from keras.applications import MobileNet
mobilenet = MobileNet(weights='imagenet') 
#Import Libraries
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
import os
print(os.listdir("../input/guitar-models-data-set/guitars/Guitars"))
%matplotlib inline
#Definition of the additional layers
from keras.layers import Dense,GlobalAveragePooling2D

base_model=MobileNet(weights='imagenet', include_top=False) 
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(6,activation='softmax')(x)
#Definition of the Model
from keras.models import Model

model=Model(inputs=base_model.input,outputs=preds)
model.summary()
#Determining the layers for additional training
for layer in model.layers[:-5]:
    layer.trainable=False
#Importing of training photos of the data set
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator=train_datagen.flow_from_directory(
    '../input/guitar-models-data-set/guitars/Guitars',
    target_size=(224,224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)
#Model Compilation
model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
#Model Training
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.n/train_generator.batch_size,
    epochs=15
)
#Definition of the outputs
print(train_generator.class_indices)
#Test photo selection
#from tkinter import *
#from tkinter.filedialog import askopenfilename

#root = Tk()
#root.update()
#filename = askopenfilename()
#root.destroy()


filename = "../input/test-guitar/silver_sky_photo12.jpg"

#Test Prediction
original = load_img(filename, target_size=(224, 224))
plt.imshow(original)
plt.show()
 
numpy_image = img_to_array(original)
image_batch = np.expand_dims(numpy_image, axis=0)

processed_image = preprocess_input(image_batch.copy())
predictions = model.predict(processed_image)

result = np.argmax(predictions)

result
#Redirecting to the shop categories
import webbrowser
if result == 0:
    webbrowser.open('https://www.andertons.co.uk/guitar-dept/electric-guitars/hollow-semi-hollow-body-guitars?#facet:&productBeginIndex:0&facetLimit:&orderBy:&pageView:grid&minPrice:&maxPrice:&pageSize:&')
elif result == 1:
    webbrowser.open('https://www.andertons.co.uk/guitar-dept/electric-guitars/les-paul?#facet:&productBeginIndex:0&facetLimit:&orderBy:&pageView:grid&minPrice:&maxPrice:&pageSize:&')
elif result == 2:
    webbrowser.open('https://www.andertons.co.uk/sg-guitars?#facet:&productBeginIndex:0&facetLimit:&orderBy:&pageView:grid&minPrice:&maxPrice:&pageSize:&')   
elif result == 3:
    webbrowser.open('https://www.andertons.co.uk/guitar-dept/electric-guitars/hollow-semi-hollow-body-guitars?#facet:&productBeginIndex:0&facetLimit:&orderBy:&pageView:grid&minPrice:&maxPrice:&pageSize:&')
elif result == 4:
    webbrowser.open('https://www.andertons.co.uk/guitar-dept/electric-guitars/stratocaster?#facet:&productBeginIndex:0&facetLimit:&orderBy:&pageView:grid&minPrice:&maxPrice:&pageSize:&')
else:
    webbrowser.open('https://www.andertons.co.uk/guitar-dept/electric-guitars/telecaster?#facet:&productBeginIndex:0&facetLimit:&orderBy:&pageView:grid&minPrice:&maxPrice:&pageSize:&')
