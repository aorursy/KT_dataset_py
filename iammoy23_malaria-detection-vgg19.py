import numpy as np
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from glob import glob
import matplotlib.pyplot as plt

#resize all the image for training
image_size = [224,224,3]

'''Training of the model was done on a smaller dataset link at the end of notebook
    TO make use of the attached public dataset use sklearn for train-test split and ofc GPU/TPU ;)'''

train_path = 'Dataset/Train'
test_path = 'Dataset/Test'
 
''' VGG19 gave the best results to the chosen parameters,
     feel free to tune them and or use other models'''

vgg19 = VGG19(input_shape=image_size, weights='imagenet',include_top=False)
 
vgg19.summary()
#the layers have been already trained on imagenet dataset 
for layer in vgg19.layers:
    layer.trainable = False

folders = glob('Dataset/Train/*')
#Adding last layers for training 
x = Flatten()(vgg19.output)

prediction = Dense(len(folders),activation='softmax')(x)

#create model object 
model = Model(inputs = vgg19.input,outputs= prediction)

model.summary()

model.compile(
    loss='categorical_crossentropy', 
    optimizer = 'adam',  #you can try other optimizers too 
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True )
test_datagen = ImageDataGenerator(rescale=1./255)

training_data = train_datagen.flow_from_directory(train_path,
                                        target_size=(224,224),
                                        batch_size =32,
                                        class_mode = 'categorical')

validation_data = test_datagen.flow_from_directory(test_path,
                                        target_size=(224,224),
                                        batch_size =32,
                                        class_mode = 'categorical')

history = model.fit(
    training_data,
    validation_data=validation_data,
    epochs=2,  #keep the epochs around 10 to get better results 
    steps_per_epoch=len(training_data),
    validation_steps=len(validation_data)
)
plt.plot(history.history['loss'],label = 'Train loss')
plt.plot(history.history['val_loss'],label='Validation loss')
plt.legend()
plt.show()
plt.savefig('Loss')

plt.plot(history.history['accuracy'],label = 'Train accuracy')
plt.plot(history.history['val_accuracy'],label='Validation accuracy')
plt.legend()
plt.show()
plt.savefig('Accuracy')
model.save('VGG_19.h5')

y_pred = model.predict(validation_data)

y_pred = np.argmax(y_pred , axis=1)
print(y_pred.shape)



#testing with custom image 
from tensorflow.keras.models import load_model
model = load_model('VGG_19.h5')

img = image.load_img('Dataset/Test/Uninfected/C3thin_original_IMG_20150608_163002_cell_93.png',target_size=(224,224))
x= image.img_to_array(img)
print(x)
print(x.shape)
x= np.expand_dims(x,axis=0)
x=x/255
image_data = preprocess_input(x)
op = np.argmax(model.predict(image_data),axis=1)

print(op.shape)

if (op[0]==1):
    print('No malaria :D')
else:
    print('Malaria')
