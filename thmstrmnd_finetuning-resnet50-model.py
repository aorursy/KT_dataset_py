import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
resnet50_model = tf.keras.applications.ResNet50()
# Check layers
resnet50_model.summary()
#Check the type of model format
type(resnet50_model)
# Convert to Keras model 
x = resnet50_model.layers[-2].output               # This line removes the final layer of 1000 inputs
predictions = Dense(5, activation='softmax')(x)    # This line adds the output/final layer of 2 nodes/inputs with softmax as activation method 
k_resnet50_model = Model(inputs=resnet50_model.input, outputs=predictions)
k_resnet50_model.summary()
# Freeze all weights. 

for layer in k_resnet50_model.layers [:-1]:
  layer.trainable = False
# Check if the dense layer with input 1 is added. 
k_resnet50_model.summary()
# Compile the new model
k_resnet50_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
train_path= '../input/bkid5/train'
valid_path= '../input/bkid5/valid'
test_path= '../input/bkid5/test'
# Specfiy shuffle=False as argument in test_batches for testing in confusion matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input).flow_from_directory(train_path, target_size=(224, 224), classes=['eggplant', 'garlic', 'ginger', 'onion', 'tomato'], batch_size=100)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input).flow_from_directory(valid_path, target_size=(224, 224), classes=['eggplant', 'garlic', 'ginger', 'onion', 'tomato'], batch_size=20)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input).flow_from_directory(test_path, target_size=(224, 224), classes=['eggplant', 'garlic', 'ginger', 'onion', 'tomato'], batch_size=20)
k_resnet50_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
k_resnet50_model.fit_generator(train_batches, steps_per_epoch=2, validation_data=valid_batches, validation_steps=2, epochs=120, verbose=2)
# Save to the model to .h5
k_resnet50_model.save('ktf2_model_v0.2.h5')