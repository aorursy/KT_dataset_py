directory = "../input/plantvillage-dataset/plantvillage dataset/segmented"

diseases = ["Strawberry___healthy", "Strawberry___Leaf_scorch"]

batch_size = 32

target_size = (299, 299)

keras_model = "strawberry_model.h5"

converted_model = "strawberry_converted.tflite"

epochs = 30
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.2)



train_generator = datagen.flow_from_directory(

    directory,

    target_size=target_size,

    classes=diseases,

    batch_size=batch_size,

    subset='training')



validation_generator = datagen.flow_from_directory(

    directory,

    target_size=target_size,

    classes=diseases,

    batch_size=batch_size,

    subset='validation')
from keras.applications.inception_v3 import InceptionV3

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D



base_model = InceptionV3(weights='imagenet', include_top=False)



x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

predictions = Dense(len(diseases), activation='softmax')(x)

                    

model = Model(inputs=base_model.input, outputs=predictions)



model.summary()



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit_generator(

    train_generator,

    validation_data=validation_generator,

    steps_per_epoch=train_generator.samples//batch_size,

    epochs=epochs,

    validation_steps=validation_generator.samples//batch_size)



model.save(keras_model)
import matplotlib.pyplot as plt



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
import tensorflow as tf



converter = tf.lite.TFLiteConverter.from_keras_model_file(

    keras_model, 

    input_shapes={'input_1': [1, 299, 299, 3]}, 

    input_arrays=['input_1'], 

    output_arrays=['dense_2/Softmax'])



converter.post_training_quantize=True

tflite_model = converter.convert()

open(converted_model, "wb").write(tflite_model)
import tensorflow as tf



interpreter = tf.lite.Interpreter(model_path=converted_model)

interpreter.allocate_tensors()



# Print input shape and type

print(interpreter.get_input_details()[0]['shape'])

print(interpreter.get_input_details()[0]['dtype'])



# Print output shape and type

print(interpreter.get_output_details()[0]['shape'])

print(interpreter.get_output_details()[0]['dtype'])