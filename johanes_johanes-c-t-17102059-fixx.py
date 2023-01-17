import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
from IPython.display import Image
import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/kaggle/input/dauntomat/train',
                                                 target_size = (64, 64),
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('/kaggle/input/dauntomat/test',
                                            target_size = (64, 64),
                                            class_mode = 'binary')
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
#to improve accuracy
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#to improve accuracy
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history=cnn.fit(
        x=training_set,
        epochs=40,
        validation_data=test_set,
        validation_steps=len(test_set))
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
import numpy as np
from keras.preprocessing import image
def predict_img(img_path):
    test_image=image.load_img(img_path,target_size=(64,64,3))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=cnn.predict(test_image)
    training_set.class_indices
    if result[0][0]==1:
        prediction ='Daun_tomato_sehat'
    else:
        prediction='Daun_tomato_sakit'
    print(prediction)
predict_img(img_path='/kaggle/input/dauntomat/train/sakit/0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B 8389.JPG')
Image(filename='/kaggle/input/dauntomat/train/sakit/0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B 8389.JPG')
predict_img(img_path='/kaggle/input/dauntomat/train/sehat/sehat (1).JPG')
Image(filename='/kaggle/input/dauntomat/train/sehat/sehat (1).JPG')
predict_img(img_path='/kaggle/input/dauntomat/train/sehat/sehat (10).JPG')
Image(filename='/kaggle/input/dauntomat/train/sehat/sehat (10).JPG')
predict_img(img_path='/kaggle/input/dauntomat/train/sakit/0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG')
Image(filename='/kaggle/input/dauntomat/train/sakit/0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG')