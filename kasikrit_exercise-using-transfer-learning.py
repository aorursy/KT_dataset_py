from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)
img_path = '../input/dogs-gone-sideways/images/train/upright/dog.11102.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
!ls ../input/dogs-gone-sideways/images/train/upright/dog.11102.jpg
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

# num_classes is the number of categories your model chooses between for each prediction
num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# The value below is either True or False.  If you choose the wrong answer, your modeling results
# won't be very good.  Recall whether the first layer should be trained/changed or not.
my_new_model.layers[0].trainable = False
# We are calling the compile command for some python object. 
# Which python object is being compiled? Fill in the answer so the compile command works.
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocess_input)

train_generator = data_generator.flow_from_directory(
        directory = '../input/dogs-gone-sideways/images/train',
        target_size=(image_size, image_size),
        batch_size=10,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        directory = '../input/dogs-gone-sideways/images/val',
        target_size=(image_size, image_size),
        class_mode='categorical')

#my_new_model.fit(train_generator, seed=10)
#my_new_model.fit(validation_generator, seed=10)

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        validation_data=validation_generator,
        validation_steps=1)
        #featurewise_center=True)